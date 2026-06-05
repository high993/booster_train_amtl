from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from amtl.objective_metadata import get_objective_metadata

np = None
plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze logged AMTL/PPO objective values for Pareto-style tradeoffs.")
    parser.add_argument("--csv", type=Path, default=None, help="Path to pareto_objectives.csv.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Training log directory containing pareto_objectives.csv.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for analysis outputs.")
    parser.add_argument(
        "--max-pairwise-objectives",
        type=int,
        default=6,
        help="Maximum number of objectives to include in the pairwise tradeoff matrix.",
    )
    args = parser.parse_args()
    if args.csv is None and args.log_dir is None:
        parser.error("Provide either --csv or --log-dir.")
    return args


def resolve_csv_path(args: argparse.Namespace) -> Path:
    if args.csv is not None:
        return args.csv
    return args.log_dir / "pareto_objectives.csv"


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def deduplicate_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    deduped: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in rows:
        iteration = int(row["iteration"])
        checkpoint_path = row["checkpoint_path"].strip()
        objective_name = row["objective_name"]
        metadata = get_objective_metadata(objective_name)
        deduped[(iteration, checkpoint_path, objective_name)] = {
            "iteration": iteration,
            "checkpoint_path": checkpoint_path,
            "objective_name": objective_name,
            "raw_value": to_float(row["raw_value"]),
            "weighted_value": to_float(row["weighted_value"]),
            "group_name": row["group_name"] or metadata.group_name,
            "is_penalty": metadata.is_penalty,
        }
    return sorted(deduped.values(), key=lambda row: (row["iteration"], row["checkpoint_path"], row["objective_name"]))


def build_iteration_table(rows: list[dict[str, object]]) -> tuple[list[int], list[str], np.ndarray, list[str]]:
    objectives = sorted({str(row["objective_name"]) for row in rows})
    iterations = sorted({int(row["iteration"]) for row in rows})
    groups_by_objective = []
    group_lookup = {str(row["objective_name"]): str(row["group_name"]) for row in rows}
    matrix = np.full((len(iterations), len(objectives)), np.nan, dtype=np.float64)
    row_lookup = {(int(row["iteration"]), str(row["objective_name"])): row for row in rows}

    for i, iteration in enumerate(iterations):
        for j, objective in enumerate(objectives):
            row = row_lookup.get((iteration, objective))
            if row is None:
                continue
            raw_value = float(row["raw_value"])
            matrix[i, j] = -raw_value if bool(row["is_penalty"]) else raw_value

    for objective in objectives:
        groups_by_objective.append(group_lookup[objective])
    return iterations, objectives, matrix, groups_by_objective


def build_checkpoint_table(rows: list[dict[str, object]]) -> tuple[list[int], list[str], list[str], np.ndarray, list[str]]:
    checkpoint_rows = [row for row in rows if str(row["checkpoint_path"])]
    if not checkpoint_rows:
        checkpoint_rows = rows

    objectives = sorted({str(row["objective_name"]) for row in checkpoint_rows})
    checkpoints = sorted({(int(row["iteration"]), str(row["checkpoint_path"])) for row in checkpoint_rows})
    group_lookup = {str(row["objective_name"]): str(row["group_name"]) for row in checkpoint_rows}
    matrix = np.full((len(checkpoints), len(objectives)), np.nan, dtype=np.float64)
    row_lookup = {
        (int(row["iteration"]), str(row["checkpoint_path"]), str(row["objective_name"])): row for row in checkpoint_rows
    }

    checkpoint_paths = []
    iterations = []
    for i, (iteration, checkpoint_path) in enumerate(checkpoints):
        iterations.append(iteration)
        checkpoint_paths.append(checkpoint_path)
        for j, objective in enumerate(objectives):
            row = row_lookup.get((iteration, checkpoint_path, objective))
            if row is None:
                continue
            raw_value = float(row["raw_value"])
            matrix[i, j] = -raw_value if bool(row["is_penalty"]) else raw_value

    groups_by_objective = [group_lookup[objective] for objective in objectives]
    return iterations, checkpoint_paths, objectives, matrix, groups_by_objective


def forward_fill_nan(matrix: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    for col in range(filled.shape[1]):
        last_value = np.nan
        for row in range(filled.shape[0]):
            if np.isnan(filled[row, col]):
                if not np.isnan(last_value):
                    filled[row, col] = last_value
            else:
                last_value = filled[row, col]
    return filled


def keep_complete_rows(iterations: list[int], labels: list[str], matrix: np.ndarray) -> tuple[list[int], list[str], np.ndarray]:
    valid_mask = ~np.isnan(matrix).any(axis=1)
    filtered_iterations = [iteration for iteration, valid in zip(iterations, valid_mask) if valid]
    filtered_labels = [label for label, valid in zip(labels, valid_mask) if valid]
    return filtered_iterations, filtered_labels, matrix[valid_mask]


def min_max_normalize(matrix: np.ndarray) -> np.ndarray:
    mins = np.nanmin(matrix, axis=0)
    maxs = np.nanmax(matrix, axis=0)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    return (matrix - mins) / denom


def standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (matrix - mean) / std


def nondominated_mask(matrix: np.ndarray) -> np.ndarray:
    num_points = matrix.shape[0]
    efficient = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        dominated = np.all(matrix >= matrix[i], axis=1) & np.any(matrix > matrix[i], axis=1)
        dominated[i] = False
        if np.any(dominated):
            efficient[i] = False
    return efficient


def plot_per_objective_curves(iterations: list[int], objectives: list[str], matrix: np.ndarray, output_dir: Path) -> None:
    cols = 3
    rows = math.ceil(len(objectives) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)
    for idx, objective in enumerate(objectives):
        axis = axes[idx // cols][idx % cols]
        axis.plot(iterations, matrix[:, idx], linewidth=1.8)
        axis.set_title(objective)
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Sign-corrected raw value")
        axis.grid(alpha=0.3)
    for idx in range(len(objectives), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "per_objective_training_curves.png", dpi=200)
    plt.close(fig)


def plot_grouped_curves(
    iterations: list[int], objectives: list[str], groups: list[str], normalized_matrix: np.ndarray, output_dir: Path
) -> None:
    grouped = defaultdict(list)
    for idx, objective in enumerate(objectives):
        grouped[groups[idx]].append(idx)

    fig, axis = plt.subplots(figsize=(10, 5))
    for group_name, indexes in sorted(grouped.items()):
        axis.plot(iterations, normalized_matrix[:, indexes].mean(axis=1), label=group_name, linewidth=2.0)
    axis.set_title("Grouped outer-objective curves")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Mean normalized objective value")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_outer_objective_curves.png", dpi=200)
    plt.close(fig)


def select_pairwise_objectives(objectives: list[str], matrix: np.ndarray, max_objectives: int) -> list[int]:
    if len(objectives) <= max_objectives:
        return list(range(len(objectives)))
    variances = np.nanvar(matrix, axis=0)
    selected = np.argsort(variances)[::-1][:max_objectives]
    return sorted(selected.tolist())


def plot_pairwise_tradeoffs(
    checkpoint_iterations: list[int],
    objectives: list[str],
    matrix: np.ndarray,
    efficient_mask: np.ndarray,
    output_dir: Path,
    max_objectives: int,
) -> None:
    selected = select_pairwise_objectives(objectives, matrix, max_objectives)
    selected_objectives = [objectives[idx] for idx in selected]
    selected_matrix = matrix[:, selected]
    dim = len(selected_objectives)
    fig, axes = plt.subplots(dim, dim, figsize=(3.2 * dim, 3.2 * dim), squeeze=False)
    for row in range(dim):
        for col in range(dim):
            axis = axes[row][col]
            if row == col:
                axis.hist(selected_matrix[:, col], bins=20, color="#4C78A8")
            elif row > col:
                axis.scatter(selected_matrix[:, col], selected_matrix[:, row], s=12, alpha=0.55, label="Checkpoint")
                axis.scatter(
                    selected_matrix[efficient_mask, col],
                    selected_matrix[efficient_mask, row],
                    s=22,
                    color="#E45756",
                    alpha=0.9,
                    label="Pareto-efficient" if row == 1 and col == 0 else None,
                )
            else:
                axis.axis("off")
            if row == dim - 1:
                axis.set_xlabel(selected_objectives[col])
            if col == 0 and row > 0:
                axis.set_ylabel(selected_objectives[row])
            axis.grid(alpha=0.2)
    fig.suptitle("Pairwise checkpoint tradeoffs", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "pairwise_objective_tradeoffs.png", dpi=200)
    plt.close(fig)


def compute_pca(matrix: np.ndarray) -> np.ndarray:
    standardized = standardize(matrix)
    centered = standardized - standardized.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


def plot_pca_trajectory(
    checkpoint_iterations: list[int],
    projection: np.ndarray,
    efficient_mask: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.plot(projection[:, 0], projection[:, 1], color="#9C755F", alpha=0.6, linewidth=1.5)
    scatter = axis.scatter(projection[:, 0], projection[:, 1], c=checkpoint_iterations, cmap="viridis", s=28)
    axis.scatter(
        projection[efficient_mask, 0],
        projection[efficient_mask, 1],
        color="#E45756",
        edgecolors="black",
        linewidths=0.4,
        s=58,
        label="Pareto-efficient",
    )
    axis.set_title("PCA trajectory of checkpoints")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.colorbar(scatter, ax=axis, label="Iteration")
    fig.tight_layout()
    fig.savefig(output_dir / "checkpoint_pca_trajectory.png", dpi=200)
    plt.close(fig)


def save_processed_rows(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_path = output_dir / "processed_objectives.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "iteration",
                "checkpoint_path",
                "objective_name",
                "group_name",
                "raw_value",
                "weighted_value",
                "is_penalty",
                "higher_is_better_raw",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "higher_is_better_raw": -float(row["raw_value"]) if bool(row["is_penalty"]) else float(row["raw_value"]),
                }
            )


def save_pareto_frontier(
    checkpoint_iterations: list[int],
    checkpoint_paths: list[str],
    objectives: list[str],
    matrix: np.ndarray,
    efficient_mask: np.ndarray,
    output_dir: Path,
) -> None:
    np.save(output_dir / "checkpoint_objective_matrix.npy", matrix)
    with open(output_dir / "pareto_efficient_checkpoints.csv", "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["iteration", "checkpoint_path"] + objectives
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, efficient in enumerate(efficient_mask):
            if not efficient:
                continue
            row = {
                "iteration": checkpoint_iterations[index],
                "checkpoint_path": checkpoint_paths[index],
            }
            row.update({objective: matrix[index, col] for col, objective in enumerate(objectives)})
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    global np
    global plt
    try:
        import numpy as np_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "numpy is required for analyze_pareto_objectives.py. Install it in the training environment and rerun "
            "the script."
        ) from exc
    np = np_module
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for analyze_pareto_objectives.py. Install it in the training environment "
            "and rerun the script."
        ) from exc
    plt = plt_module

    csv_path = resolve_csv_path(args)
    output_dir = args.output_dir or csv_path.parent / "pareto_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = deduplicate_rows(load_rows(csv_path))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")

    save_processed_rows(rows, output_dir)

    iterations, objectives, iteration_matrix, iteration_groups = build_iteration_table(rows)
    iteration_matrix = forward_fill_nan(iteration_matrix)
    plot_per_objective_curves(iterations, objectives, iteration_matrix, output_dir)
    plot_grouped_curves(iterations, objectives, iteration_groups, min_max_normalize(iteration_matrix), output_dir)

    checkpoint_iterations, checkpoint_paths, checkpoint_objectives, checkpoint_matrix, checkpoint_groups = build_checkpoint_table(rows)
    checkpoint_iterations, checkpoint_paths, checkpoint_matrix = keep_complete_rows(
        checkpoint_iterations, checkpoint_paths, checkpoint_matrix
    )
    if checkpoint_matrix.shape[0] == 0:
        raise ValueError("No complete checkpoint objective vectors were found in the logged data.")

    normalized_checkpoint_matrix = min_max_normalize(checkpoint_matrix)
    efficient_mask = nondominated_mask(normalized_checkpoint_matrix)

    plot_pairwise_tradeoffs(
        checkpoint_iterations,
        checkpoint_objectives,
        checkpoint_matrix,
        efficient_mask,
        output_dir,
        args.max_pairwise_objectives,
    )
    plot_pca_trajectory(
        checkpoint_iterations,
        compute_pca(normalized_checkpoint_matrix),
        efficient_mask,
        output_dir,
    )
    save_pareto_frontier(
        checkpoint_iterations,
        checkpoint_paths,
        checkpoint_objectives,
        checkpoint_matrix,
        efficient_mask,
        output_dir,
    )

    print(f"Loaded objective log: {csv_path}")
    print(f"Saved Pareto analysis outputs to: {output_dir}")
    print(f"Checkpoint vectors analyzed: {checkpoint_matrix.shape[0]}")
    print(f"Objectives analyzed: {checkpoint_matrix.shape[1]}")
    print(f"Pareto-efficient checkpoints: {int(efficient_mask.sum())}")


if __name__ == "__main__":
    main()
