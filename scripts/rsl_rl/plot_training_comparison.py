#!/usr/bin/env python3
"""Build smoothed TensorBoard comparisons for one or more training runs."""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util.tensor_util import make_ndarray


REWARD_COMPONENTS = (
    ("Episode_Reward/action_rate_l2", "action rate l2"),
    ("Episode_Reward/joint_limit", "joint limit"),
    ("Episode_Reward/motion_body_ang_vel", "motion body ang vel"),
    ("Episode_Reward/motion_body_lin_vel", "motion body lin vel"),
    ("Episode_Reward/motion_body_ori", "motion body ori"),
    ("Episode_Reward/motion_body_pos", "motion body pos"),
    ("Episode_Reward/motion_foot_ori", "motion foot ori"),
    ("Episode_Reward/motion_foot_pos", "motion foot pos"),
    ("Episode_Reward/motion_global_anchor_ori", "motion global anchor ori"),
    ("Episode_Reward/motion_global_anchor_pos", "motion global anchor pos"),
    ("Episode_Reward/motion_hand_ori", "motion hand ori"),
    ("Episode_Reward/motion_hand_pos", "motion hand pos"),
    ("Episode_Reward/motion_trunk_ang_vel", "motion trunk ang vel"),
    ("Episode_Reward/motion_trunk_ori", "motion trunk ori"),
    ("Episode_Reward/motion_trunk_pos", "motion trunk pos"),
    ("Episode_Reward/undesired_contacts", "undesired contacts"),
)

TERMINATIONS = (
    ("Episode_Termination/anchor_ori", "anchor ori"),
    ("Episode_Termination/anchor_pos", "anchor pos"),
    ("Episode_Termination/ee_body_pos", "ee body pos"),
    ("Episode_Termination/time_out", "time out"),
)

MOTION_ERRORS = (
    ("Metrics/motion/error_anchor_ang_vel", "error anchor ang vel"),
    ("Metrics/motion/error_anchor_lin_vel", "error anchor lin vel"),
    ("Metrics/motion/error_anchor_pos", "error anchor pos"),
    ("Metrics/motion/error_anchor_rot", "error anchor rot"),
    ("Metrics/motion/error_body_ang_vel", "error body ang vel"),
    ("Metrics/motion/error_body_lin_vel", "error body lin vel"),
    ("Metrics/motion/error_body_pos", "error body pos"),
    ("Metrics/motion/error_body_rot", "error body rot"),
    ("Metrics/motion/error_joint_pos", "error joint pos"),
    ("Metrics/motion/error_joint_vel", "error joint vel"),
)

SUMMARY_TAGS = (
    ("Train/mean_reward", "Mean reward"),
    ("Loss/entropy", "Entropy"),
    ("Metrics/motion/sampling_entropy", "Sampling entropy"),
    ("Train/mean_episode_length", "Mean episode length"),
)

REQUESTED_TAGS = {
    tag
    for tag, _ in (*SUMMARY_TAGS, *REWARD_COMPONENTS, *TERMINATIONS, *MOTION_ERRORS)
}


def parse_run(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("runs must use LABEL=PATH") from exc
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("runs must use a non-empty LABEL=PATH")
    return label.strip(), Path(path).expanduser().resolve()


def scalar_value(summary_value) -> float | None:
    if summary_value.HasField("simple_value"):
        return float(summary_value.simple_value)
    if summary_value.HasField("tensor"):
        array = np.asarray(make_ndarray(summary_value.tensor))
        if array.size == 1:
            return float(array.reshape(-1)[0])
    return None


def load_scalars(
    run_dir: Path, cache_file: Path | None = None
) -> dict[str, list[tuple[int, float]]]:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {run_dir}")

    event_signature = tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in event_files
    )
    if cache_file is not None and cache_file.exists():
        with cache_file.open("rb") as handle:
            cached = pickle.load(handle)  # noqa: S301 - cache is generated locally by this script
        if cached.get("event_signature") == event_signature:
            print(f"Using cache {cache_file}", flush=True)
            return cached["scalars"]

    by_tag_and_step: dict[str, dict[int, float]] = defaultdict(dict)
    for event_file in event_files:
        print(f"Reading {event_file}", flush=True)
        for event in EventFileLoader(str(event_file)).Load():
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                if value.tag not in REQUESTED_TAGS:
                    continue
                scalar = scalar_value(value)
                if scalar is not None and np.isfinite(scalar):
                    by_tag_and_step[value.tag][int(event.step)] = scalar

    scalars = {
        tag: sorted(values.items())
        for tag, values in by_tag_and_step.items()
    }
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as handle:
            pickle.dump(
                {"event_signature": event_signature, "scalars": scalars},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    return scalars


def bin_average(
    samples: list[tuple[int, float]], bin_width: int
) -> tuple[np.ndarray, np.ndarray]:
    bins: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for step, value in samples:
        bins[step // bin_width].append((step, value))
    ordered = sorted(bins.items())
    x = np.asarray(
        [np.mean([step for step, _ in entries]) for _, entries in ordered], dtype=float
    )
    y = np.asarray(
        [np.mean([value for _, value in entries]) for _, entries in ordered], dtype=float
    )
    return x, y


def plot_tag(ax, runs, tag: str, bin_width: int) -> bool:
    plotted = False
    for label, scalars in runs:
        samples = scalars.get(tag)
        if not samples:
            continue
        x, y = bin_average(samples, bin_width)
        ax.plot(x, y, linewidth=1.8, label=label)
        plotted = True
    ax.set_xlabel("Update")
    if not plotted:
        ax.text(0.5, 0.5, "Metric unavailable", ha="center", va="center", transform=ax.transAxes)
    return plotted


def add_figure(pdf: PdfPages, output_dir: Path, page: int, fig) -> None:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
    pdf.savefig(fig)
    fig.savefig(output_dir / f"comparison_page_{page}.png", dpi=160)
    plt.close(fig)


def add_figure_legend(fig, axes, *, ncol: int) -> None:
    handles, labels = [], []
    for ax in np.asarray(axes).reshape(-1):
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=ncol,
            frameon=True,
        )


def build_plots(runs, output_dir: Path, bin_width: int) -> Path:
    output_pdf = output_dir / "all_comparison_plots_4_runs.pdf"
    subtitle = f"{bin_width}-update averages"
    legend_columns = min(4, len(runs))

    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, 4.4))
        plot_tag(ax, runs, "Train/mean_reward", bin_width)
        ax.set_ylabel("Mean reward")
        fig.suptitle(f"Mean Reward ({subtitle})", fontsize=16, fontweight="bold")
        add_figure_legend(fig, [ax], ncol=legend_columns)
        add_figure(pdf, output_dir, 1, fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
        plot_tag(axes[0], runs, "Loss/entropy", bin_width)
        axes[0].set_title("entropy", fontweight="bold")
        plot_tag(axes[1], runs, "Metrics/motion/sampling_entropy", bin_width)
        axes[1].set_title("sampling entropy", fontweight="bold")
        fig.suptitle(f"Entropy ({subtitle})", fontsize=16, fontweight="bold")
        add_figure_legend(fig, axes, ncol=legend_columns)
        add_figure(pdf, output_dir, 2, fig)

        fig, axes = plt.subplots(4, 4, figsize=(16, 14))
        for ax, (tag, title) in zip(axes.flat, REWARD_COMPONENTS, strict=True):
            plot_tag(ax, runs, tag, bin_width)
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("Mean component reward")
        fig.suptitle(
            f"Episode Reward Components / 16 Objectives ({subtitle})",
            fontsize=16,
            fontweight="bold",
        )
        add_figure_legend(fig, axes, ncol=legend_columns)
        add_figure(pdf, output_dir, 3, fig)

        fig, ax = plt.subplots(figsize=(11.7, 4.4))
        plot_tag(ax, runs, "Train/mean_episode_length", bin_width)
        ax.set_ylabel("Mean length")
        fig.suptitle(f"Mean Episode Length ({subtitle})", fontsize=16, fontweight="bold")
        add_figure_legend(fig, [ax], ncol=legend_columns)
        add_figure(pdf, output_dir, 4, fig)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for ax, (tag, title) in zip(axes.flat, TERMINATIONS, strict=True):
            plot_tag(ax, runs, tag, bin_width)
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("Mean termination output")
        fig.suptitle(f"Episode Termination Outputs ({subtitle})", fontsize=16, fontweight="bold")
        add_figure_legend(fig, axes, ncol=legend_columns)
        add_figure(pdf, output_dir, 5, fig)

        fig, axes = plt.subplots(3, 4, figsize=(16, 11))
        for ax, (tag, title) in zip(axes.flat, MOTION_ERRORS):
            plot_tag(ax, runs, tag, bin_width)
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("Error")
        for ax in axes.flat[len(MOTION_ERRORS) :]:
            ax.set_visible(False)
        fig.suptitle(f"Motion Error Metrics ({subtitle})", fontsize=16, fontweight="bold")
        add_figure_legend(fig, axes, ncol=legend_columns)
        add_figure(pdf, output_dir, 6, fig)

    return output_pdf


def write_summary(runs, output_dir: Path, bin_width: int) -> Path:
    output_csv = output_dir / "comparison_summary.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "run",
                "metric",
                "last_update",
                "last_bin_mean",
                "best_bin_mean",
                "best_update",
            ),
        )
        writer.writeheader()
        for label, scalars in runs:
            for tag, metric in SUMMARY_TAGS:
                samples = scalars.get(tag)
                if not samples:
                    continue
                x, y = bin_average(samples, bin_width)
                best_index = int(np.nanargmax(y))
                writer.writerow(
                    {
                        "run": label,
                        "metric": metric,
                        "last_update": int(samples[-1][0]),
                        "last_bin_mean": f"{y[-1]:.9g}",
                        "best_bin_mean": f"{y[best_index]:.9g}",
                        "best_update": f"{x[best_index]:.1f}",
                    }
                )
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        metavar="LABEL=PATH",
        help="Comparison label and TensorBoard run directory; repeat for each run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-width", type=int, default=100)
    args = parser.parse_args()
    if args.bin_width <= 0:
        parser.error("--bin-width must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    load_requests = [
        (label, run_dir, args.output_dir / f"scalar_cache_{run_index}.pkl")
        for run_index, (label, run_dir) in enumerate(args.run)
    ]
    with ThreadPoolExecutor(max_workers=min(4, len(load_requests))) as executor:
        futures = [
            executor.submit(load_scalars, run_dir, cache_file)
            for _, run_dir, cache_file in load_requests
        ]

    loaded_runs = []
    for (label, _, _), future in zip(load_requests, futures, strict=True):
        scalars = future.result()
        missing = sorted(REQUESTED_TAGS - scalars.keys())
        print(
            f"Loaded {label}: {len(scalars)} requested tags; "
            f"{len(missing)} unavailable",
            flush=True,
        )
        loaded_runs.append((label, scalars))

    output_pdf = build_plots(loaded_runs, args.output_dir, args.bin_width)
    output_csv = write_summary(loaded_runs, args.output_dir, args.bin_width)
    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
