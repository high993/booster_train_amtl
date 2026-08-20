# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from tensordict import TensorDict

from amtl.modules import ActorCriticRecurrent
from amtl.modules.rnd import RandomNetworkDistillation
from amtl.storage import RolloutStorage
from amtl.utils import resolve_optimizer, string_to_callable

from amtl.actor_critic import ActorCritic


HIERARCHICAL_5_PGA_GROUPS = (
    (
        "G1_global_root_trunk_pose",
        (
            "motion_global_anchor_pos",
            "motion_global_anchor_ori",
            "motion_trunk_pos",
            "motion_trunk_ori",
        ),
        4.0 / 16.0,
    ),
    (
        "G2_whole_body_pose",
        ("motion_body_pos", "motion_body_ori"),
        2.0 / 16.0,
    ),
    (
        "G3_velocity_dynamics",
        ("motion_body_lin_vel", "motion_body_ang_vel", "motion_trunk_ang_vel"),
        3.0 / 16.0,
    ),
    (
        "G4_end_effectors",
        ("motion_foot_pos", "motion_foot_ori", "motion_hand_pos", "motion_hand_ori"),
        4.0 / 16.0,
    ),
    (
        "G5_regularizers",
        ("action_rate_l2", "joint_limit", "undesired_contacts"),
        3.0 / 16.0,
    ),
)
HIERARCHICAL_5_PGA_OBJECTIVE_COUNT = 16


def _format_numeric_tensor(tensor: torch.Tensor) -> str:
    """Return compact tensor statistics that remain useful for NaN/Inf failures."""
    try:
        value = tensor.detach()
        finite_mask = torch.isfinite(value)
        finite_count = int(finite_mask.sum().item())
        total_count = value.numel()
        nan_count = int(torch.isnan(value).sum().item())
        posinf_count = int(torch.isposinf(value).sum().item())
        neginf_count = int(torch.isneginf(value).sum().item())
        if finite_count:
            finite_values = value[finite_mask]
            finite_min = float(finite_values.min().item())
            finite_max = float(finite_values.max().item())
            finite_abs_max = float(finite_values.abs().max().item())
        else:
            finite_min = float("nan")
            finite_max = float("nan")
            finite_abs_max = float("nan")
        details = (
            f"shape={tuple(value.shape)} dtype={value.dtype} device={value.device} "
            f"finite={finite_count}/{total_count} nan={nan_count} "
            f"+inf={posinf_count} -inf={neginf_count} "
            f"finite_min={finite_min:.9e} finite_max={finite_max:.9e} "
            f"finite_abs_max={finite_abs_max:.9e}"
        )
        if total_count <= 32:
            details += f" values={value.cpu().tolist()}"
        return details
    except Exception as exc:
        return f"statistics unavailable after CUDA failure: {type(exc).__name__}: {exc}"


def _raise_numeric_error(
    stage: str,
    tensors: dict[str, torch.Tensor],
    *,
    context: str = "",
    cause: Exception | None = None,
) -> None:
    """Print numerical evidence immediately, then stop before state is corrupted."""
    header = f"[NUMERIC ERROR] stage={stage}"
    if context:
        header += f" context={context}"
    print("\n" + "=" * 96, file=sys.stderr, flush=True)
    print(header, file=sys.stderr, flush=True)
    if cause is not None:
        print(
            f"cause={type(cause).__name__}: {cause}",
            file=sys.stderr,
            flush=True,
        )
    for name, tensor in tensors.items():
        print(
            f"{name}: {_format_numeric_tensor(tensor)}",
            file=sys.stderr,
            flush=True,
        )
    print("=" * 96 + "\n", file=sys.stderr, flush=True)
    error = FloatingPointError(f"Non-finite or invalid numeric state at {stage} ({context})")
    if cause is not None:
        raise error from cause
    raise error


def set_shared_grad(shared_params, grad_vec):
    # Write a flattened gradient vector back into a parameter list.
    expected_numel = sum(p.numel() for p in shared_params)
    if grad_vec.numel() != expected_numel:
        raise ValueError(
            f"Gradient vector size mismatch: got {grad_vec.numel()} values for {expected_numel} parameters."
        )

    offset = 0
    for p in shared_params:
        _offset = offset + p.numel()
        grad_view = grad_vec[offset:_offset].view_as(p)
        if p.grad is None:
            p.grad = grad_view.clone()
        else:
            p.grad.data.copy_(grad_view)
        offset = _offset


class ProcrustesSolver:
    @staticmethod
    def apply(grads, scale_mode='min'):
        # Expects [batch, num_parameters, num_objectives] and returns the same layout
        # after objective-space alignment.
        assert (
            len(grads.shape) == 3
        ), f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable"

        with torch.no_grad():
            cov_grad_matrix_e = torch.matmul(grads.permute(0, 2, 1), grads)
            cov_grad_matrix_e = cov_grad_matrix_e.mean(0)
            
            #compute eigen vectors
            singulars, basis = torch.linalg.eigh(cov_grad_matrix_e)
            tol = (
                torch.max(singulars)
                * max(cov_grad_matrix_e.shape[-2:])
                * torch.finfo().eps
            )
            rank = sum(singulars > tol)

            order = torch.argsort(singulars, dim=-1, descending=True)
            singulars, basis = singulars[order][:rank], basis[:, order][:, :rank]

            if scale_mode == 'min':
                weights = basis * torch.sqrt(singulars[-1]).view(1, -1)
            elif scale_mode == 'median':
                weights = basis * torch.sqrt(torch.median(singulars)).view(1, -1)
            elif scale_mode == 'rmse':
                weights = basis * torch.sqrt(singulars.mean())

            weights = weights / torch.sqrt(singulars).view(1, -1)
            weights = torch.matmul(weights, basis.T)
            grads = torch.matmul(grads, weights.unsqueeze(0))

            return grads, weights, singulars


class PGASolver:
    """Principal Gradient Alignment aggregation for objective gradients.

    ``grads`` is a parameter-by-objective matrix.  The returned aggregate is a
    single gradient in parameter space, ready to be installed with
    :func:`set_shared_grad` before the optimizer step.
    """

    @staticmethod
    def apply(
        grads: torch.Tensor,
        rank: int | None = None,
        method: str = "os_nmf",
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        direction_weighting: str = "uniform",
        diagnostic_context: str = "",
        objective_names: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one aggregate gradient from several objective gradients.

        Args:
            grads: ``[parameters, objectives]`` or ``[1, parameters,
                objectives]``. Each column is the gradient of one objective.
            rank: Number of principal directions to retain. ``None`` keeps all
                numerically useful directions.
            method: One of ``"gram"``, ``"procrustes"``, or ``"os_nmf"``.
            max_iterations: Factorization iterations for ``"procrustes"`` and
                ``"os_nmf"``.
            tolerance: Relative factorization convergence tolerance.
            direction_weighting: ``"uniform"`` gives every retained basis
                direction equal weight. ``"factor_strength"`` weights each
                direction by its singular/factor strength.
        """
        if grads.ndim == 2:
            gradient_matrix = grads
            restore_singleton_dim = False
        elif grads.ndim == 3 and grads.shape[0] == 1:
            gradient_matrix = grads.squeeze(0)
            restore_singleton_dim = True
        else:
            raise ValueError(
                "grads must have shape [parameters, objectives] or "
                f"[1, parameters, objectives], got {tuple(grads.shape)}."
            )
        if not gradient_matrix.is_floating_point():
            raise TypeError("grads must have a floating-point dtype.")
        if gradient_matrix.shape[1] == 0:
            raise ValueError("grads must contain at least one objective.")
        if rank is not None and rank <= 0:
            raise ValueError("rank must be a positive integer or None.")
        if method not in {"gram", "procrustes", "os_nmf"}:
            raise ValueError('method must be "gram", "procrustes", or "os_nmf".')
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if direction_weighting not in {"uniform", "factor_strength"}:
            raise ValueError('direction_weighting must be "uniform" or "factor_strength".')
        if objective_names is not None and len(objective_names) != gradient_matrix.shape[1]:
            raise ValueError(
                "objective_names must have one entry per gradient column, got "
                f"{len(objective_names)} names for {gradient_matrix.shape[1]} columns."
            )

        def failure_tensors(**tensors: torch.Tensor) -> dict[str, torch.Tensor]:
            details = dict(tensors)
            details.setdefault("gradient_matrix", gradient_matrix)
            if objective_names is not None:
                for index, objective_name in enumerate(objective_names):
                    details[f"gradient[{index:02d}]/{objective_name}"] = gradient_matrix[:, index]
            return details

        with torch.no_grad():
            num_objectives = gradient_matrix.shape[1]
            gram = gradient_matrix.T @ gradient_matrix
            if not bool(torch.isfinite(gram).all()):
                _raise_numeric_error(
                    "pga_gram_matrix",
                    failure_tensors(gram=gram),
                    context=diagnostic_context,
                )
            try:
                eigenvalues, right_vectors = torch.linalg.eigh(gram)
            except RuntimeError as exc:
                _raise_numeric_error(
                    "pga_gram_eigendecomposition",
                    failure_tensors(gram=gram),
                    context=diagnostic_context,
                    cause=exc,
                )
            if not bool(torch.isfinite(eigenvalues).all()) or not bool(
                torch.isfinite(right_vectors).all()
            ):
                _raise_numeric_error(
                    "pga_eigensystem",
                    failure_tensors(
                        gram=gram,
                        eigenvalues=eigenvalues,
                        right_vectors=right_vectors,
                    ),
                    context=diagnostic_context,
                )
            eigenvalues = eigenvalues.clamp_min(0)
            order = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[order]
            right_vectors = right_vectors[:, order]

            eigenvalue_tolerance = (
                eigenvalues.max() * num_objectives * torch.finfo(eigenvalues.dtype).eps
            )
            useful_rank = int(torch.count_nonzero(eigenvalues > eigenvalue_tolerance).item())
            if rank is not None:
                useful_rank = min(useful_rank, rank)

            if useful_rank == 0:
                _raise_numeric_error(
                    "pga_zero_useful_rank",
                    failure_tensors(gram=gram, eigenvalues=eigenvalues),
                    context=diagnostic_context,
                )
            elif method == "gram":
                principal_weights = right_vectors[:, :useful_rank]
                singular_values = torch.sqrt(eigenvalues[:useful_rank])
                PGASolver._validate_retained_sigmas(
                    singular_values,
                    failure_tensors(gram=gram, eigenvalues=eigenvalues),
                    diagnostic_context,
                )
                if direction_weighting == "uniform":
                    direction_weights = torch.full(
                        (useful_rank,),
                        1.0 / useful_rank,
                        device=gradient_matrix.device,
                        dtype=gradient_matrix.dtype,
                    )
                else:
                    strengths = singular_values.clamp_min(torch.finfo(gradient_matrix.dtype).eps)
                    direction_weights = strengths / strengths.sum()
                objective_weights = principal_weights @ (direction_weights / singular_values)
                aggregate_grad = gradient_matrix @ objective_weights
            else:
                singular_values = torch.sqrt(eigenvalues[:useful_rank])
                PGASolver._validate_retained_sigmas(
                    singular_values,
                    failure_tensors(gram=gram, eigenvalues=eigenvalues),
                    diagnostic_context,
                )
                basis, factor = PGASolver._factorize(
                    gradient_matrix,
                    useful_rank,
                    method,
                    max_iterations,
                    tolerance,
                    diagnostic_context=diagnostic_context,
                    diagnostic_tensors=failure_tensors(
                        gram=gram,
                        eigenvalues=eigenvalues,
                        retained_sigmas=singular_values,
                    ),
                )
                if direction_weighting == "uniform":
                    basis_weights = torch.full(
                        (useful_rank,),
                        1.0 / useful_rank,
                        device=gradient_matrix.device,
                        dtype=gradient_matrix.dtype,
                    )
                else:
                    strengths = torch.linalg.vector_norm(factor, dim=1).clamp_min(
                        torch.finfo(gradient_matrix.dtype).eps
                    )
                    basis_weights = strengths / strengths.sum()
                if not bool(torch.isfinite(basis_weights).all()):
                    _raise_numeric_error(
                        "pga_basis_weights",
                        failure_tensors(
                            gram=gram,
                            eigenvalues=eigenvalues,
                            retained_sigmas=singular_values,
                            factor=factor,
                            basis_weights=basis_weights,
                        ),
                        context=diagnostic_context,
                    )
                target_grad = basis @ basis_weights
                objective_rhs = gradient_matrix.T @ target_grad
                try:
                    gram_pinv = torch.linalg.pinv(gram)
                except RuntimeError as exc:
                    _raise_numeric_error(
                        "pga_gram_pseudoinverse",
                        failure_tensors(
                            gram=gram,
                            eigenvalues=eigenvalues,
                            retained_sigmas=singular_values,
                            factor=factor,
                            basis_weights=basis_weights,
                            target_grad=target_grad,
                            objective_rhs=objective_rhs,
                        ),
                        context=diagnostic_context,
                        cause=exc,
                    )
                objective_weights = gram_pinv @ objective_rhs
                aggregate_grad = gradient_matrix @ objective_weights
                principal_weights = factor.T

            if not bool(torch.isfinite(objective_weights).all()) or not bool(
                torch.isfinite(aggregate_grad).all()
            ):
                _raise_numeric_error(
                    "pga_final_aggregation",
                    failure_tensors(
                        gram=gram,
                        eigenvalues=eigenvalues,
                        retained_sigmas=singular_values,
                        objective_weights=objective_weights,
                        aggregate_grad=aggregate_grad,
                    ),
                    context=diagnostic_context,
                )

            if restore_singleton_dim:
                aggregate_grad = aggregate_grad.unsqueeze(0)
            return aggregate_grad, objective_weights, singular_values, principal_weights

    @staticmethod
    def _validate_retained_sigmas(
        singular_values: torch.Tensor,
        diagnostic_tensors: dict[str, torch.Tensor],
        diagnostic_context: str,
    ) -> None:
        if (
            not bool(torch.isfinite(singular_values).all())
            or bool(torch.any(singular_values <= 0))
        ):
            _raise_numeric_error(
                "pga_retained_sigmas",
                {**diagnostic_tensors, "retained_sigmas": singular_values},
                context=diagnostic_context,
            )

    @staticmethod
    def _factorize(
        grads: torch.Tensor,
        rank: int,
        method: str,
        max_iterations: int,
        tolerance: float,
        *,
        diagnostic_context: str = "",
        diagnostic_tensors: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve ``min ||G - W H||_F`` with orthogonal ``W`` and optional ``H >= 0``."""
        diagnostics = diagnostic_tensors or {"gradient_matrix": grads}
        try:
            _, singular_values, right_vectors_t = torch.linalg.svd(grads, full_matrices=False)
        except RuntimeError as exc:
            _raise_numeric_error(
                "pga_initial_svd",
                diagnostics,
                context=diagnostic_context,
                cause=exc,
            )
        if not bool(torch.isfinite(singular_values).all()) or not bool(
            torch.isfinite(right_vectors_t).all()
        ):
            _raise_numeric_error(
                "pga_initial_svd_output",
                {
                    **diagnostics,
                    "factorization_sigmas": singular_values,
                    "right_vectors_t": right_vectors_t,
                },
                context=diagnostic_context,
            )
        factor = singular_values[:rank].unsqueeze(1) * right_vectors_t[:rank]
        if method == "os_nmf":
            factor = factor.abs()

        for iteration in range(max_iterations):
            factor_cross_product = grads @ factor.T
            if not bool(torch.isfinite(factor_cross_product).all()):
                _raise_numeric_error(
                    "pga_factorization_cross_product",
                    {
                        **diagnostics,
                        "factor": factor,
                        "factor_cross_product": factor_cross_product,
                    },
                    context=f"{diagnostic_context}, os_nmf_iteration={iteration}",
                )
            try:
                left_vectors, _, right_vectors_t = torch.linalg.svd(
                    factor_cross_product, full_matrices=False
                )
            except RuntimeError as exc:
                _raise_numeric_error(
                    "pga_factorization_svd",
                    {
                        **diagnostics,
                        "factor": factor,
                        "factor_cross_product": factor_cross_product,
                    },
                    context=f"{diagnostic_context}, os_nmf_iteration={iteration}",
                    cause=exc,
                )
            basis = left_vectors[:, :rank] @ right_vectors_t[:rank]
            updated_factor = basis.T @ grads
            if method == "os_nmf":
                updated_factor = updated_factor.clamp_min(0)

            change = torch.linalg.vector_norm(updated_factor - factor)
            scale = torch.linalg.vector_norm(factor).clamp_min(torch.finfo(grads.dtype).eps)
            factor = updated_factor
            if (
                not bool(torch.isfinite(basis).all())
                or not bool(torch.isfinite(factor).all())
                or not bool(torch.isfinite(change))
                or not bool(torch.isfinite(scale))
            ):
                _raise_numeric_error(
                    "pga_factorization_update",
                    {
                        **diagnostics,
                        "basis": basis,
                        "factor": factor,
                        "change": change,
                        "scale": scale,
                    },
                    context=f"{diagnostic_context}, os_nmf_iteration={iteration}",
                )
            if change <= tolerance * scale:
                break
        return basis, factor



































@dataclass
class HierarchicalPGAResult:
    aggregate_grad: torch.Tensor
    objective_weights: torch.Tensor
    group_grads: torch.Tensor
    weighted_group_grads: torch.Tensor
    group_cosine_matrix: torch.Tensor
    outer_objective_weights: torch.Tensor
    outer_singular_values: torch.Tensor
    outer_factor_strengths: torch.Tensor
    outer_factor_weights: torch.Tensor
    flat_aggregate_grad: torch.Tensor
    flat_singular_values: torch.Tensor


def aggregate_hierarchical_5_pga(
    stacked_actor_grads: torch.Tensor,
    objective_names: Sequence[str],
    *,
    rank: int | None,
    direction_weighting: str,
    diagnostic_context: str = "",
) -> HierarchicalPGAResult:
    """Aggregate 16 actor objectives through five semantic PGA groups."""
    if stacked_actor_grads.ndim != 2:
        raise ValueError(
            "stacked_actor_grads must have shape [objectives, parameters], got "
            f"{tuple(stacked_actor_grads.shape)}."
        )
    if len(objective_names) != stacked_actor_grads.shape[0]:
        raise ValueError(
            "objective_names must match stacked_actor_grads rows, got "
            f"{len(objective_names)} names and {stacked_actor_grads.shape[0]} rows."
        )
    if len(set(objective_names)) != len(objective_names):
        raise ValueError("Hierarchical-5 PGA requires unique objective names.")

    expected_names = {
        objective_name
        for _, group_objective_names, _ in HIERARCHICAL_5_PGA_GROUPS
        for objective_name in group_objective_names
    }
    actual_names = set(objective_names)
    if len(objective_names) != HIERARCHICAL_5_PGA_OBJECTIVE_COUNT or actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            "Hierarchical-5 PGA requires exactly the configured 16 objectives; "
            f"missing={missing}, unexpected={unexpected}, count={len(objective_names)}."
        )

    objective_index = {objective_name: index for index, objective_name in enumerate(objective_names)}
    group_grads = []
    inner_objective_weights = []
    group_indices = []
    for group_name, group_objective_names, _ in HIERARCHICAL_5_PGA_GROUPS:
        indices = [objective_index[objective_name] for objective_name in group_objective_names]
        group_grad, group_weights, _, _ = PGASolver.apply(
            stacked_actor_grads[indices].T.unsqueeze(0),
            rank=rank,
            method="os_nmf",
            direction_weighting=direction_weighting,
            diagnostic_context=f"{diagnostic_context}, inner_group={group_name}",
            objective_names=group_objective_names,
        )
        group_grads.append(group_grad.squeeze(0))
        inner_objective_weights.append(group_weights)
        group_indices.append(indices)

    group_grads_tensor = torch.stack(group_grads, dim=0)
    group_size_weights = group_grads_tensor.new_tensor(
        [group_weight for _, _, group_weight in HIERARCHICAL_5_PGA_GROUPS]
    )
    # PGASolver receives these weighted columns directly. There is deliberately
    # no per-column normalization between this multiplication and the outer PGA.
    weighted_group_grads = group_grads_tensor * group_size_weights.unsqueeze(1)
    group_names = [group_name for group_name, _, _ in HIERARCHICAL_5_PGA_GROUPS]
    outer_grad, outer_objective_weights, outer_singular_values, outer_principal_weights = PGASolver.apply(
        weighted_group_grads.T.unsqueeze(0),
        rank=rank,
        method="os_nmf",
        direction_weighting=direction_weighting,
        diagnostic_context=f"{diagnostic_context}, outer_groups",
        objective_names=group_names,
    )
    outer_grad = outer_grad.squeeze(0)

    objective_weights = torch.zeros(
        len(objective_names),
        dtype=stacked_actor_grads.dtype,
        device=stacked_actor_grads.device,
    )
    for group_index, indices in enumerate(group_indices):
        objective_weights[indices] = (
            outer_objective_weights[group_index]
            * group_size_weights[group_index]
            * inner_objective_weights[group_index]
        )

    group_norms = group_grads_tensor.norm(dim=1, keepdim=True)
    normalized_group_grads = torch.where(
        group_norms > 1.0e-12,
        group_grads_tensor / group_norms.clamp_min(1.0e-12),
        torch.zeros_like(group_grads_tensor),
    )
    group_cosine_matrix = normalized_group_grads @ normalized_group_grads.T

    # For OS-NMF, PGASolver returns factor.T as outer_principal_weights.
    outer_factor_strengths = outer_principal_weights.norm(dim=0)
    if direction_weighting == "uniform":
        outer_factor_weights = torch.full_like(
            outer_factor_strengths, 1.0 / len(outer_factor_strengths)
        )
    else:
        outer_factor_weights = outer_factor_strengths / outer_factor_strengths.sum().clamp_min(
            torch.finfo(outer_factor_strengths.dtype).eps
        )

    # This diagnostic uses the unchanged flat-16 solver on the same gradients.
    flat_grad, _, flat_singular_values, _ = PGASolver.apply(
        stacked_actor_grads.T.unsqueeze(0),
        rank=rank,
        method="os_nmf",
        direction_weighting=direction_weighting,
        diagnostic_context=f"{diagnostic_context}, flat_comparison",
        objective_names=objective_names,
    )

    return HierarchicalPGAResult(
        aggregate_grad=outer_grad,
        objective_weights=objective_weights,
        group_grads=group_grads_tensor,
        weighted_group_grads=weighted_group_grads,
        group_cosine_matrix=group_cosine_matrix,
        outer_objective_weights=outer_objective_weights,
        outer_singular_values=outer_singular_values,
        outer_factor_strengths=outer_factor_strengths,
        outer_factor_weights=outer_factor_weights,
        flat_aggregate_grad=flat_grad.squeeze(0),
        flat_singular_values=flat_singular_values,
    )


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic | ActorCriticRecurrent
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.00025,
        optimizer: str = "adam",
        max_grad_norm: float = 0.5,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        share_cnn_encoders: bool = False,
        device: str = "cpu",
        env=None,
        normalize_advantage_per_mini_batch: bool = False,
        amtl_apply_to: str = "actor",
        pga_rank: int | None = None,
        pga_direction_weighting: str = "uniform",
        pga_match_ppo_grad_norm: bool = False,
        min_action_std: float | None = None,
        debug_amtl: bool = False,
        debug_amtl_log_interval: int = 100,
        actor_pga_mode: str = "flat",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **legacy_kwargs,
    ) -> None:
        legacy_align_actor = legacy_kwargs.pop("align_actor_all_terms", None)
        legacy_align_critic = legacy_kwargs.pop("align_critic", None)
        if legacy_align_actor is not None or legacy_align_critic is not None:
            warnings.warn(
                "The `align_actor_all_terms` and `align_critic` flags are deprecated. "
                "Use `amtl_apply_to=\"actor\"` for actor-side AMTL.",
                stacklevel=2,
            )
            if legacy_align_actor and amtl_apply_to == "none":
                amtl_apply_to = "actor"
            if legacy_align_critic:
                warnings.warn(
                    "Critic-side AMTL is disabled; the deprecated `align_critic` flag is ignored.",
                    stacklevel=2,
                )

        if amtl_apply_to not in {"none", "actor"}:
            raise ValueError(
                f"Unsupported amtl_apply_to='{amtl_apply_to}'. This branch supports only 'none' and 'actor'."
            )
        if actor_pga_mode not in {"flat", "hierarchical_5"}:
            raise ValueError(
                "actor_pga_mode must be 'flat' or 'hierarchical_5', got "
                f"{actor_pga_mode!r}."
            )
        if min_action_std is not None and min_action_std <= 0.0:
            raise ValueError("min_action_std must be positive or None.")

        if legacy_kwargs:
            warnings.warn(
                "Ignoring unsupported PPO config fields: "
                + ", ".join(sorted(legacy_kwargs.keys()))
                + ". These settings belong to the AMP trainer path and are not used by this PPO experiment.",
                stacklevel=2,
            )
        # Device-related parameters
        self.device = device
        self.env = env
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the policy is compatible with symmetry
            if isinstance(policy, ActorCriticRecurrent):
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        self._validate_actor_critic_parameter_partition()

        # Create optimizer
        optimizer_class = resolve_optimizer(optimizer)
        self.optimizer = optimizer_class(self.policy.parameters(), lr=learning_rate)

        # Create rollout storage
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.share_cnn_encoders = share_cnn_encoders
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.amtl_apply_to = amtl_apply_to
        self.use_actor_amtl = amtl_apply_to == "actor"
        self.use_critic_amtl = False
        self.actor_pga_mode = actor_pga_mode
        self.pga_rank = pga_rank
        self.pga_direction_weighting = pga_direction_weighting
        self.pga_match_ppo_grad_norm = pga_match_ppo_grad_norm
        self.min_action_std = min_action_std
        self.debug_amtl = debug_amtl
        self.debug_amtl_log_interval = debug_amtl_log_interval
        self.update_counter = 0
        self._amtl_diagnostics_printed = False
        self._pairwise_cosine_sanity_printed = False

    def _get_reward_manager(self):
        if self.env is None:
            return None

        for env_candidate in (self.env, getattr(self.env, "unwrapped", None)):
            if env_candidate is None:
                continue
            reward_manager = getattr(env_candidate, "reward_manager", None)
            if reward_manager is not None:
                return reward_manager
        return None

    def _get_reward_terms(self) -> torch.Tensor | None:
        reward_manager = self._get_reward_manager()
        if reward_manager is None:
            return None
        if hasattr(reward_manager, "step_reward_weighted_dt"):
            return reward_manager.step_reward_weighted_dt.detach().clone()
        if hasattr(reward_manager, "step_reward_weighted"):
            return reward_manager.step_reward_weighted.detach().clone()
        return None

    def _get_reward_term_names(self) -> list[str] | None:
        reward_manager = self._get_reward_manager()
        if reward_manager is None or not hasattr(reward_manager, "active_terms"):
            return None
        return list(reward_manager.active_terms)

    def _get_critic_head_names(self, num_heads: int) -> list[str]:
        reward_term_names = self._get_reward_term_names()
        if reward_term_names is not None and len(reward_term_names) == num_heads:
            return reward_term_names
        return [f"head_{index:02d}" for index in range(num_heads)]

    def _get_actor_objective_names(self, num_objectives: int) -> list[str]:
        reward_term_names = self._get_reward_term_names()
        if reward_term_names is not None and len(reward_term_names) == num_objectives:
            return reward_term_names
        return [f"objective_{index:02d}" for index in range(num_objectives)]

    def _validate_actor_critic_parameter_partition(self) -> None:
        # AMTL temporarily zeroes and rewrites one side's gradients, so the
        # actor/critic parameter sets must stay disjoint.
        actor_param_ids = {id(param) for param in self.policy.get_actor_parameters()}
        critic_param_ids = {id(param) for param in self.policy.get_critic_parameters()}
        shared_param_count = len(actor_param_ids & critic_param_ids)
        if shared_param_count > 0:
            raise RuntimeError(
                "AMTL currently requires disjoint actor/critic parameters, "
                f"but found {shared_param_count} shared parameter(s)."
            )

    @staticmethod
    def _zero_existing_grads(params: Sequence[torch.nn.Parameter]) -> None:
        for param in params:
            if param.grad is not None:
                param.grad.data.zero_()

    def _enforce_min_action_std(self) -> None:
        """Keep learned state-independent policy noise above its configured floor."""
        if self.min_action_std is None:
            return
        std_param = self.policy.learned_action_std_parameter
        if std_param is None:
            return
        with torch.no_grad():
            if self.policy.noise_std_type == "scalar":
                if not bool(torch.isfinite(std_param).all()) or bool(torch.any(std_param <= 0)):
                    _raise_numeric_error(
                        "policy_learned_action_std_parameter",
                        {"learned_action_std_parameter": std_param},
                        context=f"update={self.update_counter}, before_min_std_clamp",
                    )
                std_param.clamp_(min=self.min_action_std)
            elif self.policy.noise_std_type == "log":
                actual_std = torch.exp(std_param)
                if not bool(torch.isfinite(actual_std).all()) or bool(torch.any(actual_std <= 0)):
                    _raise_numeric_error(
                        "policy_learned_log_std_parameter",
                        {
                            "learned_log_std_parameter": std_param,
                            "actual_action_std": actual_std,
                        },
                        context=f"update={self.update_counter}, before_min_std_clamp",
                    )
                std_param.clamp_(min=math.log(self.min_action_std))

    @staticmethod
    def _validate_policy_distribution_numerics(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        action_entropy: torch.Tensor,
        *,
        context: str,
    ) -> None:
        invalid = (
            not bool(torch.isfinite(mu).all())
            or not bool(torch.isfinite(sigma).all())
            or bool(torch.any(sigma <= 0))
            or not bool(torch.isfinite(old_mu).all())
            or not bool(torch.isfinite(old_sigma).all())
            or bool(torch.any(old_sigma <= 0))
            or not bool(torch.isfinite(action_entropy).all())
        )
        if invalid:
            _raise_numeric_error(
                "policy_action_distribution",
                {
                    "mu": mu,
                    "sigma": sigma,
                    "old_mu": old_mu,
                    "old_sigma": old_sigma,
                    "action_entropy": action_entropy,
                },
                context=context,
            )

    @staticmethod
    def _collect_flat_grad(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
        return torch.cat(
            [
                param.grad.flatten().clone() if param.grad is not None else torch.zeros_like(param).flatten()
                for param in params
            ]
        )

    @staticmethod
    def _flatten_grad_list(
        grads: Sequence[torch.Tensor | None], params: Sequence[torch.nn.Parameter]
    ) -> torch.Tensor:
        return torch.cat(
            [
                grad.reshape(-1) if grad is not None else torch.zeros_like(param).reshape(-1)
                for grad, param in zip(grads, params, strict=True)
            ]
        )

    @staticmethod
    def _parameter_count(params: Sequence[torch.nn.Parameter]) -> int:
        return sum(param.numel() for param in params)

    @staticmethod
    def _pairwise_cosine_stats(grad_matrix: torch.Tensor) -> dict[str, float]:
        if grad_matrix.ndim != 2:
            raise ValueError(f"Expected [num_objectives, num_parameters], got {tuple(grad_matrix.shape)}.")
        num_objectives = grad_matrix.shape[0]
        if num_objectives <= 1:
            return {
                "mean": 1.0,
                "min": 1.0,
                "max": 1.0,
                "conflict_fraction": 0.0,
                "near_zero_row_count": 0.0,
            }

        row_norms = grad_matrix.norm(p=2, dim=1, keepdim=True)
        normalized_grads = grad_matrix / row_norms.clamp_min(1.0e-12)
        near_zero_rows = row_norms.squeeze(1) <= 1.0e-12
        if near_zero_rows.any():
            normalized_grads = normalized_grads.clone()
            normalized_grads[near_zero_rows] = 0.0

        cosine_matrix = normalized_grads @ normalized_grads.T
        pair_indices = torch.triu_indices(num_objectives, num_objectives, offset=1, device=grad_matrix.device)
        pairwise_cosines = cosine_matrix[pair_indices[0], pair_indices[1]]
        if pairwise_cosines.numel() == 0:
            return {
                "mean": 1.0,
                "min": 1.0,
                "max": 1.0,
                "conflict_fraction": 0.0,
                "near_zero_row_count": float(near_zero_rows.sum().item()),
            }

        return {
            "mean": pairwise_cosines.mean().item(),
            "min": pairwise_cosines.min().item(),
            "max": pairwise_cosines.max().item(),
            "conflict_fraction": (pairwise_cosines < 0).float().mean().item(),
            "near_zero_row_count": float(near_zero_rows.sum().item()),
        }

    @classmethod
    def _mean_pairwise_cosine(cls, grad_matrix: torch.Tensor) -> float:
        return cls._pairwise_cosine_stats(grad_matrix)["mean"]

    @staticmethod
    def _safe_cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor, eps: float = 1.0e-12) -> float:
        norm_a = vec_a.norm()
        norm_b = vec_b.norm()
        denom = norm_a * norm_b
        if denom.item() <= eps:
            return 0.0
        return (torch.dot(vec_a, vec_b) / denom).item()

    @staticmethod
    def _cosine_against_reference(
        grad_matrix: torch.Tensor, reference_grad: torch.Tensor, eps: float = 1.0e-12
    ) -> torch.Tensor:
        if grad_matrix.ndim != 2:
            raise ValueError(f"Expected [num_objectives, num_parameters], got {tuple(grad_matrix.shape)}.")
        if reference_grad.ndim != 1:
            raise ValueError(f"Expected reference gradient [num_parameters], got {tuple(reference_grad.shape)}.")

        reference_norm = reference_grad.norm()
        row_norms = grad_matrix.norm(dim=1)
        dots = grad_matrix @ reference_grad
        denom = row_norms * reference_norm
        return torch.where(denom > eps, dots / denom.clamp_min(eps), torch.zeros_like(dots))

    def _print_pairwise_cosine_sanity_once(self, device: torch.device | str) -> None:
        if self._pairwise_cosine_sanity_printed or self.gpu_global_rank != 0:
            return

        test = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], device=device)
        row_norms = test.norm(p=2, dim=1, keepdim=True)
        normalized_test = test / row_norms.clamp_min(1.0e-12)
        cosine_matrix = normalized_test @ normalized_test.T
        pair_indices = torch.triu_indices(test.shape[0], test.shape[0], offset=1, device=test.device)
        pairwise_cosines = cosine_matrix[pair_indices[0], pair_indices[1]]
        cosine_stats = self._pairwise_cosine_stats(test)

        print(f"pairwise_cosine_sanity_test_shape={tuple(test.shape)}")
        print(f"pairwise_cosine_sanity_test_pairs={pairwise_cosines.detach().cpu().tolist()}")
        print(f"pairwise_cosine_sanity_test_mean={cosine_stats['mean']}")
        self._pairwise_cosine_sanity_printed = True

    def _print_amtl_diagnostics_once(
        self,
        actor_amtl_enabled: bool,
        critic_amtl_enabled: bool,
        num_actor_objectives: int,
        stacked_actor_grads: torch.Tensor,
        aligned_actor_grads: torch.Tensor,
        aligned_actor_grad_matrix: torch.Tensor,
        aligned_actor_grad: torch.Tensor,
        raw_actor_cosine_stats: dict[str, float],
        aligned_actor_cosine_stats: dict[str, float],
        actor_amtl_vs_ppo_cosine: float,
        actor_amtl_projection_ratio: float,
        actor_amtl_orthogonal_ratio: float,
        objective_vs_ppo_cosines: torch.Tensor,
        amtl_vs_objective_cosines: torch.Tensor,
        num_critic_objectives: int,
        stacked_critic_grads: torch.Tensor | None,
        procrustes_critic_input: torch.Tensor | None,
        aligned_critic_grad: torch.Tensor | None,
        actor_params: Sequence[torch.nn.Parameter],
        critic_params: Sequence[torch.nn.Parameter],
    ) -> None:
        if self._amtl_diagnostics_printed or self.gpu_global_rank != 0:
            return
        self._print_pairwise_cosine_sanity_once(stacked_actor_grads.device)
        print(f"actor_amtl_enabled={actor_amtl_enabled}")
        print(f"num_actor_objectives={num_actor_objectives}")
        print(f"number_of_actor_objectives={stacked_actor_grads.shape[0]}")
        print(f"number_of_actor_gradients={stacked_actor_grads.shape[0]}")
        print(f"stacked_actor_gradient_shape={tuple(stacked_actor_grads.shape)}")
        print(f"stacked_actor_grads.shape={tuple(stacked_actor_grads.shape)}")
        print(f"aligned_actor_grads.shape={tuple(aligned_actor_grads.shape)}")
        print(f"aligned_actor_grad_matrix.shape={tuple(aligned_actor_grad_matrix.shape)}")
        print(f"aligned_actor_grad.shape={tuple(aligned_actor_grad.shape)}")
        print(f"actor_parameter_count={self._parameter_count(actor_params)}")
        print("actor_alignment_includes_std/log_std=False")
        print(f"actor_has_separate_std/log_std={bool(self.policy.get_actor_std_parameters())}")
        print(f"critic_amtl_enabled={critic_amtl_enabled}")
        print(f"num_critic_objectives={num_critic_objectives}")
        print(f"critic_parameter_count={self._parameter_count(critic_params)}")
        if stacked_critic_grads is not None:
            print(f"stacked_critic_grads.shape={tuple(stacked_critic_grads.shape)}")
        if procrustes_critic_input is not None:
            print(f"procrustes_critic_input.shape={tuple(procrustes_critic_input.shape)}")
        if aligned_critic_grad is not None:
            print(f"aligned_critic_grad.shape={tuple(aligned_critic_grad.shape)}")
        print(f"actor_raw_mean_cosine={raw_actor_cosine_stats['mean']}")
        print(f"actor_raw_min_cosine={raw_actor_cosine_stats['min']}")
        print(f"actor_raw_max_cosine={raw_actor_cosine_stats['max']}")
        print(f"actor_raw_conflict_fraction={raw_actor_cosine_stats['conflict_fraction']}")
        print(f"actor_raw_near_zero_row_count={raw_actor_cosine_stats['near_zero_row_count']}")
        print(f"actor_aligned_mean_cosine={aligned_actor_cosine_stats['mean']}")
        print(f"actor_aligned_min_cosine={aligned_actor_cosine_stats['min']}")
        print(f"actor_aligned_max_cosine={aligned_actor_cosine_stats['max']}")
        print(f"actor_aligned_conflict_fraction={aligned_actor_cosine_stats['conflict_fraction']}")
        print(f"actor_aligned_near_zero_row_count={aligned_actor_cosine_stats['near_zero_row_count']}")
        print(f"actor_amtl_vs_ppo_cosine={actor_amtl_vs_ppo_cosine}")
        print(f"actor_amtl_projection_ratio={actor_amtl_projection_ratio}")
        print(f"actor_amtl_orthogonal_ratio={actor_amtl_orthogonal_ratio}")
        print(
            "actor_objective_vs_ppo_cosine_min_mean_max="
            f"({objective_vs_ppo_cosines.min().item()}, {objective_vs_ppo_cosines.mean().item()}, {objective_vs_ppo_cosines.max().item()})"
        )
        print(
            "actor_amtl_vs_objective_cosine_min_mean_max="
            f"({amtl_vs_objective_cosines.min().item()}, {amtl_vs_objective_cosines.mean().item()}, {amtl_vs_objective_cosines.max().item()})"
        )
        self._amtl_diagnostics_printed = True
    '''
    def _flatten_grad_list(
        self, grads: Sequence[torch.Tensor | None], params: Sequence[torch.nn.Parameter]
    ) -> torch.Tensor:
        flat_grads = []
        for grad, param in zip(grads, params):
            if grad is None:
                flat_grads.append(torch.zeros_like(param).reshape(-1))
            else:
                flat_grads.append(grad.reshape(-1))
        return torch.cat(flat_grads)

    def _set_flat_gradients(self, params: Sequence[torch.nn.Parameter], flat_grad: torch.Tensor) -> None:
        offset = 0
        for param in params:
            numel = param.numel()
            param_grad = flat_grad[offset : offset + numel].view_as(param)
            if param.grad is None:
                param.grad = param_grad.clone()
            else:
                param.grad.copy_(param_grad)
            offset += numel

    def _align_loss_gradients(
        self, losses: Sequence[torch.Tensor], params: Sequence[torch.nn.Parameter]
    ) -> torch.Tensor:
        if len(losses) == 0:
            return torch.zeros(sum(param.numel() for param in params), device=self.device)

        task_grads = []
        for loss in losses:
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            task_grads.append(self._flatten_grad_list(grads, params))

        aligned_grads = [grad.clone() for grad in task_grads]
        for task_index, grad in enumerate(aligned_grads):
            for other_index, other_grad in enumerate(task_grads):
                if task_index == other_index:
                    continue
                dot_product = torch.dot(grad, other_grad)
                if dot_product < 0:
                    grad_norm_sq = torch.dot(other_grad, other_grad).clamp_min(1.0e-12)
                    grad -= dot_product / grad_norm_sq * other_grad

        return torch.stack(aligned_grads, dim=0).mean(dim=0)
    '''

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        # Create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            value_size=getattr(self.policy, "num_critic_heads", 1),
            device=self.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        reward_terms = self._get_reward_terms()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            timeout_mask = extras["time_outs"].unsqueeze(1).to(self.device)
            if self.transition.values.shape[-1] > 1 and reward_terms is not None:
                timeout_bonus_by_term = self.gamma * self.transition.values * timeout_mask
                self.transition.rewards += timeout_bonus_by_term.sum(dim=-1)
                reward_terms = reward_terms + timeout_bonus_by_term
            else:
                timeout_bonus = self.gamma * self.transition.values.sum(dim=-1) * extras["time_outs"].to(self.device)
                self.transition.rewards += timeout_bonus

        self.transition.reward_terms = reward_terms

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        # Compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    # Kept only as a historical reference while this branch uses the
    # actor-only `update()` implementation below.
    def _update_experimental_actor_amtl_legacy(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        debug_logged_this_update = False
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
            advantages_by_term_batch,
            returns_by_term_batch,
        ) in generator:
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    scalar_advantage_std = advantages_batch.std() + 1e-8
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / scalar_advantage_std
                    if advantages_by_term_batch is not None:
                        term_mean = advantages_by_term_batch.mean(dim=0, keepdim=True)
                        advantages_by_term_batch = (
                            advantages_by_term_batch - term_mean
                        ) / scalar_advantage_std

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
                if advantages_by_term_batch is not None:
                    advantages_by_term_batch = advantages_by_term_batch.repeat(num_aug, 1)
                if returns_by_term_batch is not None:
                    returns_by_term_batch = returns_by_term_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate




            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_loss = surrogate.mean()
            surrogate_losses_by_term = None

            if advantages_by_term_batch is not None:
                surrogate_by_term = -advantages_by_term_batch * ratio.unsqueeze(-1)
                surrogate_losses_by_term = surrogate_by_term.reshape(-1, surrogate_by_term.shape[-1]).mean(dim=0)
                surrogate_loss = surrogate_losses_by_term.mean()

            critic_losses_by_term = None
            if returns_by_term_batch is not None:
                if value_batch.shape[-1] != returns_by_term_batch.shape[-1]:
                    raise ValueError(
                        f"Critic output dimension ({value_batch.shape[-1]}) does not match "
                        f"the number of reward terms ({returns_by_term_batch.shape[-1]})."
                    )
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_by_term_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_by_term_batch).pow(2)
                    critic_losses_by_term = torch.max(value_losses, value_losses_clipped).reshape(
                        -1, value_batch.shape[-1]
                    ).mean(dim=0)
                else:
                    critic_losses_by_term = (returns_by_term_batch - value_batch).pow(2).reshape(
                        -1, value_batch.shape[-1]
                    ).mean(dim=0)
                value_loss = critic_losses_by_term.mean()
            else:
                if value_batch.shape[-1] != 1:
                    raise ValueError(
                        "Multi-head critic requires per-term returns, but rollout storage did not provide them."
                    )
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

            actor_auxiliary_loss = -self.entropy_coef * entropy_batch.mean()


            '''
            # Surrogate loss by term now that we have the option to do gradient surgery on each term separately
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            surrogate_losses_by_term = None
        
            
            #for ppo
            if advantages_by_term_batch is not None:
                ratio_by_term = ratio.unsqueeze(-1)
                clipped_ratio_by_term = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param).unsqueeze(-1)
                surrogate_by_term = -advantages_by_term_batch * ratio_by_term
                surrogate_clipped_by_term = -advantages_by_term_batch * clipped_ratio_by_term
                surrogate_losses_by_term = torch.max(surrogate_by_term, surrogate_clipped_by_term).mean(dim=0)
                surrogate_loss = surrogate_losses_by_term.mean()
                #print("advantages_by_term_batch:", advantages_by_term_batch.shape)
           '''


            # Value function loss
            # if self.use_clipped_value_loss:
            #     value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
            #         -self.clip_param, self.clip_param
            #     )
            #     value_losses = (value_batch - returns_batch).pow(2)
            #     value_losses_clipped = (value_clipped - returns_batch).pow(2)
            #     value_loss = torch.max(value_losses, value_losses_clipped).mean()
            # else:
            #     value_loss = (returns_batch - value_batch).pow(2).mean()

            # auxiliary_loss = self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    actor_auxiliary_loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            actor_params = list(self.policy.get_actor_parameters())
            critic_params = list(self.policy.get_critic_parameters())
            actor_objective_grad_norms = None
            actor_amtl_objectives = 0
            final_actor_aligned_grad_norm = None
            final_critic_grad_norm = None

            if surrogate_losses_by_term is not None:
                if self.align_actor_all_terms:
                    expected_num_actor_terms = len(MOTION_AMTL_TERM_NAMES) + len(REGULARIZER_TERM_NAMES)
                    if len(surrogate_losses_by_term) != expected_num_actor_terms:
                        raise ValueError(
                            f"Expected {expected_num_actor_terms} actor objective losses for full-term AMTL, "
                            f"got {len(surrogate_losses_by_term)}."
                        )

                    grads = []
                    actor_objective_grad_norms = []
                    for loss in surrogate_losses_by_term:
                        self._zero_existing_grads(actor_params)
                        loss.backward(retain_graph=True)
                        grad = self._collect_flat_grad(actor_params)
                        grads.append(grad)
                        actor_objective_grad_norms.append(grad.norm().item())

                    self._zero_existing_grads(actor_params)
                    grads = torch.stack(grads, dim=0)
                    grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0))
                    aligned_actor_grad = grads[0].sum(-1)
                    final_actor_aligned_grad_norm = aligned_actor_grad.norm().item()
                    actor_amtl_objectives = grads.shape[-1]
                    set_shared_grad(actor_params, aligned_actor_grad)
                else:
                    hybrid_term_indices = self._get_hybrid_term_indices()
                    params = actor_params

                    if hybrid_term_indices is None:
                        grads = []
                        actor_objective_grad_norms = []
                        for loss in surrogate_losses_by_term:
                            self._zero_existing_grads(params)
                            loss.backward(retain_graph=True)
                            grad = self._collect_flat_grad(params)
                            grads.append(grad)
                            actor_objective_grad_norms.append(grad.norm().item())

                        self._zero_existing_grads(params)
                        grads = torch.stack(grads, dim=0)
                        grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0))
                        aligned_actor_grad = grads[0].sum(-1)
                        final_actor_aligned_grad_norm = aligned_actor_grad.norm().item()
                        actor_amtl_objectives = grads.shape[-1]
                        set_shared_grad(params, aligned_actor_grad)
                    else:
                        motion_indices, regularizer_indices = hybrid_term_indices

                        motion_grads = []
                        actor_objective_grad_norms = []
                        for idx in motion_indices:
                            self._zero_existing_grads(params)
                            surrogate_losses_by_term[idx].backward(retain_graph=True)
                            grad = self._collect_flat_grad(params)
                            motion_grads.append(grad)
                            actor_objective_grad_norms.append(grad.norm().item())

                        self._zero_existing_grads(params)
                        motion_grads = torch.stack(motion_grads, dim=0)
                        aligned_motion_grads, weights, singulars = ProcrustesSolver.apply(
                            motion_grads.T.unsqueeze(0)
                        )
                        aligned_motion_grad = aligned_motion_grads[0].sum(-1)

                        regularizer_loss = surrogate_losses_by_term[regularizer_indices].sum()
                        regularizer_loss.backward(retain_graph=True)
                        regularizer_grad = self._collect_flat_grad(params)

                        self._zero_existing_grads(params)
                        total_actor_grad = aligned_motion_grad + regularizer_grad
                        final_actor_aligned_grad_norm = total_actor_grad.norm().item()
                        actor_amtl_objectives = motion_grads.shape[0]
                        set_shared_grad(params, total_actor_grad)

                actor_auxiliary_loss.backward()
            else:
                (surrogate_loss + actor_auxiliary_loss).backward()
                actor_amtl_objectives = 1

            if critic_losses_by_term is not None and self.align_critic:
                critic_grads = []
                for loss in critic_losses_by_term:
                    self._zero_existing_grads(critic_params)
                    (self.value_loss_coef * loss).backward(retain_graph=True)
                    critic_grads.append(self._collect_flat_grad(critic_params))

                self._zero_existing_grads(critic_params)
                critic_grads = torch.stack(critic_grads, dim=0)
                aligned_critic_grads, weights, singulars = ProcrustesSolver.apply(critic_grads.T.unsqueeze(0))
                aligned_critic_grad = aligned_critic_grads[0].sum(-1)
                set_shared_grad(critic_params, aligned_critic_grad)
                final_critic_grad_norm = aligned_critic_grad.norm().item()
            else:
                (self.value_loss_coef * value_loss).backward()
                final_critic_grad_norm = self._collect_flat_grad(critic_params).norm().item()

            if (
                self.debug_amtl
                and not debug_logged_this_update
                and self.gpu_global_rank == 0
                and self.update_counter % max(self.debug_amtl_log_interval, 1) == 0
            ):
                critic_objectives = len(critic_losses_by_term) if critic_losses_by_term is not None else 1
                critic_value_losses = (
                    critic_losses_by_term.detach().cpu().tolist()
                    if critic_losses_by_term is not None
                    else [value_loss.item()]
                )
                print("Actor AMTL: ENABLED" if surrogate_losses_by_term is not None else "Actor AMTL: DISABLED")
                print(f"Actor AMTL objectives: {actor_amtl_objectives}")
                print(f"Critic heads: {value_batch.shape[-1]}")
                print(f"Critic objectives: {critic_objectives}")
                print(f"Critic AMTL: {'ENABLED' if self.align_critic else 'DISABLED'}")
                print("Critic training: normal mean per-head MSE" if not self.align_critic else "Critic training: AMTL-aligned per-head MSE")
                if actor_objective_grad_norms is not None:
                    print(f"Actor gradient norms per objective: {actor_objective_grad_norms}")
                print(f"Critic value loss per head: {critic_value_losses}")
                print(f"Final actor aligned gradient norm: {final_actor_aligned_grad_norm}")
                print(f"Final critic gradient norm: {final_critic_grad_norm}")
                debug_logged_this_update = True

            # self.optimizer.step()

            # grads = []
            # for loss in surrogate_losses_by_term:
            #     for p in self.policy.parameters():
            #         if p.grad is not None:
            #             p.grad.data.zero_()
            #     loss.backward(retain_graph=True)
            #     grad = torch.cat([p.grad.flatten().clone() if p.grad is not None else torch.zeros_like(p).flatten() for p in self.policy.parameters()])
            #     grads.append(grad)
            #
            # for p in self.policy.parameters():
            #     if p.grad is not None:
            #         p.grad.data.zero_()
            #
            # grads = torch.stack(grads, dim=0)
            # grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0))
            # grad, weights = grads[0].sum(-1), weights.sum(-1)
            # set_shared_grad(self.policy.parameters(), grad)
            #
            # self.optimizer.step()












            '''
            if surrogate_losses_by_term is not None:
                params = [param for param in self.policy.parameters() if param.requires_grad]
                aligned_surrogate_grad = self._align_loss_gradients(list(surrogate_losses_by_term.unbind()), params)
                auxiliary_grads = torch.autograd.grad(auxiliary_loss, params, allow_unused=True)
                combined_grad = aligned_surrogate_grad + self._flatten_grad_list(auxiliary_grads, params)
                self._set_flat_gradients(params, combined_grad)
            else:
                loss = surrogate_loss + auxiliary_loss
                loss.backward()
            '''

            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()


















            # Apply the gradients for PPO
          #  nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
           # self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        self.update_counter += 1
        return loss_dict

    def update(self) -> dict[str, float]:
        """AMTL-enabled PPO update.

        The actor aggregates per-objective PPO surrogate gradients with PGA.
        The critic always uses the standard mean value-loss gradient.
        """

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl_divergence = 0.0
        mean_actor_amtl_grad_norm = 0.0
        mean_num_actor_objectives = 0.0
        mean_num_critic_objectives = 0.0
        mean_actor_raw_mean_cosine = 0.0
        mean_actor_raw_min_cosine = 0.0
        mean_actor_raw_max_cosine = 0.0
        mean_actor_raw_conflict_fraction = 0.0
        mean_actor_aligned_mean_cosine = 0.0
        mean_actor_aligned_min_cosine = 0.0
        mean_actor_aligned_max_cosine = 0.0
        mean_actor_aligned_conflict_fraction = 0.0
        mean_actor_alignment_mean_cosine = 0.0
        mean_actor_amtl_vs_ppo_cosine = 0.0
        mean_actor_amtl_projection_on_ppo = 0.0
        mean_actor_amtl_projection_ratio = 0.0
        mean_actor_amtl_parallel_norm = 0.0
        mean_actor_amtl_orthogonal_norm = 0.0
        mean_actor_amtl_orthogonal_ratio = 0.0
        mean_actor_objective_vs_ppo_cosine_min = 0.0
        mean_actor_objective_vs_ppo_cosine_max = 0.0
        mean_actor_objective_vs_ppo_cosine_mean = 0.0
        mean_actor_amtl_vs_objective_cosine_min = 0.0
        mean_actor_amtl_vs_objective_cosine_max = 0.0
        mean_actor_amtl_vs_objective_cosine_mean = 0.0
        mean_actor_pga_retained_rank = 0.0
        mean_actor_pga_sigma_min = 0.0
        mean_actor_pga_sigma_max = 0.0
        mean_actor_pga_sigma_condition_number = 0.0
        mean_actor_scale = 0.0
        mean_critic_amtl_grad_norm = 0.0
        critic_value_loss_sums: torch.Tensor | None = None
        action_entropy_sums: torch.Tensor | None = None
        actor_objective_vs_ppo_cosine_sums: torch.Tensor | None = None
        actor_amtl_vs_objective_cosine_sums: torch.Tensor | None = None
        actor_pga_objective_weight_sums: torch.Tensor | None = None
        actor_pga_objective_contribution_norm_sums: torch.Tensor | None = None
        actor_pga_objective_projection_sums: torch.Tensor | None = None
        actor_pga_singular_value_sums: torch.Tensor | None = None
        actor_pga_energy_sums: torch.Tensor | None = None
        actor_pga_energy_fraction_sums: torch.Tensor | None = None
        actor_pga_svd_direction_loading_sums: torch.Tensor | None = None
        actor_pga_svd_direction_energy_fraction_sums: torch.Tensor | None = None
        actor_hierarchical_diagnostic_sums: dict[str, torch.Tensor] = {}
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        debug_logged_this_update = False
        pga_call_index = 0
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
            advantages_by_term_batch,
            returns_by_term_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    scalar_advantage_std = advantages_batch.std() + 1e-8
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / scalar_advantage_std
                    if advantages_by_term_batch is not None:
                        term_mean = advantages_by_term_batch.mean(dim=0, keepdim=True)
                        advantages_by_term_batch = (
                            advantages_by_term_batch - term_mean
                        ) / scalar_advantage_std

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
                if advantages_by_term_batch is not None:
                    advantages_by_term_batch = advantages_by_term_batch.repeat(num_aug, 1)
                if returns_by_term_batch is not None:
                    returns_by_term_batch = returns_by_term_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            action_entropy_batch = self.policy.distribution.entropy()[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]
            self._validate_policy_distribution_numerics(
                mu_batch,
                sigma_batch,
                old_mu_batch,
                old_sigma_batch,
                action_entropy_batch,
                context=f"update={self.update_counter}, minibatch={pga_call_index}",
            )

            kl_mean_value = 0.0
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

                    kl_mean_value = kl_mean.item()

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * clipped_ratio
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            surrogate_losses_by_term = None
            if advantages_by_term_batch is not None:
                surrogate_by_term = -advantages_by_term_batch * ratio.unsqueeze(-1)
                surrogate_clipped_by_term = -advantages_by_term_batch * clipped_ratio.unsqueeze(-1)
                surrogate_losses_by_term = torch.max(surrogate_by_term, surrogate_clipped_by_term).reshape(
                    -1, advantages_by_term_batch.shape[-1]
                ).mean(dim=0)

            critic_losses_by_term = None
            if returns_by_term_batch is not None:
                if value_batch.shape[-1] != returns_by_term_batch.shape[-1]:
                    raise ValueError(
                        f"Critic output dimension ({value_batch.shape[-1]}) does not match "
                        f"the number of reward terms ({returns_by_term_batch.shape[-1]})."
                    )
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_by_term_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_by_term_batch).pow(2)
                    critic_losses_by_term = torch.max(value_losses, value_losses_clipped).reshape(
                        -1, value_batch.shape[-1]
                    ).mean(dim=0)
                else:
                    critic_losses_by_term = (returns_by_term_batch - value_batch).pow(2).reshape(
                        -1, value_batch.shape[-1]
                    ).mean(dim=0)
                value_loss = critic_losses_by_term.mean()
                num_critic_objectives = len(critic_losses_by_term)
            else:
                if value_batch.shape[-1] != 1:
                    raise ValueError(
                        "Multi-head critic requires per-term returns, but rollout storage did not provide them."
                    )
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

            actor_auxiliary_loss = -self.entropy_coef * entropy_batch.mean()
            if self.symmetry:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                if not self.symmetry["use_data_augmentation"]:
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                symmetry_loss = torch.nn.MSELoss()(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    actor_auxiliary_loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            # PGA is applied only to the policy-mean network. Exploration std
            # follows the ordinary scalar PPO surrogate plus entropy.
            actor_params = list(self.policy.get_actor_mean_parameters())
            actor_std_params = list(self.policy.get_actor_std_parameters())
            critic_params = list(self.policy.get_critic_parameters())

            actor_amtl_grad_norm = 0.0
            actor_raw_mean_cosine = 0.0
            actor_raw_min_cosine = 0.0
            actor_raw_max_cosine = 0.0
            actor_raw_conflict_fraction = 0.0
            actor_aligned_mean_cosine = 0.0
            actor_aligned_min_cosine = 0.0
            actor_aligned_max_cosine = 0.0
            actor_aligned_conflict_fraction = 0.0
            actor_alignment_mean_cosine = 0.0
            actor_amtl_vs_ppo_cosine = 0.0
            actor_amtl_projection_on_ppo = 0.0
            actor_amtl_projection_ratio = 0.0
            actor_amtl_parallel_norm = 0.0
            actor_amtl_orthogonal_norm = 0.0
            actor_amtl_orthogonal_ratio = 0.0
            actor_objective_vs_ppo_cosine_min = 0.0
            actor_objective_vs_ppo_cosine_max = 0.0
            actor_objective_vs_ppo_cosine_mean = 0.0
            actor_amtl_vs_objective_cosine_min = 0.0
            actor_amtl_vs_objective_cosine_max = 0.0
            actor_amtl_vs_objective_cosine_mean = 0.0
            actor_objective_vs_ppo_cosines: torch.Tensor | None = None
            actor_amtl_vs_objective_cosines: torch.Tensor | None = None
            actor_pga_objective_weights: torch.Tensor | None = None
            actor_pga_objective_contribution_norms: torch.Tensor | None = None
            actor_pga_objective_projections: torch.Tensor | None = None
            actor_pga_singular_values: torch.Tensor | None = None
            actor_pga_energies: torch.Tensor | None = None
            actor_pga_energy_fractions: torch.Tensor | None = None
            actor_pga_svd_direction_loadings: torch.Tensor | None = None
            actor_pga_svd_direction_energy_fractions: torch.Tensor | None = None
            actor_pga_retained_rank = 0.0
            actor_pga_sigma_min = 0.0
            actor_pga_sigma_max = 0.0
            actor_pga_sigma_condition_number = 0.0
            actor_hierarchical_diagnostics: dict[str, torch.Tensor] = {}
            actor_scale = 0.0
            num_actor_objectives = 1
            num_critic_objectives = 1
            stacked_actor_grads: torch.Tensor | None = None
            aligned_actor_grads: torch.Tensor | None = None
            aligned_actor_grad_matrix: torch.Tensor | None = None
            aligned_actor_grad: torch.Tensor | None = None
            raw_actor_cosine_stats = {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "conflict_fraction": 0.0,
                "near_zero_row_count": 0.0,
            }
            aligned_actor_cosine_stats = dict(raw_actor_cosine_stats)
            stacked_critic_grads: torch.Tensor | None = None
            procrustes_critic_input: torch.Tensor | None = None
            aligned_critic_grad: torch.Tensor | None = None

            if self.use_actor_amtl:
                if surrogate_losses_by_term is None:
                    raise ValueError("Actor-only AMTL requires per-objective advantages, but none were provided.")

                standard_actor_grad_list = torch.autograd.grad(
                    surrogate_loss, actor_params, retain_graph=True, allow_unused=True
                )
                standard_actor_grad = self._flatten_grad_list(standard_actor_grad_list, actor_params)
                actor_grads = []
                num_actor_objectives = len(surrogate_losses_by_term)
                for loss in surrogate_losses_by_term:
                    grad_list = torch.autograd.grad(
                        loss, actor_params, retain_graph=True, allow_unused=True
                    )
                    actor_grads.append(self._flatten_grad_list(grad_list, actor_params))

                self._zero_existing_grads(actor_params)
                stacked_actor_grads = torch.stack(actor_grads, dim=0)
                raw_actor_cosine_stats = self._pairwise_cosine_stats(stacked_actor_grads)
                objective_names = self._get_actor_objective_names(num_actor_objectives)
                # PGA expects columns to be objective gradients. This is the
                # same gradient layout used in sample_example.txt:
                # [1, parameters, objectives]. Its aggregate replaces the
                # actor gradients immediately before the optimizer step.
                pga_context = (
                    f"update={self.update_counter}, minibatch={pga_call_index}, "
                    f"mode={self.actor_pga_mode}, rank={self.pga_rank}, "
                    f"weighting={self.pga_direction_weighting}"
                )
                hierarchical_result: HierarchicalPGAResult | None = None
                if self.actor_pga_mode == "flat":
                    aggregate_grad, actor_pga_objective_weights, singular_values, _ = PGASolver.apply(
                        stacked_actor_grads.T.unsqueeze(0),
                        rank=self.pga_rank,
                        method="os_nmf",
                        direction_weighting=self.pga_direction_weighting,
                        diagnostic_context=pga_context,
                        objective_names=objective_names,
                    )
                    aligned_actor_grad = aggregate_grad.squeeze(0)
                else:
                    hierarchical_result = aggregate_hierarchical_5_pga(
                        stacked_actor_grads,
                        objective_names,
                        rank=self.pga_rank,
                        direction_weighting=self.pga_direction_weighting,
                        diagnostic_context=pga_context,
                    )
                    aligned_actor_grad = hierarchical_result.aggregate_grad
                    actor_pga_objective_weights = hierarchical_result.objective_weights
                    # Keep the existing 16-objective spectrum tags comparable
                    # with flat runs. Applied outer-PGA spectra use dedicated
                    # actor_hierarchical/outer_pga tags below.
                    singular_values = hierarchical_result.flat_singular_values
                    hierarchical_grad_norm_before_ppo_match = aligned_actor_grad.norm()
                pga_call_index += 1
                if self.pga_match_ppo_grad_norm:
                    aligned_norm = aligned_actor_grad.norm()
                    norm_scale = torch.where(
                        aligned_norm > 1.0e-12,
                        standard_actor_grad.norm() / aligned_norm.clamp_min(1.0e-12),
                        torch.ones_like(aligned_norm),
                    )
                    aligned_actor_grad = aligned_actor_grad * norm_scale
                    actor_pga_objective_weights = actor_pga_objective_weights * norm_scale
                if hierarchical_result is not None:
                    group_norms = hierarchical_result.group_grads.norm(dim=1)
                    weighted_group_norms = hierarchical_result.weighted_group_grads.norm(dim=1)
                    group_vs_final_cosines = self._cosine_against_reference(
                        hierarchical_result.group_grads, aligned_actor_grad
                    )
                    num_groups = len(HIERARCHICAL_5_PGA_GROUPS)
                    outer_singular_values = aligned_actor_grad.new_zeros(num_groups)
                    outer_singular_values[
                        : len(hierarchical_result.outer_singular_values)
                    ] = hierarchical_result.outer_singular_values
                    outer_energies = outer_singular_values.square()
                    outer_energy_fractions = outer_energies / outer_energies.sum().clamp_min(1.0e-12)
                    outer_factor_strengths = torch.zeros_like(outer_singular_values)
                    outer_factor_strengths[
                        : len(hierarchical_result.outer_factor_strengths)
                    ] = hierarchical_result.outer_factor_strengths
                    outer_factor_weights = torch.zeros_like(outer_singular_values)
                    outer_factor_weights[
                        : len(hierarchical_result.outer_factor_weights)
                    ] = hierarchical_result.outer_factor_weights
                    outer_sigma_min = hierarchical_result.outer_singular_values.min()
                    outer_sigma_max = hierarchical_result.outer_singular_values.max()
                    actor_hierarchical_diagnostics.update(
                        {
                            "actor_hierarchical_final_grad_norm": aligned_actor_grad.norm(),
                            "actor_hierarchical_final_grad_norm_before_ppo_match": (
                                hierarchical_grad_norm_before_ppo_match
                            ),
                            "actor_hierarchical_vs_flat_cosine": aligned_actor_grad.new_tensor(
                                self._safe_cosine_similarity(
                                    aligned_actor_grad, hierarchical_result.flat_aggregate_grad
                                )
                            ),
                            "actor_hierarchical_outer_pga_retained_rank": aligned_actor_grad.new_tensor(
                                float(len(hierarchical_result.outer_singular_values))
                            ),
                            "actor_hierarchical_outer_pga_sigma_min": outer_sigma_min,
                            "actor_hierarchical_outer_pga_sigma_max": outer_sigma_max,
                            "actor_hierarchical_outer_pga_sigma_condition_number": (
                                outer_sigma_max
                                / outer_sigma_min.clamp_min(
                                    torch.finfo(outer_sigma_min.dtype).tiny
                                )
                            ),
                        }
                    )
                    for group_index, (group_name, _, group_weight) in enumerate(
                        HIERARCHICAL_5_PGA_GROUPS
                    ):
                        actor_hierarchical_diagnostics[
                            f"actor_hierarchical_group_norm_before_outer_weight/{group_name}"
                        ] = group_norms[group_index]
                        actor_hierarchical_diagnostics[
                            f"actor_hierarchical_outer_input_weight/{group_name}"
                        ] = aligned_actor_grad.new_tensor(group_weight)
                        actor_hierarchical_diagnostics[
                            f"actor_hierarchical_weighted_group_norm/{group_name}"
                        ] = weighted_group_norms[group_index]
                        actor_hierarchical_diagnostics[
                            f"actor_hierarchical_group_vs_final_cosine/{group_name}"
                        ] = group_vs_final_cosines[group_index]
                        actor_hierarchical_diagnostics[
                            f"actor_hierarchical_outer_pga_objective_weight/{group_name}"
                        ] = hierarchical_result.outer_objective_weights[group_index]
                        for other_group_index, (other_group_name, _, _) in enumerate(
                            HIERARCHICAL_5_PGA_GROUPS
                        ):
                            actor_hierarchical_diagnostics[
                                f"actor_hierarchical_group_cosine/{group_name}__{other_group_name}"
                            ] = hierarchical_result.group_cosine_matrix[
                                group_index, other_group_index
                            ]
                    for direction_index in range(num_groups):
                        direction_name = f"direction_{direction_index:02d}"
                        actor_hierarchical_diagnostics[
                            f"actor_hierarchical_outer_pga_singular_value/{direction_name}"
                        ] = outer_singular_values[direction_index]
                        actor_hierarchical_diagnostics[
                            f"actor_hierarchical_outer_pga_energy/{direction_name}"
                        ] = outer_energies[direction_index]
                        actor_hierarchical_diagnostics[
                            f"actor_hierarchical_outer_pga_energy_fraction/{direction_name}"
                        ] = outer_energy_fractions[direction_index]
                        actor_hierarchical_diagnostics[
                            f"actor_hierarchical_outer_pga_factor_strength/{direction_name}"
                        ] = outer_factor_strengths[direction_index]
                        actor_hierarchical_diagnostics[
                            f"actor_hierarchical_outer_pga_factor_weight/{direction_name}"
                        ] = outer_factor_weights[direction_index]
                if (
                    not bool(torch.isfinite(aligned_actor_grad).all())
                    or not bool(torch.isfinite(actor_pga_objective_weights).all())
                ):
                    _raise_numeric_error(
                        "pga_norm_matching",
                        {
                            "stacked_actor_grads": stacked_actor_grads,
                            "standard_actor_grad": standard_actor_grad,
                            "aligned_actor_grad": aligned_actor_grad,
                            "objective_weights": actor_pga_objective_weights,
                        },
                        context=(
                            f"update={self.update_counter}, minibatch={pga_call_index - 1}, "
                            f"rank={self.pga_rank}, weighting={self.pga_direction_weighting}"
                        ),
                    )

                actor_pga_retained_rank = float(len(singular_values))
                actor_pga_sigma_min = float(singular_values.min().item())
                actor_pga_sigma_max = float(singular_values.max().item())
                actor_pga_sigma_condition_number = (
                    actor_pga_sigma_max / max(actor_pga_sigma_min, torch.finfo(singular_values.dtype).tiny)
                )

                # Singular values are ordered by PGA direction strength. Pad
                # numerically discarded directions with zeros so every update
                # logs a stable, one-entry-per-objective spectrum.
                actor_pga_singular_values = torch.zeros(
                    num_actor_objectives,
                    device=singular_values.device,
                    dtype=singular_values.dtype,
                )
                actor_pga_singular_values[: len(singular_values)] = singular_values
                sigma_square_limit = math.sqrt(torch.finfo(singular_values.dtype).max)
                if bool(torch.any(singular_values.abs() > sigma_square_limit)):
                    _raise_numeric_error(
                        "pga_sigma_energy_square_overflow",
                        {
                            "retained_sigmas": singular_values,
                            "stacked_actor_grads": stacked_actor_grads,
                        },
                        context=(
                            f"update={self.update_counter}, minibatch={pga_call_index - 1}, "
                            f"float_square_limit={sigma_square_limit:.9e}"
                        ),
                    )
                actor_pga_energies = actor_pga_singular_values.square()
                actor_pga_energy_fractions = actor_pga_energies / actor_pga_energies.sum().clamp_min(1.0e-12)
                if not bool(torch.isfinite(actor_pga_energy_fractions).all()):
                    _raise_numeric_error(
                        "pga_sigma_energy_logging",
                        {
                            "retained_sigmas": singular_values,
                            "padded_sigmas": actor_pga_singular_values,
                            "sigma_energies": actor_pga_energies,
                            "sigma_energy_fractions": actor_pga_energy_fractions,
                        },
                        context=f"update={self.update_counter}, minibatch={pga_call_index - 1}",
                    )

                # Decompose each ranked SVD direction into named objectives.
                # ``right_vectors[:, k]`` is the objective-space loading for
                # the same direction whose energy is ``singular_values[k]``.
                # Eigenvector signs are arbitrary, so log absolute loadings
                # and squared fractions rather than unstable signed values.
                objective_gram = stacked_actor_grads @ stacked_actor_grads.T
                if not bool(torch.isfinite(objective_gram).all()):
                    _raise_numeric_error(
                        "pga_logging_gram_matrix",
                        {
                            "stacked_actor_grads": stacked_actor_grads,
                            "objective_gram": objective_gram,
                        },
                        context=f"update={self.update_counter}, minibatch={pga_call_index - 1}",
                    )
                try:
                    _, right_vectors = torch.linalg.eigh(objective_gram)
                except RuntimeError as exc:
                    _raise_numeric_error(
                        "pga_logging_eigendecomposition",
                        {
                            "stacked_actor_grads": stacked_actor_grads,
                            "objective_gram": objective_gram,
                        },
                        context=f"update={self.update_counter}, minibatch={pga_call_index - 1}",
                        cause=exc,
                    )
                right_vectors = right_vectors.flip(dims=(1,))
                actor_pga_svd_direction_loadings = right_vectors.abs()
                actor_pga_svd_direction_energy_fractions = right_vectors.square()

                # PGA returns ``aggregate = sum_i weight_i * gradient_i``.
                # Log both the magnitude of each addend and its signed share
                # along the final update direction. The latter sums to one
                # (up to numerical precision), while a negative value means
                # that objective is cancelling the final update.
                weighted_objective_grads = (
                    stacked_actor_grads * actor_pga_objective_weights.unsqueeze(1)
                )
                actor_pga_objective_contribution_norms = torch.linalg.vector_norm(
                    weighted_objective_grads, dim=1
                )
                actor_pga_objective_projections = (
                    weighted_objective_grads @ aligned_actor_grad
                ) / aligned_actor_grad.square().sum().clamp_min(1.0e-12)

                # Retain the existing diagnostics interface. PGA produces one
                # aggregate rather than a transformed gradient per objective,
                # so its per-objective diagnostic matrix is the original
                # objective matrix and pairwise statistics are unchanged.
                aligned_actor_grads = stacked_actor_grads.T.unsqueeze(0)
                aligned_actor_grad_matrix = stacked_actor_grads
                aligned_actor_cosine_stats = raw_actor_cosine_stats
                standard_actor_grad_norm = standard_actor_grad.norm()
                aligned_actor_grad_norm = aligned_actor_grad.norm()
                aligned_vs_ppo_dot = torch.dot(aligned_actor_grad, standard_actor_grad)
                ppo_unit = (
                    standard_actor_grad / standard_actor_grad_norm.clamp_min(1.0e-12)
                    if standard_actor_grad_norm.item() > 1.0e-12
                    else torch.zeros_like(standard_actor_grad)
                )
                parallel_actor_grad = torch.dot(aligned_actor_grad, ppo_unit) * ppo_unit
                orthogonal_actor_grad = aligned_actor_grad - parallel_actor_grad
                actor_objective_vs_ppo_cosines = self._cosine_against_reference(
                    stacked_actor_grads, standard_actor_grad
                )
                actor_amtl_vs_objective_cosines = self._cosine_against_reference(
                    stacked_actor_grads, aligned_actor_grad
                )
                actor_amtl_grad_norm = aligned_actor_grad_norm.item()
                actor_raw_mean_cosine = raw_actor_cosine_stats["mean"]
                actor_raw_min_cosine = raw_actor_cosine_stats["min"]
                actor_raw_max_cosine = raw_actor_cosine_stats["max"]
                actor_raw_conflict_fraction = raw_actor_cosine_stats["conflict_fraction"]
                actor_aligned_mean_cosine = aligned_actor_cosine_stats["mean"]
                actor_aligned_min_cosine = aligned_actor_cosine_stats["min"]
                actor_aligned_max_cosine = aligned_actor_cosine_stats["max"]
                actor_aligned_conflict_fraction = aligned_actor_cosine_stats["conflict_fraction"]
                actor_alignment_mean_cosine = actor_aligned_mean_cosine
                actor_amtl_vs_ppo_cosine = self._safe_cosine_similarity(aligned_actor_grad, standard_actor_grad)
                actor_amtl_projection_on_ppo = (
                    aligned_vs_ppo_dot / standard_actor_grad_norm.clamp_min(1.0e-12)
                ).item()
                actor_amtl_projection_ratio = (
                    aligned_vs_ppo_dot / standard_actor_grad_norm.square().clamp_min(1.0e-12)
                ).item()
                actor_amtl_parallel_norm = parallel_actor_grad.norm().item()
                actor_amtl_orthogonal_norm = orthogonal_actor_grad.norm().item()
                actor_amtl_orthogonal_ratio = (
                    orthogonal_actor_grad.norm() / aligned_actor_grad_norm.clamp_min(1.0e-12)
                ).item()
                actor_objective_vs_ppo_cosine_min = actor_objective_vs_ppo_cosines.min().item()
                actor_objective_vs_ppo_cosine_max = actor_objective_vs_ppo_cosines.max().item()
                actor_objective_vs_ppo_cosine_mean = actor_objective_vs_ppo_cosines.mean().item()
                actor_amtl_vs_objective_cosine_min = actor_amtl_vs_objective_cosines.min().item()
                actor_amtl_vs_objective_cosine_max = actor_amtl_vs_objective_cosines.max().item()
                actor_amtl_vs_objective_cosine_mean = actor_amtl_vs_objective_cosines.mean().item()
                actor_scale = (aligned_actor_grad_norm / (standard_actor_grad.norm() + 1.0e-8)).item()
                set_shared_grad(actor_params, aligned_actor_grad)

                # Do not expose exploration std to the per-objective PGA
                # solve. Its gradient comes from normal scalar PPO plus the
                # entropy regularizer (symmetry has no std dependency).
                if actor_std_params:
                    std_grad_list = torch.autograd.grad(
                        surrogate_loss + actor_auxiliary_loss,
                        actor_std_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    set_shared_grad(
                        actor_std_params,
                        self._flatten_grad_list(std_grad_list, actor_std_params),
                    )

                # Add mean-network auxiliary gradients (for example symmetry)
                # on top of the aligned PPO gradient. Entropy is independent
                # of the mean for the state-independent Normal policy.
                mean_aux_grad_list = torch.autograd.grad(
                    actor_auxiliary_loss,
                    actor_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                for param, auxiliary_grad in zip(actor_params, mean_aux_grad_list, strict=True):
                    if auxiliary_grad is not None:
                        param.grad.add_(auxiliary_grad)
            else:
                (surrogate_loss + actor_auxiliary_loss).backward()

            # Critic training always follows the normal mean value-loss path;
            # gradient alignment is intentionally actor-only.
            (self.value_loss_coef * value_loss).backward()
            critic_grad_norm = self._collect_flat_grad(critic_params).norm().item()

            if self.use_actor_amtl and stacked_actor_grads is not None and aligned_actor_grads is not None and aligned_actor_grad_matrix is not None and aligned_actor_grad is not None:
                self._print_amtl_diagnostics_once(
                    self.use_actor_amtl,
                    self.use_critic_amtl,
                    num_actor_objectives,
                    stacked_actor_grads,
                    aligned_actor_grads,
                    aligned_actor_grad_matrix,
                    aligned_actor_grad,
                    raw_actor_cosine_stats,
                    aligned_actor_cosine_stats,
                    actor_amtl_vs_ppo_cosine,
                    actor_amtl_projection_ratio,
                    actor_amtl_orthogonal_ratio,
                    actor_objective_vs_ppo_cosines if actor_objective_vs_ppo_cosines is not None else torch.zeros(1, device=self.device),
                    actor_amtl_vs_objective_cosines if actor_amtl_vs_objective_cosines is not None else torch.zeros(1, device=self.device),
                    num_critic_objectives,
                    stacked_critic_grads,
                    procrustes_critic_input,
                    aligned_critic_grad,
                    actor_params,
                    critic_params,
                )

            if (
                self.debug_amtl
                and not debug_logged_this_update
                and self.gpu_global_rank == 0
                and self.update_counter % max(self.debug_amtl_log_interval, 1) == 0
            ):
                critic_objectives = len(critic_losses_by_term) if critic_losses_by_term is not None else 1
                critic_value_losses = (
                    critic_losses_by_term.detach().cpu().tolist()
                    if critic_losses_by_term is not None
                    else [value_loss.item()]
                )
                print(f"Actor AMTL: {'ON' if self.use_actor_amtl else 'OFF'}")
                print(f"Actor AMTL objectives: {num_actor_objectives}")
                print(f"Actor AMTL gradient norm: {actor_amtl_grad_norm}")
                print(f"Actor alignment mean cosine: {actor_alignment_mean_cosine}")
                print(f"Actor scale: {actor_scale}")
                print(f"Critic heads: {value_batch.shape[-1]}")
                print(f"Critic objectives: {critic_objectives}")
                print(f"Critic AMTL: {'ON' if self.use_critic_amtl else 'OFF'}")
                print(
                    "Critic training: AMTL-aligned per-head MSE"
                    if self.use_critic_amtl
                    else "Critic training: normal mean per-head MSE"
                )
                print(f"Critic value loss per head: {critic_value_losses}")
                print(f"Critic gradient norm: {critic_grad_norm}")
                debug_logged_this_update = True

            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._enforce_min_action_std()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_kl_divergence += kl_mean_value
            mean_actor_amtl_grad_norm += actor_amtl_grad_norm
            mean_num_actor_objectives += num_actor_objectives
            mean_critic_amtl_grad_norm += critic_grad_norm if self.use_critic_amtl else 0.0
            mean_num_critic_objectives += num_critic_objectives
            mean_actor_raw_mean_cosine += actor_raw_mean_cosine
            mean_actor_raw_min_cosine += actor_raw_min_cosine
            mean_actor_raw_max_cosine += actor_raw_max_cosine
            mean_actor_raw_conflict_fraction += actor_raw_conflict_fraction
            mean_actor_aligned_mean_cosine += actor_aligned_mean_cosine
            mean_actor_aligned_min_cosine += actor_aligned_min_cosine
            mean_actor_aligned_max_cosine += actor_aligned_max_cosine
            mean_actor_aligned_conflict_fraction += actor_aligned_conflict_fraction
            mean_actor_alignment_mean_cosine += actor_alignment_mean_cosine
            mean_actor_amtl_vs_ppo_cosine += actor_amtl_vs_ppo_cosine
            mean_actor_amtl_projection_on_ppo += actor_amtl_projection_on_ppo
            mean_actor_amtl_projection_ratio += actor_amtl_projection_ratio
            mean_actor_amtl_parallel_norm += actor_amtl_parallel_norm
            mean_actor_amtl_orthogonal_norm += actor_amtl_orthogonal_norm
            mean_actor_amtl_orthogonal_ratio += actor_amtl_orthogonal_ratio
            mean_actor_objective_vs_ppo_cosine_min += actor_objective_vs_ppo_cosine_min
            mean_actor_objective_vs_ppo_cosine_max += actor_objective_vs_ppo_cosine_max
            mean_actor_objective_vs_ppo_cosine_mean += actor_objective_vs_ppo_cosine_mean
            mean_actor_amtl_vs_objective_cosine_min += actor_amtl_vs_objective_cosine_min
            mean_actor_amtl_vs_objective_cosine_max += actor_amtl_vs_objective_cosine_max
            mean_actor_amtl_vs_objective_cosine_mean += actor_amtl_vs_objective_cosine_mean
            mean_actor_pga_retained_rank += actor_pga_retained_rank
            mean_actor_pga_sigma_min += actor_pga_sigma_min
            mean_actor_pga_sigma_max += actor_pga_sigma_max
            mean_actor_pga_sigma_condition_number += actor_pga_sigma_condition_number
            mean_actor_scale += actor_scale
            if actor_objective_vs_ppo_cosines is not None:
                if actor_objective_vs_ppo_cosine_sums is None:
                    actor_objective_vs_ppo_cosine_sums = torch.zeros_like(actor_objective_vs_ppo_cosines)
                actor_objective_vs_ppo_cosine_sums += actor_objective_vs_ppo_cosines.detach()
            if actor_amtl_vs_objective_cosines is not None:
                if actor_amtl_vs_objective_cosine_sums is None:
                    actor_amtl_vs_objective_cosine_sums = torch.zeros_like(actor_amtl_vs_objective_cosines)
                actor_amtl_vs_objective_cosine_sums += actor_amtl_vs_objective_cosines.detach()
            if actor_pga_objective_weights is not None:
                if actor_pga_objective_weight_sums is None:
                    actor_pga_objective_weight_sums = torch.zeros_like(actor_pga_objective_weights)
                actor_pga_objective_weight_sums += actor_pga_objective_weights.detach()
            if actor_pga_objective_contribution_norms is not None:
                if actor_pga_objective_contribution_norm_sums is None:
                    actor_pga_objective_contribution_norm_sums = torch.zeros_like(
                        actor_pga_objective_contribution_norms
                    )
                actor_pga_objective_contribution_norm_sums += actor_pga_objective_contribution_norms.detach()
            if actor_pga_objective_projections is not None:
                if actor_pga_objective_projection_sums is None:
                    actor_pga_objective_projection_sums = torch.zeros_like(actor_pga_objective_projections)
                actor_pga_objective_projection_sums += actor_pga_objective_projections.detach()
            if actor_pga_singular_values is not None:
                if actor_pga_singular_value_sums is None:
                    actor_pga_singular_value_sums = torch.zeros_like(actor_pga_singular_values)
                actor_pga_singular_value_sums += actor_pga_singular_values.detach()
            if actor_pga_energies is not None:
                if actor_pga_energy_sums is None:
                    actor_pga_energy_sums = torch.zeros_like(actor_pga_energies)
                actor_pga_energy_sums += actor_pga_energies.detach()
            if actor_pga_energy_fractions is not None:
                if actor_pga_energy_fraction_sums is None:
                    actor_pga_energy_fraction_sums = torch.zeros_like(actor_pga_energy_fractions)
                actor_pga_energy_fraction_sums += actor_pga_energy_fractions.detach()
            if actor_pga_svd_direction_loadings is not None:
                if actor_pga_svd_direction_loading_sums is None:
                    actor_pga_svd_direction_loading_sums = torch.zeros_like(actor_pga_svd_direction_loadings)
                actor_pga_svd_direction_loading_sums += actor_pga_svd_direction_loadings.detach()
            if actor_pga_svd_direction_energy_fractions is not None:
                if actor_pga_svd_direction_energy_fraction_sums is None:
                    actor_pga_svd_direction_energy_fraction_sums = torch.zeros_like(
                        actor_pga_svd_direction_energy_fractions
                    )
                actor_pga_svd_direction_energy_fraction_sums += (
                    actor_pga_svd_direction_energy_fractions.detach()
                )
            for diagnostic_name, diagnostic_value in actor_hierarchical_diagnostics.items():
                if diagnostic_name not in actor_hierarchical_diagnostic_sums:
                    actor_hierarchical_diagnostic_sums[diagnostic_name] = torch.zeros_like(
                        diagnostic_value
                    )
                actor_hierarchical_diagnostic_sums[diagnostic_name] += diagnostic_value.detach()
            if action_entropy_sums is None:
                action_entropy_sums = torch.zeros_like(action_entropy_batch.mean(dim=0))
            action_entropy_sums += action_entropy_batch.detach().mean(dim=0)
            if critic_losses_by_term is not None:
                if critic_value_loss_sums is None:
                    critic_value_loss_sums = torch.zeros_like(critic_losses_by_term)
                critic_value_loss_sums += critic_losses_by_term.detach()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_kl_divergence /= num_updates
        mean_actor_amtl_grad_norm /= num_updates
        mean_num_actor_objectives /= num_updates
        mean_critic_amtl_grad_norm /= num_updates
        mean_num_critic_objectives /= num_updates
        mean_actor_raw_mean_cosine /= num_updates
        mean_actor_raw_min_cosine /= num_updates
        mean_actor_raw_max_cosine /= num_updates
        mean_actor_raw_conflict_fraction /= num_updates
        mean_actor_aligned_mean_cosine /= num_updates
        mean_actor_aligned_min_cosine /= num_updates
        mean_actor_aligned_max_cosine /= num_updates
        mean_actor_aligned_conflict_fraction /= num_updates
        mean_actor_alignment_mean_cosine /= num_updates
        mean_actor_amtl_vs_ppo_cosine /= num_updates
        mean_actor_amtl_projection_on_ppo /= num_updates
        mean_actor_amtl_projection_ratio /= num_updates
        mean_actor_amtl_parallel_norm /= num_updates
        mean_actor_amtl_orthogonal_norm /= num_updates
        mean_actor_amtl_orthogonal_ratio /= num_updates
        mean_actor_objective_vs_ppo_cosine_min /= num_updates
        mean_actor_objective_vs_ppo_cosine_max /= num_updates
        mean_actor_objective_vs_ppo_cosine_mean /= num_updates
        mean_actor_amtl_vs_objective_cosine_min /= num_updates
        mean_actor_amtl_vs_objective_cosine_max /= num_updates
        mean_actor_amtl_vs_objective_cosine_mean /= num_updates
        mean_actor_pga_retained_rank /= num_updates
        mean_actor_pga_sigma_min /= num_updates
        mean_actor_pga_sigma_max /= num_updates
        mean_actor_pga_sigma_condition_number /= num_updates
        mean_actor_scale /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "actor_ppo_loss": mean_surrogate_loss,
            "entropy": mean_entropy,
            "kl_divergence": mean_kl_divergence,
            "actor_amtl_grad_norm": mean_actor_amtl_grad_norm,
            "critic_amtl_grad_norm": mean_critic_amtl_grad_norm,
            "num_actor_objectives": mean_num_actor_objectives,
            "num_critic_objectives": mean_num_critic_objectives,
            "actor_raw_mean_cosine": mean_actor_raw_mean_cosine,
            "actor_raw_min_cosine": mean_actor_raw_min_cosine,
            "actor_raw_max_cosine": mean_actor_raw_max_cosine,
            "actor_raw_conflict_fraction": mean_actor_raw_conflict_fraction,
            "actor_aligned_mean_cosine": mean_actor_aligned_mean_cosine,
            "actor_aligned_min_cosine": mean_actor_aligned_min_cosine,
            "actor_aligned_max_cosine": mean_actor_aligned_max_cosine,
            "actor_aligned_conflict_fraction": mean_actor_aligned_conflict_fraction,
            "actor_alignment_mean_cosine": mean_actor_alignment_mean_cosine,
            "actor_amtl_vs_ppo_cosine": mean_actor_amtl_vs_ppo_cosine,
            "actor_amtl_projection_on_ppo": mean_actor_amtl_projection_on_ppo,
            "actor_amtl_projection_ratio": mean_actor_amtl_projection_ratio,
            "actor_amtl_parallel_norm": mean_actor_amtl_parallel_norm,
            "actor_amtl_orthogonal_norm": mean_actor_amtl_orthogonal_norm,
            "actor_amtl_orthogonal_ratio": mean_actor_amtl_orthogonal_ratio,
            "actor_objective_vs_ppo_cosine_min": mean_actor_objective_vs_ppo_cosine_min,
            "actor_objective_vs_ppo_cosine_max": mean_actor_objective_vs_ppo_cosine_max,
            "actor_objective_vs_ppo_cosine_mean": mean_actor_objective_vs_ppo_cosine_mean,
            "actor_amtl_vs_objective_cosine_min": mean_actor_amtl_vs_objective_cosine_min,
            "actor_amtl_vs_objective_cosine_max": mean_actor_amtl_vs_objective_cosine_max,
            "actor_amtl_vs_objective_cosine_mean": mean_actor_amtl_vs_objective_cosine_mean,
            "actor_pga_retained_rank": mean_actor_pga_retained_rank,
            "actor_pga_sigma_min": mean_actor_pga_sigma_min,
            "actor_pga_sigma_max": mean_actor_pga_sigma_max,
            "actor_pga_sigma_condition_number": mean_actor_pga_sigma_condition_number,
            "actor_pga_hierarchical_enabled": 1.0 if self.actor_pga_mode == "hierarchical_5" else 0.0,
            "actor_scale": mean_actor_scale,
            "actor_amtl_enabled": 1.0 if self.use_actor_amtl else 0.0,
            "critic_amtl_enabled": 1.0 if self.use_critic_amtl else 0.0,
        }
        if actor_objective_vs_ppo_cosine_sums is not None:
            objective_names = self._get_actor_objective_names(len(actor_objective_vs_ppo_cosine_sums))
            for objective_name, cosine_value in zip(
                objective_names, actor_objective_vs_ppo_cosine_sums / num_updates, strict=True
            ):
                loss_dict[f"actor_objective_vs_ppo_cosine/{objective_name}"] = cosine_value.item()
        if actor_amtl_vs_objective_cosine_sums is not None:
            objective_names = self._get_actor_objective_names(len(actor_amtl_vs_objective_cosine_sums))
            for objective_name, cosine_value in zip(
                objective_names, actor_amtl_vs_objective_cosine_sums / num_updates, strict=True
            ):
                loss_dict[f"actor_amtl_vs_objective_cosine/{objective_name}"] = cosine_value.item()
        if actor_pga_objective_weight_sums is not None:
            objective_names = self._get_actor_objective_names(len(actor_pga_objective_weight_sums))
            for objective_name, weight in zip(
                objective_names, actor_pga_objective_weight_sums / num_updates, strict=True
            ):
                loss_dict[f"actor_pga_objective_weight/{objective_name}"] = weight.item()
        if actor_pga_objective_contribution_norm_sums is not None:
            objective_names = self._get_actor_objective_names(len(actor_pga_objective_contribution_norm_sums))
            for objective_name, contribution_norm in zip(
                objective_names, actor_pga_objective_contribution_norm_sums / num_updates, strict=True
            ):
                loss_dict[f"actor_pga_objective_contribution_norm/{objective_name}"] = contribution_norm.item()
        if actor_pga_objective_projection_sums is not None:
            objective_names = self._get_actor_objective_names(len(actor_pga_objective_projection_sums))
            for objective_name, projection in zip(
                objective_names, actor_pga_objective_projection_sums / num_updates, strict=True
            ):
                loss_dict[f"actor_pga_objective_projection/{objective_name}"] = projection.item()
        if actor_pga_singular_value_sums is not None:
            for direction_index, singular_value in enumerate(actor_pga_singular_value_sums / num_updates):
                loss_dict[f"actor_pga_singular_value/direction_{direction_index:02d}"] = singular_value.item()
        if actor_pga_energy_sums is not None:
            for direction_index, energy in enumerate(actor_pga_energy_sums / num_updates):
                loss_dict[f"actor_pga_energy/direction_{direction_index:02d}"] = energy.item()
        if actor_pga_energy_fraction_sums is not None:
            for direction_index, energy_fraction in enumerate(actor_pga_energy_fraction_sums / num_updates):
                loss_dict[f"actor_pga_energy_fraction/direction_{direction_index:02d}"] = energy_fraction.item()
        if actor_pga_svd_direction_loading_sums is not None:
            objective_names = self._get_actor_objective_names(actor_pga_svd_direction_loading_sums.shape[0])
            mean_loadings = actor_pga_svd_direction_loading_sums / num_updates
            for direction_index, direction_loadings in enumerate(mean_loadings.T):
                for objective_name, loading in zip(objective_names, direction_loadings, strict=True):
                    loss_dict[
                        f"actor_pga_svd_direction_loading/direction_{direction_index:02d}/{objective_name}"
                    ] = loading.item()
        if actor_pga_svd_direction_energy_fraction_sums is not None:
            objective_names = self._get_actor_objective_names(
                actor_pga_svd_direction_energy_fraction_sums.shape[0]
            )
            mean_energy_fractions = actor_pga_svd_direction_energy_fraction_sums / num_updates
            for direction_index, direction_fractions in enumerate(mean_energy_fractions.T):
                for objective_name, energy_fraction in zip(objective_names, direction_fractions, strict=True):
                    loss_dict[
                        f"actor_pga_svd_direction_energy_fraction/direction_{direction_index:02d}/{objective_name}"
                    ] = energy_fraction.item()
        for diagnostic_name, diagnostic_sum in sorted(actor_hierarchical_diagnostic_sums.items()):
            loss_dict[diagnostic_name] = (diagnostic_sum / num_updates).item()
        if critic_value_loss_sums is not None:
            critic_head_names = self._get_critic_head_names(len(critic_value_loss_sums))
            for head_name, head_loss in zip(critic_head_names, critic_value_loss_sums / num_updates, strict=True):
                loss_dict[f"critic_value_loss/{head_name}"] = head_loss.item()
        if action_entropy_sums is not None:
            num_actions = len(action_entropy_sums)
            digits = max(2, len(str(num_actions - 1)))
            for action_index, entropy_value in enumerate(action_entropy_sums / num_updates):
                loss_dict[f"Policy/action_entropy/action_{action_index:0{digits}d}"] = entropy_value.item()
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        self.update_counter += 1
        return loss_dict

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
