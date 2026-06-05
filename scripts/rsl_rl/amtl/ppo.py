# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from collections.abc import Sequence
from itertools import chain
from tensordict import TensorDict

from amtl.modules import ActorCriticRecurrent, AmpDiscriminator, AmpReplayBuffer
from amtl.modules.rnd import RandomNetworkDistillation
from amtl.objective_metadata import get_objective_metadata
from amtl.storage import RolloutStorage
from amtl.utils import resolve_optimizer, string_to_callable

from amtl.actor_critic import ActorCritic

def set_shared_grad(shared_params, grad_vec):
    offset = 0
    for p in shared_params:
        _offset = offset + p.numel()
        if p.grad is not None:
            p.grad.data = grad_vec[offset:_offset].view_as(p.grad)
        offset = _offset


    


class ProcrustesSolver:
    @staticmethod
    def apply(grads, scale_mode='min'):
        assert (
            len(grads.shape) == 3
        ), f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable"

        with torch.no_grad():
            cov_grad_matrix_e = torch.matmul(grads.permute(0, 2, 1), grads)
            cov_grad_matrix_e = cov_grad_matrix_e.mean(0)

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
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        share_cnn_encoders: bool = False,
        device: str = "cpu",
        env=None,
        normalize_advantage_per_mini_batch: bool = False,
        beta_reg: float = 0.5,
        max_reg_scale: float = 1.0,
        use_blended_actor_update: bool = True,
        ppo_blend_weight: float = 0.8,
        amtl_blend_weight: float = 0.2,
        blend_schedule: str = "constant",
        blend_transition_start: int = 5000,
        blend_transition_end: int = 15000,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # AMP parameters
        use_amp: bool = False,
        amp_style_weight: float = 0.9,
        amp_task_weight: float = 0.1,
        amp_discriminator_lr: float = 2.0e-5,
        amp_grad_penalty_weight: float = 10.0,
        amp_replay_buffer_size: int = 200000,
        amp_batch_size: int = 4096,
        amp_as_separate_objective: bool = False,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
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

        # Create optimizer
        optimizer_class = resolve_optimizer(optimizer)
        self.optimizer = optimizer_class(self.policy.parameters(), lr=learning_rate)

        self.use_amp = use_amp
        self.amp_style_weight = amp_style_weight
        self.amp_task_weight = amp_task_weight
        self.amp_grad_penalty_weight = amp_grad_penalty_weight
        self.amp_batch_size = amp_batch_size
        self.amp_as_separate_objective = amp_as_separate_objective
        self.amp_discriminator: AmpDiscriminator | None = None
        self.amp_optimizer: optim.Optimizer | None = None
        self.amp_replay_buffer: AmpReplayBuffer | None = None
        self._amp_episode_task_sums: torch.Tensor | None = None
        self._amp_episode_style_sums: torch.Tensor | None = None

        if self.use_amp:
            motion_command = self._get_motion_command()
            if motion_command is None:
                raise RuntimeError("AMP is enabled, but no motion command was found on the environment.")
            amp_feature_dim = motion_command.amp_feature_dim
            self.amp_discriminator = AmpDiscriminator(state_feature_dim=amp_feature_dim).to(self.device)
            self.amp_optimizer = optim.Adam(self.amp_discriminator.parameters(), lr=amp_discriminator_lr)
            self.amp_replay_buffer = AmpReplayBuffer(
                capacity=amp_replay_buffer_size,
                transition_feature_dim=amp_feature_dim * 2,
                device=self.device,
            )

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
        self.beta_reg = beta_reg
        self.max_reg_scale = max_reg_scale
        self.use_blended_actor_update = use_blended_actor_update
        self.ppo_blend_weight = ppo_blend_weight
        self.amtl_blend_weight = amtl_blend_weight
        self.blend_schedule = blend_schedule
        self.blend_transition_start = blend_transition_start
        self.blend_transition_end = blend_transition_end
        self.update_counter = 0
        self._hybrid_term_indices: tuple[list[int], list[int]] | None = None
        self.latest_objective_cosine_matrix: torch.Tensor | None = None
        self.log_dir: str | None = None
        self._gradient_debug_dump_written = False

    def set_log_dir(self, log_dir: str | None) -> None:
        self.log_dir = log_dir

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

    def _get_motion_command(self):
        if self.env is None:
            return None

        for env_candidate in (self.env, getattr(self.env, "unwrapped", None)):
            if env_candidate is None:
                continue
            command_manager = getattr(env_candidate, "command_manager", None)
            if command_manager is None:
                continue
            try:
                return command_manager.get_term("motion")
            except Exception:
                continue
        return None

    def _get_max_episode_length_s(self) -> float:
        if self.env is None:
            return 1.0

        for env_candidate in (self.env, getattr(self.env, "unwrapped", None)):
            if env_candidate is None:
                continue
            value = getattr(env_candidate, "max_episode_length_s", None)
            if value is not None:
                return float(value)
        return 1.0

    def _get_amp_transition_features(self, state_features_t: torch.Tensor, state_features_tp1: torch.Tensor) -> torch.Tensor:
        return torch.cat([state_features_t, state_features_tp1], dim=-1)

    def _append_amp_reward_terms(
        self,
        reward_terms: torch.Tensor | None,
        amp_style_reward: torch.Tensor,
    ) -> torch.Tensor | None:
        if reward_terms is None:
            return None

        scaled_reward_terms = reward_terms * self.amp_task_weight
        if not self.amp_as_separate_objective:
            return scaled_reward_terms

        style_term = (self.amp_style_weight * amp_style_reward).unsqueeze(-1)
        return torch.cat([scaled_reward_terms, style_term], dim=-1)

    def _update_amp_episode_logs(
        self,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
        amp_task_reward: torch.Tensor,
        amp_style_reward: torch.Tensor,
    ) -> None:
        if self._amp_episode_task_sums is None or self._amp_episode_style_sums is None:
            return

        self._amp_episode_task_sums += amp_task_reward
        self._amp_episode_style_sums += amp_style_reward

        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if done_ids.numel() == 0:
            return

        max_episode_length_s = self._get_max_episode_length_s()
        episode_log = extras.get("episode")
        if episode_log is None:
            episode_log = extras.get("log")
        if episode_log is None:
            episode_log = {}
            extras["episode"] = episode_log

        episode_log["Episode_Reward/amp_task"] = (
            self._amp_episode_task_sums[done_ids].mean() / max_episode_length_s
        )
        episode_log["Episode_Reward/amp_style"] = (
            self._amp_episode_style_sums[done_ids].mean() / max_episode_length_s
        )

        self._amp_episode_task_sums[done_ids] = 0.0
        self._amp_episode_style_sums[done_ids] = 0.0

    def _compute_amp_style_reward(self, state_features_t: torch.Tensor, state_features_tp1: torch.Tensor) -> torch.Tensor:
        if not self.use_amp or self.amp_discriminator is None:
            raise RuntimeError("AMP style reward requested while AMP is disabled.")

        policy_scores = self.amp_discriminator(state_features_t, state_features_tp1)
        return torch.clamp(1.0 - 0.25 * torch.square(policy_scores - 1.0), min=0.0)

    def _update_amp_discriminator(self) -> dict[str, float]:
        if not self.use_amp or self.amp_discriminator is None or self.amp_optimizer is None or self.amp_replay_buffer is None:
            return {
                "amp_discriminator": 0.0,
                "amp_grad_penalty": 0.0,
                "amp_ref_score": 0.0,
                "amp_policy_score": 0.0,
            }

        if len(self.amp_replay_buffer) == 0:
            return {
                "amp_discriminator": 0.0,
                "amp_grad_penalty": 0.0,
                "amp_ref_score": 0.0,
                "amp_policy_score": 0.0,
            }

        motion_command = self._get_motion_command()
        if motion_command is None:
            raise RuntimeError("AMP discriminator update requires the motion command.")

        policy_transition_batch = self.amp_replay_buffer.sample(self.amp_batch_size)
        batch_size = policy_transition_batch.shape[0]
        ref_features_t, ref_features_tp1 = motion_command.sample_amp_reference_transitions(batch_size)
        ref_transition_batch = self._get_amp_transition_features(ref_features_t, ref_features_tp1)

        self.amp_optimizer.zero_grad()
        ref_scores = self.amp_discriminator.forward_transition(ref_transition_batch)
        policy_scores = self.amp_discriminator.forward_transition(policy_transition_batch)

        ref_loss = torch.mean(torch.square(ref_scores - 1.0))
        policy_loss = torch.mean(torch.square(policy_scores + 1.0))

        alpha = torch.rand(batch_size, 1, device=self.device)
        interpolated_transition = alpha * ref_transition_batch + (1.0 - alpha) * policy_transition_batch
        interpolated_transition.requires_grad_(True)
        interpolated_scores = self.amp_discriminator.forward_transition(interpolated_transition)
        grad_outputs = torch.ones_like(interpolated_scores, device=self.device)
        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated_transition,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty = self.amp_grad_penalty_weight * torch.mean((gradients.norm(2, dim=-1) - 1.0) ** 2)

        amp_loss = ref_loss + policy_loss + grad_penalty
        amp_loss.backward()

        return {
            "amp_discriminator": amp_loss.item(),
            "amp_grad_penalty": grad_penalty.item(),
            "amp_ref_score": ref_scores.mean().item(),
            "amp_policy_score": policy_scores.mean().item(),
        }

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
        reward_term_names = list(reward_manager.active_terms)
        if self.use_amp and self.amp_as_separate_objective:
            reward_term_names.append("amp_style")
        return reward_term_names

    def _get_hybrid_term_indices(self) -> tuple[list[int], list[int]] | None:
        if self._hybrid_term_indices is not None:
            return self._hybrid_term_indices

        reward_term_names = self._get_reward_term_names()
        if reward_term_names is None:
            return None

        motion_indices = []
        regularizer_indices = []
        for index, name in enumerate(reward_term_names):
            metadata = get_objective_metadata(name)
            if metadata.is_penalty:
                regularizer_indices.append(index)
            else:
                motion_indices.append(index)

        if len(motion_indices) == 0 or len(regularizer_indices) == 0:
            self._hybrid_term_indices = None
            return self._hybrid_term_indices

        self._hybrid_term_indices = (motion_indices, regularizer_indices)
        return self._hybrid_term_indices

    @staticmethod
    def _zero_existing_grads(params: Sequence[torch.nn.Parameter]) -> None:
        for param in params:
            if param.grad is not None:
                param.grad.data.zero_()

    @staticmethod
    def _collect_flat_grad(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
        return torch.cat(
            [
                param.grad.flatten().clone() if param.grad is not None else torch.zeros_like(param).flatten()
                for param in params
            ]
        )

    @staticmethod
    def _compute_parameter_grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
        grad_sq_sum = 0.0
        for param in params:
            if param.grad is None:
                continue
            grad_sq_sum += torch.sum(param.grad.detach() ** 2).item()
        return grad_sq_sum ** 0.5

    @staticmethod
    def _compute_flat_subset_grad_norm(
        params: Sequence[torch.nn.Parameter],
        grad_vec: torch.Tensor,
        subset_params: Sequence[torch.nn.Parameter],
    ) -> float:
        subset_param_ids = {id(param) for param in subset_params}
        grad_sq_sum = 0.0
        offset = 0
        for param in params:
            next_offset = offset + param.numel()
            if id(param) in subset_param_ids:
                grad_slice = grad_vec[offset:next_offset]
                grad_sq_sum += torch.sum(grad_slice.detach() ** 2).item()
            offset = next_offset
        return grad_sq_sum ** 0.5

    def _get_effective_blend_weights(self) -> tuple[float, float]:
        if self.blend_schedule == "constant":
            return self.ppo_blend_weight, self.amtl_blend_weight
        if self.blend_schedule == "ppo_to_amtl":
            if self.update_counter <= self.blend_transition_start:
                return 1.0, 0.0
            if self.update_counter >= self.blend_transition_end:
                return self.ppo_blend_weight, self.amtl_blend_weight
            transition_span = max(1, self.blend_transition_end - self.blend_transition_start)
            alpha = (self.update_counter - self.blend_transition_start) / transition_span
            effective_ppo = (1.0 - alpha) * 1.0 + alpha * self.ppo_blend_weight
            effective_amtl = (1.0 - alpha) * 0.0 + alpha * self.amtl_blend_weight
            return effective_ppo, effective_amtl
        raise ValueError(f"Unsupported blend_schedule: {self.blend_schedule}")

    @staticmethod
    def _accumulate_objective_gradient_metrics(
        sum_grad_norms: dict[str, float],
        sum_grad_fractions: dict[str, float],
        sum_aligned_projections: dict[str, float],
        objective_names: Sequence[str],
        grads: torch.Tensor,
        aligned_grad: torch.Tensor,
    ) -> None:
        if grads.numel() == 0 or len(objective_names) == 0:
            return

        grad_norms = grads.norm(dim=1)
        total_grad_norm = grad_norms.sum().clamp_min(1.0e-8)
        aligned_grad_norm = aligned_grad.norm().clamp_min(1.0e-8)
        projections = torch.sum(grads * aligned_grad.unsqueeze(0), dim=1) / (
            grad_norms * aligned_grad_norm
        ).clamp_min(1.0e-8)

        for objective_name, grad_norm, grad_fraction, projection in zip(
            objective_names,
            grad_norms,
            grad_norms / total_grad_norm,
            projections,
            strict=True,
        ):
            sum_grad_norms[objective_name] = sum_grad_norms.get(objective_name, 0.0) + grad_norm.item()
            sum_grad_fractions[objective_name] = sum_grad_fractions.get(objective_name, 0.0) + grad_fraction.item()
            sum_aligned_projections[objective_name] = (
                sum_aligned_projections.get(objective_name, 0.0) + projection.item()
            )

    @staticmethod
    def _compute_objective_fraction_summary(grads: torch.Tensor) -> dict[str, float]:
        if grads.numel() == 0 or grads.shape[0] == 0:
            return {
                "top_objective_grad_fraction": 0.0,
                "bottom_objective_grad_fraction": 0.0,
                "objective_grad_fraction_entropy": 0.0,
            }

        grad_norms = grads.norm(dim=1)
        grad_fractions = grad_norms / grad_norms.sum().clamp_min(1.0e-8)
        safe_fractions = grad_fractions.clamp_min(1.0e-8)
        return {
            "top_objective_grad_fraction": grad_fractions.max().item(),
            "bottom_objective_grad_fraction": grad_fractions.min().item(),
            "objective_grad_fraction_entropy": (-(safe_fractions * safe_fractions.log()).sum()).item(),
        }

    def _write_first_update_gradient_debug(
        self,
        payload: dict,
    ) -> None:
        if self._gradient_debug_dump_written or self.log_dir is None:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "gradient_debug_first_update.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        self._gradient_debug_dump_written = True

    @staticmethod
    def _compute_objective_cosine_stats(
        grads: torch.Tensor,
    ) -> tuple[dict[str, float], torch.Tensor]:
        num_objectives = grads.shape[0]
        if num_objectives == 0:
            zero_matrix = torch.zeros((0, 0), device=grads.device)
            return {
                "objective_mean_cosine": 0.0,
                "objective_median_cosine": 0.0,
                "objective_min_cosine": 0.0,
                "objective_max_cosine": 0.0,
                "objective_std_cosine": 0.0,
                "objective_conflict_fraction": 0.0,
            }, zero_matrix

        normalized_grads = grads / grads.norm(dim=1, keepdim=True).clamp_min(1.0e-8)
        cosine_matrix = torch.matmul(normalized_grads, normalized_grads.T).detach()

        if num_objectives < 2:
            return {
                "objective_mean_cosine": 1.0,
                "objective_median_cosine": 1.0,
                "objective_min_cosine": 1.0,
                "objective_max_cosine": 1.0,
                "objective_std_cosine": 0.0,
                "objective_conflict_fraction": 0.0,
            }, cosine_matrix

        pair_indices = torch.triu_indices(num_objectives, num_objectives, offset=1, device=grads.device)
        pairwise_cosines = cosine_matrix[pair_indices[0], pair_indices[1]]

        return {
            "objective_mean_cosine": pairwise_cosines.mean().item(),
            "objective_median_cosine": pairwise_cosines.median().item(),
            "objective_min_cosine": pairwise_cosines.min().item(),
            "objective_max_cosine": pairwise_cosines.max().item(),
            "objective_std_cosine": pairwise_cosines.std(unbiased=False).item(),
            "objective_conflict_fraction": (pairwise_cosines < 0).float().mean().item(),
        }, cosine_matrix

    @staticmethod
    def _compute_objective_matrix_stats(grads: torch.Tensor) -> dict[str, float]:
        num_objectives = grads.shape[0]
        if num_objectives == 0:
            return {
                "objective_effective_rank": 0.0,
                "objective_sv1_ratio": 0.0,
                "objective_sv2_ratio": 0.0,
                "objective_sv3_ratio": 0.0,
                "objective_pca_var1": 0.0,
                "objective_pca_var2": 0.0,
                "objective_pca_var3": 0.0,
            }

        singular_values = torch.linalg.svdvals(grads)
        singular_sum = singular_values.sum().clamp_min(1.0e-8)
        singular_ratios = singular_values / singular_sum
        effective_rank = torch.exp(
            -(singular_ratios * torch.log(singular_ratios.clamp_min(1.0e-8))).sum()
        ).item()

        centered_grads = grads - grads.mean(dim=0, keepdim=True)
        if num_objectives > 1:
            centered_singular_values = torch.linalg.svdvals(centered_grads)
            pca_variance = torch.square(centered_singular_values)
            pca_total_variance = pca_variance.sum().clamp_min(1.0e-8)
            pca_ratios = pca_variance / pca_total_variance
        else:
            pca_ratios = torch.ones(1, device=grads.device)

        def get_ratio(values: torch.Tensor, index: int) -> float:
            if index < values.numel():
                return values[index].item()
            return 0.0

        return {
            "objective_effective_rank": effective_rank,
            "objective_sv1_ratio": get_ratio(singular_ratios, 0),
            "objective_sv2_ratio": get_ratio(singular_ratios, 1),
            "objective_sv3_ratio": get_ratio(singular_ratios, 2),
            "objective_pca_var1": get_ratio(pca_ratios, 0),
            "objective_pca_var2": get_ratio(pca_ratios, 1),
            "objective_pca_var3": get_ratio(pca_ratios, 2),
        }
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
            self.device,
        )
        if self.use_amp:
            self._amp_episode_task_sums = torch.zeros(num_envs, device=self.device)
            self._amp_episode_style_sums = torch.zeros(num_envs, device=self.device)

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
        if self.use_amp:
            motion_command = self._get_motion_command()
            if motion_command is None:
                raise RuntimeError("AMP rollout collection requires the motion command.")
            self.transition.amp_state_features = motion_command.get_amp_policy_features().detach()
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
        reward_terms = self._get_reward_terms()
        self.transition.dones = dones
        learning_rewards = rewards.clone()

        if self.use_amp:
            motion_command = self._get_motion_command()
            if motion_command is None or self.transition.amp_state_features is None:
                raise RuntimeError("AMP reward computation requires cached policy features and the motion command.")
            next_amp_state_features = motion_command.get_amp_policy_features().detach()
            amp_style_reward = self._compute_amp_style_reward(
                self.transition.amp_state_features,
                next_amp_state_features,
            ).detach()
            amp_task_reward = self.amp_task_weight * learning_rewards
            amp_style_reward_weighted = self.amp_style_weight * amp_style_reward
            learning_rewards = amp_task_reward + amp_style_reward_weighted
            rewards.copy_(learning_rewards)
            reward_terms = self._append_amp_reward_terms(reward_terms, amp_style_reward)
            self._update_amp_episode_logs(dones, extras, amp_task_reward, amp_style_reward_weighted)
            if self.amp_replay_buffer is not None:
                self.amp_replay_buffer.add(
                    self._get_amp_transition_features(self.transition.amp_state_features, next_amp_state_features)
                )

        self.transition.rewards = learning_rewards.clone()

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            timeout_bonus = self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )
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

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_kl_divergence = 0
        mean_objective_mean_cosine = 0
        mean_objective_median_cosine = 0
        mean_objective_min_cosine = 0
        mean_objective_max_cosine = 0
        mean_objective_std_cosine = 0
        mean_objective_conflict_fraction = 0
        mean_objective_effective_rank = 0
        mean_objective_sv1_ratio = 0
        mean_objective_sv2_ratio = 0
        mean_objective_sv3_ratio = 0
        mean_objective_pca_var1 = 0
        mean_objective_pca_var2 = 0
        mean_objective_pca_var3 = 0
        mean_aligned_grad_norm = 0
        mean_regularizer_grad_norm = 0
        mean_reg_scale = 0
        mean_scaled_regularizer_grad_norm = 0
        mean_total_actor_grad_norm = 0
        mean_actor_mean_grad_norm = 0
        mean_actor_std_grad_norm = 0
        mean_actor_std_to_mean_grad_ratio = 0
        mean_num_aligned_terms = 0
        mean_num_regularizer_terms = 0
        mean_ppo_per_objective_proxy_grad_norm = 0
        mean_amtl_actor_grad_norm = 0
        mean_blended_actor_grad_norm = 0
        mean_ppo_per_objective_proxy_amtl_grad_cosine = 0
        mean_ppo_true_actor_grad_norm = 0
        mean_ppo_true_amtl_grad_cosine = 0
        mean_ppo_true_to_amtl_norm_ratio = 0
        mean_ppo_true_minus_amtl_grad_norm = 0
        mean_ppo_true_minus_amtl_relative_norm = 0
        mean_amtl_projection_on_ppo_true = 0
        mean_ppo_true_projection_on_amtl = 0
        mean_ppo_true_actor_mean_grad_norm = 0
        mean_ppo_true_actor_std_grad_norm = 0
        mean_amtl_actor_mean_grad_norm = 0
        mean_amtl_actor_std_grad_norm = 0
        mean_ppo_true_std_to_mean_grad_ratio = 0
        mean_amtl_std_to_mean_grad_ratio = 0
        mean_top_objective_grad_fraction = 0
        mean_bottom_objective_grad_fraction = 0
        mean_objective_grad_fraction_entropy = 0
        mean_logged_ppo_blend_weight = 0
        mean_logged_amtl_blend_weight = 0
        mean_objective_grad_norms: dict[str, float] = {}
        mean_objective_grad_fractions: dict[str, float] = {}
        mean_objective_aligned_projections: dict[str, float] = {}
        objective_cosine_matrix_sum: torch.Tensor | None = None
        objective_cosine_matrix_count = 0
        mean_amp_discriminator_loss = 0.0
        mean_amp_grad_penalty = 0.0
        mean_amp_ref_score = 0.0
        mean_amp_policy_score = 0.0
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
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                    if advantages_by_term_batch is not None:
                        term_mean = advantages_by_term_batch.mean(dim=0, keepdim=True)
                        term_std = advantages_by_term_batch.std(dim=0, keepdim=True)
                        advantages_by_term_batch = (advantages_by_term_batch - term_mean) / (term_std + 1e-8)

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

            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                # Reduce the KL divergence across all GPUs so the logged value matches scheduler inputs.
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
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
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            # g_ppo_true diagnostics must use the untouched scalar PPO actor loss:
            # - scalar_advantage source tensor: advantages_batch
            # - ratio source tensor: ratio
            # - clipped_ratio source tensor: torch.clamp(ratio, 1-clip, 1+clip)
            # - advantages_by_term_batch is intentionally not used here
            scalar_surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            surrogate_loss = scalar_surrogate_loss
            surrogate_losses_by_term = None

            if advantages_by_term_batch is not None:
                ratio_by_term = ratio.unsqueeze(-1)
                clipped_ratio_by_term = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param).unsqueeze(-1)
                surrogate_by_term = -advantages_by_term_batch * ratio_by_term
                surrogate_clipped_by_term = -advantages_by_term_batch * clipped_ratio_by_term
                surrogate_losses_by_term = torch.max(surrogate_by_term, surrogate_clipped_by_term).mean(dim=0)
                surrogate_loss = surrogate_losses_by_term.mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            auxiliary_loss = self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

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
                    auxiliary_loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
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
            aligned_grad_norm_value = 0.0
            regularizer_grad_norm_value = 0.0
            reg_scale_value = 0.0
            scaled_regularizer_grad_norm_value = 0.0
            total_actor_grad_norm_value = 0.0
            actor_mean_grad_norm_value = 0.0
            actor_std_grad_norm_value = 0.0
            actor_std_to_mean_grad_ratio_value = 0.0
            num_aligned_terms_value = 0.0
            num_regularizer_terms_value = 0.0
            objective_mean_cosine_value = 0.0
            objective_median_cosine_value = 0.0
            objective_min_cosine_value = 0.0
            objective_max_cosine_value = 0.0
            objective_std_cosine_value = 0.0
            objective_conflict_fraction_value = 0.0
            objective_effective_rank_value = 0.0
            objective_sv1_ratio_value = 0.0
            objective_sv2_ratio_value = 0.0
            objective_sv3_ratio_value = 0.0
            objective_pca_var1_value = 0.0
            objective_pca_var2_value = 0.0
            objective_pca_var3_value = 0.0
            ppo_actor_grad_norm_value = 0.0
            ppo_true_actor_grad_norm_value = 0.0
            amtl_actor_grad_norm_value = 0.0
            blended_actor_grad_norm_value = 0.0
            ppo_amtl_grad_cosine_value = 0.0
            ppo_true_amtl_grad_cosine_value = 0.0
            ppo_true_to_amtl_norm_ratio_value = 0.0
            ppo_true_minus_amtl_grad_norm_value = 0.0
            ppo_true_minus_amtl_relative_norm_value = 0.0
            amtl_projection_on_ppo_true_value = 0.0
            ppo_true_projection_on_amtl_value = 0.0
            ppo_true_actor_mean_grad_norm_value = 0.0
            ppo_true_actor_std_grad_norm_value = 0.0
            amtl_actor_mean_grad_norm_value = 0.0
            amtl_actor_std_grad_norm_value = 0.0
            ppo_true_std_to_mean_grad_ratio_value = 0.0
            amtl_std_to_mean_grad_ratio_value = 0.0
            top_objective_grad_fraction_value = 0.0
            bottom_objective_grad_fraction_value = 0.0
            objective_grad_fraction_entropy_value = 0.0
            aligned_objective_names: list[str] = []
            regularizer_names: list[str] = []
            raw_objective_grad_norms: dict[str, float] = {}
            raw_objective_grad_fractions: dict[str, float] = {}
            ppo_blend_weight_value = 1.0
            amtl_blend_weight_value = 0.0
            amp_stats = self._update_amp_discriminator() if self.use_amp else None

            if surrogate_losses_by_term is not None:
                hybrid_term_indices = self._get_hybrid_term_indices()
                reward_term_names = self._get_reward_term_names()
                params = list(self.policy.parameters())
                self._zero_existing_grads(params)
                scalar_surrogate_loss.backward(retain_graph=True)
                ppo_true_actor_grad = self._collect_flat_grad(params)
                self._zero_existing_grads(params)
                ppo_true_actor_grad_norm_value = ppo_true_actor_grad.norm().item()
                self._zero_existing_grads(params)
                surrogate_loss.backward(retain_graph=True)
                ppo_actor_grad = self._collect_flat_grad(params)
                self._zero_existing_grads(params)
                ppo_actor_grad_norm_value = ppo_actor_grad.norm().item()
                ppo_true_actor_mean_grad_norm_value = self._compute_flat_subset_grad_norm(
                    params, ppo_true_actor_grad, self.policy.get_actor_mean_parameters()
                )
                ppo_true_actor_std_grad_norm_value = self._compute_flat_subset_grad_norm(
                    params, ppo_true_actor_grad, self.policy.get_actor_std_parameters()
                )
                ppo_true_std_to_mean_grad_ratio_value = (
                    ppo_true_actor_std_grad_norm_value / (ppo_true_actor_mean_grad_norm_value + 1.0e-8)
                )
                aligned_objective_names: list[str] = []
                regularizer_names: list[str] = []
                raw_objective_grad_norms: dict[str, float] = {}
                raw_objective_grad_fractions: dict[str, float] = {}

                if hybrid_term_indices is None:
                    grads = []
                    num_aligned_terms_value = float(len(surrogate_losses_by_term))
                    if reward_term_names is not None:
                        aligned_objective_names = reward_term_names[: len(surrogate_losses_by_term)]
                    for loss in surrogate_losses_by_term:
                        self._zero_existing_grads(params)
                        loss.backward(retain_graph=True)
                        grads.append(self._collect_flat_grad(params))

                    self._zero_existing_grads(params)
                    grads = torch.stack(grads, dim=0)
                    grad_norms = grads.norm(dim=1)
                    grad_fractions = grad_norms / grad_norms.sum().clamp_min(1.0e-8)
                    if aligned_objective_names:
                        raw_objective_grad_norms = {
                            name: value.item() for name, value in zip(aligned_objective_names, grad_norms, strict=True)
                        }
                        raw_objective_grad_fractions = {
                            name: value.item()
                            for name, value in zip(aligned_objective_names, grad_fractions, strict=True)
                        }
                    objective_cosine_stats, objective_cosine_matrix = self._compute_objective_cosine_stats(grads)
                    objective_matrix_stats = self._compute_objective_matrix_stats(grads)
                    objective_fraction_summary = self._compute_objective_fraction_summary(grads)
                    objective_mean_cosine_value = objective_cosine_stats["objective_mean_cosine"]
                    objective_median_cosine_value = objective_cosine_stats["objective_median_cosine"]
                    objective_min_cosine_value = objective_cosine_stats["objective_min_cosine"]
                    objective_max_cosine_value = objective_cosine_stats["objective_max_cosine"]
                    objective_std_cosine_value = objective_cosine_stats["objective_std_cosine"]
                    objective_conflict_fraction_value = objective_cosine_stats["objective_conflict_fraction"]
                    objective_effective_rank_value = objective_matrix_stats["objective_effective_rank"]
                    objective_sv1_ratio_value = objective_matrix_stats["objective_sv1_ratio"]
                    objective_sv2_ratio_value = objective_matrix_stats["objective_sv2_ratio"]
                    objective_sv3_ratio_value = objective_matrix_stats["objective_sv3_ratio"]
                    objective_pca_var1_value = objective_matrix_stats["objective_pca_var1"]
                    objective_pca_var2_value = objective_matrix_stats["objective_pca_var2"]
                    objective_pca_var3_value = objective_matrix_stats["objective_pca_var3"]
                    top_objective_grad_fraction_value = objective_fraction_summary["top_objective_grad_fraction"]
                    bottom_objective_grad_fraction_value = objective_fraction_summary["bottom_objective_grad_fraction"]
                    objective_grad_fraction_entropy_value = objective_fraction_summary["objective_grad_fraction_entropy"]
                    objective_cosine_matrix_sum = (
                        objective_cosine_matrix.clone()
                        if objective_cosine_matrix_sum is None
                        else objective_cosine_matrix_sum + objective_cosine_matrix
                    )
                    objective_cosine_matrix_count += 1
                    original_grads = grads
                    grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0))
                    grad = grads[0].sum(-1)
                    if reward_term_names is not None:
                        objective_names = reward_term_names[: len(surrogate_losses_by_term)]
                        self._accumulate_objective_gradient_metrics(
                            mean_objective_grad_norms,
                            mean_objective_grad_fractions,
                            mean_objective_aligned_projections,
                            objective_names,
                            original_grads,
                            aligned_grad=grad,
                        )
                    amtl_actor_grad = grad
                    aligned_grad_norm_value = amtl_actor_grad.norm().item()
                    amtl_actor_grad_norm_value = aligned_grad_norm_value
                    amtl_actor_mean_grad_norm_value = self._compute_flat_subset_grad_norm(
                        params, amtl_actor_grad, self.policy.get_actor_mean_parameters()
                    )
                    amtl_actor_std_grad_norm_value = self._compute_flat_subset_grad_norm(
                        params, amtl_actor_grad, self.policy.get_actor_std_parameters()
                    )
                    amtl_std_to_mean_grad_ratio_value = (
                        amtl_actor_std_grad_norm_value / (amtl_actor_mean_grad_norm_value + 1.0e-8)
                    )
                    effective_ppo_blend_weight, effective_amtl_blend_weight = self._get_effective_blend_weights()
                    if self.use_blended_actor_update:
                        ppo_blend_grad = ppo_true_actor_grad
                        ppo_blend_grad_norm = ppo_true_actor_grad.norm()
                        amtl_actor_grad_norm = amtl_actor_grad.norm()
                        ppo_blend_grad_normed = ppo_blend_grad / (ppo_blend_grad_norm + 1.0e-8)
                        amtl_actor_grad_normed = amtl_actor_grad / (amtl_actor_grad_norm + 1.0e-8)
                        blended_actor_grad = (
                            effective_ppo_blend_weight * ppo_blend_grad_normed
                            + effective_amtl_blend_weight * amtl_actor_grad_normed
                        ) * ppo_blend_grad_norm
                        ppo_blend_weight_value = effective_ppo_blend_weight
                        amtl_blend_weight_value = effective_amtl_blend_weight
                    else:
                        blended_actor_grad = amtl_actor_grad
                        ppo_blend_weight_value = 0.0
                        amtl_blend_weight_value = 1.0
                    blended_actor_grad_norm_value = blended_actor_grad.norm().item()
                    total_actor_grad_norm_value = blended_actor_grad_norm_value
                    if ppo_actor_grad_norm_value > 0.0 and amtl_actor_grad_norm_value > 0.0:
                        ppo_amtl_grad_cosine_value = torch.dot(ppo_actor_grad, amtl_actor_grad).item() / (
                            ppo_actor_grad_norm_value * amtl_actor_grad_norm_value + 1.0e-8
                        )
                    if ppo_true_actor_grad_norm_value > 0.0 and amtl_actor_grad_norm_value > 0.0:
                        ppo_true_amtl_grad_cosine_value = torch.dot(ppo_true_actor_grad, amtl_actor_grad).item() / (
                            ppo_true_actor_grad_norm_value * amtl_actor_grad_norm_value + 1.0e-8
                        )
                        ppo_true_to_amtl_norm_ratio_value = (
                            ppo_true_actor_grad_norm_value / (amtl_actor_grad_norm_value + 1.0e-8)
                        )
                        ppo_true_minus_amtl_grad_norm_value = (ppo_true_actor_grad - amtl_actor_grad).norm().item()
                        ppo_true_minus_amtl_relative_norm_value = (
                            ppo_true_minus_amtl_grad_norm_value / (ppo_true_actor_grad_norm_value + 1.0e-8)
                        )
                        amtl_projection_on_ppo_true_value = torch.dot(
                            amtl_actor_grad, ppo_true_actor_grad / (ppo_true_actor_grad_norm_value + 1.0e-8)
                        ).item()
                        ppo_true_projection_on_amtl_value = torch.dot(
                            ppo_true_actor_grad, amtl_actor_grad / (amtl_actor_grad_norm_value + 1.0e-8)
                        ).item()
                    set_shared_grad(params, blended_actor_grad)
                else:
                    motion_indices, regularizer_indices = hybrid_term_indices
                    num_aligned_terms_value = float(len(motion_indices))
                    num_regularizer_terms_value = float(len(regularizer_indices))
                    if reward_term_names is not None:
                        aligned_objective_names = [reward_term_names[idx] for idx in motion_indices]
                        regularizer_names = [reward_term_names[idx] for idx in regularizer_indices]

                    motion_grads = []
                    for idx in motion_indices:
                        self._zero_existing_grads(params)
                        surrogate_losses_by_term[idx].backward(retain_graph=True)
                        motion_grads.append(self._collect_flat_grad(params))

                    self._zero_existing_grads(params)
                    motion_grads = torch.stack(motion_grads, dim=0)
                    motion_grad_norms = motion_grads.norm(dim=1)
                    motion_grad_fractions = motion_grad_norms / motion_grad_norms.sum().clamp_min(1.0e-8)
                    if aligned_objective_names:
                        raw_objective_grad_norms = {
                            name: value.item()
                            for name, value in zip(aligned_objective_names, motion_grad_norms, strict=True)
                        }
                        raw_objective_grad_fractions = {
                            name: value.item()
                            for name, value in zip(aligned_objective_names, motion_grad_fractions, strict=True)
                        }
                    objective_cosine_stats, objective_cosine_matrix = self._compute_objective_cosine_stats(motion_grads)
                    objective_matrix_stats = self._compute_objective_matrix_stats(motion_grads)
                    objective_fraction_summary = self._compute_objective_fraction_summary(motion_grads)
                    objective_mean_cosine_value = objective_cosine_stats["objective_mean_cosine"]
                    objective_median_cosine_value = objective_cosine_stats["objective_median_cosine"]
                    objective_min_cosine_value = objective_cosine_stats["objective_min_cosine"]
                    objective_max_cosine_value = objective_cosine_stats["objective_max_cosine"]
                    objective_std_cosine_value = objective_cosine_stats["objective_std_cosine"]
                    objective_conflict_fraction_value = objective_cosine_stats["objective_conflict_fraction"]
                    objective_effective_rank_value = objective_matrix_stats["objective_effective_rank"]
                    objective_sv1_ratio_value = objective_matrix_stats["objective_sv1_ratio"]
                    objective_sv2_ratio_value = objective_matrix_stats["objective_sv2_ratio"]
                    objective_sv3_ratio_value = objective_matrix_stats["objective_sv3_ratio"]
                    objective_pca_var1_value = objective_matrix_stats["objective_pca_var1"]
                    objective_pca_var2_value = objective_matrix_stats["objective_pca_var2"]
                    objective_pca_var3_value = objective_matrix_stats["objective_pca_var3"]
                    top_objective_grad_fraction_value = objective_fraction_summary["top_objective_grad_fraction"]
                    bottom_objective_grad_fraction_value = objective_fraction_summary["bottom_objective_grad_fraction"]
                    objective_grad_fraction_entropy_value = objective_fraction_summary["objective_grad_fraction_entropy"]
                    objective_cosine_matrix_sum = (
                        objective_cosine_matrix.clone()
                        if objective_cosine_matrix_sum is None
                        else objective_cosine_matrix_sum + objective_cosine_matrix
                    )
                    objective_cosine_matrix_count += 1
                    aligned_motion_grads, weights, singulars = ProcrustesSolver.apply(
                        motion_grads.T.unsqueeze(0)
                    )
                    aligned_motion_grad = aligned_motion_grads[0].sum(-1)
                    if reward_term_names is not None:
                        motion_objective_names = [reward_term_names[idx] for idx in motion_indices]
                        self._accumulate_objective_gradient_metrics(
                            mean_objective_grad_norms,
                            mean_objective_grad_fractions,
                            mean_objective_aligned_projections,
                            motion_objective_names,
                            motion_grads,
                            aligned_motion_grad,
                        )

                    regularizer_loss = surrogate_losses_by_term[regularizer_indices].sum()
                    self._zero_existing_grads(params)
                    regularizer_loss.backward(retain_graph=True)
                    regularizer_grad = self._collect_flat_grad(params)

                    motion_grad_norm = aligned_motion_grad.norm()
                    regularizer_grad_norm = regularizer_grad.norm()
                    reg_scale = self.beta_reg * (
                        motion_grad_norm / (regularizer_grad_norm + 1.0e-8)
                    )
                    reg_scale = torch.clamp(reg_scale, max=self.max_reg_scale)
                    scaled_regularizer_grad = reg_scale * regularizer_grad

                    self._zero_existing_grads(params)
                    amtl_actor_grad = aligned_motion_grad + scaled_regularizer_grad
                    aligned_grad_norm_value = motion_grad_norm.item()
                    amtl_actor_grad_norm_value = amtl_actor_grad.norm().item()
                    amtl_actor_mean_grad_norm_value = self._compute_flat_subset_grad_norm(
                        params, amtl_actor_grad, self.policy.get_actor_mean_parameters()
                    )
                    amtl_actor_std_grad_norm_value = self._compute_flat_subset_grad_norm(
                        params, amtl_actor_grad, self.policy.get_actor_std_parameters()
                    )
                    amtl_std_to_mean_grad_ratio_value = (
                        amtl_actor_std_grad_norm_value / (amtl_actor_mean_grad_norm_value + 1.0e-8)
                    )
                    regularizer_grad_norm_value = regularizer_grad_norm.item()
                    reg_scale_value = reg_scale.item()
                    scaled_regularizer_grad_norm_value = scaled_regularizer_grad.norm().item()
                    effective_ppo_blend_weight, effective_amtl_blend_weight = self._get_effective_blend_weights()
                    if self.use_blended_actor_update:
                        ppo_blend_grad = ppo_true_actor_grad
                        ppo_blend_grad_norm = ppo_true_actor_grad.norm()
                        amtl_actor_grad_norm = amtl_actor_grad.norm()
                        ppo_blend_grad_normed = ppo_blend_grad / (ppo_blend_grad_norm + 1.0e-8)
                        amtl_actor_grad_normed = amtl_actor_grad / (amtl_actor_grad_norm + 1.0e-8)
                        blended_actor_grad = (
                            effective_ppo_blend_weight * ppo_blend_grad_normed
                            + effective_amtl_blend_weight * amtl_actor_grad_normed
                        ) * ppo_blend_grad_norm
                        ppo_blend_weight_value = effective_ppo_blend_weight
                        amtl_blend_weight_value = effective_amtl_blend_weight
                    else:
                        blended_actor_grad = amtl_actor_grad
                        ppo_blend_weight_value = 0.0
                        amtl_blend_weight_value = 1.0
                    blended_actor_grad_norm_value = blended_actor_grad.norm().item()
                    total_actor_grad_norm_value = blended_actor_grad_norm_value
                    if ppo_actor_grad_norm_value > 0.0 and amtl_actor_grad_norm_value > 0.0:
                        ppo_amtl_grad_cosine_value = torch.dot(ppo_actor_grad, amtl_actor_grad).item() / (
                            ppo_actor_grad_norm_value * amtl_actor_grad_norm_value + 1.0e-8
                        )
                    if ppo_true_actor_grad_norm_value > 0.0 and amtl_actor_grad_norm_value > 0.0:
                        ppo_true_amtl_grad_cosine_value = torch.dot(ppo_true_actor_grad, amtl_actor_grad).item() / (
                            ppo_true_actor_grad_norm_value * amtl_actor_grad_norm_value + 1.0e-8
                        )
                        ppo_true_to_amtl_norm_ratio_value = (
                            ppo_true_actor_grad_norm_value / (amtl_actor_grad_norm_value + 1.0e-8)
                        )
                        ppo_true_minus_amtl_grad_norm_value = (ppo_true_actor_grad - amtl_actor_grad).norm().item()
                        ppo_true_minus_amtl_relative_norm_value = (
                            ppo_true_minus_amtl_grad_norm_value / (ppo_true_actor_grad_norm_value + 1.0e-8)
                        )
                        amtl_projection_on_ppo_true_value = torch.dot(
                            amtl_actor_grad, ppo_true_actor_grad / (ppo_true_actor_grad_norm_value + 1.0e-8)
                        ).item()
                        ppo_true_projection_on_amtl_value = torch.dot(
                            ppo_true_actor_grad, amtl_actor_grad / (amtl_actor_grad_norm_value + 1.0e-8)
                        ).item()
                    set_shared_grad(params, blended_actor_grad)

                auxiliary_loss.backward()
            else:
                params = list(self.policy.parameters())
                self._zero_existing_grads(params)
                scalar_surrogate_loss.backward(retain_graph=True)
                ppo_true_actor_grad = self._collect_flat_grad(params)
                self._zero_existing_grads(params)
                ppo_true_actor_grad_norm_value = ppo_true_actor_grad.norm().item()
                self._zero_existing_grads(params)
                surrogate_loss.backward(retain_graph=True)
                ppo_actor_grad = self._collect_flat_grad(params)
                self._zero_existing_grads(params)
                ppo_actor_grad_norm_value = ppo_actor_grad.norm().item()
                amtl_actor_grad_norm_value = ppo_actor_grad_norm_value
                ppo_true_actor_mean_grad_norm_value = self._compute_flat_subset_grad_norm(
                    params, ppo_true_actor_grad, self.policy.get_actor_mean_parameters()
                )
                ppo_true_actor_std_grad_norm_value = self._compute_flat_subset_grad_norm(
                    params, ppo_true_actor_grad, self.policy.get_actor_std_parameters()
                )
                ppo_true_std_to_mean_grad_ratio_value = (
                    ppo_true_actor_std_grad_norm_value / (ppo_true_actor_mean_grad_norm_value + 1.0e-8)
                )
                amtl_actor_mean_grad_norm_value = ppo_true_actor_mean_grad_norm_value
                amtl_actor_std_grad_norm_value = ppo_true_actor_std_grad_norm_value
                amtl_std_to_mean_grad_ratio_value = ppo_true_std_to_mean_grad_ratio_value
                blended_actor_grad_norm_value = ppo_actor_grad_norm_value
                total_actor_grad_norm_value = blended_actor_grad_norm_value
                ppo_amtl_grad_cosine_value = 1.0 if ppo_actor_grad_norm_value > 0.0 else 0.0
                ppo_true_amtl_grad_cosine_value = 1.0 if ppo_true_actor_grad_norm_value > 0.0 else 0.0
                ppo_true_to_amtl_norm_ratio_value = 1.0 if ppo_true_actor_grad_norm_value > 0.0 else 0.0
                ppo_true_minus_amtl_grad_norm_value = 0.0
                ppo_true_minus_amtl_relative_norm_value = 0.0
                amtl_projection_on_ppo_true_value = ppo_true_actor_grad_norm_value
                ppo_true_projection_on_amtl_value = ppo_true_actor_grad_norm_value
                ppo_blend_weight_value = 1.0
                amtl_blend_weight_value = 0.0
                set_shared_grad(params, ppo_true_actor_grad)
                auxiliary_loss.backward()

            if not self._gradient_debug_dump_written and self.update_counter == 0:
                debug_payload = {
                    "g_ppo_true_norm": ppo_true_actor_grad_norm_value,
                    "g_amtl_norm": amtl_actor_grad_norm_value,
                    "cosine": ppo_true_amtl_grad_cosine_value,
                    "norm_ratio": ppo_true_to_amtl_norm_ratio_value,
                    "relative_difference_norm": ppo_true_minus_amtl_relative_norm_value,
                    "ppo_true_minus_amtl_grad_norm": ppo_true_minus_amtl_grad_norm_value,
                    "first_10_g_ppo_true": ppo_true_actor_grad[:10].detach().cpu().tolist(),
                    "first_10_g_amtl": (
                        amtl_actor_grad[:10].detach().cpu().tolist()
                        if surrogate_losses_by_term is not None
                        else ppo_true_actor_grad[:10].detach().cpu().tolist()
                    ),
                    "aligned_objective_names": aligned_objective_names if "aligned_objective_names" in locals() else [],
                    "regularizer_names": regularizer_names if "regularizer_names" in locals() else [],
                    "num_aligned_terms": num_aligned_terms_value,
                    "num_regularizer_terms": num_regularizer_terms_value,
                    "raw_per_objective_gradient_norms": raw_objective_grad_norms if "raw_objective_grad_norms" in locals() else {},
                    "raw_per_objective_gradient_fractions": (
                        raw_objective_grad_fractions if "raw_objective_grad_fractions" in locals() else {}
                    ),
                    "g_ppo_true_checks": {
                        "advantage_source_tensor": "advantages_batch",
                        "ratio_source_tensor": "ratio",
                        "clipped_ratio_source_tensor": "torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)",
                        "uses_advantages_by_term": False,
                    },
                    "g_amtl_checks": {
                        "definition": "aligned objective gradient + reg_scale * regularizer_gradient",
                        "regularizer_scaling_uses_beta_reg": True,
                        "regularizer_scaling_uses_max_reg_scale": True,
                    },
                    "gradient_isolation_checks": {
                        "same_tensor_object": (
                            False if surrogate_losses_by_term is None else ppo_true_actor_grad.data_ptr() == amtl_actor_grad.data_ptr()
                        ),
                        "same_shape": (
                            True if surrogate_losses_by_term is None else list(ppo_true_actor_grad.shape) == list(amtl_actor_grad.shape)
                        ),
                        "ppo_true_has_nan_or_inf": not torch.isfinite(ppo_true_actor_grad).all().item(),
                        "amtl_has_nan_or_inf": (
                            False if surrogate_losses_by_term is None else not torch.isfinite(amtl_actor_grad).all().item()
                        ),
                        "ppo_true_nonzero_norm": ppo_true_actor_grad_norm_value > 0.0,
                        "amtl_nonzero_norm": amtl_actor_grad_norm_value > 0.0,
                        "includes_value_loss": False,
                        "includes_entropy_loss": False,
                        "includes_auxiliary_loss": False,
                    },
                }
                self._write_first_update_gradient_debug(debug_payload)

            actor_mean_grad_norm_value = self._compute_parameter_grad_norm(self.policy.get_actor_mean_parameters())
            actor_std_grad_norm_value = self._compute_parameter_grad_norm(self.policy.get_actor_std_parameters())
            actor_std_to_mean_grad_ratio_value = actor_std_grad_norm_value / (actor_mean_grad_norm_value + 1.0e-8)

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
            if self.amp_optimizer:
                self.amp_optimizer.step()
           # self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_kl_divergence += kl_mean.item()
            mean_objective_mean_cosine += objective_mean_cosine_value
            mean_objective_median_cosine += objective_median_cosine_value
            mean_objective_min_cosine += objective_min_cosine_value
            mean_objective_max_cosine += objective_max_cosine_value
            mean_objective_std_cosine += objective_std_cosine_value
            mean_objective_conflict_fraction += objective_conflict_fraction_value
            mean_objective_effective_rank += objective_effective_rank_value
            mean_objective_sv1_ratio += objective_sv1_ratio_value
            mean_objective_sv2_ratio += objective_sv2_ratio_value
            mean_objective_sv3_ratio += objective_sv3_ratio_value
            mean_objective_pca_var1 += objective_pca_var1_value
            mean_objective_pca_var2 += objective_pca_var2_value
            mean_objective_pca_var3 += objective_pca_var3_value
            mean_aligned_grad_norm += aligned_grad_norm_value
            mean_regularizer_grad_norm += regularizer_grad_norm_value
            mean_reg_scale += reg_scale_value
            mean_scaled_regularizer_grad_norm += scaled_regularizer_grad_norm_value
            mean_total_actor_grad_norm += total_actor_grad_norm_value
            mean_actor_mean_grad_norm += actor_mean_grad_norm_value
            mean_actor_std_grad_norm += actor_std_grad_norm_value
            mean_actor_std_to_mean_grad_ratio += actor_std_to_mean_grad_ratio_value
            mean_num_aligned_terms += num_aligned_terms_value
            mean_num_regularizer_terms += num_regularizer_terms_value
            mean_ppo_per_objective_proxy_grad_norm += ppo_actor_grad_norm_value
            mean_ppo_true_actor_grad_norm += ppo_true_actor_grad_norm_value
            mean_amtl_actor_grad_norm += amtl_actor_grad_norm_value
            mean_blended_actor_grad_norm += blended_actor_grad_norm_value
            mean_ppo_per_objective_proxy_amtl_grad_cosine += ppo_amtl_grad_cosine_value
            mean_ppo_true_amtl_grad_cosine += ppo_true_amtl_grad_cosine_value
            mean_ppo_true_to_amtl_norm_ratio += ppo_true_to_amtl_norm_ratio_value
            mean_ppo_true_minus_amtl_grad_norm += ppo_true_minus_amtl_grad_norm_value
            mean_ppo_true_minus_amtl_relative_norm += ppo_true_minus_amtl_relative_norm_value
            mean_amtl_projection_on_ppo_true += amtl_projection_on_ppo_true_value
            mean_ppo_true_projection_on_amtl += ppo_true_projection_on_amtl_value
            mean_ppo_true_actor_mean_grad_norm += ppo_true_actor_mean_grad_norm_value
            mean_ppo_true_actor_std_grad_norm += ppo_true_actor_std_grad_norm_value
            mean_amtl_actor_mean_grad_norm += amtl_actor_mean_grad_norm_value
            mean_amtl_actor_std_grad_norm += amtl_actor_std_grad_norm_value
            mean_ppo_true_std_to_mean_grad_ratio += ppo_true_std_to_mean_grad_ratio_value
            mean_amtl_std_to_mean_grad_ratio += amtl_std_to_mean_grad_ratio_value
            mean_top_objective_grad_fraction += top_objective_grad_fraction_value
            mean_bottom_objective_grad_fraction += bottom_objective_grad_fraction_value
            mean_objective_grad_fraction_entropy += objective_grad_fraction_entropy_value
            mean_logged_ppo_blend_weight += ppo_blend_weight_value
            mean_logged_amtl_blend_weight += amtl_blend_weight_value
            if amp_stats is not None:
                mean_amp_discriminator_loss += amp_stats["amp_discriminator"]
                mean_amp_grad_penalty += amp_stats["amp_grad_penalty"]
                mean_amp_ref_score += amp_stats["amp_ref_score"]
                mean_amp_policy_score += amp_stats["amp_policy_score"]
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
        mean_kl_divergence /= num_updates
        mean_objective_mean_cosine /= num_updates
        mean_objective_median_cosine /= num_updates
        mean_objective_min_cosine /= num_updates
        mean_objective_max_cosine /= num_updates
        mean_objective_std_cosine /= num_updates
        mean_objective_conflict_fraction /= num_updates
        mean_objective_effective_rank /= num_updates
        mean_objective_sv1_ratio /= num_updates
        mean_objective_sv2_ratio /= num_updates
        mean_objective_sv3_ratio /= num_updates
        mean_objective_pca_var1 /= num_updates
        mean_objective_pca_var2 /= num_updates
        mean_objective_pca_var3 /= num_updates
        mean_aligned_grad_norm /= num_updates
        mean_regularizer_grad_norm /= num_updates
        mean_reg_scale /= num_updates
        mean_scaled_regularizer_grad_norm /= num_updates
        mean_total_actor_grad_norm /= num_updates
        mean_actor_mean_grad_norm /= num_updates
        mean_actor_std_grad_norm /= num_updates
        mean_actor_std_to_mean_grad_ratio /= num_updates
        mean_num_aligned_terms /= num_updates
        mean_num_regularizer_terms /= num_updates
        mean_ppo_per_objective_proxy_grad_norm /= num_updates
        mean_ppo_true_actor_grad_norm /= num_updates
        mean_amtl_actor_grad_norm /= num_updates
        mean_blended_actor_grad_norm /= num_updates
        mean_ppo_per_objective_proxy_amtl_grad_cosine /= num_updates
        mean_ppo_true_amtl_grad_cosine /= num_updates
        mean_ppo_true_to_amtl_norm_ratio /= num_updates
        mean_ppo_true_minus_amtl_grad_norm /= num_updates
        mean_ppo_true_minus_amtl_relative_norm /= num_updates
        mean_amtl_projection_on_ppo_true /= num_updates
        mean_ppo_true_projection_on_amtl /= num_updates
        mean_ppo_true_actor_mean_grad_norm /= num_updates
        mean_ppo_true_actor_std_grad_norm /= num_updates
        mean_amtl_actor_mean_grad_norm /= num_updates
        mean_amtl_actor_std_grad_norm /= num_updates
        mean_ppo_true_std_to_mean_grad_ratio /= num_updates
        mean_amtl_std_to_mean_grad_ratio /= num_updates
        mean_top_objective_grad_fraction /= num_updates
        mean_bottom_objective_grad_fraction /= num_updates
        mean_objective_grad_fraction_entropy /= num_updates
        mean_logged_ppo_blend_weight /= num_updates
        mean_logged_amtl_blend_weight /= num_updates
        mean_amp_discriminator_loss /= num_updates
        mean_amp_grad_penalty /= num_updates
        mean_amp_ref_score /= num_updates
        mean_amp_policy_score /= num_updates
        for objective_name in list(mean_objective_grad_norms):
            mean_objective_grad_norms[objective_name] /= num_updates
        for objective_name in list(mean_objective_grad_fractions):
            mean_objective_grad_fractions[objective_name] /= num_updates
        for objective_name in list(mean_objective_aligned_projections):
            mean_objective_aligned_projections[objective_name] /= num_updates
        self.latest_objective_cosine_matrix = None
        if objective_cosine_matrix_sum is not None and objective_cosine_matrix_count > 0:
            self.latest_objective_cosine_matrix = (
                objective_cosine_matrix_sum / objective_cosine_matrix_count
            ).detach().cpu()
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
            "kl_divergence": mean_kl_divergence,
            "objective_mean_cosine": mean_objective_mean_cosine,
            "objective_median_cosine": mean_objective_median_cosine,
            "objective_min_cosine": mean_objective_min_cosine,
            "objective_max_cosine": mean_objective_max_cosine,
            "objective_std_cosine": mean_objective_std_cosine,
            "objective_conflict_fraction": mean_objective_conflict_fraction,
            "objective_effective_rank": mean_objective_effective_rank,
            "objective_sv1_ratio": mean_objective_sv1_ratio,
            "objective_sv2_ratio": mean_objective_sv2_ratio,
            "objective_sv3_ratio": mean_objective_sv3_ratio,
            "objective_pca_var1": mean_objective_pca_var1,
            "objective_pca_var2": mean_objective_pca_var2,
            "objective_pca_var3": mean_objective_pca_var3,
            "aligned_grad_norm": mean_aligned_grad_norm,
            "regularizer_grad_norm": mean_regularizer_grad_norm,
            "reg_scale": mean_reg_scale,
            "scaled_regularizer_grad_norm": mean_scaled_regularizer_grad_norm,
            "total_actor_grad_norm": mean_total_actor_grad_norm,
            "actor_mean_grad_norm": mean_actor_mean_grad_norm,
            "actor_std_grad_norm": mean_actor_std_grad_norm,
            "actor_std_to_mean_grad_ratio": mean_actor_std_to_mean_grad_ratio,
            "num_aligned_terms": mean_num_aligned_terms,
            "num_regularizer_terms": mean_num_regularizer_terms,
            "ppo_per_objective_proxy_grad_norm": mean_ppo_per_objective_proxy_grad_norm,
            "ppo_true_actor_grad_norm": mean_ppo_true_actor_grad_norm,
            "amtl_actor_grad_norm": mean_amtl_actor_grad_norm,
            "blended_actor_grad_norm": mean_blended_actor_grad_norm,
            "ppo_per_objective_proxy_amtl_grad_cosine": mean_ppo_per_objective_proxy_amtl_grad_cosine,
            "ppo_true_amtl_grad_cosine": mean_ppo_true_amtl_grad_cosine,
            "ppo_true_to_amtl_norm_ratio": mean_ppo_true_to_amtl_norm_ratio,
            "ppo_true_minus_amtl_grad_norm": mean_ppo_true_minus_amtl_grad_norm,
            "ppo_true_minus_amtl_relative_norm": mean_ppo_true_minus_amtl_relative_norm,
            "amtl_projection_on_ppo_true": mean_amtl_projection_on_ppo_true,
            "ppo_true_projection_on_amtl": mean_ppo_true_projection_on_amtl,
            "ppo_true_actor_mean_grad_norm": mean_ppo_true_actor_mean_grad_norm,
            "ppo_true_actor_std_grad_norm": mean_ppo_true_actor_std_grad_norm,
            "amtl_actor_mean_grad_norm": mean_amtl_actor_mean_grad_norm,
            "amtl_actor_std_grad_norm": mean_amtl_actor_std_grad_norm,
            "ppo_true_std_to_mean_grad_ratio": mean_ppo_true_std_to_mean_grad_ratio,
            "amtl_std_to_mean_grad_ratio": mean_amtl_std_to_mean_grad_ratio,
            "top_objective_grad_fraction": mean_top_objective_grad_fraction,
            "bottom_objective_grad_fraction": mean_bottom_objective_grad_fraction,
            "objective_grad_fraction_entropy": mean_objective_grad_fraction_entropy,
            "ppo_blend_weight": mean_logged_ppo_blend_weight,
            "amtl_blend_weight": mean_logged_amtl_blend_weight,
            "blend_uses_ppo_true": 1.0,
            "amp_discriminator": mean_amp_discriminator_loss,
            "amp_grad_penalty": mean_amp_grad_penalty,
            "amp_ref_score": mean_amp_ref_score,
            "amp_policy_score": mean_amp_policy_score,
        }
        for objective_name, value in mean_objective_grad_norms.items():
            loss_dict[f"objective_grad_norm/{objective_name}"] = value
        for objective_name, value in mean_objective_grad_fractions.items():
            loss_dict[f"objective_grad_fraction/{objective_name}"] = value
        for objective_name, value in mean_objective_aligned_projections.items():
            loss_dict[f"objective_aligned_projection/{objective_name}"] = value
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
        if self.amp_discriminator:
            model_params.append(self.amp_discriminator.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        state_index = 1
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[state_index])
            state_index += 1
        if self.amp_discriminator:
            self.amp_discriminator.load_state_dict(model_params[state_index])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        if self.amp_discriminator:
            grads += [param.grad.view(-1) for param in self.amp_discriminator.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        if self.amp_discriminator:
            all_params = chain(all_params, self.amp_discriminator.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
