# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from collections.abc import Sequence
from itertools import chain
from tensordict import TensorDict

from amtl.modules import ActorCriticRecurrent
from amtl.modules.rnd import RandomNetworkDistillation
from amtl.storage import RolloutStorage
from amtl.utils import resolve_optimizer, string_to_callable

from amtl.actor_critic import ActorCritic


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
        amtl_apply_to: str = "critic",
        debug_amtl: bool = False,
        debug_amtl_log_interval: int = 100,
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
                "Use `amtl_apply_to=\"critic\"` for the critic-only AMTL setup.",
                stacklevel=2,
            )
            if legacy_align_actor:
                raise ValueError("Actor-side AMTL is disabled in this critic-only PPO branch.")
            if legacy_align_critic and amtl_apply_to == "none":
                amtl_apply_to = "critic"

        if amtl_apply_to not in {"none", "critic"}:
            raise ValueError(
                f"Unsupported amtl_apply_to='{amtl_apply_to}'. This branch supports only 'none' and 'critic'."
            )

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
        # This branch only supports critic-side AMTL. The actor always stays on
        # the standard PPO clipped surrogate path.
        self.amtl_apply_to = amtl_apply_to
        self.use_actor_amtl = False
        self.use_critic_amtl = amtl_apply_to == "critic"
        self.debug_amtl = debug_amtl
        self.debug_amtl_log_interval = debug_amtl_log_interval
        self.update_counter = 0

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

    def _validate_actor_critic_parameter_partition(self) -> None:
        # Critic-only AMTL temporarily zeroes and rewrites critic grads, so the
        # actor/critic parameter sets must stay disjoint.
        actor_param_ids = {id(param) for param in self.policy.get_actor_parameters()}
        critic_param_ids = {id(param) for param in self.policy.get_critic_parameters()}
        shared_param_count = len(actor_param_ids & critic_param_ids)
        if shared_param_count > 0:
            raise RuntimeError(
                "Critic-only AMTL currently requires disjoint actor/critic parameters, "
                f"but found {shared_param_count} shared parameter(s)."
            )

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
    # critic-only `update()` implementation below.
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
        """Critic-only AMTL update.

        The actor uses the standard PPO clipped surrogate. Only the per-head
        critic losses participate in AMTL gradient alignment.
        """

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl_divergence = 0.0
        mean_critic_amtl_grad_norm = 0.0
        critic_value_loss_sums: torch.Tensor | None = None
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

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
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                    if advantages_by_term_batch is not None:
                        term_mean = advantages_by_term_batch.mean(dim=0, keepdim=True)
                        term_std = advantages_by_term_batch.std(dim=0, keepdim=True)
                        advantages_by_term_batch = (advantages_by_term_batch - term_mean) / (term_std + 1e-8)

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
            entropy_batch = self.policy.entropy[:original_batch_size]

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

            # Actor AMTL is OFF here by design. The actor uses standard PPO with
            # the scalar summed advantage and the clipped surrogate objective.
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * clipped_ratio
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

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
            critic_params = list(self.policy.get_critic_parameters())

            # Actor PPO, entropy, and optional symmetry loss accumulate first.
            (surrogate_loss + actor_auxiliary_loss).backward()

            critic_grad_norm = 0.0
            if critic_losses_by_term is not None and self.use_critic_amtl:
                critic_grads = []
                for loss_index, loss in enumerate(critic_losses_by_term):
                    # Isolate each critic head gradient before alignment.
                    self._zero_existing_grads(critic_params)
                    retain_graph = loss_index < len(critic_losses_by_term) - 1
                    (self.value_loss_coef * loss).backward(retain_graph=retain_graph)
                    critic_grads.append(self._collect_flat_grad(critic_params))

                self._zero_existing_grads(critic_params)
                critic_grads = torch.stack(critic_grads, dim=0)
                # After alignment the solver still returns [1, num_parameters, num_heads],
                # so summing over the last axis produces one flattened critic gradient.
                aligned_critic_grads, _, _ = ProcrustesSolver.apply(critic_grads.T.unsqueeze(0))
                aligned_critic_grad = aligned_critic_grads[0].sum(-1)
                set_shared_grad(critic_params, aligned_critic_grad)
                critic_grad_norm = aligned_critic_grad.norm().item()
            else:
                # Baseline path: plain mean value loss without AMTL alignment.
                (self.value_loss_coef * value_loss).backward()
                critic_grad_norm = self._collect_flat_grad(critic_params).norm().item()

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
                print("Actor AMTL: OFF (standard PPO clipped surrogate)")
                print(f"Critic heads: {value_batch.shape[-1]}")
                print(f"Critic objectives: {critic_objectives}")
                print(f"Critic AMTL: {'ON' if self.use_critic_amtl else 'OFF'}")
                print(
                    "Critic training: AMTL-aligned per-head MSE"
                    if self.use_critic_amtl
                    else "Critic training: normal mean per-head MSE"
                )
                print(f"Critic value loss per head: {critic_value_losses}")
                print(f"Critic AMTL gradient norm: {critic_grad_norm}")
                debug_logged_this_update = True

            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_kl_divergence += kl_mean_value
            mean_critic_amtl_grad_norm += critic_grad_norm
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
        mean_critic_amtl_grad_norm /= num_updates
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
            "critic_amtl_grad_norm": mean_critic_amtl_grad_norm,
            "actor_amtl_enabled": 1.0 if self.use_actor_amtl else 0.0,
            "critic_amtl_enabled": 1.0 if self.use_critic_amtl else 0.0,
        }
        if critic_value_loss_sums is not None:
            critic_head_names = self._get_critic_head_names(len(critic_value_loss_sums))
            for head_name, head_loss in zip(critic_head_names, critic_value_loss_sums / num_updates, strict=True):
                loss_dict[f"critic_value_loss/{head_name}"] = head_loss.item()
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
