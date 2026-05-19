# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from itertools import chain
from tensordict import TensorDict

from amtl.modules import ActorCriticRecurrent
from amtl.modules.rnd import RandomNetworkDistillation
from amtl.storage import RolloutStorage
from amtl.utils import resolve_optimizer, string_to_callable

from amtl.actor_critic import ActorCritic






















class PPO:
    """On-policy actor-critic trainer with PPO and APA actor objectives."""

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
        schedule: str = "fixed",
        desired_kl: float | None = None,
        device: str = "cpu",
        env=None,
        normalize_advantage_per_mini_batch: bool = False,
        actor_loss_mode: str = "apa",
        apa_beta: float = 0.1,
        apa_normalize_advantage: bool = True,
        apa_target_shift_clip: float | None = None,
        ref_policy_sync_interval: int | None = None,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
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

        # On-policy components
        self.policy = policy
        self.policy.to(self.device)

        # Create optimizer
        optimizer_cls = resolve_optimizer(optimizer)
        self.optimizer = optimizer_cls(self.policy.parameters(), lr=learning_rate)

        # Create rollout storage
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        # On-policy parameters
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
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.actor_loss_mode = actor_loss_mode.lower()
        if self.actor_loss_mode not in {"ppo", "apa"}:
            raise ValueError(f"Unknown actor loss mode: {actor_loss_mode}. Expected one of: 'ppo', 'apa'.")
        if self.actor_loss_mode == "apa" and (self.schedule != "fixed" or self.desired_kl is not None):
            warnings.warn(
                "APA uses a frozen reference policy. Disabling PPO-style adaptive KL scheduling by setting "
                "`schedule='fixed'` and `desired_kl=None`.",
            )
            self.schedule = "fixed"
            self.desired_kl = None
        self.apa_beta = apa_beta
        self.apa_normalize_advantage = apa_normalize_advantage
        if apa_target_shift_clip is not None and apa_target_shift_clip <= 0:
            raise ValueError("`apa_target_shift_clip` must be positive or None.")
        self.apa_target_shift_clip = apa_target_shift_clip
        if ref_policy_sync_interval is not None and ref_policy_sync_interval <= 0:
            raise ValueError("`ref_policy_sync_interval` must be a positive integer or None.")
        self.ref_policy_sync_interval = ref_policy_sync_interval
        self.update_counter = 0
        self.ref_policy: ActorCritic | ActorCriticRecurrent | None = None
        if self.actor_loss_mode == "apa":
            self.sync_reference_policy()

    def sync_reference_policy(self, state_dict: dict | None = None) -> None:
        if self.actor_loss_mode != "apa":
            self.ref_policy = None
            return

        if self.ref_policy is None:
            self.ref_policy = copy.deepcopy(self.policy)

        if state_dict is None:
            state_dict = self.policy.state_dict()
        self.ref_policy.load_state_dict(state_dict)
        self.ref_policy.to(self.device)
        self.ref_policy.eval()
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _squeeze_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
        if log_probs.ndim > 1 and log_probs.shape[-1] == 1:
            return log_probs.squeeze(-1)
        return log_probs

    @staticmethod
    def _collapse_advantages(advantages: torch.Tensor) -> torch.Tensor:
        if advantages.ndim > 1 and advantages.shape[-1] != 1:
            advantages = advantages.sum(dim=-1)
        return advantages.squeeze(-1)

    def _compute_actor_loss(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        masks_batch: torch.Tensor | None,
        hidden_states_batch: tuple,
        actions_log_prob_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        advantages_by_term_batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.actor_loss_mode == "apa":
            if self.ref_policy is None:
                raise RuntimeError("APA actor loss requires a frozen reference policy.")

            logp = self._squeeze_log_probs(actions_log_prob_batch)
            with torch.no_grad():
                self.ref_policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
                logp_ref = self.ref_policy.get_actions_log_prob(actions_batch)
                logp_ref = self._squeeze_log_probs(logp_ref)
                ref_mu = self.ref_policy.action_mean
                ref_sigma = self.ref_policy.action_std
            # APA uses the rollout advantages directly instead of the per-term
            # decomposition, which can overweight dense auxiliary penalties.
            adv = self._collapse_advantages(advantages_batch).detach()
            raw_adv_std = adv.std(unbiased=False)
            if self.apa_normalize_advantage:
                adv = (adv - adv.mean()) / (raw_adv_std + 1e-8)
            target_shift = self.apa_beta * adv
            if self.apa_target_shift_clip is not None:
                target_shift = target_shift.clamp(-self.apa_target_shift_clip, self.apa_target_shift_clip)
            apa_target = logp_ref + target_shift
            actor_loss = (logp - apa_target).pow(2).mean()

            with torch.no_grad():
                current_mu = self.policy.action_mean
                current_sigma = self.policy.action_std

                current_sigma_safe = current_sigma.clamp_min(1.0e-8)
                ref_sigma_safe = ref_sigma.clamp_min(1.0e-8)

                ref_kl = torch.sum(
                    torch.log(current_sigma_safe / ref_sigma_safe)
                    + (ref_sigma_safe.pow(2) + (ref_mu - current_mu).pow(2))
                    / (2.0 * current_sigma_safe.pow(2))
                    - 0.5,
                    dim=-1,
                ).mean()



            return actor_loss, {
                "apa_ref_kl": ref_kl.detach(),
                "apa_adv_std": raw_adv_std.detach(),
            }

        return (
            self._compute_ppo_actor_loss(
                actions_log_prob_batch,
                old_actions_log_prob_batch,
                advantages_batch,
                advantages_by_term_batch,
            ),
            {},
        )

    def _compute_ppo_actor_loss(
        self,
        actions_log_prob_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        advantages_by_term_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        if advantages_by_term_batch is not None:
            ratio_by_term = ratio.unsqueeze(-1)
            clipped_ratio_by_term = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param).unsqueeze(-1)
            surrogate_by_term = -advantages_by_term_batch * ratio_by_term
            surrogate_clipped_by_term = -advantages_by_term_batch * clipped_ratio_by_term
            return torch.max(surrogate_by_term, surrogate_clipped_by_term).mean(dim=0).sum()

        scalar_advantages = torch.squeeze(advantages_batch, dim=-1)
        surrogate = -scalar_advantages * ratio
        surrogate_clipped = -scalar_advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        return torch.max(surrogate, surrogate_clipped).mean()

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

    @staticmethod
    def _is_tensor_finite(tensor: torch.Tensor) -> bool:
        return bool(torch.isfinite(tensor).all().item())

    @staticmethod
    def _are_gradients_finite(parameters) -> bool:
        for param in parameters:
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return False
        return True

    @staticmethod
    def _are_parameters_finite(parameters) -> bool:
        for param in parameters:
            if not torch.isfinite(param.data).all():
                return False
        return True

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        value_shape: tuple[int] | list[int] = (1,),
    ) -> None:
        # Create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            value_shape,
            self.device,
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
            timeout_bonus = self.gamma * self.transition.values * timeout_mask
            self.transition.rewards += timeout_bonus.sum(dim=-1)
            if reward_terms is not None and timeout_bonus.shape[-1] == reward_terms.shape[-1]:
                reward_terms = reward_terms + timeout_bonus

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
        self.update_counter += 1
        ref_policy_synced = 0.0
        if (
            self.actor_loss_mode == "apa"
            and self.ref_policy_sync_interval is not None
            and self.update_counter % self.ref_policy_sync_interval == 0
        ):
            self.sync_reference_policy()
            ref_policy_synced = 1.0

        mean_value_loss = 0
        mean_actor_loss = 0
        mean_entropy = 0
        mean_apa_ref_kl = 0 if self.actor_loss_mode == "apa" else None
        mean_apa_adv_std = 0 if self.actor_loss_mode == "apa" else None
        skipped_updates = 0
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
            self.policy.update_distribution(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
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




            actor_loss, actor_metrics = self._compute_actor_loss(
                obs_batch,
                actions_batch,
                masks_batch,
                hidden_states_batch,
                actions_log_prob_batch,
                old_actions_log_prob_batch,
                advantages_batch,
                advantages_by_term_batch,
            )

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

            # Compute the gradients for the actor-critic update
            self.optimizer.zero_grad()

            loss = actor_loss + auxiliary_loss
            if not self._is_tensor_finite(loss):
                skipped_updates += 1
                continue
            loss.backward()
            if not self._are_gradients_finite(self.policy.parameters()):
                self.optimizer.zero_grad()
                skipped_updates += 1
                continue


            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                if not self._are_gradients_finite(self.rnd.parameters()):
                    self.optimizer.zero_grad()
                    self.rnd_optimizer.zero_grad()
                    skipped_updates += 1
                    continue

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad()
                if self.rnd_optimizer:
                    self.rnd_optimizer.zero_grad()
                skipped_updates += 1
                continue
            self.optimizer.step()
            if not self._are_parameters_finite(self.policy.parameters()):
                raise RuntimeError("Non-finite policy parameters detected after optimizer step.")
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
                if not self._are_parameters_finite(self.rnd.parameters()):
                    raise RuntimeError("Non-finite RND parameters detected after optimizer step.")

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_actor_loss += actor_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_apa_ref_kl is not None:
                mean_apa_ref_kl += actor_metrics["apa_ref_kl"].item()
            if mean_apa_adv_std is not None:
                mean_apa_adv_std += actor_metrics["apa_adv_std"].item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_actor_loss /= num_updates
        mean_entropy /= num_updates
        if mean_apa_ref_kl is not None:
            mean_apa_ref_kl /= num_updates
        if mean_apa_adv_std is not None:
            mean_apa_adv_std /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "actor" if self.actor_loss_mode == "apa" else "surrogate": mean_actor_loss,
            "entropy": mean_entropy,
            "skipped_updates": float(skipped_updates),
        }
        if self.actor_loss_mode == "apa":
            loss_dict["apa_loss"] = mean_actor_loss
            loss_dict["apa_ref_kl"] = mean_apa_ref_kl
            loss_dict["apa_adv_std"] = mean_apa_adv_std
            if self.ref_policy_sync_interval is not None:
                loss_dict["apa_ref_synced"] = ref_policy_synced
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.ref_policy is not None:
            model_params.append(self.ref_policy.state_dict())
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        offset = 1
        if self.ref_policy is not None:
            self.sync_reference_policy(model_params[offset])
            offset += 1
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[offset])

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
