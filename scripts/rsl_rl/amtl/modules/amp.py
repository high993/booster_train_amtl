from __future__ import annotations

import torch
import torch.nn as nn


class AmpDiscriminator(nn.Module):
    """Least-squares AMP discriminator over consecutive state features."""

    def __init__(self, state_feature_dim: int, hidden_dims: tuple[int, ...] = (1024, 512, 256)) -> None:
        super().__init__()

        input_dim = state_feature_dim * 2
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state_features_t: torch.Tensor, state_features_tp1: torch.Tensor) -> torch.Tensor:
        transition_features = torch.cat([state_features_t, state_features_tp1], dim=-1)
        return self.network(transition_features).squeeze(-1)

    def forward_transition(self, transition_features: torch.Tensor) -> torch.Tensor:
        return self.network(transition_features).squeeze(-1)


class AmpReplayBuffer:
    """Ring buffer of policy-generated AMP transition features."""

    def __init__(self, capacity: int, transition_feature_dim: int, device: str | torch.device = "cpu") -> None:
        if capacity <= 0:
            raise ValueError(f"AMP replay buffer capacity must be positive. Received: {capacity}.")

        self.capacity = capacity
        self.transition_feature_dim = transition_feature_dim
        self.device = device
        self.storage = torch.zeros(capacity, transition_feature_dim, device=device)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, transition_features: torch.Tensor) -> None:
        if transition_features.numel() == 0:
            return

        flat_transitions = transition_features.reshape(-1, self.transition_feature_dim).detach()
        num_new = flat_transitions.shape[0]

        if num_new >= self.capacity:
            self.storage.copy_(flat_transitions[-self.capacity :])
            self.position = 0
            self.size = self.capacity
            return

        first_chunk = min(self.capacity - self.position, num_new)
        self.storage[self.position : self.position + first_chunk] = flat_transitions[:first_chunk]

        remaining = num_new - first_chunk
        if remaining > 0:
            self.storage[:remaining] = flat_transitions[first_chunk:]

        self.position = (self.position + num_new) % self.capacity
        self.size = min(self.capacity, self.size + num_new)

    def sample(self, batch_size: int) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("AMP replay buffer is empty.")
        if batch_size <= 0:
            raise ValueError(f"AMP batch size must be positive. Received: {batch_size}.")

        sample_size = min(batch_size, self.size)
        indices = torch.randint(0, self.size, (sample_size,), device=self.device)
        return self.storage[indices]
