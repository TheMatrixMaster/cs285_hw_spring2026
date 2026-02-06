"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)
        self.mse_loss = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,        # (bs, state_dim)
        action_chunk: torch.Tensor, # (bs, chunk_size, action_dim)
    ) -> torch.Tensor:
        bs = state.shape[0]
        pred_chunk = self.net(state)
        action_chunk = action_chunk.reshape(bs, -1)
        return self.mse_loss(pred_chunk, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        bs = state.shape[0]
        return self.net(state).reshape(bs, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        self.chunk_action_dim = chunk_size * action_dim
        input_dim = state_dim + self.chunk_action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.chunk_action_dim))
        self.net = nn.Sequential(*layers)
        self.mse_loss = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,        # (bs, state_dim)
        action_chunk: torch.Tensor, # (bs, chunk_size, action_dim)
    ) -> torch.Tensor:
        bs = state.shape[0]
        action_chunk = action_chunk.reshape(bs, -1)
        noise = torch.randn_like(action_chunk)                  # sample noise from std normal
        tau = torch.rand(bs, 1, device=state.device)            # sample tau uniformly from [0, 1]
        noised_chunk = action_chunk * tau + noise * (1 - tau)   # interpolate between noise (tau=0) and action_chunk (tau=1)
        net_input = torch.cat([state, noised_chunk], dim=-1)    # (bs, state_dim + chunk_action_dim)
        v_pred = self.net(net_input)                            # vector field prediction (bs, chunk_action_dim)
        v_target = (action_chunk - noise).detach()              # target vector field
        return self.mse_loss(v_pred, v_target)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        bs = state.shape[0]
        noise = torch.randn(bs, self.chunk_action_dim, device=state.device) # (bs, chunk_action_dim)
        dt = 1.0 / num_steps
        action_chunk = noise

        for _ in range(num_steps):
            net_input = torch.cat([state, action_chunk], dim=-1) # (bs, state_dim + chunk_action_dim)
            v = self.net(net_input) # (bs, chunk_action_dim)
            action_chunk = action_chunk + v * dt

        return action_chunk.reshape(bs, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
