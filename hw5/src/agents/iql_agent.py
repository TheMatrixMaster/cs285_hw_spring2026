from typing import Optional
import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        expectile: float,
    ):
        super().__init__()

        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.expectile = expectile

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.actor(observation).mode  # Take the mean (mode) action
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action[0])

    @staticmethod
    def iql_expectile_loss(
        adv: torch.Tensor, expectile: float,
    ) -> torch.Tensor:
        """
        Compute the expectile loss for IQL
        """
        # TODO(student): Implement the expectile loss
        return torch.abs(expectile - (adv > 0).float()) * adv**2

    @torch.compile
    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update V(s) with expectile regression
        """
        # TODO(student): Compute the value loss
        v = self.value(observations)
        q = torch.min(self.target_critic(observations, actions), dim=0).values.detach()
        loss = self.iql_expectile_loss(v - q, self.expectile).mean()

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        return {
            "v_loss": loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        batch_size = observations.shape[0]
        num_critic_networks = len(self.critic.net.mlps)

        with torch.no_grad():
            next_v = self.value(next_observations)

            if next_v.shape == (batch_size,):
                next_v = next_v[None].expand((num_critic_networks, batch_size)).contiguous()

            assert next_v.shape == (num_critic_networks, batch_size), next_v.shape
            target_values = rewards + (1.0 - dones) * self.discount * next_v

        q_values = self.critic(observations, actions)
        assert q_values.shape == (num_critic_networks, batch_size), q_values.shape

        loss = F.mse_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q_values.mean(),
            "q_max": q_values.max(),
            "q_min": q_values.min(),
        }

    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor using advantage-weighted regression
        """
        # TODO(student): Compute the actor loss
        num_critic_networks = len(self.critic.net.mlps)
        batch_size = observations.shape[0]
        
        with torch.no_grad():
            q_values: torch.Tensor = self.critic(observations, actions)
            assert q_values.shape == (num_critic_networks, batch_size), q_values.shape

            v: torch.Tensor = self.value(observations)
            assert v.shape == (batch_size,), v.shape

            adv = q_values.min(dim=0).values - v
            weights = -torch.clamp(torch.exp(self.alpha * adv), max=100)

        dist: distributions.Distribution = self.actor(observations)
        log_prob = dist.log_prob(actions)

        loss = (weights * log_prob).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": loss,
            "mse": torch.mean((dist.mode - actions) ** 2),
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_v(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        tau = self.target_update_rate
        critic_nets = self.critic.net.mlps
        target_critic_nets = self.target_critic.net.mlps
        for target_critic, critic in zip(target_critic_nets, critic_nets):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
