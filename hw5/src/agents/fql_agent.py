from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        batch_size = observation.shape[0]
        acs = torch.randn(batch_size, self.action_dim, device=observation.device)
        action = self.onestep_actor(obs=observation, acs=acs)
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        batch_size = observation.shape[0]
        dt = 1 / self.flow_steps
        action = noise

        for step in range(self.flow_steps):
            t = torch.full((batch_size, 1), fill_value = step * dt, device=observation.device)
            action += dt * self.bc_actor(obs=observation, acs=action, times=t)

        action = torch.clamp(action, -1, 1)
        return action

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
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        batch_size = observations.shape[0]
        num_critic_networks = len(self.critic.net.mlps)

        with torch.no_grad():
            acs = torch.randn_like(actions)
            next_actions = self.onestep_actor(obs=next_observations, acs=acs)
            next_actions = torch.clamp(next_actions, -1, 1)

            next_qs = self.target_critic(next_observations, next_actions)
            next_qs = next_qs.mean(dim=0)

            if next_qs.shape == (batch_size,):
                next_qs = (
                    next_qs[None].expand((num_critic_networks, batch_size)).contiguous()
                )

            assert next_qs.shape == (num_critic_networks, batch_size), next_qs.shape
            target_values = rewards + (1.0 - dones) * self.discount * next_qs

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
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        batch_size = observations.shape[0]

        acs = torch.randn_like(actions)
        t = torch.rand(batch_size, 1, device=observations.device)

        actions_t = (1 - t) * acs + t * actions
        v_pred = self.bc_actor(obs=observations, acs=actions_t, times=t)
        v_true = actions - acs

        loss = F.mse_loss(v_pred, v_true)   # this already divides by action_dim

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        num_critic_networks = len(self.critic.net.mlps)
        batch_size = observations.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(actions)
            acs_true = self.get_bc_action(observations, noise)

        acs_pred = self.onestep_actor(obs=observations, acs=noise)
        distill_loss = self.alpha * F.mse_loss(acs_pred, acs_true)  # this already divides by action_dim

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        for critic in self.critic.net.mlps:
            for param in critic.parameters():
                param.requires_grad = False

        q_values = self.critic(observations, torch.clamp(acs_pred, -1, 1))
        assert q_values.shape == (num_critic_networks, batch_size), q_values.shape

        for critic in self.critic.net.mlps:
            for param in critic.parameters():
                param.requires_grad = True

        q_loss = -q_values.mean()

        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        mse = F.mse_loss(acs_pred, acs_true).detach()

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
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
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
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
