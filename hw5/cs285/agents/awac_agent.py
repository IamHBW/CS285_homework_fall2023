from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn


from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        batch_size = observations.shape[0]
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            next_action_distribution = self.actor(next_observations.view(batch_size,-1))
            next_qa_values = self.target_critic(next_observations.view(batch_size,-1))

            # Use the actor to compute a critic backup
            next_probs = next_action_distribution.probs
            next_qs = (next_probs * next_qa_values).sum(dim=-1)
            assert dones.shape == (batch_size,), dones.shape
            next_qs = next_qs * (1 - dones.float())
            next_qs = next_qs.squeeze(dim=-1)
            assert next_qs.shape == (batch_size,), next_qs.shape

            # TODO(student): Compute the TD target
            target_values = rewards.squeeze(dim=-1) + self.discount * next_qs
            assert target_values.shape == (batch_size,), target_values.shape

        
        # TODO(student): Compute Q(s, a) and loss similar to DQN
        qa_values = self.critic(observations.view(batch_size,-1))
        q_values = torch.gather(qa_values,-1,actions.unsqueeze(-1)).squeeze(dim=-1)
        assert q_values.shape == target_values.shape

        loss = self.critic_loss(q_values,target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    @torch.no_grad()
    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        qa_values = self.critic(observations)
        q_values = torch.gather(qa_values,-1,actions.unsqueeze(-1)).squeeze(dim=-1)
        if action_dist is not None:
            action_probs = action_dist.probs
            values = (action_probs * qa_values).sum(dim=-1)
        else:
            raise NotImplementedError
        advantages = q_values - values
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        action_dist = self.actor(observations)
        action_log_prob = action_dist.log_prob(actions)
        advantage = self.compute_advantage(observations,actions,action_dist)
        loss = - (action_log_prob * (advantage / self.temperature).exp()).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
