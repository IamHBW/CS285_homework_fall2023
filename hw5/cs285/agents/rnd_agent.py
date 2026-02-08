import torch
from torch import nn
import numpy as np

from typing import Callable, List, Tuple

from cs285.agents.dqn_agent import DQNAgent
import cs285.infrastructure.pytorch_util as ptu

def init_network(model):
    if isinstance(model, nn.Linear):
        model.weight.data.normal_()
        model.bias.data.normal_()

class RNDAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        num_actions: int,
        make_rnd_network: Callable[[Tuple[int, ...]], nn.Module],
        make_rnd_network_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        make_target_rnd_network: Callable[[Tuple[int, ...]], nn.Module],
        rnd_weight: float,
        rnd_dim: int,
        rnd_alpha: float,
        num_exploration_steps: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, make_critic=make_critic,make_optimizer=make_optimizer,make_lr_schedule=make_lr_schedule,**kwargs
        )
        self.rnd_weight = rnd_weight
        self.num_exploration_steps = num_exploration_steps

        self.rnd_net = make_rnd_network(observation_shape)
        self.rnd_target_net = make_target_rnd_network(observation_shape)
        self.rnd_dim = rnd_dim
        self.rnd_alpha = rnd_alpha

        self.rnd_target_net.apply(init_network)

        # Freeze target network
        for p in self.rnd_target_net.parameters():
            p.requires_grad_(False)

        self.rnd_optimizer = make_rnd_network_optimizer(
            self.rnd_net.parameters()
        )
        self.exploration_critic = make_critic(observation_shape, num_actions)
        self.exploration_critic_optimizer = make_optimizer(self.exploration_critic.parameters())

        self.exploration_target_critic = make_critic(observation_shape, num_actions)
        self.update_exploration_target_critic()

        self.exploration_lr_scheduler = make_lr_schedule(self.exploration_critic_optimizer)

        self.register_buffer(
            "rnd_error_mean", torch.zeros(self.rnd_dim, device=ptu.device)
        )
        self.register_buffer(
            "rnd_error_std", torch.ones(self.rnd_dim, device=ptu.device)
        )

    def update_rnd(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Update the RND network using the observations.
        """
        # TODO(student): update the RND network
        rnd_pred = self.rnd_net(obs)
        rnd_target = self.rnd_target_net(obs)
        assert rnd_target.shape == rnd_pred.shape
        loss = nn.functional.mse_loss(rnd_pred,rnd_target)

        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()

        return loss.item()

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        is_truncated: torch.Tensor,
        step: int,
    ):
        with torch.no_grad():
            # TODO(student): Compute RND bonus for batch and modify rewards
            rnd_predictions = self.rnd_net(observations)
            rnd_targets = self.rnd_target_net(observations)
            # Compute per-sample MSE (reduction='none')
            assert rnd_predictions.shape == rnd_targets.shape
            rnd_error = nn.functional.mse_loss(rnd_predictions, rnd_targets, reduction='none') #(batch_size,rnd_dim)
            self.update_rnd_statictics(rnd_error)
            rnd_error = (rnd_error - self.rnd_error_mean) / (self.rnd_error_std + 1e-8)
            rnd_error = rnd_error.mean(dim=-1)
            assert rnd_error.shape == rewards.shape
            rewards = rewards + self.rnd_weight * rnd_error

        metrics = super().update(
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            is_truncated,
            step,
        )

        # Update the RND network.
        rnd_loss = self.update_rnd(observations)
        metrics["rnd_loss"] = rnd_loss

        metrics.update(
            self.update_exploration_critic(
                observations,
                actions,
                rnd_error,
                next_observations,
                dones,
                is_truncated,
            )
        )

        if step % self.target_update_period == 0:
            self.update_exploration_target_critic()

        return metrics
    
    def get_action(self, observation: np.ndarray, steps: int, epsilon: float = 0.0 ) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # Use exploration critic during first num_exploration_steps
        if steps < self.num_exploration_steps:
            critic = self.exploration_critic
        else:
            critic = self.critic
            
        if np.random.rand() > epsilon:
            qa_values = critic(observation)
            action = torch.argmax(qa_values,dim=1)
        else:
            action = torch.randint(high=self.num_actions,size=(1,)) 

        return ptu.to_numpy(action).squeeze(0).item()
    
    def update_exploration_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        rnd_error: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        is_truncated: torch.Tensor,
    ) -> dict:
        """Update the exploration critic, and return stats for logging."""
        loss, metrics, _ = self.compute_exploration_critic_loss(
            obs, action, rnd_error, next_obs, done, is_truncated
        )
        
        self.exploration_critic_optimizer.zero_grad()
        loss.backward()
        exploration_grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.exploration_critic.parameters(), self.clip_grad_norm or float("inf")
        )
        metrics["exploration_grad_norm"] = exploration_grad_norm.item()
        self.exploration_critic_optimizer.step()
        self.exploration_lr_scheduler.step()

        return metrics


    def compute_exploration_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        is_truncated: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Compute the loss for the exploration critic.

        Returns:
         - loss: torch.Tensor, the MSE loss for the critic
         - metrics: dict, a dictionary of metrics to log
         - variables: dict, a dictionary of variables that can be used in subsequent calculations
        """
        (batch_size,) = reward.shape
        with torch.no_grad():
            next_qa_values = self.exploration_target_critic(next_obs.view(batch_size,-1))

            next_qa_values_online = self.exploration_critic(next_obs.view(batch_size,-1))
            next_action = torch.argmax(next_qa_values_online,dim=-1)
            
            reward = reward.unsqueeze(-1)
            next_q_values = torch.gather(next_qa_values,-1,next_action.unsqueeze(-1))
            terminal = done.float() * (1 - is_truncated.float())
            bootstrap = 1 - terminal
            next_q_values = next_q_values * bootstrap.unsqueeze(-1)
            next_q_values = next_q_values.squeeze(dim=-1)
            assert next_q_values.shape == (batch_size,), next_q_values.shape

            target_values = reward.squeeze(dim=-1) + self.discount * next_q_values
            assert target_values.shape == (batch_size,), target_values.shape
        
        qa_values = self.exploration_critic(obs.view(batch_size,-1))
        q_values = torch.gather(qa_values,-1,action.unsqueeze(-1)).squeeze(dim=-1)
        loss = self.critic_loss(q_values,target_values)

        return (
            loss,
            {
                "exploration_critic_loss": loss.item(),
                "exploration_q_values": q_values.mean().item(),
                "exploration_target_values": target_values.mean().item(),
            },
            {
                "exploration_qa_values": qa_values,
                "exploration_q_values": q_values,
            },
        )
    
    def update_rnd_statictics(self,rnd_error: torch.Tensor):
        batch_mean = rnd_error.mean(dim=0)
        batch_std = rnd_error.std(dim=0)

        self.rnd_error_mean = (1 - self.rnd_alpha) * self.rnd_error_mean + self.rnd_alpha * batch_mean
        self.rnd_error_std = (1 - self.rnd_alpha) * self.rnd_error_std + self.rnd_alpha * batch_std

    def update_exploration_target_critic(self):
        self.exploration_target_critic.load_state_dict(self.exploration_critic.state_dict())
    
    def num_aux_plots(self) -> int:
        return 1
    
    def plot_aux(
        self,
        axes: List,
    ) -> dict:
        """
        Plot the RND prediction error for the observations.
        """
        import matplotlib.pyplot as plt
        assert len(axes) == 1
        ax: plt.Axes = axes[0]

        with torch.no_grad():
            # Assume a state space of [0, 1] x [0, 1]
            x = torch.linspace(0, 1, 100)
            y = torch.linspace(0, 1, 100)
            xx, yy = torch.meshgrid(x, y)

            inputs = ptu.from_numpy(np.stack([xx.flatten(), yy.flatten()], axis=1))
            targets = self.rnd_target_net(inputs)
            predictions = self.rnd_net(inputs)

            errors = torch.norm(predictions - targets, dim=-1)
            errors = torch.reshape(errors, xx.shape)

            # Log scale, aligned with normal axes
            from matplotlib import cm
            ax.imshow(ptu.to_numpy(errors).T, extent=[0, 1, 0, 1], origin="lower", cmap="hot")
            plt.colorbar(ax.images[0], ax=ax)
