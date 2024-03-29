# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
#

import torch
from torch import nn
from torch.distributions import Categorical

from warp_drive.training.param_scheduler import ParamScheduler

_EPSILON = 1e-10  # small number to prevent indeterminate division
from algs.utils import normalize_batch

class A2C:
    """
    The Advantage Actor-Critic Class
    https://arxiv.org/abs/1602.01783
    """

    def __init__(
        self,
        discount_factor_gamma=1.0,
        normalize_advantage=False,
        normalize_return=False,
        vf_loss_coeff=0.01,
        entropy_coeff=0.01,
    ):
        assert 0 <= discount_factor_gamma <= 1
        self.discount_factor_gamma = discount_factor_gamma
        self.normalize_advantage = normalize_advantage
        self.normalize_return = normalize_return
        # Create vf_loss and entropy coefficient schedules
        self.vf_loss_coeff_schedule = ParamScheduler(vf_loss_coeff)
        self.entropy_coeff_schedule = ParamScheduler(entropy_coeff)

    def compute_loss_and_metrics(
        self,
        timestep=None,
        actions_batch=None,
        rewards_batch=None,
        done_flags_batch=None,
        action_probabilities_batch=None,
        value_functions_batch=None,
        perform_logging=False,
    ):
        assert timestep is not None
        assert actions_batch is not None
        assert rewards_batch is not None
        assert done_flags_batch is not None
        assert action_probabilities_batch is not None
        assert value_functions_batch is not None

        # Detach value_functions_batch from the computation graph
        # for return and advantage computations.
        value_functions_batch_detached = value_functions_batch.detach()

        # Value objective.
        returns_batch = torch.zeros_like(rewards_batch)

        returns_batch[-1] = (
            done_flags_batch[-1][:, None] * rewards_batch[-1]
            + (1 - done_flags_batch[-1][:, None]) * value_functions_batch_detached[-1]
        )
        for step in range(-2, -returns_batch.shape[0] - 1, -1):
            future_return = (
                done_flags_batch[step][:, None] * torch.zeros_like(rewards_batch[step])
                + (1 - done_flags_batch[step][:, None])
                * self.discount_factor_gamma
                * returns_batch[step + 1]
            )
            returns_batch[step] = rewards_batch[step] + future_return

        # Normalize across the agents and env dimensions
        if self.normalize_return:
            normalized_returns_batch = (
                returns_batch - returns_batch.mean(dim=(1, 2), keepdim=True)
            ) / (returns_batch.std(dim=(1, 2), keepdim=True) + torch.tensor(_EPSILON))
        else:
            normalized_returns_batch = returns_batch

        vf_loss = nn.MSELoss()(normalized_returns_batch, value_functions_batch)

        # Policy objective
        advantages_batch = normalized_returns_batch - value_functions_batch_detached

        # Normalize across the agents and env dimensions
        normalize_batch(advantages_batch, self.normalize_advantage)

        log_prob = 0.0
        mean_entropy = 0.0
        for idx in range(actions_batch.shape[-1]):
            m = Categorical(action_probabilities_batch[idx])
            mean_entropy += m.entropy().mean()
            log_prob += m.log_prob(actions_batch[..., idx])

        policy_loss = (-log_prob * normalized_advantages_batch).mean()

        # Total loss
        vf_loss_coeff_t = self.vf_loss_coeff_schedule.get_param_value(timestep)
        entropy_coeff_t = self.entropy_coeff_schedule.get_param_value(timestep)
        loss = policy_loss + vf_loss_coeff_t * vf_loss - entropy_coeff_t * mean_entropy

        variance_explained = max(
            torch.tensor(-1.0),
            (
                1
                - (
                    normalized_advantages_batch.detach().var()
                    / (normalized_returns_batch.detach().var() + torch.tensor(_EPSILON))
                )
            ),
        )

        if perform_logging:
            metrics = {
                "VF loss coefficient": vf_loss_coeff_t,
                "Entropy coefficient": entropy_coeff_t,
                "Total loss": loss.item(),
                "Policy loss": policy_loss.item(),
                "Value function loss": vf_loss.item(),
                "Mean rewards": rewards_batch.mean().item(),
                "Max. rewards": rewards_batch.max().item(),
                "Min. rewards": rewards_batch.min().item(),
                "Mean value function": value_functions_batch.mean().item(),
                "Mean advantages": advantages_batch.mean().item(),
                "Mean (norm.) advantages": normalized_advantages_batch.mean().item(),
                "Mean (discounted) returns": returns_batch.mean().item(),
                "Mean normalized returns": normalized_returns_batch.mean().item(),
                "Mean entropy": mean_entropy.item(),
                "Variance explained by the value function": variance_explained.item(),
            }
            # mean of the standard deviation of sampled actions
            std_over_agent_per_action = (
                actions_batch.float().std(axis=2).mean(axis=(0, 1))
            )
            std_over_time_per_action = (
                actions_batch.float().std(axis=0).mean(axis=(0, 1))
            )
            std_over_env_per_action = (
                actions_batch.float().std(axis=1).mean(axis=(0, 1))
            )
            for idx, _ in enumerate(std_over_agent_per_action):
                std_action = {
                    f"Std. of action_{idx} over agents": std_over_agent_per_action[
                        idx
                    ].item(),
                    f"Std. of action_{idx} over envs": std_over_env_per_action[
                        idx
                    ].item(),
                    f"Std. of action_{idx} over time": std_over_time_per_action[
                        idx
                    ].item(),
                }
                metrics.update(std_action)
        else:
            metrics = {}
        return loss, metrics
