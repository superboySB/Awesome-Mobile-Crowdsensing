# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
#

import torch
from torch import nn
from torch.distributions import Categorical
from torch import Tensor
from algs.utils import normalize_batch, _EPSILON
from warp_drive.training.param_scheduler import ParamScheduler


class PPO:
    """
    The Proximal Policy Optimization Class
    https://arxiv.org/abs/1707.06347
    """

    def __init__(
            self,
            discount_factor_gamma=1.0,
            clip_param=0.2,
            normalize_advantage=False,
            normalize_return=False,
            vf_loss_coeff=0.01,
            entropy_coeff=0.01,
            lambda_gae=0.95,
            use_gae=False,
    ):
        assert 0 <= discount_factor_gamma <= 1
        self.discount_factor_gamma = discount_factor_gamma
        assert 0 <= clip_param <= 1
        self.clip_param = clip_param
        self.normalize_advantage = normalize_advantage
        self.normalize_return = normalize_return
        # Create vf_loss and entropy coefficient schedules
        self.vf_loss_coeff_schedule = ParamScheduler(vf_loss_coeff)
        self.entropy_coeff_schedule = ParamScheduler(entropy_coeff)
        self.lambda_gae = lambda_gae
        self.use_gae = use_gae

    def compute_loss_and_metrics(
            self,
            timestep=None,
            actions_batch: Tensor = None,
            rewards_batch: Tensor = None,
            done_flags_batch: Tensor = None,
            action_probabilities_batch: Tensor = None,
            value_functions_batch: Tensor = None,
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
        with torch.no_grad():
            returns_batch, advantages_batch, deltas_batch = (torch.zeros_like(rewards_batch),) * 3
            returns_batch[-1] = value_functions_batch_detached[-1]
            prev_advantage = torch.zeros_like(returns_batch[0])
            num_of_steps = returns_batch.shape[0]
            prev_value = value_functions_batch_detached[-1]
            for step in reversed(range(num_of_steps - 1)):
                if self.use_gae:
                    deltas_batch[step] = (rewards_batch[step] +
                                          self.discount_factor_gamma * (1 - done_flags_batch[step][:, None]) *
                                          prev_value - value_functions_batch_detached[
                                              step]
                                          )
                    advantages_batch[step] = deltas_batch[step] + self.discount_factor_gamma * self.lambda_gae * (
                            1 - done_flags_batch[step][:, None]) * prev_advantage
                    returns_batch[step] = value_functions_batch_detached[step] + advantages_batch[step]
                else:
                    if step == num_of_steps - 1:
                        continue
                    future_return = (
                            done_flags_batch[step][:, None] * torch.zeros_like(rewards_batch[step])
                            + (1 - done_flags_batch[step][:, None])
                            * self.discount_factor_gamma
                            * returns_batch[step + 1]
                    )
                    returns_batch[step] = rewards_batch[step] + future_return

        # Normalize across the agents and env dimensions
        normalized_returns_batch = normalize_batch(returns_batch, self.normalize_return)
        vf_loss = nn.MSELoss()(normalized_returns_batch, value_functions_batch)
        # Normalize advantages if required
        normalized_advantages_batch = normalize_batch(advantages_batch, self.normalize_advantage)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reference = actions_batch.squeeze(-1)
        log_prob = torch.zeros_like(reference, device=device, dtype=torch.float32)
        mean_entropy = torch.tensor(0.0, device=device)
        for idx in range(actions_batch.shape[-1]):
            m = Categorical(action_probabilities_batch[idx])
            mean_entropy += m.entropy().mean()
            log_prob += m.log_prob(actions_batch[..., idx])

        old_log_prob = log_prob.detach()
        ratio = torch.exp(log_prob - old_log_prob)

        surr1 = ratio * normalized_advantages_batch
        surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * normalized_advantages_batch
        )
        policy_surr = torch.minimum(surr1, surr2)
        policy_loss = -1.0 * policy_surr.mean()

        # Total loss
        vf_loss_coefficient_t = self.vf_loss_coeff_schedule.get_param_value(timestep)
        entropy_coefficient_t = self.entropy_coeff_schedule.get_param_value(timestep)
        loss = policy_loss + vf_loss_coefficient_t * vf_loss - entropy_coefficient_t * mean_entropy

        if perform_logging:
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
            metrics = {
                "VF loss coefficient": float(vf_loss_coefficient_t),
                "Entropy coefficient": float(entropy_coefficient_t),
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
