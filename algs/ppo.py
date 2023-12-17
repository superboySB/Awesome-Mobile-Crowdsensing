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
from algs.utils import normalize_batch, _EPSILON, shuffle_and_divide_tuple
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
            vf_loss_coefficient=0.01,
            entropy_coefficient=0.01,
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
        self.vf_loss_coeff_schedule = ParamScheduler(vf_loss_coefficient)
        self.entropy_coeff_schedule = ParamScheduler(entropy_coefficient)
        self.lambda_gae = lambda_gae
        self.use_gae = use_gae

    def compute_loss_and_metrics(
            self,
            timestep=None,
            observations_batch: Tensor = None,
            actions_batch: Tensor = None,
            rewards_batch: Tensor = None,
            done_flags_batch: Tensor = None,
            model: torch.nn.Module = None,
            perform_logging: bool = False,
            **kwargs
    ):
        assert timestep is not None
        assert actions_batch is not None
        assert rewards_batch is not None
        assert done_flags_batch is not None
        # assert action_probabilities_batch is not None
        # assert value_functions_batch is not None

        # Detach value_functions_batch from the computation graph
        # for return and advantage computations.
        # Policy evaluation for the entire batch
        l2_loss_func = nn.MSELoss()
        vf_loss_coeff_t = self.vf_loss_coeff_schedule.get_param_value(timestep)
        entropy_coeff_t = self.entropy_coeff_schedule.get_param_value(timestep)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, value_functions_batch = model(
            obs=observations_batch
        )
        value_functions_batch_detached = value_functions_batch.detach()

        # Value objective.
        with torch.no_grad():
            returns_batch, advantages_batch, deltas_batch = (torch.zeros_like(rewards_batch),) * 3
            returns_batch[-1] = (
                    done_flags_batch[-1][:, None] * rewards_batch[-1]
                    + (~ done_flags_batch[-1][:, None]) * value_functions_batch_detached[-1]
            )
            prev_advantage = torch.zeros_like(returns_batch[0])
            num_of_steps = returns_batch.shape[0]
            prev_value = value_functions_batch_detached[-1]
            for step in reversed(range(num_of_steps - 1)):
                if self.use_gae:
                    deltas_batch[step] = (rewards_batch[step] +
                                          self.discount_factor_gamma * (~ done_flags_batch[step][:, None]) *
                                          prev_value - value_functions_batch_detached[
                                              step]
                                          )
                    advantages_batch[step] = deltas_batch[step] + self.discount_factor_gamma * self.lambda_gae * (
                        ~ done_flags_batch[step][:, None]) * prev_advantage
                    returns_batch[step] = value_functions_batch_detached[step] + advantages_batch[step]
                else:
                    if step == num_of_steps - 1:
                        continue
                    future_return = (
                            (~ done_flags_batch[step][:, None])
                            * self.discount_factor_gamma
                            * returns_batch[step + 1]
                    )
                    returns_batch[step] = rewards_batch[step] + future_return
        # send model into this function, and input mini-batch randomized observations
        batch_to_update = (observations_batch,
                           actions_batch,
                           value_functions_batch,
                           returns_batch,
                           advantages_batch
                           )
        if kwargs['num_mini_batches'] is not None:
            num_mini_batches = kwargs['num_mini_batches']
            divided_data = shuffle_and_divide_tuple(batch_to_update, num_mini_batches)
        else:
            divided_data = [batch_to_update]
        batch_loss, batch_policy_loss, batch_vf_loss, batch_mean_entropy = (torch.tensor(0.0, device=device),) * 4
        for mini_batch in divided_data:
            (observations_batch, actions_batch, value_functions_batch,
             returns_batch, advantages_batch) = mini_batch
            action_probabilities_batch, new_value_functions_batch = model(observations_batch)
            log_prob = torch.zeros_like(actions_batch.squeeze(-1), device=device, dtype=torch.float32)
            mean_entropy = torch.tensor(0.0, device=device)
            for idx in range(actions_batch.shape[-1]):
                m = Categorical(action_probabilities_batch[idx])
                mean_entropy += m.entropy().mean()
                log_prob += m.log_prob(actions_batch[..., idx])

            old_log_prob = log_prob.detach()
            ratio = torch.exp(log_prob - old_log_prob)
            normalized_advantages_batch = normalize_batch(advantages_batch, self.normalize_advantage)
            surr1 = ratio * normalized_advantages_batch
            surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * normalized_advantages_batch
            )
            policy_surr = torch.minimum(surr1, surr2)
            policy_loss = -1.0 * policy_surr.mean()

            # Clip the value to reduce variability during Critic training
            new_value_functions_batch_clipped = (new_value_functions_batch +
                                                 torch.clamp(new_value_functions_batch,
                                                             -self.clip_param,
                                                             self.clip_param))
            returns_batch = normalize_batch(returns_batch, self.normalize_return)
            vf_loss = l2_loss_func(returns_batch, new_value_functions_batch)
            vf_loss_clipped = l2_loss_func(returns_batch, new_value_functions_batch_clipped)
            vf_loss = 0.5 * torch.max(vf_loss, vf_loss_clipped).mean()
            # Total loss
            loss = policy_loss + vf_loss_coeff_t * vf_loss - entropy_coeff_t * mean_entropy
            batch_loss += loss.mean()
            batch_vf_loss += vf_loss.detach()
            batch_policy_loss += policy_loss.detach()
            batch_mean_entropy += mean_entropy.detach()

        if perform_logging:
            variance_explained = max(
                torch.tensor(-1.0),
                (
                        1
                        - (
                                advantages_batch.detach().var()
                                / (returns_batch.detach().var() + torch.tensor(_EPSILON))
                        )
                ),
            )
            metrics = {
                "VF loss coefficient": float(vf_loss_coeff_t),
                "Entropy coefficient": float(entropy_coeff_t),
                "Total loss": batch_loss.item(),
                "Policy loss": batch_policy_loss.item(),
                "Value function loss": batch_vf_loss.item(),
                "Mean rewards": rewards_batch.mean().item(),
                "Max. rewards": rewards_batch.max().item(),
                "Min. rewards": rewards_batch.min().item(),
                "Mean value function": value_functions_batch.mean().item(),
                "Mean advantages": advantages_batch.mean().item(),
                "Mean (norm.) advantages": advantages_batch.mean().item(),
                "Mean (discounted) returns": returns_batch.mean().item(),
                "Mean normalized returns": returns_batch.mean().item(),
                "Mean entropy": batch_mean_entropy.item(),
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
        return batch_loss, metrics
