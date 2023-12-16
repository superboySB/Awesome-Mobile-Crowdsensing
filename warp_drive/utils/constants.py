# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

# String names used for observations, actions, rewards and done flags
# Single source of truth


class Constants:
    """
    Constants for WarpDrive
    """

    OBSERVATIONS = "observations"
    ACTIONS = "sampled_actions"
    REWARDS = "rewards"
    GLOBAL_REWARDS = "global_rewards"
    DONE_FLAGS = "done_flags"
    PROCESSED_OBSERVATIONS = "processed_observations"
    ACTION_MASK = "action_mask"
    AGENT_ENERGY = "agent_energy"
    COVERAGE_METRIC_NAME = "target_coverage"
    DATA_METRIC_NAME = "collected_data_ratio"
    ENERGY_METRIC_NAME = "mean_energy_consumption"
    AOI_METRIC_NAME = "mean_aoi"
    MAIN_METRIC_NAME = "fresh_equivalent_coverage"
    STATE = "state"
