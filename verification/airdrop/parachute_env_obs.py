import logging
import gym
import torch
from gym.spaces import Discrete, MultiDiscrete, Box
from typing import Any, List
import numpy as np

UP, DOWN, LEFT, RIGHT, STOP = 0, 1, 2, 3, 4
logger = logging.getLogger(__name__)
ID = 'id'
POSITION = 'position'
CARRIED_AGENTS = 'carried_agents'
DEPLOYED = 'deployed'
GRID_SIZE = 20
EPISODE_LENGTH = 60
TARGETS = "big_agent_targets"
TARGET = "small_agent_target"
SURVEILLANCE_AOI = "surveillance_aoi"
EMERGENCY_AOI = "emergency_aoi"
BIG_AGENT_RANGE = 8
SMALL_AGENT_RANGE = 4
NUM_BIG_AGENTS = 2
NUM_SMALL_AGENTS = 4
NUM_MOVEMENTS = 5  # up, down, left, right, stop
NUM_CLUSTERS = 5
CLUSTER_RADIUS = 3
MAX_VALUE = 10


class MultiAgentGridWorld(gym.Env):
    """Please use the following members to generate reward functions"""
    device: torch.device
    big_agent_shape: tuple
    small_agent_shape: tuple
    grid_size: int
    group_factor: float
    self_factor: float
    num_big_agents: int
    num_emergencies: int
    log_action: bool
    num_small_agents: int
    carried_small_agents: int
    max_timesteps: int
    figure: Any
    seed: int
    big_vec_mode: bool
    small_vec_mode: bool
    big_agent_buffer_size: int
    coordinate_dim: int
    observation_space: dict
    big_action_space: MultiDiscrete
    small_action_space: Discrete
    action_space: dict
    emergency_poi_grid: np.ndarray  # shape: (grid_size, grid_size)
    poi_grid: np.ndarray  # shape: (grid_size, grid_size)
    emergency_positions: np.ndarray  # shape: (num_emergencies, 2)
    emer_ids: np.ndarray  # shape: (num_emergencies,)
    emergency_mapping: dict
    emergency_start_time: np.ndarray  # shape: (num_emergencies,)
    num_poi: int
    big_agents: List[dict]
    small_agents: List[dict]
    emergency_aoi: np.ndarray  # shape: (grid_size, grid_size)
    aoi_grid_by_time: np.ndarray  # shape: (max_timesteps, grid_size, grid_size)
    emergency_state_time: np.ndarray  # shape: (num_emergencies,)
    emergency_assign_status: np.ndarray  # shape: (num_emergencies,)
    timestep: int
    small_agent_targets: List[List[Any]]
    small_agent_trajectories: List[List[Any]]
    big_agent_trajectories: List[List[Any]]

    def _get_vec_observation(self, agent):
        if CARRIED_AGENTS in agent:
            # big agent, self-position, position of carried agents, and emergency (x,y) buffer.
            self_position = agent[POSITION] / self.grid_size
            carried_agents_pos = np.zeros((self.carried_small_agents * self.coordinate_dim))
            big_agent_id = agent[ID]
            my_small_agent = self.small_agents[big_agent_id * self.coordinate_dim:
                                               (big_agent_id + 1) * self.coordinate_dim]
            for i, small_agent in enumerate(my_small_agent):
                if small_agent[DEPLOYED]:
                    delta_pos = (small_agent[POSITION] - agent[POSITION]) / self.grid_size
                    carried_agents_pos[
                    i * self.coordinate_dim:(i + 1) * self.coordinate_dim
                    ] = delta_pos
                else:
                    carried_agents_pos[i * self.coordinate_dim:(i + 1) * self.coordinate_dim] = self_position
            target_obs = np.zeros((self.big_agent_buffer_size * self.coordinate_dim))
            for i, target in enumerate(agent[TARGETS][:self.big_agent_buffer_size]):
                target_obs[i * self.coordinate_dim:(i + 1) * self.coordinate_dim] = target / self.grid_size
            return np.concatenate([self_position, carried_agents_pos, target_obs])
        else:
            self_position = agent[POSITION] / self.grid_size
            if agent[TARGET] is not None:
                target_position = agent[TARGET] / self.grid_size
            else:
                target_position = np.zeros(self.coordinate_dim)
            return np.concatenate([self_position, target_position])

    def _get_grid_observation(self, agent, agents, observation_range):
        pass

    def _get_observation(self):
        """
        Combines observations for all agents. Small agents get vector observations,
        and big agents get grid observations.
        """
        observations = {}

        for big_agent_id, big_agent in enumerate(self.big_agents):
            if self.big_vec_mode:
                observations[f'big_{big_agent_id}'] = self._get_vec_observation(big_agent)
            else:
                observations[f'big_{big_agent_id}'] = self._get_grid_observation(big_agent, self.big_agents,
                                                                                 BIG_AGENT_RANGE)

        for small_agent_id, small_agent in enumerate(self.small_agents):
            if small_agent[DEPLOYED]:
                if self.small_vec_mode:
                    observations[f'small_{small_agent_id}'] = self._get_vec_observation(small_agent)
                else:
                    observations[f'small_{small_agent_id}'] = self._get_grid_observation(small_agent, self.small_agents,
                                                                                         SMALL_AGENT_RANGE)
            else:
                # Small agents not deployed will have masked observation.
                observations[f'small_{small_agent_id}'] = np.zeros(self.small_agent_shape)

        return observations
