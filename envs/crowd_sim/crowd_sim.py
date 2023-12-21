import logging
import os
from typing import Optional, Tuple, Dict, List, Union, Any, Union

import numpy as np
import torch
import folium
import wandb
from copy import deepcopy
import time
import pandas as pd
from folium.plugins import TimestampedGeoJson, AntPath
from gym import spaces
from ray.rllib.env import GroupAgentsWrapper

from run_configs.mcs_configs_python import PROJECT_NAME
from warp_drive.utils.common import get_project_root
from warp_drive.utils.env_registrar import EnvironmentRegistrar
from ray.rllib.utils.typing import MultiAgentDict, EnvActionType, EnvObsType, EnvInfoDict
from shapely.geometry import Point
from pytorch_lightning import seed_everything
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.base_env import dummy_group_id
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from tabulate import tabulate

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext
from warp_drive.utils.recursive_obs_dict_to_spaces_dict import (
    BIG_NUMBER
)
from envs.crowd_sim.env_wrapper import CUDAEnvWrapper
from warp_drive.training.data_loader import create_and_push_data_placeholders
from .utils import *

grid_size = 10

COVERAGE_METRIC_NAME = Constants.COVERAGE_METRIC_NAME
DATA_METRIC_NAME = Constants.DATA_METRIC_NAME
ENERGY_METRIC_NAME = Constants.ENERGY_METRIC_NAME
MAIN_METRIC_NAME = Constants.MAIN_METRIC_NAME
AOI_METRIC_NAME = Constants.AOI_METRIC_NAME
_AGENT_ENERGY = Constants.AGENT_ENERGY
_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_GLOBAL_REWARD = Constants.GLOBAL_REWARDS
_STATE = Constants.STATE
teams_name = ("cars_", "drones_")
policy_mapping_dict = {
    "SanFrancisco": {
        "description": "two team cooperate to collect data",
        "team_prefix": teams_name,
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}
excluded_keys = {"trainer", "env_params", "map_name"}
logging.getLogger().setLevel(logging.WARN)


class Pair:
    """
    A simple class to store a pair of values, used for index and distance
    """
    distance: float
    index: int

    def __init__(self):
        self.distance = 0.0
        self.index = 0


def get_theta(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    # 使用 arctan2 计算角度
    angles = np.arctan2(dy, dx)

    # 将角度转换为 0 到 2π 之间
    angles = np.mod(angles + 2 * np.pi, 2 * np.pi)
    return angles


class CrowdSim:
    """
    The Mobile Crowdsensing Environment
    """

    name = "CrowdSim"

    def __init__(
            self,
            num_drones=2,
            num_cars=2,
            num_agents_observed=5,
            seed=None,
            env_backend="cpu",
            dynamic_zero_shot=False,
            env_config=None,
            centralized=True
    ):
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        self.bool_dtype = np.bool_
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)
        self.config = env_config()
        # Seeding
        self.np_random: np.random = np.random
        if seed is not None:
            self.seed(seed)

        self.centralized = centralized
        self.num_drones = num_drones
        self.num_cars = num_cars
        self.num_agents = self.num_drones + self.num_cars
        self.num_sensing_targets = self.config.env.human_num
        self.aoi_threshold = self.config.env.aoi_threshold
        self.num_agents_observed = num_agents_observed

        self.episode_length = self.config.env.num_timestep
        self.step_time = self.config.env.step_time
        self.start_timestamp = self.config.env.start_timestamp
        self.end_timestamp = self.config.env.end_timestamp

        self.nlon = self.config.env.nlon
        self.nlat = self.config.env.nlat
        self.lower_left = self.config.env.lower_left
        self.upper_right = self.config.env.upper_right
        from movingpandas.geometry_utils import measure_distance_geodesic
        self.max_distance_x: float = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                               Point(self.upper_right[0], self.lower_left[1]))
        self.max_distance_y: float = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                               Point(self.lower_left[0], self.upper_right[1]))
        self.human_df = pd.read_csv(self.config.env.dataset_dir)
        logging.info("Finished reading {} rows".format(len(self.human_df)))
        self.human_df['t'] = pd.to_datetime(self.human_df['timestamp'], unit='s')  # s表示时间戳转换
        self.human_df['aoi'] = -1  # 加入aoi记录aoi
        self.human_df['energy'] = -1  # 加入energy记录energy
        self.agent_speed = {'car': self.config.env.car_velocity, 'drone': self.config.env.drone_velocity}
        points_x, points_y, num_centers, num_points, num_points_per_center = (None,) * 5
        if dynamic_zero_shot:
            num_centers = int(self.num_sensing_targets * 0.05)
            num_points_per_center = 3
            max_distance_from_center = 10
            int_arrays = [
                self.np_random.randint(0, int(self.max_distance_x), (num_centers, 1)).astype(np.int_),
                self.np_random.randint(0, int(self.max_distance_y), (num_centers, 1)).astype(np.int_)
            ]
            centers = np.concatenate(int_arrays, axis=1)
            points_x = np.zeros((num_centers * num_points_per_center,), dtype=int)
            points_y = np.zeros((num_centers * num_points_per_center,), dtype=int)
            for i, (cx, cy) in enumerate(centers):
                for j in range(num_points_per_center):
                    index = i * num_points_per_center + j
                    points_x[index] = self.np_random.randint(max(cx - max_distance_from_center, 0),
                                                             min(cx + max_distance_from_center + 1,
                                                                 int(self.max_distance_x)))
                    points_y[index] = self.np_random.randint(max(cy - max_distance_from_center, 0),
                                                             min(cy + max_distance_from_center + 1,
                                                                 int(self.max_distance_y)))
            self.num_sensing_targets += (num_centers * num_points_per_center)
        # human infos
        unique_ids = np.arange(0, self.num_sensing_targets)  # id from 0 to 91
        unique_timestamps = np.arange(self.start_timestamp, self.end_timestamp + self.step_time, self.step_time)
        id_to_index = {my_id: index for index, my_id in enumerate(unique_ids)}
        timestamp_to_index = {timestamp: index for index, timestamp in enumerate(unique_timestamps)}
        self.target_x_time_list = np.zeros([self.episode_length + 1, self.num_sensing_targets], dtype=np.int_)
        self.target_y_time_list = np.zeros([self.episode_length + 1, self.num_sensing_targets], dtype=np.int_)
        self.target_aoi_timelist = np.ones([self.episode_length + 1, self.num_sensing_targets], dtype=np.int_)
        self.target_coveraged_timelist = np.zeros([self.episode_length + 1, self.num_sensing_targets], dtype=np.bool_)

        # Fill the new array with data from the full DataFrame
        for _, row in self.human_df.iterrows():
            id_index = id_to_index.get(row['id'], None)
            timestamp_index = timestamp_to_index.get(row['timestamp'], None)
            if not (id_index is None or timestamp_index is None):
                self.target_x_time_list[timestamp_index, id_index] = row['x']
                self.target_y_time_list[timestamp_index, id_index] = row['y']
            else:
                raise ValueError("Got invalid rows:", row)
        if dynamic_zero_shot:
            self.target_x_time_list[:, self.num_sensing_targets - num_centers * num_points_per_center:] = points_x
            self.target_y_time_list[:, self.num_sensing_targets - num_centers * num_points_per_center:] = points_y
            # rebuild DataFrame from longitude and latitude

        x1 = self.target_x_time_list[:-1, :]
        y1 = self.target_y_time_list[:-1, :]
        x2 = self.target_x_time_list[1:, :]
        y2 = self.target_y_time_list[1:, :]
        self.target_theta_timelist = get_theta(x1, y1, x2, y2)
        self.target_theta_timelist = self.float_dtype(
            np.vstack([self.target_theta_timelist, self.target_theta_timelist[-1, :]]))

        # Check if there are any NaN values in the array
        assert not np.isnan(self.target_x_time_list).any()
        assert not np.isnan(self.target_y_time_list).any()
        assert not np.isnan(self.target_theta_timelist).any()

        # agent infos
        self.timestep = 0
        self.starting_location_x = self.nlon / 2
        self.starting_location_y = self.nlat / 2
        self.max_uav_energy = self.config.env.max_uav_energy
        self.agent_energy_timelist = np.full([self.episode_length + 1, self.num_agents],
                                             fill_value=self.max_uav_energy, dtype=self.float_dtype)
        self.agent_x_time_list = np.full([self.episode_length + 1, self.num_agents],
                                         fill_value=self.starting_location_x, dtype=self.float_dtype)
        self.agent_y_time_list = np.full([self.episode_length + 1, self.num_agents],
                                         fill_value=self.starting_location_y, dtype=self.float_dtype)
        self.data_collection = 0
        # Types and Status of vehicles
        self.agent_types = self.int_dtype(np.ones([self.num_agents, ]))
        self.cars = {}
        self.drones = {}
        for agent_id in range(self.num_agents):
            if agent_id < self.num_cars:
                self.agent_types[agent_id] = 0  # Car
                self.cars[agent_id] = True
            else:
                self.agent_types[agent_id] = 1  # Drone
                self.drones[agent_id] = True

        # These will be set during reset (see below)
        self.timestep = None

        # Defining observation and action spaces
        # obs = self type(1) + energy (1) + (num_agents - 1) * (homo_pos(2) + hetero_pos(2)) +
        # neighbor_aoi_grids (10 * 10) = 122
        self.observation_space = None  # Note: this will be set via the env_wrapper
        # state = (type,energy,x,y) * self.num_agents + neighbor_aoi_grids (10 * 10)
        self.state_dim = self.num_agents * 4 + 100
        self.global_state = np.zeros((self.state_dim,), dtype=self.float_dtype)
        self.drone_action_space_dx = self.float_dtype(self.config.env.drone_action_space[:, 0])
        self.drone_action_space_dy = self.float_dtype(self.config.env.drone_action_space[:, 1])
        self.car_action_space_dx = self.float_dtype(self.config.env.car_action_space[:, 0])
        self.car_action_space_dy = self.float_dtype(self.config.env.car_action_space[:, 1])
        self.action_space = spaces.Dict({
            agent_id: spaces.Discrete(len(self.drone_action_space_dx))
            if self.agent_types[agent_id] == 1
            else spaces.Discrete(len(self.car_action_space_dx))
            for agent_id in range(self.num_agents)
        })
        # Used in generate_observation()
        # When use_full_observation is True, then all the agents will have info of
        # all the other agents, otherwise, each agent will only have info of
        # its k-nearest agents (k = num_other_agents_observed)
        self.init_obs = None  # Will be set later in generate_observation()

        # Distance margin between agents for non-zero rewards
        # If a tagger is closer than this to a runner, the tagger
        # gets a positive reward, and the runner a negative reward
        self.drone_sensing_range = self.float_dtype(self.config.env.drone_sensing_range)
        self.car_sensing_range = self.float_dtype(self.config.env.car_sensing_range)
        self.drone_car_comm_range = self.float_dtype(self.config.env.drone_car_comm_range)

        self.global_distance_matrix = np.full((self.num_agents, self.num_agents + self.num_sensing_targets), np.nan)

        # Rewards and penalties
        self.energy_factor = self.config.env.energy_factor

        # These will also be set via the env_wrapper
        self.env_backend = env_backend

        # [may not necessary] Copy drones dict for applying at reset (with limited energy reserve)
        # self.drones_at_reset = copy.deepcopy(self.drones)

    def seed(self, seed=None):
        """
        Seeding the environment with a desired seed
        Note: this uses the code in
        https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        """
        self.np_random.seed(seed)
        return [seed]

    def history_reset(self):
        # Reset time to the beginning
        self.timestep = 0
        # Re-initialize the global state
        # for agent_id in range(self.num_agents):
        self.agent_x_time_list[self.timestep, :] = self.starting_location_x
        self.agent_y_time_list[self.timestep, :] = self.starting_location_y
        self.agent_energy_timelist[self.timestep, :] = self.max_uav_energy
        # for target_id in range(self.num_sensing_targets):
        self.target_aoi_timelist = np.zeros([self.episode_length + 1, self.num_sensing_targets], dtype=np.int_)
        self.target_aoi_timelist[self.timestep, :] = 1
        # reset global distance matrix
        self.calculate_global_distance_matrix()
        # for logging
        self.data_collection = 0
        # print("Reset target coverage timelist")
        self.target_coveraged_timelist = np.zeros([self.episode_length + 1, self.num_sensing_targets], dtype=np.bool_)

    def reset(self):
        """
        Env reset().
        """
        self.history_reset()
        return self.generate_observation_and_update_state()

    def calculate_global_distance_matrix(self):
        """
        Calculate the global distance matrix for all agents and targets prior to environment running.
        """
        # Vectorized computation for distances between agents
        agent_x_diff = self.agent_x_time_list[self.timestep, np.newaxis, :] - \
                       self.agent_x_time_list[self.timestep, :, np.newaxis]
        agent_y_diff = self.agent_y_time_list[self.timestep, np.newaxis, :] - \
                       self.agent_y_time_list[self.timestep, :, np.newaxis]
        agent_distances = np.sqrt(agent_x_diff ** 2 + agent_y_diff ** 2)
        np.fill_diagonal(agent_distances, -1)  # Avoid division by zero for same agents

        # Vectorized computation for distances between agents and targets
        target_x_diff = self.agent_x_time_list[self.timestep, :, np.newaxis] - \
                        self.target_x_time_list[self.timestep, np.newaxis, :]
        target_y_diff = self.agent_y_time_list[self.timestep, :, np.newaxis] - \
                        self.target_y_time_list[self.timestep, np.newaxis, :]
        target_distances = np.sqrt(target_x_diff ** 2 + target_y_diff ** 2)

        # Combine agent and target distances
        self.global_distance_matrix = np.block([
            [agent_distances, target_distances],
            [target_distances.T, np.zeros((self.num_sensing_targets, self.num_sensing_targets))]
        ])

        return self.global_distance_matrix

    def get_relative_positions(self, agent_id, other_agent_ids):
        """
        Calculate the relative positions of other agents to the specified agent.

        Args:
        - agent_id: The ID of the agent relative to whom positions are calculated.
        - other_agent_ids: Array of other agent IDs.

        Returns:
        - A 2D NumPy array containing the relative positions.
        """

        # Positions of the specified agent
        agent_x = self.agent_x_time_list[self.timestep, agent_id]
        agent_y = self.agent_y_time_list[self.timestep, agent_id]

        # Positions of other agents
        other_agents_x = self.agent_x_time_list[self.timestep, other_agent_ids]
        other_agents_y = self.agent_y_time_list[self.timestep, other_agent_ids]

        # Calculate relative positions
        relative_positions_x = (other_agents_x - agent_x) / self.nlon
        relative_positions_y = (other_agents_y - agent_y) / self.nlat

        # Stack the relative positions
        relative_positions = np.stack((relative_positions_x, relative_positions_y), axis=-1)

        return relative_positions

    def generate_observation_and_update_state(self) -> Dict[int, np.ndarray]:
        # Generate agent nearest targets IDs
        agent_nearest_targets_ids = np.argsort(self.global_distance_matrix[:, :self.num_agents], axis=-1, kind='stable')

        # Self info for all agents (num_agents, 2)
        self_parts = np.vstack((self.agent_types,
                                self.agent_energy_timelist[self.timestep] / self.max_uav_energy)).T

        # Initialize homoge_parts and hetero_parts (num_agents, num_agents_observed, 2)
        homoge_parts = np.zeros((self.num_agents, self.num_agents_observed, 2), dtype=self.float_dtype)
        hetero_parts = np.zeros((self.num_agents, self.num_agents_observed, 2), dtype=self.float_dtype)

        # Calculate homoge_parts and hetero_parts
        for agent_id in range(self.num_agents):
            nearest_targets = agent_nearest_targets_ids[agent_id, 1:]
            homoge_agents = nearest_targets[self.agent_types[nearest_targets] == self.agent_types[agent_id]]
            hetero_agents = nearest_targets[self.agent_types[nearest_targets] != self.agent_types[agent_id]]
            homoge_parts[agent_id, :len(homoge_agents), :] = self.get_relative_positions(agent_id, homoge_agents)
            hetero_parts[agent_id, :len(hetero_agents), :] = self.get_relative_positions(agent_id, hetero_agents)

        # Generate AoI grid parts for each agent
        aoi_grid_parts = self.generate_aoi_grid(self.agent_x_time_list[self.timestep, :],
                                                self.agent_y_time_list[self.timestep, :],
                                                self.drone_car_comm_range * 2,
                                                self.drone_car_comm_range * 2,
                                                grid_size)

        # Generate global state AoI grid
        state_aoi_grid = self.generate_aoi_grid(int(self.max_distance_x // 2),
                                                int(self.max_distance_y // 2),
                                                self.drone_car_comm_range * 4,
                                                self.drone_car_comm_range * 4,
                                                grid_size)

        # Merge parts for observations
        observations = self.float_dtype(np.hstack((self_parts, homoge_parts.reshape(self.num_agents, -1),
                                                   hetero_parts.reshape(self.num_agents, -1),
                                                   aoi_grid_parts.reshape(self.num_agents, -1))))

        agents_state = np.hstack([self_parts, self.agent_x_time_list[self.timestep, :].reshape(-1, 1) / self.nlon,
                                  self.agent_y_time_list[self.timestep, :].reshape(-1, 1) / self.nlat])
        # Global state
        self.global_state = self.float_dtype(np.concatenate([agents_state.ravel(), state_aoi_grid.ravel()]))
        observations = {agent_id: observations[agent_id] for agent_id in range(self.num_agents)}
        return observations

    def generate_aoi_grid(self, grid_centers_x: Union[np.ndarray, int, float],
                          grid_centers_y: Union[np.ndarray, int, float],
                          sensing_range_x: int, sensing_range_y: int, grid_size: int) -> np.ndarray:
        """
        Generate AoI grid parts for each agent. (Grid_centers can be vectored or scalar)
        """
        # AOI grid for all agents
        if isinstance(grid_centers_x, (float, int)):
            grid_centers_x = np.array([grid_centers_x], dtype=self.float_dtype)
        if isinstance(grid_centers_y, (float, int)):
            grid_centers_y = np.array([grid_centers_y], dtype=self.float_dtype)
        grid_centers = np.stack((grid_centers_x, grid_centers_y), axis=-1)
        # Define grid min and max for each axis
        shifts = np.array([sensing_range_x // 2, sensing_range_y // 2], dtype=self.float_dtype)
        grid_min = grid_centers - shifts
        grid_max = grid_centers + shifts
        # Target positions
        point_xy = np.stack((self.target_x_time_list[self.timestep], self.target_y_time_list[self.timestep]), axis=-1)
        # Discretize target positions
        discrete_points_xy = np.floor((point_xy[None, :, :] - grid_min[:, None, :]) / (
                grid_max[:, None, :] - grid_min[:, None, :]) * grid_size).astype(int)
        # Initialize AoI grid parts
        num_agents = grid_centers.shape[0]
        aoi_grid_parts = np.zeros((num_agents, grid_size, grid_size), dtype=self.float_dtype)
        grid_point_count = np.zeros((num_agents, grid_size, grid_size), dtype=int)

        # Iterate over each agent
        for agent_id in range(num_agents):
            # Create a boolean mask for valid points for this agent
            valid_mask = (discrete_points_xy[agent_id, :, 0] >= 0) & (discrete_points_xy[agent_id, :, 0] < grid_size) & \
                         (discrete_points_xy[agent_id, :, 1] >= 0) & (discrete_points_xy[agent_id, :, 1] < grid_size)
            # Filter the points and AoI values using the mask
            filtered_points = discrete_points_xy[agent_id, valid_mask]
            filtered_aoi_values = self.target_aoi_timelist[self.timestep, valid_mask]
            # Accumulate counts and AoI values for this agent
            np.add.at(aoi_grid_parts[agent_id], (filtered_points[:, 0], filtered_points[:, 1]), filtered_aoi_values)
            np.add.at(grid_point_count[agent_id], (filtered_points[:, 0], filtered_points[:, 1]), 1)

        # Normalize AoI values
        grid_point_count_nonzero = np.where(grid_point_count > 0, grid_point_count, 1)
        aoi_grid_parts = aoi_grid_parts / grid_point_count_nonzero / self.episode_length
        aoi_grid_parts[grid_point_count == 0] = 0
        return aoi_grid_parts

    def calculate_energy_consume(self, move_time, agent_id):
        stop_time = self.step_time - move_time
        if agent_id in self.cars:
            idle_cost = 17.49
            energy_factor = 7.4
            return (idle_cost + energy_factor) * self.agent_speed['car'] * move_time + idle_cost * stop_time
        elif agent_id in self.drones:
            # configs
            Pu = 0.5  # the average transmitted power of each user, W,  e.g. mobile phone
            P0 = 79.8563  # blade profile power, W
            P1 = 88.6279  # derived power, W
            U_tips = 120  # tip speed of the rotor blade of the UAV,m/s
            v0 = 4.03  # the mean rotor induced velocity in the hovering state,m/s
            d0 = 0.6  # fuselage drag ratio
            rho = 1.225  # density of air,kg/m^3
            s0 = 0.05  # the rotor solidity
            A = 0.503  # the area of the rotor disk, m^2
            vt = self.config.env.drone_velocity  # velocity of the UAV, m/s

            flying_energy = P0 * (1 + 3 * vt ** 2 / U_tips ** 2) + \
                            P1 * np.sqrt((np.sqrt(1 + vt ** 4 / (4 * v0 ** 4)) - vt ** 2 / (2 * v0 ** 2))) + \
                            0.5 * d0 * rho * s0 * A * vt ** 3

            hovering_energy = P0 + P1
            return move_time * flying_energy + stop_time * hovering_energy
        else:
            raise NotImplementedError("Energy model not supported for the agent.")

    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        self.timestep += 1
        assert isinstance(actions, dict)
        assert len(actions) == self.num_agents
        over_range = False
        for agent_id in range(self.num_agents):

            # is_stopping = True if actions[agent_id] == 0 else False
            if self.agent_types[agent_id] == 0:
                dx, dy = self.car_action_space_dx[actions[agent_id]], self.car_action_space_dy[actions[agent_id]]
            else:
                dx, dy = self.drone_action_space_dx[actions[agent_id]], self.drone_action_space_dy[actions[agent_id]]

            new_x = self.agent_x_time_list[self.timestep - 1, agent_id] + dx
            new_y = self.agent_y_time_list[self.timestep - 1, agent_id] + dy
            if new_x <= self.max_distance_x and new_y <= self.max_distance_y:
                self.agent_x_time_list[self.timestep, agent_id] = new_x
                self.agent_y_time_list[self.timestep, agent_id] = new_y
                # calculate distance between last time step and current time step
                distance = np.sqrt((new_x - self.agent_x_time_list[self.timestep - 1, agent_id]) ** 2
                                   + (new_y - self.agent_y_time_list[self.timestep - 1, agent_id]) ** 2)
                move_time = distance / (
                    self.agent_speed['car'] if self.agent_types[agent_id] == 0 else self.agent_speed['drone'])
                consume_energy = self.calculate_energy_consume(move_time, agent_id)
            else:
                self.agent_x_time_list[self.timestep, agent_id] = self.agent_x_time_list[self.timestep - 1, agent_id]
                self.agent_y_time_list[self.timestep, agent_id] = self.agent_y_time_list[self.timestep - 1, agent_id]
                consume_energy = 0
                over_range = True

            self.agent_energy_timelist[self.timestep, agent_id] = self.agent_energy_timelist[
                                                                      self.timestep - 1, agent_id] - consume_energy

        self.calculate_global_distance_matrix()

        # Calculate the minimum distance from each drone to its nearest car
        drone_nearest_car_id = np.argmin(self.global_distance_matrix[self.num_cars:self.num_agents, :self.num_cars],
                                         axis=1)
        drone_car_min_distance = self.global_distance_matrix[self.num_cars:self.num_agents, np.arange(self.num_cars)][
            np.arange(self.num_drones), drone_nearest_car_id]

        # Calculate the nearest agent for each target and their distances
        target_nearest_agent_ids = np.argmin(self.global_distance_matrix[:, self.num_agents:], axis=0)
        target_nearest_agent_distances = self.global_distance_matrix[
            np.arange(self.num_agents), target_nearest_agent_ids + self.num_agents]

        # Calculate rewards based on agent type and distance
        is_drone = self.agent_types == 1
        is_car = self.agent_types == 0

        # Reward calculation for drones
        drone_condition = (is_drone & (target_nearest_agent_distances <= self.drone_sensing_range) & (
                drone_car_min_distance <= self.drone_car_comm_range))

        # Reward calculation for cars
        car_condition = (is_car & (target_nearest_agent_distances <= self.drone_sensing_range))

        # Update AOI for targets
        increase_aoi_flags = ~np.any(np.vstack([car_condition, drone_condition]), axis=0)
        self.target_aoi_timelist[self.timestep] = np.where(increase_aoi_flags,
                                                           self.target_aoi_timelist[self.timestep - 1] + 1, 1)

        # Initialize rewards
        if self.centralized:
            # Calculate Global reward where each target AoI is recorded.
            global_reward = np.sum(
                (self.target_aoi_timelist[self.timestep - 1] - self.target_aoi_timelist[self.timestep])
                / self.episode_length)
            rew = np.full(self.num_agents, global_reward)
        else:
            rew = np.zeros(self.num_agents)
            rew += (car_condition * (self.target_aoi_timelist[self.timestep - 1] - 1) / self.episode_length)
            rew += (drone_condition * (self.target_aoi_timelist[self.timestep - 1] - 1) / self.episode_length)

        # Update coverage and data collection
        self.target_coveraged_timelist[self.timestep - 1] = np.where(~increase_aoi_flags, True, False)
        self.data_collection += np.sum(
            self.target_aoi_timelist[self.timestep - 1] - self.target_aoi_timelist[self.timestep])

        # Convert rewards back to a dictionary format if needed
        rew_dict = {agent_id: rew_val for agent_id, rew_val in enumerate(rew)}

        obs = self.generate_observation_and_update_state()

        done = {
            "__all__": (self.timestep >= self.episode_length) or over_range
        }

        return obs, rew_dict, done, self.collect_info()

    def collect_info(self) -> Dict[str, float]:
        if isinstance(self, CUDACrowdSim):
            self.target_aoi_timelist[self.timestep, :] = self.cuda_data_manager.pull_data_from_device(
                "target_aoi").mean(axis=0)
            self.agent_energy_timelist[self.timestep, :] = self.cuda_data_manager.pull_data_from_device(
                _AGENT_ENERGY).mean(axis=0)
            self.target_coveraged_timelist[self.timestep, :] = self.cuda_data_manager.pull_data_from_device(
                "target_coverage").mean(axis=0)
        self.data_collection += np.sum(
            self.target_aoi_timelist[self.timestep] - self.target_aoi_timelist[self.timestep - 1])
        coverage = np.sum(self.target_coveraged_timelist) / (self.episode_length * self.num_sensing_targets)
        mean_aoi = np.mean(self.target_aoi_timelist[self.timestep])
        freshness_factor = 1 - np.mean(np.clip(self.float_dtype(self.target_aoi_timelist[self.timestep]) /
                                               self.aoi_threshold, a_min=0, a_max=1) ** 2)
        logging.debug(f"freshness_factor: {freshness_factor}, mean_aoi: {mean_aoi}")
        mean_energy = np.mean(self.agent_energy_timelist[self.timestep])
        energy_consumption_ratio = mean_energy / self.max_uav_energy
        info = {AOI_METRIC_NAME: mean_aoi,
                ENERGY_METRIC_NAME: energy_consumption_ratio,
                "freshness_factor": freshness_factor,
                DATA_METRIC_NAME: self.data_collection / (self.episode_length * self.num_sensing_targets),
                COVERAGE_METRIC_NAME: coverage,
                MAIN_METRIC_NAME: freshness_factor * coverage
                }
        return info


    def render(self, output_file=None, plot_loop=False, moving_line=False):
        import geopandas as gpd
        import movingpandas as mpd
        mixed_df = self.human_df.copy()
        if isinstance(self, CUDACrowdSim):
            self.agent_x_time_list[self.timestep, :] = self.cuda_data_manager.pull_data_from_device("agent_x")[0]
            self.agent_y_time_list[self.timestep, :] = self.cuda_data_manager.pull_data_from_device("agent_y")[0]
        if self.timestep == self.config.env.num_timestep:
            # output final trajectory
            # 可将机器人traj，可以载入到human的dataframe中，id从-1开始递减
            for i in range(self.num_agents):
                x_list = self.agent_x_time_list[:, i]
                y_list = self.agent_y_time_list[:, i]
                id_list = np.full_like(x_list, -i - 1)
                aoi_list = np.full_like(x_list, -1)
                energy_list = self.agent_energy_timelist[:, i]
                timestamp_list = [self.start_timestamp + i * self.step_time for i in range(self.episode_length + 1)]
                x_distance_list = x_list * self.max_distance_x / self.nlon + self.max_distance_x / self.nlon / 2
                y_distance_list = y_list * self.max_distance_y / self.nlat + self.max_distance_y / self.nlat / 2
                max_longitude = abs(self.lower_left[0] - self.upper_right[0])
                max_latitude = abs(self.lower_left[1] - self.upper_right[1])
                longitude_list = x_list * max_longitude / self.nlon + max_longitude / self.nlon / 2 + self.lower_left[0]
                latitude_list = y_list * max_latitude / self.nlat + max_latitude / self.nlat / 2 + self.lower_left[1]

                data = {"id": id_list, "longitude": longitude_list, "latitude": latitude_list,
                        "x": x_list, "y": y_list, "x_distance": x_distance_list, "y_distance": y_distance_list,
                        "timestamp": timestamp_list, "aoi": aoi_list, "energy": energy_list}
                robot_df = pd.DataFrame(data)
                robot_df['t'] = pd.to_datetime(robot_df['timestamp'], unit='s')  # s表示时间戳转换
                mixed_df = pd.concat([mixed_df, robot_df])

            # ------------------------------------------------------------------------------------
            # 建立moving pandas轨迹，也可以选择调用高级API继续清洗轨迹。
            mixed_gdf = gpd.GeoDataFrame(mixed_df, geometry=gpd.points_from_xy(mixed_df.longitude, mixed_df.latitude),
                                         crs=4326)
            mixed_gdf = mixed_gdf.set_index('t').tz_localize(None)  # tz=time zone, 以本地时间为准
            mixed_gdf = mixed_gdf.sort_values(by=["id", "t"], ascending=[True, True])
            trajs = mpd.TrajectoryCollection(mixed_gdf, 'id')

            start_point = trajs.trajectories[0].get_start_location()

            # 经纬度反向
            m = folium.Map(location=[start_point.y, start_point.x], tiles="cartodbpositron", zoom_start=14, max_zoom=24)

            m.add_child(folium.LatLngPopup())
            minimap = folium.plugins.MiniMap()
            m.add_child(minimap)
            folium.TileLayer('Stamen Terrain',
                             attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL').add_to(
                m)

            folium.TileLayer('Stamen Toner',
                             attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL').add_to(
                m)

            folium.TileLayer('cartodbpositron',
                             attr='Map tiles by Carto, under CC BY 3.0. Data by OpenStreetMap, under ODbL').add_to(m)

            folium.TileLayer('OpenStreetMap', attr='© OpenStreetMap contributors').add_to(m)

            # 锁定范围
            grid_geo_json = get_border(self.upper_right, self.lower_left)
            color = "red"
            border = folium.GeoJson(grid_geo_json,
                                    style_function=lambda feature, clr=color: {
                                        'fillColor': color,
                                        'color': "black",
                                        'weight': 2,
                                        'dashArray': '5,5',
                                        'fillOpacity': 0,
                                    })
            m.add_child(border)

            for index, traj in enumerate(trajs.trajectories):
                if 0 > traj.df['id'].iloc[0] >= (-self.num_cars):
                    name = f"Agent {self.num_agents - index - 1} (Car)"
                elif traj.df['id'].iloc[0] < (-self.num_cars):
                    name = f"Agent {self.num_agents - index - 1} (Drone)"
                else:
                    name = f"Human {traj.df['id'].iloc[0]}"

                def rand_byte():
                    """
                    return a random integer between 0 and 255 ( a byte)
                    """
                    return np.random.randint(0, 255)

                color = '#%02X%02X%02X' % (rand_byte(), rand_byte(), rand_byte())  # black

                # point
                features = traj_to_timestamped_geojson(index, traj, self.num_cars, self.num_drones, color)
                TimestampedGeoJson(
                    {
                        "type": "FeatureCollection",
                        "features": features,
                    },
                    period="PT15S",
                    add_last_point=True,
                    transition_time=5,
                    loop=plot_loop,
                ).add_to(m)  # sub_map

                # line
                if index < self.num_agents:
                    geo_col = traj.to_point_gdf().geometry
                    xy = [[y, x] for x, y in zip(geo_col.x, geo_col.y)]
                    f1 = folium.FeatureGroup(name)
                    if moving_line:
                        AntPath(locations=xy, color=color, weight=4, opacity=0.7, dash_array=[100, 20],
                                delay=1000).add_to(f1)
                    else:
                        folium.PolyLine(locations=xy, color=color, weight=4, opacity=0.7).add_to(f1)
                    f1.add_to(m)

            folium.LayerControl().add_to(m)

            m.get_root().render()
            m.get_root().save(output_file)
            logging.info(f"{output_file} saved!")


class CUDACrowdSim(CrowdSim, CUDAEnvironmentContext):
    """
    CUDA version of the CrowdSim environment.
    Note: this class subclasses the Python environment class CrowdSim,
    and also the CUDAEnvironmentContext
    """

    def history_reset(self):
        super().history_reset()

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        # TODO: Note: valid status
        #  是否暂时移出游戏，目前valid的主要原因是lost connection，引入能耗之后可能会有电量耗尽
        data_dict = DataFeed()
        # add all data with add_data_list method
        data_dict.add_data_list([("agent_types", self.int_dtype(self.agent_types)),  # [n_agents, ]
                                 ("car_action_space_dx", self.float_dtype(self.car_action_space_dx)),
                                 ("car_action_space_dy", self.float_dtype(self.car_action_space_dy)),
                                 ("drone_action_space_dx", self.float_dtype(self.drone_action_space_dx)),
                                 ("drone_action_space_dy", self.float_dtype(self.drone_action_space_dy)),
                                 ("agent_x", self.float_dtype(np.full([self.num_agents, ], self.starting_location_x)),
                                  True),
                                 ("agent_x_range", self.float_dtype(self.nlon)),
                                 ("agent_y", self.float_dtype(np.full([self.num_agents, ], self.starting_location_y)),
                                  True),
                                 ("agent_y_range", self.float_dtype(self.nlat)),
                                 ("agent_energy", self.float_dtype(self.agent_energy_timelist[self.timestep, :]), True),
                                 ("agent_energy_range", self.float_dtype(self.max_uav_energy)),
                                 ("num_targets", self.int_dtype(self.num_sensing_targets)),
                                 ("num_agents_observed", self.int_dtype(self.num_agents_observed)),
                                 ("target_x", self.float_dtype(self.target_x_time_list), True),
                                 # [self.episode_length + 1, self.num_sensing_targets]
                                 ("target_y", self.float_dtype(self.target_y_time_list), True),
                                 # [self.episode_length + 1, self.num_sensing_targets]
                                 ("target_aoi", self.int_dtype(np.ones([self.num_sensing_targets, ])), True),
                                 ("target_coverage", self.bool_dtype(np.zeros([self.num_sensing_targets, ])), True),
                                 ("valid_status", self.bool_dtype(np.ones([self.num_agents, ])), True),
                                 ("neighbor_agent_ids", self.int_dtype(np.full([self.num_agents, ], -1)), True),
                                 ("car_sensing_range", self.float_dtype(self.car_sensing_range)),
                                 ("drone_sensing_range", self.float_dtype(self.drone_sensing_range)),
                                 ("drone_car_comm_range", self.float_dtype(self.drone_car_comm_range)),
                                 ("neighbor_agent_distances",
                                  self.float_dtype(np.zeros([self.num_agents, self.num_agents - 1])), True),
                                 ("neighbor_agent_ids_sorted",
                                  self.int_dtype(np.zeros([self.num_agents, self.num_agents - 1])), True),
                                 ("max_distance_x", self.float_dtype(self.max_distance_x)),
                                 ("max_distance_y", self.float_dtype(self.max_distance_y)),
                                 ("slot_time", self.float_dtype(self.step_time)),
                                 ("agent_speed", self.int_dtype(list(self.agent_speed.values()))),
                                 ])
        return data_dict

    def step(self, actions=None) -> Tuple[Dict, Dict]:
        # logging.debug(f"Timestep in CUDACrowdSim {self.timestep}")
        args = [
            _STATE,
            _OBSERVATIONS,
            _ACTIONS,
            _REWARDS,
            _GLOBAL_REWARD,
            "agent_types",
            "car_action_space_dx",
            "car_action_space_dy",
            "drone_action_space_dx",
            "drone_action_space_dy",
            "agent_x",
            "agent_x_range",
            "agent_y",
            "agent_y_range",
            _AGENT_ENERGY,
            "agent_energy_range",
            "num_targets",
            "num_agents_observed",
            "target_x",
            "target_y",
            "target_aoi",
            "target_coverage",
            "valid_status",
            "neighbor_agent_ids",
            "car_sensing_range",
            "drone_sensing_range",
            "drone_car_comm_range",
            # "neighbor_pairs",
            "neighbor_agent_distances",
            "neighbor_agent_ids_sorted",
            "_done_",
            "_timestep_",
            ("n_agents", "meta"),
            ("episode_length", "meta"),
            "max_distance_x",
            "max_distance_y",
            "slot_time",
            "agent_speed",
        ]
        if self.env_backend == "pycuda":
            self.cuda_step(
                *self.cuda_step_function_feed(args),
                block=self.cuda_function_manager.block,
                grid=self.cuda_function_manager.grid,
            )
        else:
            raise Exception("CUDACrowdSim expects env_backend = 'pycuda' ")
        # Pull data from the device
        # dones = self.cuda_data_manager.pull_data_from_device("_done_")
        # Update environment state (note self.global_state is vectorized for RLlib)
        dones = self.bool_dtype(self.cuda_data_manager.pull_data_from_device("_done_"))
        self.global_state = self.float_dtype(self.cuda_data_manager.pull_data_from_device(_STATE))
        self.timestep = int(self.cuda_data_manager.pull_data_from_device("_timestep_").mean())
        return dones, self.collect_info()


class RLlibCUDACrowdSim(MultiAgentEnv):
    def __init__(self, run_config: dict):
        super().__init__()
        logging.debug("received run_config: " + str(run_config))
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self.iter: int = 0
        requirements = ["env_params", "trainer"]
        for requirement in requirements:
            if requirement not in run_config:
                raise Exception(f"RLlibCUDACrowdSim expects '{requirement}' in run_config")
        # assert 'env_registrar' in run_config['env_params'], 'env_registrar must be specified in env_params'
        assert 'env_setup' in run_config['env_params'], 'env_setup must be specified in env_params'
        additional_params = run_config["env_params"]
        logging.debug("additional_params: " + str(additional_params))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(additional_params.get("gpu_id", 0))
        run_config['env_config'] = additional_params['env_setup']
        self.logging_config = additional_params.get('logging_config', None)
        self.trainer_params = run_config["trainer"]
        self.env_registrar = EnvironmentRegistrar()
        self.centralized = additional_params.get('centralized', False)
        if "mock" not in additional_params:
            self.env_registrar.add_cuda_env_src_path(CUDACrowdSim.name,
                                                     os.path.join(
                                                         get_project_root(),
                                                         "envs",
                                                         "crowd_sim",
                                                         "crowd_sim_step.cu"
                                                     )
                                                     )
        # self.env_registrar = additional_params['env_registrar']
        train_batch_size = self.trainer_params['train_batch_size']
        self.num_envs = self.trainer_params['num_envs']
        num_mini_batches = self.trainer_params['num_mini_batches']
        if (train_batch_size // self.num_envs) < num_mini_batches:
            logging.error("Batch per env must be larger than num_mini_batches, Exiting...")
            return

        # Using dictionary comprehension to exclude specific keys
        new_dict = {k: v for k, v in run_config.items() if k not in excluded_keys}
        self.env = CUDACrowdSim(**new_dict)
        self.env_backend = self.env.env_backend = "pycuda"
        self.action_space: spaces.Space = next(iter(self.env.action_space.values()))
        # manually setting observation space
        self.agents = []
        if "mock" not in additional_params:
            self.env_wrapper: CUDAEnvWrapper = CUDAEnvWrapper(
                self.env,
                num_envs=self.num_envs,
                env_backend=self.env_backend,
                env_registrar=self.env_registrar
            )
            self.global_state = self.env_wrapper.env.global_state
        else:
            self.env_wrapper = None
            self.global_state = None
        for name in teams_name:
            self.agents += [f"{name}{i}" for i in range(getattr(self.env, "num_" + name.strip("_")))]
        self.num_agents = len(self.agents)
        self.obs_ref = self.env.reset()[0]
        self.observation_space: spaces.Dict = spaces.Dict({"obs": binary_search_bound(self.obs_ref)})
        try:
            state_space = binary_search_bound(self.global_state)
            self.observation_space["state"] = state_space
        except AttributeError:
            pass
        if "mock" not in additional_params and self.env_wrapper.env_backend == "pycuda":
            from warp_drive.cuda_managers.pycuda_function_manager import (
                PyCUDASampler,
            )

            self.cuda_sample_controller = PyCUDASampler(
                self.env_wrapper.cuda_function_manager
            )

            policy_tag_to_agent_id_map = {
                "car": list(self.env_wrapper.env.cars),
                "drone": list(self.env_wrapper.env.drones),
            }

            # Seeding
            if self.trainer_params is not None:
                seed = self.trainer_params.get("seed", np.int32(time.time()))
                seed_everything(seed)
                self.cuda_sample_controller.init_random(seed)
            if self.logging_config is not None:
                setup_wandb(self.logging_config)

            # Create and push data placeholders to the device
            create_and_push_data_placeholders(
                env_wrapper=self.env_wrapper,
                action_sampler=self.cuda_sample_controller,
                policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
                push_data_batch_placeholders=False,  # we use lightning to batch
            )


    def get_env_info(self):
        """
        return a dict of env_info
        """
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env.config.env.num_timestep,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

    def close(self):
        if wandb.run is not None:
            wandb.finish()

    def vector_reset(self) -> List[EnvObsType]:
        """
        Reset all environments.
        """
        self.env_wrapper.reset()
        # current_observation shape [n_agent, dim_obs]
        current_observation = self.env_wrapper.cuda_data_manager.pull_data_from_device(_OBSERVATIONS)
        # convert it into a dict of {agent_id: {"obs": observation}}
        obs_list = []
        for env_index in range(self.num_envs):
            obs_list.append(get_rllib_multi_agent_obs(current_observation[env_index], self.global_state, self.agents))
        return obs_list

    def reset(self) -> MultiAgentDict:
        """
        Reset the environment and return the initial observations.
        """
        return self.vector_reset()[0]

    def vector_step(
            self, actions: List[EnvActionType]
    ) -> Tuple[List[EnvObsType], List[Dict], List[Dict], List[EnvInfoDict]]:
        """
        step all cuda environments at the same time.
        """
        # convert MultiAgentDict to an ndarray with shape [num_agent, action_index]
        vectorized_actions = np.expand_dims(np.vstack([np.array(list(action_dict.values()))
                                                       for action_dict in actions]), axis=-1)
        # np.asarray(list(map(int, {1: 2, 2: 5, 3: 8}.values()))), 30% faster than
        # actions_ls = [int(actions[agent_id]) for agent_id in actions.keys()]
        # upload actions to device
        self.env_wrapper.cuda_data_manager.data_on_device_via_torch(_ACTIONS)[:] = (
            torch.tensor(vectorized_actions).cuda())
        dones, info = self.env_wrapper.step()
        # current_observation shape [n_envs, n_agent, dim_obs]
        next_obs = self.env_wrapper.cuda_data_manager.pull_data_from_device(_OBSERVATIONS)
        if self.centralized:
            reward = np.tile(self.env_wrapper.cuda_data_manager.pull_data_from_device(_GLOBAL_REWARD),
                             reps=self.num_agents).reshape(-1, self.num_agents)
        else:
            reward = self.env_wrapper.cuda_data_manager.pull_data_from_device(_REWARDS)
        # convert observation to dict {EnvID: {AgentID: Action}...}
        obs_list, reward_list, info_list = [], [], []
        for env_index in range(self.num_envs):
            one_obs, one_reward = get_rllib_obs_and_reward(self.agents, self.global_state,
                                                           next_obs[env_index], reward[env_index])
            obs_list.append(one_obs)
            reward_list.append(one_reward)
            info_list.append({})
        if all(dones):
            logging.info("All OK!")
            log_env_metrics(self.iter, info)
        return obs_list, reward_list, dones, info_list

    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        new_dict = action_dict.copy()
        obs_list, reward_list, dones, info_list = self.vector_step([new_dict] * self.num_envs)
        return obs_list[0], reward_list[0], dones[0], info_list[0]

    def render(self, mode=None):
        logging.debug("render called")
        self.env.render("/workspace/saved_data/new_logs.html", False, False)


def get_rllib_multi_agent_obs(current_observation, state, agents: list[Any]) -> MultiAgentDict:
    return {agent_name: {"obs": current_observation[i], "state": state} for i, agent_name in enumerate(agents)}


class RLlibCUDACrowdSimWrapper(VectorEnv):
    def __init__(self, env: Union[RLlibCUDACrowdSim, GroupAgentsWrapper]):
        if isinstance(env, GroupAgentsWrapper):
            # For IQL (No Grouping) Compatibility
            actual_env = env.env
            self.group_wrapper: GroupAgentsWrapper = env
        else:
            actual_env = env
            self.group_wrapper = None
        self.observation_space = actual_env.observation_space
        self.action_space = actual_env.action_space
        self.num_envs = actual_env.num_envs
        self.num_agents = actual_env.num_agents
        self.agents = actual_env.agents
        self.env_wrapper = actual_env.env_wrapper
        self.obs_at_reset = None
        self.env = actual_env


    def vector_step(
            self, actions: List[EnvActionType]
    ) -> Tuple[List[EnvObsType], List[Dict], List[Dict], List[EnvInfoDict]]:
        logging.debug("vector_step() called")
        logging.debug(f"Current Timestep: {self.env.env.timestep}")
        if self.group_wrapper is not None:
            actions = [self.group_wrapper._ungroup_items(item) for item in actions]
        # logging.debug(actions)
        step_result = self.env.vector_step(actions)

        if self.group_wrapper is not None:
            obs_list, reward_list, done_list, info_list = step_result
            obs_group_list = [self.group_wrapper._group_items(item) for item in obs_list]
            reward_group_list = [self.group_wrapper._group_items(item) for item in reward_list]
            for rewards in reward_group_list:
                for agent_id, rew in rewards.items():
                    if isinstance(rew, list):
                        rewards[agent_id] = sum(rew)
            info_group_list = [{dummy_group_id: item} for item in info_list]
            logging.debug(done_list[:10])
            logging.debug(reward_group_list[:10])
            return obs_group_list, reward_group_list, done_list, info_group_list
        else:
            return step_result

    def vector_reset(self) -> List[EnvObsType]:
        logging.debug("vector_reset() called")
        reset_obs_list = self.env.vector_reset()
        # print(reset_obs_list[0])
        # print(self.group_wrapper._group_items(reset_obs_list[0]))
        if self.group_wrapper is not None:
            return [self.group_wrapper._group_items(item) for item in reset_obs_list]
        else:
            return reset_obs_list

    def reset_at(self, index: Optional[int] = None) -> EnvObsType:
        # logging.debug(f"resetting environment {index} called")
        if index == 0:
            self.obs_at_reset = self.vector_reset()
        return self.obs_at_reset[index]

    def try_render_at(self, index: Optional[int] = None) -> \
            Optional[np.ndarray]:
        self.env.render()

    def stop(self):
        pass


def binary_search_bound(array: np.ndarray) -> spaces.Box:
    """
    :param array: reference observation
    """
    x = float(BIG_NUMBER)
    box = spaces.Box(low=-x, high=x, shape=array.shape, dtype=array.dtype)
    low_high_valid = (box.low < 0).all() and (box.high > 0).all()
    # This loop avoids issues with overflow to make sure low/high are good.
    while not low_high_valid:
        x //= 2
        box = spaces.Box(low=-x, high=x, shape=array.shape, dtype=array.dtype)
        low_high_valid = (box.low < 0).all() and (box.high > 0).all()
    return box


def setup_wandb(logging_config: dict):
    wandb.init(project=PROJECT_NAME, name=logging_config['expr_name'], group=logging_config['group'],
               tags=[logging_config['dataset']] + logging_config['tag']
               if logging_config['tag'] is not None else [], dir=logging_config['logging_dir'],
               resume=logging_config['resume'])
    # prefix = 'env/'
    wandb.define_metric(COVERAGE_METRIC_NAME, summary="max")
    wandb.define_metric(ENERGY_METRIC_NAME, summary="min")
    wandb.define_metric(DATA_METRIC_NAME, summary="max")
    wandb.define_metric(MAIN_METRIC_NAME, summary="max")
    wandb.define_metric(AOI_METRIC_NAME, summary="min")


def get_rllib_obs_and_reward(agents: list[Any], state: np.ndarray, obs: dict[int, np.ndarray],
                             reward: dict[int, int]) -> Tuple[Dict, Dict]:
    """
    :param agents: list of agent names
    :param state: global state
    :param obs: observation
    :param reward: reward
    :return: obs_dict, reward_dict

    obs_dict: {agent_name: {"obs": observation, "state": state}}
    reward_dict: {agent_name: reward}
    """
    obs_dict = {}
    reward_dict = {}
    for i, key in enumerate(agents):
        reward_dict[key] = reward[i]
        obs_dict[key] = {
            "obs": np.array(obs[i]),
            "state": state
        }
    return obs_dict, reward_dict


def log_env_metrics(cur_iter, info: dict):
    # Create table data
    table_data = [["Metric Name", "Value"]]
    table_data.extend([[key, value] for key, value in info.items()])
    # Print the table
    logging.info(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    if wandb.run is not None:
        wandb.log(info)


class RLlibCrowdSim(MultiAgentEnv):
    def __init__(self, run_config: dict):
        # print("passed run_config is", run_config)
        super().__init__()
        self.iter = 0
        self.iter: int = 0
        requirements = ["env_params", "trainer"]
        for requirement in requirements:
            if requirement not in run_config:
                raise Exception(f"RLlibCUDACrowdSim expects '{requirement}' in run_config")
        # assert 'env_registrar' in run_config['env_params'], 'env_registrar must be specified in env_params'
        assert 'env_setup' in run_config['env_params'], 'env_setup must be specified in env_params'
        additional_params = run_config["env_params"]
        run_config['env_config'] = additional_params['env_setup']
        self.logging_config = additional_params.get('logging_config', None)
        new_dict = {k: v for k, v in run_config.items() if k not in excluded_keys}
        self.env = CrowdSim(**run_config)
        self.action_space: spaces.Space = next(iter(self.env.action_space.values()))
        # manually setting observation space
        self.obs_ref = self.env.reset()[0]
        self.agents = []
        for name in teams_name:
            self.agents += [f"{name}{i}" for i in range(getattr(self.env, "num_" + name.strip("_")))]
        self.num_agents = len(self.agents)
        self.observation_space: spaces.Dict = spaces.Dict({"obs": binary_search_bound(self.obs_ref)})
        try:
            state_space = binary_search_bound(self.env.global_state)
            self.observation_space["state"] = state_space
        except AttributeError:
            pass
        if self.logging_config is not None:
            setup_wandb(self.logging_config)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> MultiAgentDict:
        self.timestep = 0
        current_observation = self.env.reset()
        # current_observation shape [n_agent, dim_obs]
        # convert it into a dict of {agent_id: {"obs": observation}}
        return get_rllib_multi_agent_obs(current_observation, self.env.global_state, self.agents)

    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        # convert MultiAgentDict to an ndarray with shape [num_agent, action_index]
        # np.asarray(list(map(int, {1: 2, 2: 5, 3: 8}.values()))), 30% faster than
        # actions_ls = [int(actions[agent_id]) for agent_id in actions.keys()]
        raw_action_dict = {i: value for i, value in enumerate(action_dict.values())}
        obs, reward, dones, info = self.env.step(raw_action_dict)
        # current_observation shape [n_agent, dim_obs]
        obs_dict, reward_dict = get_rllib_obs_and_reward(self.agents, self.env.global_state, obs, reward)
        if dones["__all__"]:
            log_env_metrics(self.iter, info)
        # else:
        logging.debug("wandb not detected")
        # truncated = {"__all__": False}
        # figure out a way to transfer metrics information out, in marllib, info can only be the subset of obs.
        # see 1.8.0 ray/rllib/env/base_env.py L437
        return obs_dict, reward_dict, dones, {}

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env.config.env.num_timestep,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

    def close(self):
        if wandb.run is not None:
            wandb.finish()

    def render(self, mode=None) -> None:
        pass
        # self.env.render()


LARGE_DATASET_NAME = 'SanFrancisco'
