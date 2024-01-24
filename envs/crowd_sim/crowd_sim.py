"""
The Mobile Crowdsensing Environment
"""
from datetime import datetime
import logging
import os
import pprint
import warnings
import random
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List, Any, Union

import numpy as np
import torch
import folium
import wandb
import time
import pandas as pd
from folium import DivIcon
from folium.plugins import TimestampedGeoJson, AntPath
from gym import spaces
from gym.spaces import Discrete, Box, MultiDiscrete
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

state_aoi_caption = "State AoI"

emergency_caption = "Emergency Grid Example"

aoi_caption = "AoI Grid Example"

FRESHNESS_FACTOR = "freshness_factor"

OVERALL_AOI = "overall_aoi"

EMERGENCY_METRIC = "response_delay"

SURVEILLANCE_METRIC = "surveillance_aoi"

VALID_HANDLING_RATIO = "valid_handling_ratio"

user_override_params = ['env_config', 'dynamic_zero_shot', 'use_2d_state', 'all_random',
                        'num_drones', 'num_cars', 'cut_points', 'fix_target', 'gen_interval', 'no_refresh']

grid_size = 10

COVERAGE_METRIC_NAME = Constants.COVERAGE_METRIC_NAME
DATA_METRIC_NAME = Constants.DATA_METRIC_NAME
ENERGY_METRIC_NAME = Constants.ENERGY_METRIC_NAME
MAIN_METRIC_NAME = Constants.MAIN_METRIC_NAME
AOI_METRIC_NAME = Constants.AOI_METRIC_NAME
_VECTOR_STATE = Constants.VECTOR_STATE
_IMAGE_STATE = Constants.IMAGE_STATE
_AGENT_ENERGY = Constants.AGENT_ENERGY
_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_GLOBAL_REWARD = Constants.GLOBAL_REWARDS
_STATE = Constants.STATE
teams_name = ("cars_", "drones_")
map_dict = {
    "description": "two team cooperate to collect data",
    "team_prefix": teams_name,
    "all_agents_one_policy": True,
    "one_agent_one_policy": True,
}
policy_mapping_dict = {
    "SanFrancisco": map_dict,
    "KAIST": map_dict,
}
excluded_keys = {"trainer", "env_params", "map_name"}
logging.getLogger().setLevel(logging.WARN)


def convert_to_lat_lon_units(distance, latitude):
    # Earth's radius in kilometers
    earth_radius_km = 6371

    # Convert distance to kilometers (if it's in a different unit)
    distance_km = distance  # Replace this with the actual conversion if needed

    # Calculate the conversion factor for latitude (degrees to kilometers)
    lat_to_km_conversion_factor = (2 * np.pi * earth_radius_km) / 360

    # Calculate the maximum latitude change equivalent to the desired distance
    max_lat_change = distance_km / lat_to_km_conversion_factor

    # Calculate the equivalent longitude change at the given latitude
    lon_to_lat_conversion_factor = (2 * np.pi * earth_radius_km * np.cos(np.deg2rad(latitude))) / 360
    max_lon_change = distance_km / lon_to_lat_conversion_factor

    return max_lat_change, max_lon_change


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
    """
    Calculate the angle between two points
    """
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
            # num_agents_observed=5,
            seed=None,
            env_backend="cpu",
            single_type_agent=True,
            dynamic_zero_shot=False,
            use_2d_state=False,
            fix_target=False,
            env_config=None,
            centralized=True,
            all_random=False,
            cut_points=-1,
            gen_interval=30,
            no_refresh=False,
    ):
        self.float_dtype = np.float32
        self.single_type_agent = single_type_agent
        self.int_dtype = np.int32
        self.no_refresh = no_refresh
        self.bool_dtype = np.bool_
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)
        self.fix_target = fix_target
        self.config = env_config()
        # Seeding
        self.np_random: np.random = np.random
        if seed is not None:
            self.seed(seed)

        self.centralized = centralized
        self.use_2d_state = use_2d_state
        self.num_drones = num_drones
        self.num_cars = num_cars
        self.num_agents = self.num_drones + self.num_cars
        self.gen_interval = gen_interval
        self.num_sensing_targets = self.config.env.human_num
        if cut_points != -1:
            self.num_sensing_targets = min(self.num_sensing_targets, cut_points)
        self.aoi_threshold = self.config.env.aoi_threshold
        self.emergency_threshold = self.config.env.emergency_threshold
        self.num_agents_observed = self.num_agents - 1
        self.all_random = all_random
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
        self.human_df: pd.DataFrame = pd.read_csv(self.config.env.dataset_dir)
        logging.debug("Finished reading {} rows".format(len(self.human_df)))
        # Hack, reduce points for better effect.
        if cut_points != -1:
            self.human_df = self.human_df[self.human_df['id'] < cut_points]
        self.human_df['t'] = pd.to_datetime(self.human_df['timestamp'], unit='s')  # s表示时间戳转换
        self.human_df['aoi'] = -1  # 加入aoi记录aoi
        self.human_df['energy'] = -1  # 加入energy记录energy
        self.max_distance_x = min(self.max_distance_x, max(self.human_df['x']))
        self.max_distance_y = min(self.max_distance_y, max(self.human_df['y']))
        self.agent_speed = {'car': self.config.env.car_velocity, 'drone': self.config.env.drone_velocity}
        points_x, points_y, self.num_centers, self.num_points, self.num_points_per_center = (None,) * 5
        self.dynamic_zero_shot = dynamic_zero_shot
        if self.all_random:
            self.num_centers = self.num_sensing_targets
            self.num_points_per_center = 1
            points_x, points_y = self.generate_emergency(self.num_centers, self.num_points_per_center)
            self.zero_shot_start = 0
            self.emergency_count = 0
            self.points_per_gen = 0
        else:
            self.zero_shot_start = self.num_sensing_targets
            self.points_per_gen = self.num_agents - 1 if self.num_agents > 1 else 1
            self.aoi_schedule = np.repeat(np.arange(self.gen_interval, self.episode_length, self.gen_interval),
                                          repeats=self.points_per_gen)
            logging.debug(f"AoI Schedule: {self.aoi_schedule}")
            generation_time = int(self.aoi_schedule.shape[0] / self.points_per_gen)
            if self.dynamic_zero_shot:
                self.num_centers = generation_time * self.points_per_gen
                self.num_points_per_center = 1
                points_x, points_y = self.generate_emergency(self.num_centers, self.num_points_per_center)
                self.emergency_count = (self.num_centers * self.num_points_per_center)
                self.num_sensing_targets += self.emergency_count
                # add visualization parts of dynamic generated points.
            else:
                self.zero_shot_start = 0
                self.emergency_count = 0
                self.points_per_gen = 0
        # human infos
        unique_ids = np.arange(0, self.num_sensing_targets)  # id from 0 to 91
        unique_timestamps = np.arange(self.start_timestamp, self.end_timestamp + self.step_time, self.step_time)
        id_to_index = {my_id: index for index, my_id in enumerate(unique_ids)}
        timestamp_to_index = {timestamp: index for index, timestamp in enumerate(unique_timestamps)}
        self.target_x_time_list = np.zeros([self.episode_length + 1, self.num_sensing_targets], dtype=np.int_)
        self.target_y_time_list = np.zeros([self.episode_length + 1, self.num_sensing_targets], dtype=np.int_)
        self.target_aoi_timelist = np.ones([self.episode_length + 1, self.num_sensing_targets], dtype=np.int_)
        self.target_coveraged_timelist = np.zeros([self.episode_length + 1, self.num_sensing_targets], dtype=np.bool_)
        if not self.all_random:
            # Fill the new array with data from the full DataFrame
            last_id = -1
            first_row_of_last_id = None
            logging.debug(f"target fix status: {fix_target}")
            for row in self.human_df.itertuples():
                id_index = id_to_index.get(row.id, None)
                if last_id != id_index:
                    first_row_of_last_id = row
                    last_id = id_index
                timestamp_index = timestamp_to_index.get(row.timestamp, None)
                if not (id_index is None or timestamp_index is None):
                    if fix_target:
                        self.target_x_time_list[timestamp_index, id_index] = first_row_of_last_id.x
                        self.target_y_time_list[timestamp_index, id_index] = first_row_of_last_id.y
                    else:
                        self.target_x_time_list[timestamp_index, id_index] = row.x
                        self.target_y_time_list[timestamp_index, id_index] = row.y
                else:
                    raise ValueError("Got invalid rows:", row)

            if dynamic_zero_shot:
                self.target_x_time_list[:, self.num_sensing_targets -
                                           self.num_centers * self.num_points_per_center:] = points_x
                self.target_y_time_list[:, self.num_sensing_targets -
                                           self.num_centers * self.num_points_per_center:] = points_y
                # rebuild DataFrame from longitude and latitude
        else:
            self.target_x_time_list[:, :] = points_x
            self.target_y_time_list[:, :] = points_y

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
        self.drone_action_space_dx = self.float_dtype(self.config.env.drone_action_space[:, 0])
        self.drone_action_space_dy = self.float_dtype(self.config.env.drone_action_space[:, 1])
        self.car_action_space_dx = self.float_dtype(self.config.env.car_action_space[:, 0])
        self.car_action_space_dy = self.float_dtype(self.config.env.car_action_space[:, 1])
        self.action_space = spaces.Dict()
        self.emergency_slots = self.points_per_gen
        for agent_id in range(self.num_agents):
            # note one action for not choosing any emergency.
            if self.agent_types[agent_id] == 1:
                self.action_space[agent_id] = Discrete(len(self.drone_action_space_dx))
                # self.action_space[agent_id] = MultiDiscrete([len(self.drone_action_space_dx), self.emergency_slots + 1])
            else:
                self.action_space[agent_id] = Discrete(len(self.car_action_space_dx))
                # self.action_space[agent_id] = MultiDiscrete([len(self.car_action_space_dx), self.emergency_slots + 1])
        if isinstance(self.action_space[0], MultiDiscrete):
            self.action_dim = self.action_space[0].nvec[0]
        else:
            self.action_dim = self.action_space[0].n

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
        if self.dynamic_zero_shot:
            self.emergency_allocation_table = np.full([self.emergency_count, ], -1, dtype=self.int_dtype)
        self.agent_emergency_table = np.full([self.episode_length + 1, self.num_agents], fill_value=-1,
                                             dtype=self.int_dtype)
        self.agent_actions_time_list = np.zeros([self.episode_length + 1, self.num_agents, self.action_dim],
                                                dtype=self.int_dtype)
        self.agent_rewards_time_list = np.zeros([self.episode_length + 1, self.num_agents], dtype=self.float_dtype)
        self.data_collection = 0

        # These will be set during reset (see below)
        self.timestep = None

        # Defining observation and action spaces
        # obs = self type(1) + energy (1) + (num_agents - 1) * (homo_pos(2) + hetero_pos(2)) +
        # neighbor_aoi_grids (10 * 10) = 122
        self.observation_space = None  # Note: this will be set via the env_wrapper
        # state = (type,energy,x,y) * self.num_agents + neighbor_aoi_grids (10 * 10)
        self.vector_state_dim = (self.num_agents + 4) * self.num_agents + self.emergency_count * 4 + 1
        self.image_state_dim = 100
        if self.use_2d_state:
            self.global_state = {
                _IMAGE_STATE: np.zeros(self.image_state_dim).reshape(-1, grid_size, grid_size),
                _VECTOR_STATE: np.zeros(self.vector_state_dim),
            }
        else:
            self.global_state = np.zeros((self.vector_state_dim + self.image_state_dim,),
                                         dtype=self.float_dtype)
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
        # List of available colors excluding orange and red
        self.available_colors = ['blue', "darkgreen", "darkred", "purple", 'black', "magenta", 'darkblue', "teal",
                                 "brown", 'gray']

        # Shuffle the list of available colors
        random.shuffle(self.available_colors)
        # Initialize an index to keep track of the selected color
        self.selected_color_index = 0
        self.queue_feature = 2
        self.queue_length = 10

    def get_next_color(self):
        # Function to get the next color
        # Check if we have used all colors, shuffle again if needed
        if self.selected_color_index >= len(self.available_colors):
            random.shuffle(self.available_colors)
            self.selected_color_index = 0
        # Get the next color
        next_color = self.available_colors[self.selected_color_index]
        self.selected_color_index += 1
        return next_color

    def generate_emergency(self, num_centers, num_points_per_center):
        max_distance_from_center = 10
        max_distance_x = self.max_distance_x
        max_distance_y = self.max_distance_y

        # Generate random center coordinates
        centers_x = self.np_random.randint(0, int(max_distance_x), (num_centers,))
        centers_y = self.np_random.randint(0, int(max_distance_y), (num_centers,))

        # Generate random relative offsets for all points
        offsets_x = self.np_random.randint(-max_distance_from_center, max_distance_from_center + 1,
                                           (num_centers, num_points_per_center))
        offsets_y = self.np_random.randint(-max_distance_from_center, max_distance_from_center + 1,
                                           (num_centers, num_points_per_center))

        # Calculate point coordinates for all points
        points_x = np.clip(centers_x[:, np.newaxis] + offsets_x, 0, int(max_distance_x) - 1).reshape(-1)
        points_y = np.clip(centers_y[:, np.newaxis] + offsets_y, 0, int(max_distance_y) - 1).reshape(-1)
        logging.debug(f"new x,y examples {points_x[:5]}, {points_y[:5]}")
        return points_x, points_y

    def seed(self, seed=None):
        """
        Seeding the environment with a desired seed
        Note: this uses the code in
        https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        """
        self.np_random.seed(seed)
        return [seed]

    def history_reset(self):
        """
        Reset the history of the environment, including the global state, the global distance matrix,
        the target coverage / aoi, and agent x,y,energy.
        """
        # Reset time to the beginning
        self.timestep = 0
        # Re-initialize the global state
        # for agent_id in range(self.num_agents):
        self.agent_x_time_list[self.timestep, :] = self.starting_location_x
        self.agent_y_time_list[self.timestep, :] = self.starting_location_y
        self.agent_energy_timelist[self.timestep, :] = self.max_uav_energy
        if self.dynamic_zero_shot:
            self.emergency_allocation_table[:] = -1
        # for target_id in range(self.num_sensing_targets):
        self.target_aoi_timelist = np.zeros([self.episode_length + 1, self.num_sensing_targets], dtype=np.int_)
        self.target_aoi_timelist[self.timestep, :] = 1
        # for logging
        self.data_collection = 0
        # print("Reset target coverage timelist")
        self.target_coveraged_timelist = np.zeros([self.episode_length + 1, self.num_sensing_targets], dtype=np.bool_)

    def reset(self):
        """
        Env reset().
        """
        logging.debug("CrowdSim reset() is called")
        self.history_reset()
        self.targets_regen()
        return self.generate_observation_and_update_state()

    def targets_regen(self):
        if self.dynamic_zero_shot and not self.all_random:
            logging.debug("Emergency points reset completed!")
            points_x, points_y = self.generate_emergency(self.num_centers, self.num_points_per_center)
            self.target_x_time_list[:, self.zero_shot_start:] = points_x
            self.target_y_time_list[:, self.zero_shot_start:] = points_y
        elif self.all_random:
            points_x, points_y = self.generate_emergency(self.num_centers, self.num_points_per_center)
            self.target_x_time_list[:, :] = points_x
            self.target_y_time_list[:, :] = points_y

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
        # reset global distance matrix
        self.calculate_global_distance_matrix()
        # Self info for all agents (num_agents, 2)
        agents_state = np.vstack([np.diag(np.ones(self.num_agents)), self.agent_types,
                                  self.agent_energy_timelist[self.timestep] / self.max_uav_energy,
                                  self.agent_x_time_list[self.timestep] / self.max_distance_x,
                                  self.agent_y_time_list[self.timestep] / self.max_distance_y]).T

        # Generate agent nearest targets IDs
        agent_nearest_targets_ids = np.argsort(self.global_distance_matrix[:, :self.num_agents], axis=-1, kind='stable')

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
        aoi_grid_parts = self.generate_aoi_grid(self.target_x_time_list[self.timestep, :self.zero_shot_start],
                                                self.target_y_time_list[self.timestep, :self.zero_shot_start],
                                                np.zeros(self.zero_shot_start, dtype=self.int_dtype),
                                                self.target_aoi_timelist[self.timestep, :self.zero_shot_start],
                                                self.agent_x_time_list[self.timestep, :],
                                                self.agent_y_time_list[self.timestep, :],
                                                self.drone_car_comm_range * 2,
                                                self.drone_car_comm_range * 2,
                                                grid_size)

        # TODO: this full_queue is mock, no actual prediction is provided.
        full_queue = np.zeros((self.num_agents, self.queue_feature))
        if self.dynamic_zero_shot:
            current_aoi = self.target_aoi_timelist[self.timestep]
            valid_zero_shots_mask = (current_aoi > 1) & (np.arange(self.num_sensing_targets) > self.zero_shot_start) & \
                                    np.concatenate([np.zeros(self.zero_shot_start, dtype=self.bool_dtype),
                                                    self.timestep > self.aoi_schedule]) & \
                                    (self.target_coveraged_timelist[self.timestep] == 0)
            # find valid emergencies only
            # zero_shot_aois, zero_shot_x, zero_shot_y = (current_aoi[valid_zero_shots_mask],
            #                                             self.target_x_time_list[self.timestep, valid_zero_shots_mask],
            #                                             self.target_y_time_list[self.timestep, valid_zero_shots_mask])
            # select the part of global distance matrix, where distances between all zero_shot points and all agents
            # are included
            # zero_shot_distance_matrix = self.global_distance_matrix[self.num_agents:][valid_zero_shots_mask,
            #                             :self.num_agents]
            # select the nearest agents for each zero_shot points
            # zero_shot_nearest_agents_ids = np.argsort(zero_shot_distance_matrix, axis=-1, kind='stable')
            # calculate distance of each zero_shot points to each agent and argsort
            # zero_shot_distance_to_agents = np.argsort(zero_shot_distance_matrix, axis=0, kind='stable')[:10]
            # fill the queue of each agent according to argsort result, features are listed in the order (x,y,aoi,distance)
            # for agent_id in range(self.num_agents):
            #     if len(zero_shot_distance_to_agents[:, agent_id]) > 0:
            #         agent_queue = np.zeros((self.queue_length, self.queue_feature), dtype=self.float_dtype)
            #         agent_queue[:, 0] = zero_shot_x[zero_shot_distance_to_agents[:, agent_id]]
            #         agent_queue[:, 1] = zero_shot_y[zero_shot_distance_to_agents[:, agent_id]]
            #         agent_queue[:, 2] = zero_shot_aois[zero_shot_distance_to_agents[:, agent_id]]
            #         agent_queue[:, 3] = zero_shot_distance_matrix[zero_shot_distance_to_agents[:, agent_id], agent_id]
            #         full_queue[agent_id, :] = agent_queue.reshape(-1)

        # Generate global state AoI grid
        state_aoi_grid = self.generate_aoi_grid(self.target_x_time_list[self.timestep, :self.zero_shot_start],
                                                self.target_y_time_list[self.timestep, :self.zero_shot_start],
                                                np.zeros(self.zero_shot_start, dtype=self.int_dtype),
                                                self.target_aoi_timelist[self.timestep, :self.zero_shot_start],
                                                int(self.max_distance_x // 2),
                                                int(self.max_distance_y // 2),
                                                int(self.max_distance_x),
                                                int(self.max_distance_y),
                                                grid_size)
        vector_obs = self.float_dtype(np.hstack((agents_state, homoge_parts.reshape(self.num_agents, -1),
                                                 hetero_parts.reshape(self.num_agents, -1), full_queue)))

        # Global state
        if self.dynamic_zero_shot:
            emergency_status = np.where(self.timestep > self.aoi_schedule,
                                        self.target_coveraged_timelist[self.timestep, self.zero_shot_start:],
                                        -1)
            emergency_state = np.vstack([self.target_x_time_list[self.timestep, self.zero_shot_start:] / self.nlon,
                                         self.target_y_time_list[self.timestep, self.zero_shot_start:] / self.nlat,
                                         self.target_aoi_timelist[self.timestep, self.zero_shot_start:],
                                         emergency_status]).T
            vector_state = np.concatenate([agents_state.ravel(), emergency_state.ravel(), np.array([self.timestep])])
        else:
            vector_state = np.concatenate([agents_state.ravel(), np.array([self.timestep])])
        if self.use_2d_state:
            self.global_state = {
                _VECTOR_STATE: self.float_dtype(vector_state),
                _IMAGE_STATE: self.float_dtype(state_aoi_grid),
            }
            for array in self.global_state.values():
                logging.debug("Global state shape: {}".format(array.shape))
            observations = {agent_id: {
                _VECTOR_STATE: vector_obs[agent_id],
                _IMAGE_STATE: aoi_grid_parts[agent_id][np.newaxis]
            } for agent_id in range(self.num_agents)
            }
            for array in observations[0].values():
                logging.debug("Observation shape: {}".format(array.shape))
            return observations
        else:
            # Merge parts for observations
            aoi_grids = aoi_grid_parts.reshape(self.num_agents, -1)
            observations = self.float_dtype(np.hstack((vector_obs,
                                                       aoi_grids.reshape(self.num_agents, -1))))
            self.global_state = self.float_dtype(np.concatenate([vector_state,
                                                                 state_aoi_grid.ravel()]))
            observations = {agent_id: observations[agent_id] for agent_id in range(self.num_agents)}
        return observations

    def generate_agent_pos_grid(self, grid_centers_x: Union[np.ndarray, int, float],
                                grid_centers_y: Union[np.ndarray, int, float], my_grid_size: int) -> np.ndarray:
        """
        Generate Discrete agent positions for each agent. (Grid_centers can be vectored or scalar)
        """
        # Target positions
        point_xy = np.stack((self.agent_x_time_list[self.timestep], self.agent_y_time_list[self.timestep]), axis=-1)
        discrete_points_xy = self.generate_discrete_points(grid_centers_x, grid_centers_y, my_grid_size, point_xy,
                                                           self.max_distance_x, self.max_distance_y)
        agent_pos_grid = np.zeros((my_grid_size, my_grid_size), dtype=self.float_dtype)
        # select index with discrete_points_xy
        agent_pos_grid[discrete_points_xy[:, 0], discrete_points_xy[:, 1]] = 1
        return agent_pos_grid

    def generate_aoi_grid(self, x_list: np.ndarray, y_list: np.ndarray,
                          aoi_schedule: np.ndarray, aoi_list: np.ndarray,
                          grid_centers_x: Union[np.ndarray, int, float],
                          grid_centers_y: Union[np.ndarray, int, float],
                          sensing_range_x: int, sensing_range_y: int, grid_size: int) -> np.ndarray:
        """
        Generate AoI grid parts for each agent. (Grid_centers can be vectored or scalar)
        """

        # Target positions
        point_xy = np.stack((x_list, y_list), axis=-1)
        discrete_points_xy = self.generate_discrete_points(grid_centers_x, grid_centers_y, grid_size, point_xy,
                                                           sensing_range_x, sensing_range_y)
        # Initialize AoI grid parts
        num_entries = grid_centers_x.shape[0] if isinstance(grid_centers_x, np.ndarray) else 1
        aoi_grid_parts = np.zeros((num_entries, grid_size, grid_size), dtype=self.float_dtype)
        grid_point_count = np.zeros((num_entries, grid_size, grid_size), dtype=self.int_dtype)

        # Iterate over each agent
        for agent_id in range(num_entries):
            # Create a boolean mask for valid points for this agent
            valid_mask = (discrete_points_xy[agent_id, :, 0] >= 0) & (discrete_points_xy[agent_id, :, 0] < grid_size) & \
                         (discrete_points_xy[agent_id, :, 1] >= 0) & (discrete_points_xy[agent_id, :, 1] < grid_size) & \
                         (aoi_schedule < self.timestep)
            # Filter the points and AoI values using the mask
            filtered_points = discrete_points_xy[agent_id, valid_mask]
            # is_zero_shot = np.arange(self.num_sensing_targets) > self.zero_shot_start
            filtered_aoi_values = aoi_list[valid_mask]
            # filtered_aoi_values[is_zero_shot] *= 1.5
            # TODO: strong aoi for emgergency is not added.
            # Accumulate counts and AoI values for this agent
            np.add.at(aoi_grid_parts[agent_id], (filtered_points[:, 0], filtered_points[:, 1]), filtered_aoi_values)
            np.add.at(grid_point_count[agent_id], (filtered_points[:, 0], filtered_points[:, 1]), 1)

        # Normalize AoI values
        grid_point_count_nonzero = np.where(grid_point_count > 0, grid_point_count, 1)
        aoi_grid_parts = aoi_grid_parts / grid_point_count_nonzero / self.episode_length
        aoi_grid_parts[grid_point_count == 0] = 0
        return aoi_grid_parts

    def generate_discrete_points(self, grid_centers_x, grid_centers_y, grid_size, point_xy, sensing_range_x,
                                 sensing_range_y):
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
        # Discretize target positions
        discrete_points_xy = np.floor((point_xy[None, :, :] - grid_min[:, None, :]) / (
                grid_max[:, None, :] - grid_min[:, None, :]) * grid_size).astype(int)
        return discrete_points_xy

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
            # 158J at vt=18m/s
            flying_energy = P0 * (1 + 3 * vt ** 2 / U_tips ** 2) + \
                            P1 * np.sqrt((np.sqrt(1 + vt ** 4 / (4 * v0 ** 4)) - vt ** 2 / (2 * v0 ** 2))) + \
                            0.5 * d0 * rho * s0 * A * vt ** 3

            # 168.48J at vt=0m/s
            hovering_energy = P0 + P1
            return move_time * flying_energy + stop_time * hovering_energy
        else:
            raise NotImplementedError("Energy model not supported for the agent.")

    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        # TODO: CPU step is outdated.
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
        # TODO: for dynamic points, once the AoI is cleared, its AoI will not rise again.
        self.target_aoi_timelist[self.timestep] = np.where(increase_aoi_flags,
                                                           self.target_aoi_timelist[self.timestep - 1] + 1, 1)
        # TODO: CPU version does not have extra reward for emergency targets.
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
        freshness_factor = 1 - np.mean(np.clip(self.float_dtype(self.target_aoi_timelist[self.timestep]) /
                                               self.aoi_threshold, a_min=0, a_max=1) ** 2)
        logging.debug(f"{FRESHNESS_FACTOR}: {freshness_factor}")
        # mean_energy = np.mean(self.agent_energy_timelist[self.timestep])
        # energy_remaining_ratio = mean_energy / self.max_uav_energy
        info = {
            ENERGY_METRIC_NAME: 1 - np.mean(self.agent_energy_timelist[self.timestep]) / self.max_uav_energy,
            FRESHNESS_FACTOR: freshness_factor,
        }
        if self.dynamic_zero_shot and not self.all_random:
            info[SURVEILLANCE_METRIC] = np.mean(self.target_aoi_timelist[self.timestep, :-self.emergency_count])
            emergency_aoi = self.target_aoi_timelist[self.timestep, -self.emergency_count:]
            info[EMERGENCY_METRIC] = np.mean(emergency_aoi)
            info[VALID_HANDLING_RATIO] = np.mean(emergency_aoi < self.emergency_threshold)
            # info[OVERALL_AOI] = (info[SURVEILLANCE_METRIC] + info[EMERGENCY_METRIC]) / 2
            # logging.debug(f"Emergency: {info[EMERGENCY_METRIC]}")
        else:
            mean_aoi = np.mean(self.target_aoi_timelist[self.timestep])
            # self.data_collection += np.sum(
            #     self.target_aoi_timelist[self.timestep] - self.target_aoi_timelist[self.timestep - 1])
            coverage = np.sum(self.target_coveraged_timelist) / (self.episode_length * self.num_sensing_targets)
            info.update({
                # DATA_METRIC_NAME: self.data_collection / (self.episode_length * self.num_sensing_targets),
                AOI_METRIC_NAME: mean_aoi,
                COVERAGE_METRIC_NAME: coverage,
                MAIN_METRIC_NAME: freshness_factor * coverage
            })
        return info

    def render(self, output_file=None, plot_loop=False, moving_line=False):

        def custom_style_function(feature):
            return {
                "color": feature["properties"]["style"]["color"],  # Use the color from the properties
                "weight": 2,
                "radius": 10,  # Adjust the marker size as needed
            }

        # add html suffix if output_file does not have one
        if output_file is not None and not output_file.endswith(".html"):
            output_file += ".html"

        logging.debug(f"Current Render {self.timestep} / {self.episode_length} timestep")
        if isinstance(self, CUDACrowdSim):
            self.agent_x_time_list[self.timestep, :] = self.cuda_data_manager.pull_data_from_device("agent_x")[0]
            self.agent_y_time_list[self.timestep, :] = self.cuda_data_manager.pull_data_from_device("agent_y")[0]
            agent_emergency_allocation = self.cuda_data_manager.pull_data_from_device("emergency_allocation_table")[0]
            self.agent_rewards_time_list[self.timestep, :] = self.cuda_data_manager.pull_data_from_device(_REWARDS)[0]
            self.agent_actions_time_list[self.timestep, :] = self.cuda_data_manager.pull_data_from_device(_ACTIONS)[0]
            if self.dynamic_zero_shot:
                # get emergency index where the allocation table is not -1
                num_of_normal_targets = self.num_sensing_targets - self.emergency_count
                for i in range(self.num_agents):
                    emergency_index = agent_emergency_allocation[i]
                    if emergency_index != -1:
                        self.emergency_allocation_table[emergency_index - num_of_normal_targets] = i
                self.agent_emergency_table[self.timestep] = agent_emergency_allocation
            # agent_emergency_allocation will now be the index of emergency points, where the index of
            # agent_emergency_allocation are the allocated agents.
            # agent table
            # 0 1 2 3
            # 201 202 200 203
            # Emergency Table
            # 200 201 202 203
            # 2 0 1 3

        if self.timestep == self.config.env.num_timestep:
            # output final trajectory
            # 可将机器人traj，可以载入到human的dataframe中，id从-1开始递减
            import geopandas as gpd
            import movingpandas as mpd
            mixed_df = self.human_df.copy()
            aoi_list = np.full(self.episode_length + 1, -1, dtype=np.int_)
            timestamp_list = [self.start_timestamp + t * self.step_time for t in range(self.episode_length + 1)]
            max_longitude = abs(self.lower_left[0] - self.upper_right[0])
            max_latitude = abs(self.lower_left[1] - self.upper_right[1])
            for i in range(self.num_agents):
                x_list = self.agent_x_time_list[:, i]
                y_list = self.agent_y_time_list[:, i]
                id_list = np.full_like(x_list, -i - 1)
                energy_list = self.agent_energy_timelist[:, i]
                robot_df = self.xy_to_dataframe(aoi_list, energy_list, id_list, max_latitude,
                                                max_longitude, timestamp_list, x_list, y_list)
                robot_df['reward'] = self.agent_rewards_time_list[:, i]
                robot_df['selection'] = self.agent_actions_time_list[:, i, 0]
                # Add reward timelist for rendering.
                mixed_df = pd.concat([mixed_df, robot_df])
            # add emergency targets.
            # pull emergency coordinates from device to host, since points in host are already refreshed
            all_zero_shot_x = self.cuda_data_manager.pull_data_from_device("target_x")[0][:, self.zero_shot_start:]
            all_zero_shot_y = self.cuda_data_manager.pull_data_from_device("target_y")[0][:, self.zero_shot_start:]
            if self.dynamic_zero_shot:
                for i in range(self.zero_shot_start, self.num_sensing_targets):
                    logging.debug(f"Creation Time: {self.aoi_schedule[i - self.zero_shot_start]}")
                    delay_list = np.full_like(self.target_aoi_timelist[:, i], self.target_aoi_timelist[:, i][-1])
                    x_list = all_zero_shot_x[:, i - self.zero_shot_start]
                    y_list = all_zero_shot_y[:, i - self.zero_shot_start]
                    id_list = np.full_like(x_list, i)
                    energy_list = np.zeros_like(x_list)
                    robot_df = self.xy_to_dataframe(delay_list, energy_list, id_list, max_latitude,
                                                    max_longitude, timestamp_list, x_list, y_list)
                    robot_df['creation_time'] = self.aoi_schedule[i - self.zero_shot_start]
                    if self.dynamic_zero_shot:
                        robot_df['allocation'] = int(self.emergency_allocation_table[i - self.zero_shot_start])
                    else:
                        robot_df['allocation'] = -1
                    robot_df['episode_length'] = self.episode_length
                    mixed_df = pd.concat([mixed_df, robot_df])
            # ------------------------------------------------------------------------------------
            # 建立moving pandas轨迹，也可以选择调用高级API继续清洗轨迹。
            mixed_gdf = gpd.GeoDataFrame(mixed_df, geometry=gpd.points_from_xy(mixed_df.longitude, mixed_df.latitude),
                                         crs=4326)
            mixed_gdf = mixed_gdf.set_index('t').tz_localize(None)  # tz=time zone, 以本地时间为准
            mixed_gdf = mixed_gdf.sort_values(by=["id", "t"], ascending=[True, True])
            trajectories = mpd.TrajectoryCollection(mixed_gdf, 'id')

            start_point = trajectories.trajectories[0].get_start_location()

            # 经纬度反向
            my_render_map: folium.Map = folium.Map(location=[start_point.y, start_point.x], tiles="cartodbpositron",
                                                   zoom_start=14, max_zoom=24, control_scale=True)

            my_render_map.add_child(folium.LatLngPopup())
            minimap = folium.plugins.MiniMap()
            my_render_map.add_child(minimap)
            # folium.TileLayer('Stamen Terrain',
            #                  attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL'
            #                  ).add_to(my_render_map)
            #
            # folium.TileLayer('Stamen Toner',
            #                  attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL'
            #                  ).add_to(my_render_map)

            folium.TileLayer('cartodbpositron',
                             attr='Map tiles by Carto, under CC BY 3.0. Data by OpenStreetMap, under ODbL'
                             ).add_to(my_render_map)

            folium.TileLayer('OpenStreetMap', attr='© OpenStreetMap contributors').add_to(my_render_map)

            # 锁定范围
            grid_geo_json = get_border(self.upper_right, self.lower_left)
            color = "red"
            border = folium.GeoJson(grid_geo_json,
                                    style_function=lambda feature, clr=color: {
                                        # 'fillColor': color,
                                        'color': "black",
                                        'weight': 2,
                                        'dashArray': '5,5',
                                        'fillOpacity': 0,
                                    })
            my_render_map.add_child(border)
            all_features = []
            for index, traj in tqdm(enumerate(trajectories)):
                is_car = 0 > traj.df['id'].iloc[0] >= (-self.num_cars)
                is_drone = traj.df['id'].iloc[0] < (-self.num_cars)
                if is_car:
                    name = f"Agent {self.num_agents - index - 1} (Car)"
                elif is_drone:
                    name = f"Agent {self.num_agents - index - 1} (Drone)"
                else:
                    name = f"PoI {traj.df['id'].iloc[0]}"

                # Define your color logic here
                if self.dynamic_zero_shot:
                    if index < self.num_agents:
                        color = self.get_next_color()
                    elif (self.dynamic_zero_shot and self.num_agents <= index < self.num_agents +
                          self.zero_shot_start):
                        color = "orange"
                    else:
                        # emergency targets
                        color = "red"
                else:
                    if index < self.num_agents:
                        color = self.get_next_color()
                    else:
                        color = "orange"

                # Create features for the current trajectory
                features = traj_to_timestamped_geojson(index,
                                                       traj,
                                                       self.num_cars,
                                                       self.num_drones,
                                                       color,
                                                       index < self.num_agents,
                                                       self.fix_target,
                                                       color == 'red')
                if is_car or is_drone:
                    # create a feature group
                    TimestampedGeoJson(
                        {
                            "type": "FeatureCollection",
                            "features": features,
                        },
                        period="PT5S",  # Adjust the time interval as needed
                        add_last_point=True,
                        transition_time=5,
                        loop=plot_loop  # Apply the custom GeoJSON options
                    ).add_to(my_render_map)
                else:
                    # link thre name with trajectories
                    all_features.extend(features)

            # Point Set Mapping:
            # point_set = folium.FeatureGroup(name="My Point Set")
            #
            # # Define the coordinates of your points
            # point_coordinates = [(latitude1, longitude1), (latitude2, longitude2), (latitude3, longitude3)]
            #
            # # Add points to the FeatureGroup
            # for coord in point_coordinates:
            #     folium.CircleMarker(location=coord, radius=6, color='blue', fill=True, fill_color='blue').add_to(
            #         point_set)
            #
            # # Add the FeatureGroup to the map
            # point_set.add_to(my_render_map)

            # Create a single TimestampedGeoJson with all features
            TimestampedGeoJson(
                {
                    "type": "FeatureCollection",
                    "features": all_features,
                },
                period="PT5S",  # Adjust the time interval as needed
                add_last_point=True,
                transition_time=5,
                loop=False  # Apply the custom GeoJSON options
            ).add_to(my_render_map)

            folium.LayerControl().add_to(my_render_map)
            # collect environment metric info and add it to folium
            info = self.collect_info()
            if self.dynamic_zero_shot:
                # the metric should include surveillance_aoi, response_delay, overall_aoi, not mean_aoi
                info_str = f"Energy: {info[ENERGY_METRIC_NAME]:.2f},<br>" \
                           f"Freshness: {info[FRESHNESS_FACTOR]:.2f},<br>" \
                           f"Surveillance: {info[SURVEILLANCE_METRIC]:.2f},<br>" \
                           f"Response Delay: {info[EMERGENCY_METRIC]:.2f},<br>" \
                           f"Valid Ratio: {info[VALID_HANDLING_RATIO]:.2f},<br>" \
                           f"Total Reward: {np.sum(self.agent_rewards_time_list):.2f}"
                # f"Overall AoI: {info[OVERALL_AOI]:.2f}"
            else:
                info_str = f"Energy Ratio: {info[ENERGY_METRIC_NAME]:.2f}, " \
                           f"Freshness: {info[FRESHNESS_FACTOR]:.4f}, " \
                           f"Coverage: {info[COVERAGE_METRIC_NAME]:.4f},<br>" \
                           f"Mean AoI: {info[AOI_METRIC_NAME]:.2f},<br>" \
                           f"Fresh Coverage: {info[MAIN_METRIC_NAME]:.2f}" \
                           f"Total Reward: {np.sum(self.agent_rewards_time_list):.2f}"
                # f"Data Collect: {info[DATA_METRIC_NAME]:.4f}, " \
            folium.map.Marker(
                [self.upper_right[1], self.upper_right[0]],
                icon=DivIcon(
                    icon_size=(180, 60),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size: 12pt">{info_str}</div>',
                )
            ).add_to(my_render_map)
            my_render_map.get_root().render()
            my_render_map.get_root().save(output_file)
            logging.info(f"{output_file} saved!")

    def xy_to_dataframe(self, aoi_list, energy_list, id_list, max_latitude, max_longitude, timestamp_list, x_list,
                        y_list):
        """
        Convert x, y to dataframe, which contains longitude, latitude, x, y, x_distance, y_distance, timestamp, aoi
        For later use in folium.
        """
        x_distance_list = x_list * self.max_distance_x / self.nlon + self.max_distance_x / self.nlon / 2
        y_distance_list = y_list * self.max_distance_y / self.nlat + self.max_distance_y / self.nlat / 2
        longitude_list = x_list * max_longitude / self.nlon + max_longitude / self.nlon / 2 + self.lower_left[0]
        latitude_list = y_list * max_latitude / self.nlat + max_latitude / self.nlat / 2 + self.lower_left[1]
        data = {"id": id_list, "longitude": longitude_list, "latitude": latitude_list,
                "x": x_list, "y": y_list, "x_distance": x_distance_list, "y_distance": y_distance_list,
                "timestamp": timestamp_list, "aoi": aoi_list}
        if energy_list is not None:
            data["energy"] = energy_list
        else:
            data['energy'] = -1
        robot_df = pd.DataFrame(data)
        robot_df['t'] = pd.to_datetime(robot_df['timestamp'], unit='s')  # s表示时间戳转换
        return robot_df


class CUDACrowdSim(CrowdSim, CUDAEnvironmentContext):
    """
    CUDA version of the CrowdSim environment.
    Note: this class subclasses the Python environment class CrowdSim,
    and also the CUDAEnvironmentContext
    """

    def history_reset(self):
        """
        Empty the history of the environment, inlcuding those related with metric tracking.
        """
        super().history_reset()

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        # Note, get_data_dictionary is called only once, so dummy values are allowed.
        # TODO: Note: valid status
        #  是否暂时移出游戏，目前valid的主要原因是lost connection，引入能耗之后可能会有电量耗尽
        data_dict = DataFeed()
        logging.debug("Creating data dictionary for CUDACrowdSim")
        # add all data with add_data_list method,
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
                                 ("agent_energy", self.float_dtype(np.full([self.num_agents, ], self.max_uav_energy)),
                                  True),
                                 ("agent_energy_range", self.float_dtype(self.max_uav_energy)),
                                 ("num_targets", self.int_dtype(self.num_sensing_targets)),
                                 ("num_agents_observed", self.int_dtype(self.num_agents_observed)),
                                 ("target_x", self.float_dtype(self.target_x_time_list), True),
                                 # [self.episode_length + 1, self.num_sensing_targets]
                                 ("target_y", self.float_dtype(self.target_y_time_list), True),
                                 # [self.episode_length + 1, self.num_sensing_targets]
                                 ("aoi_schedule", self.int_dtype(self.aoi_schedule)),
                                 ("emergency_per_gen", self.int_dtype(self.points_per_gen)),
                                 ("emergency_allocation_table", self.int_dtype(np.full([self.num_agents, ], -1)), True),
                                 ("target_aoi", self.int_dtype(np.ones([self.num_sensing_targets, ])), True),
                                 ("emergency_index", self.int_dtype(np.full(
                                     [self.num_agents, self.emergency_count], -1)), True),
                                 ("emergency_dis", self.float_dtype(np.zeros(
                                     [self.num_agents, self.emergency_count])), True),
                                 ("emergency_dis_to_target_index", self.int_dtype(np.zeros(
                                     [self.num_agents, ])), True),
                                 ("emergency_dis_to_target", self.float_dtype(np.zeros(
                                     [self.num_agents, ])), True),
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
                                 ("max_distance_x", self.int_dtype(self.max_distance_x)),
                                 ("max_distance_y", self.int_dtype(self.max_distance_y)),
                                 ("slot_time", self.float_dtype(self.step_time)),
                                 ("agent_speed", self.int_dtype(list(self.agent_speed.values()))),
                                 ("dynamic_zero_shot", self.int_dtype(self.dynamic_zero_shot)),
                                 ("zero_shot_start", self.int_dtype(self.zero_shot_start)),
                                 ("single_type_agent", self.int_dtype(self.single_type_agent)),
                                 ("agents_over_range", self.bool_dtype(np.zeros([self.num_agents, ])), True),
                                 ])
        # WARNING: single bool value seems to fail pycuda.
        return data_dict

    def step(self, actions=None) -> Tuple[Dict, Dict]:
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
            "aoi_schedule",
            "emergency_per_gen",
            "emergency_allocation_table",
            "target_aoi",
            "emergency_index",
            "emergency_dis",
            "emergency_dis_to_target_index",
            "emergency_dis_to_target",
            "target_coverage",
            "valid_status",
            "neighbor_agent_ids",
            "car_sensing_range",
            "drone_sensing_range",
            "drone_car_comm_range",
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
            "dynamic_zero_shot",
            "zero_shot_start",
            "single_type_agent",
            "agents_over_range",
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
        # self.global_state = self.float_dtype(self.cuda_data_manager.pull_data_from_device(_STATE))
        self.timestep = int(self.cuda_data_manager.pull_data_from_device("_timestep_")[0])
        logging.debug(f"Timestep in CUDACrowdSim {self.timestep}")
        logging.debug(f"Dones in CUDACrowdSim {dones[:5]}")
        if not self.dynamic_zero_shot:
            self.target_coveraged_timelist[self.timestep, :] = self.cuda_data_manager.pull_data_from_device(
                "target_coverage")[0]
        self.target_aoi_timelist[self.timestep, :] = self.cuda_data_manager.pull_data_from_device(
            "target_aoi")[0]
        self.agent_energy_timelist[self.timestep, :] = self.cuda_data_manager.pull_data_from_device(
            _AGENT_ENERGY)[0]
        info = self.collect_info() if all(dones) else {}
        return dones, info


def get_space(new_space: spaces.Dict, obs_example: Union[dict, np.ndarray], update_key: str):
    if isinstance(obs_example, np.ndarray):
        new_space[update_key] = binary_search_bound(obs_example)
    elif isinstance(obs_example, dict):
        new_space[update_key] = spaces.Dict()
        for key, value in obs_example.items():
            if len(value.shape) != 1:
                # 2d or 3d observation space
                new_space[update_key][key] = Box(low=np.full_like(value, -np.inf),
                                                 high=np.full_like(value, np.inf))
            else:
                new_space[update_key][key] = binary_search_bound(value)
        logging.debug(new_space)
    return new_space


class RLlibCUDACrowdSim(MultiAgentEnv):
    def __init__(self, run_config: dict):
        super().__init__()

        logging.debug("received run_config: %s", pprint.pformat(run_config))
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self.eval_interval = 5
        self.evaluate_count_down: int = self.eval_interval
        requirements = ["env_params", "trainer"]
        for requirement in requirements:
            if requirement not in run_config:
                raise Exception(f"RLlibCUDACrowdSim expects '{requirement}' in run_config")
        # assert 'env_registrar' in run_config['env_params'], 'env_registrar must be specified in env_params'
        assert 'env_config' in run_config['env_params'], 'env_config must be specified in env_params'
        additional_params = run_config["env_params"]
        logging.debug("additional_params: " + pprint.pformat(additional_params))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(additional_params.get("gpu_id", 0))
        torch.cuda.current_device()
        torch.cuda._initialized = True
        for item in user_override_params:
            run_config[item] = additional_params[item]
        self.trainer_params = run_config["trainer"]
        self.render_file_name = additional_params.get("render_file_name", None)
        self.logging_config = additional_params.get("logging_config", None)
        self.centralized = additional_params.get("centralized", False)
        self.is_render = additional_params.get("render", False)
        self.is_local = additional_params.get("local_mode", False)
        self.env_registrar = EnvironmentRegistrar()
        if "mock" not in additional_params:
            self.env_registrar.add_cuda_env_src_path(CUDACrowdSim.name,
                                                     os.path.join(
                                                         get_project_root(),
                                                         "envs",
                                                         "crowd_sim",
                                                         "crowd_sim_step.cu"
                                                     )
                                                     )
        train_batch_size = self.trainer_params['train_batch_size']
        self.num_envs = self.trainer_params['num_envs']
        num_mini_batches = self.trainer_params['num_mini_batches']
        if (train_batch_size // self.num_envs) < num_mini_batches:
            logging.error("Batch per env must be larger than num_mini_batches, Exiting...")
            return
        # Using dictionary comprehension to exclude specific keys
        new_dict = {k: v for k, v in run_config.items() if k not in excluded_keys}
        logging.debug(new_dict)
        self.env = CUDACrowdSim(**new_dict)
        self.float_dtype = self.env.float_dtype
        self.use_2d_state = self.env.use_2d_state
        self.env_backend = self.env.env_backend = "pycuda"
        self.action_space: spaces.Space = next(iter(self.env.action_space.values()))
        # manually setting observation space
        self.agents = []
        if "mock" not in additional_params:
            if self.is_render:
                self.num_envs = 4
                warnings.warn("render=True, num_envs is always equal to 4, and user input is ignored.")
            elif self.is_local:
                # pass
                self.num_envs = 4
                warnings.warn("local_mode=True, num_envs is always equal to 4, and user input is ignored.")
            self.env_wrapper: CUDAEnvWrapper = CUDAEnvWrapper(
                self.env,
                num_envs=self.num_envs,
                env_backend=self.env_backend,
                env_registrar=self.env_registrar
            )
        else:
            self.env_wrapper = None
        for name in teams_name:
            self.agents += [f"{name}{i}" for i in range(getattr(self.env, "num_" + name.strip("_")))]
        self.num_agents = len(self.agents)
        logging.debug("Total Number of Agents: {}".format(self.num_agents))
        self.obs_ref = self.env.reset()[0]
        self.global_state = self.env.global_state
        self.observation_space = spaces.Dict()
        get_space(self.observation_space, self.obs_ref, "obs")
        try:
            if self.global_state is not None:
                state = self.global_state
                get_space(self.observation_space, state, "state")
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
            if self.use_2d_state:
                self.obs_vec_dim = self.env.observation_space[0][_VECTOR_STATE].shape[-1]
            else:
                self.obs_vec_dim = self.env.observation_space[0].shape[-1]
            self.state_vec_dim = self.env.vector_state_dim

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
        if self.env.dynamic_zero_shot and (not self.env.no_refresh):
            # One episode delays all newly generated points, that is, these points are prepared
            #  for the next episodes. But it should be fine.
            self.env.targets_regen()
            obs_for_agents = list(self.env.generate_observation_and_update_state().values())
            if isinstance(obs_for_agents[0], dict):
                obs_for_agents = [np.concatenate([array.ravel() for array in item.values()]) for item in obs_for_agents]
            new_obs = np.stack(obs_for_agents)
            zero_shot_start = self.env.zero_shot_start
            logging.debug("Modifying Emergency Points on CUDA!")
            points_x, points_y = (torch.tensor(self.env.target_x_time_list[:, zero_shot_start:]),
                                  torch.tensor(self.env.target_y_time_list[:, zero_shot_start:]))
            self.env_wrapper.cuda_data_manager.data_on_device_via_torch("target_x_at_reset")[:, :,
            zero_shot_start:] = points_x
            self.env_wrapper.cuda_data_manager.data_on_device_via_torch("target_y_at_reset")[:, :,
            zero_shot_start:] = points_y
            self.env_wrapper.cuda_data_manager.data_on_device_via_torch(_OBSERVATIONS + "_at_reset")[:] = (
                torch.tensor(new_obs).cuda())
        # current_observation shape [n_agent, dim_obs]
        current_observation = self.pull_vec_from_device_to_list(_OBSERVATIONS, self.obs_vec_dim)
        state_list = self.pull_vec_from_device_to_list(_STATE, self.env.vector_state_dim)
        # convert it into a dict of {agent_id: {"obs": observation}}
        obs_list = []
        for env_index in range(self.num_envs):
            obs_list.append(get_rllib_multi_agent_obs(current_observation[env_index],
                                                      state_list[env_index], self.agents))
        # WARN：The ordering is wrong, debug only. Reset Emergency Points

        return obs_list

    def pull_vec_from_device_to_list(self, name: str, split_dim) -> list[np.ndarray]:
        """
        Convenient Wrapper to pull data from device and convert it to a list of ndarray
        For later usage in ray.
        """
        my_vector = self.float_dtype(self.env_wrapper.cuda_data_manager.pull_data_from_device(name))
        if self.use_2d_state:
            all_vector = my_vector[..., :split_dim]
            if name == _STATE:
                shape = (self.num_envs, -1, grid_size, grid_size)
            else:
                shape = (self.num_envs, self.num_agents, -1, grid_size, grid_size)
            all_images = my_vector[..., split_dim:].reshape(*shape)
            my_vector = [{_VECTOR_STATE: all_vector[i], _IMAGE_STATE: all_images[i]}
                         for i in range(self.num_envs)]
        return my_vector

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
        if isinstance(self.action_space, MultiDiscrete):
            vectorized_actions = np.stack([np.array(list(action_dict.values())) for action_dict in actions])
        else:
            vectorized_actions = np.expand_dims(np.vstack([np.array(list(action_dict.values()))
                                                           for action_dict in actions]), axis=-1)
        # np.asarray(list(map(int, {1: 2, 2: 5, 3: 8}.values()))), 30% faster than
        # actions_ls = [int(actions[agent_id]) for agent_id in actions.keys()]
        # upload actions to device
        self.env_wrapper.cuda_data_manager.data_on_device_via_torch(_ACTIONS)[:] = (
            torch.tensor(vectorized_actions).cuda())
        dones, info = self.env_wrapper.step()
        # current_observation shape [n_envs, n_agent, dim_obs]
        next_obs = self.pull_vec_from_device_to_list(_OBSERVATIONS, self.obs_vec_dim)
        state_list = self.pull_vec_from_device_to_list(_STATE, self.env.vector_state_dim)

        if self.centralized:
            reward = np.repeat(self.env_wrapper.cuda_data_manager.pull_data_from_device(_GLOBAL_REWARD),
                               repeats=self.num_agents).reshape(-1, self.num_agents)
        else:
            reward = self.env_wrapper.cuda_data_manager.pull_data_from_device(_REWARDS)
        # convert observation to dict {EnvID: {AgentID: Action}...}
        # rewards_all_zero = np.all(reward == 0)
        obs_list, reward_list, info_list = [], [], []
        for env_index in range(self.num_envs):
            # if not rewards_all_zero and np.mean(reward[env_index]) == 0:
            #     print(f"Reward all zero in env {env_index}")
            one_obs, one_reward = get_rllib_obs_and_reward(self.agents, state_list[env_index],
                                                           next_obs[env_index], reward[env_index])
            obs_list.append(one_obs)
            reward_list.append(one_reward)
            info_list.append({})
        if self.evaluate_count_down <= 0:
            self.render()
        if all(dones):
            logging.debug("All OK!")
            if self.evaluate_count_down <= 0:
                self.evaluate_count_down = self.eval_interval
            else:
                self.evaluate_count_down -= 1
            log_env_metrics(info, self.evaluate_count_down)
        return obs_list, reward_list, dones, info_list

    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        new_dict = action_dict.copy()
        obs_list, reward_list, dones, info_list = self.vector_step([new_dict] * self.num_envs)
        return obs_list[0], reward_list[0], dones[0], info_list[0]

    def render(self, mode=None):
        logging.debug("render called")
        # add datetime to trajectory
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.env.render(f'{self.render_file_name}_{datetime_str}', True, False)

    def log_aoi_grid(self):
        pass
        # logging.debug(f"aoi_grid: {next_obs[0][_IMAGE_STATE][0][0]}, emergency_grid: {next_obs[0][_IMAGE_STATE][0][1]}")
        # if wandb.run is not None and self.evaluate_count_down % 50 == 0 and self.env.timestep % 20 == 0:
        #     for i, item in enumerate(next_obs[0][_IMAGE_STATE] if self.use_2d_state else next_obs[0]):
        #         if self.use_2d_state:
        #             aoi_grid = wandb.Image(item[0], caption=aoi_caption)
        #         else:
        #             aoi_grid = wandb.Image(item[self.obs_vec_dim: self.obs_vec_dim + grid_size * grid_size
        #                         ].reshape(1, grid_size, grid_size), caption=aoi_caption)
        #         wandb.log({f"grids/aoi_agent_{i}": aoi_grid})
        #     first_state = state_list[0]
        #     if self.use_2d_state:
        #         state_aoi = wandb.Image(first_state[_IMAGE_STATE][0], caption=state_aoi_caption)
        #         emergency_aoi = wandb.Image(first_state[_IMAGE_STATE][1], caption=emergency_caption)
        #     else:
        #         state_aoi = wandb.Image(first_state[self.state_vec_dim:self.state_vec_dim + grid_size * grid_size
        #                                 ].reshape(grid_size, grid_size), caption=state_aoi_caption)
        #         emergency_aoi = wandb.Image(first_state[self.state_vec_dim + grid_size * grid_size:
        #                                   ].reshape(grid_size, grid_size), caption=emergency_caption)
        #     wandb.log({"grids/state_aoi": state_aoi, "grids/emergency_aoi": emergency_aoi})
        # assert np.array_equal(next_obs[0][0][20 + grid_size * grid_size:],state_list[0][self.state_vec_dim + grid_size * grid_size:])
        # assert np.array_equal(next_obs[0][0][20 + grid_size * grid_size:],next_obs[0][-1][20 + grid_size * grid_size:])


def get_rllib_multi_agent_obs(current_observation, state, agents: list[Any]) -> MultiAgentDict:
    if isinstance(current_observation, np.ndarray):
        return {agent_name: {"obs": current_observation[i], "state": state} for i, agent_name in enumerate(agents)}
    elif isinstance(current_observation, dict):
        return {agent_name: {"obs": {key: value[i] for key, value in current_observation.items()}, "state": state}
                for i, agent_name in enumerate(agents)}


class RLlibCUDACrowdSimWrapper(VectorEnv):
    def __init__(self, env: Union[RLlibCUDACrowdSim, GroupAgentsWrapper]):
        if isinstance(env, GroupAgentsWrapper):
            # For IQL (No Grouping) Compatibility
            actual_env = env.env
            self.group_wrapper: GroupAgentsWrapper = env
        else:
            actual_env = env
            self.group_wrapper = None
        super().__init__(actual_env.observation_space, actual_env.action_space, actual_env.num_envs)
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
        logging.debug(actions[0])
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
            logging.debug(done_list[:3])
            logging.debug(reward_group_list[:3])
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
        logging.debug(f"resetting environment {index} called")
        if index == 0 or self.obs_at_reset is None:
            self.obs_at_reset = self.vector_reset()
        return self.obs_at_reset[index]

    def try_render_at(self, index: Optional[int] = None) -> \
            Optional[np.ndarray]:
        self.env.render()
        return np.zeros(1)

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
    if not logging_config:
        return
    wandb.init(project=PROJECT_NAME, name=logging_config['expr_name'], group=logging_config['group'],
               tags=[logging_config['dataset']] + logging_config['tag']
               if logging_config['tag'] is not None else [], dir=logging_config['logging_dir'],
               config=logging_config, resume=logging_config['resume'])
    # prefix = 'env/'
    define_metrics_crowdsim()


def define_metrics_crowdsim():
    for item in [COVERAGE_METRIC_NAME, DATA_METRIC_NAME, MAIN_METRIC_NAME, FRESHNESS_FACTOR]:
        wandb.define_metric(item, summary="max")
    for item in [AOI_METRIC_NAME, ENERGY_METRIC_NAME, SURVEILLANCE_METRIC, EMERGENCY_METRIC, OVERALL_AOI]:
        wandb.define_metric(item, summary="min")


def get_rllib_obs_and_reward(agents: list[Any], state: Union[np.ndarray, dict],
                             obs: Union[np.ndarray, dict[int, np.ndarray]],
                             reward: Union[dict[int, int], list[int]]) -> Tuple[Dict, Dict]:
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
            "state": state
        }
        if isinstance(obs, np.ndarray):
            obs_dict[key]["obs"] = obs[i]
        elif isinstance(obs, dict):
            obs_dict[key]["obs"] = {k: v[i] for k, v in obs.items()}
    return obs_dict, reward_dict


def log_env_metrics(info: dict, episode_count: int):
    # Create table data
    table_data = [["Metric Name", "Value"]]
    table_data.extend([[key, value] for key, value in info.items()])
    # Print the table
    logging.info(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    if wandb.run is not None:
        wandb.log(info)
        # wandb.log({'evaluation_count_down': episode_count})


class RLlibCrowdSim(MultiAgentEnv):
    def __init__(self, run_config: dict):
        # print("passed run_config is", run_config)
        super().__init__()
        self.episode_count: int = 0
        requirements = ["env_params", "trainer", "render_file_name"]
        for requirement in requirements:
            if requirement not in run_config:
                raise Exception(f"RLlibCUDACrowdSim expects '{requirement}' in run_config")
        # assert 'env_registrar' in run_config['env_params'], 'env_registrar must be specified in env_params'
        assert 'env_config' in run_config['env_params'], 'env_config must be specified in env_params'
        additional_params = run_config["env_params"]
        run_config['env_config'] = additional_params['env_config']
        self.logging_config = additional_params.get('logging_config', None)
        self.render_file_name = additional_params['render_file_name']
        new_dict = {k: v for k, v in run_config.items() if k not in excluded_keys}
        self.env = CrowdSim(**new_dict)
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
            log_env_metrics(info, self.episode_count)
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
