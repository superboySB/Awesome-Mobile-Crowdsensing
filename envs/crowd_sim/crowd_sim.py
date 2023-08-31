import copy
import heapq
import GPUtil
import numpy as np
import pandas as pd
import logging
import folium

from gym import spaces
from shapely.geometry import Point
from folium.plugins import TimestampedGeoJson, AntPath
from datasets.KAIST.env_config import BaseEnvConfig

from .utils import *
from .mdp import *

try:
    num_gpus_available = len(GPUtil.getAvailable())
    print(f"Inside covid19_env.py: {num_gpus_available} GPUs are available.")
    if num_gpus_available == 0:
        print("No GPUs found! Running the simulation on a CPU.")
    else:
        from warp_drive.utils.constants import Constants
        from warp_drive.utils.data_feed import DataFeed
        from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

        _OBSERVATIONS = Constants.OBSERVATIONS
        _ACTIONS = Constants.ACTIONS
        _REWARDS = Constants.REWARDS
except ModuleNotFoundError:
    print(
        "Warning: The 'WarpDrive' package is not found and cannot be used! "
        "If you wish to use WarpDrive, please run "
        "'pip install rl-warp-drive' first."
    )
except ValueError:
    print("No GPUs found! Running the simulation on a CPU.")


_LOC_X = "loc_x"
_LOC_Y = "loc_y"
_DIR = "direction"
_ENR = "energy"  # only in drones/cars
_AOI = "aoi"  # only in humans


class CrowdSim:
    """
    The game of tag on a continuous circular 2D space.
    There are some taggers trying to tag several runners.
    The taggers want to get as close as possible to the runner, while the runner
    wants to get as far away from them as possible.
    Once a runner is tagged, he exits the game if runner_exits_game_after_tagged is True
    otherwise he continues to run around (and the tagger can catch him again)
    """
    
    name = "CrowdSim"
    
    def __init__(
        self,
        num_aerial_agents=2,
        num_ground_agents=2,
        num_targets_observed = 50,
        seed=None,
        env_backend="cpu",
    ):
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)
        self.config = BaseEnvConfig()

        assert num_aerial_agents > 0
        self.num_aerial_agents = num_aerial_agents
        self.num_ground_agents = num_ground_agents
        self.num_agents = self.num_aerial_agents + self.num_ground_agents
        self.num_sensing_targets = self.config.env.human_num

        
        self.episode_length =  self.config.env.num_timestep
        self.step_time = self.config.env.step_time
        self.start_timestamp = self.config.env.start_timestamp
        self.end_timestamp = self.config.env.end_timestamp
        self.max_uav_energy = self.config.env.max_uav_energy

        self.nlon = self.config.env.nlon
        self.nlat = self.config.env.nlat
        self.grid_diagonal = np.sqrt(self.nlon**2 + self.nlat**2)
        self.lower_left = self.config.env.lower_left
        self.upper_right = self.config.env.upper_right
        self.human_df = pd.read_csv(self.config.env.dataset_dir)
        logging.info("Finished reading {} rows".format(len(self.human_df)))
        self.human_df['t'] = pd.to_datetime(self.human_df['timestamp'], unit='s')  # s表示时间戳转换
        self.human_df['aoi'] = -1  # 加入aoi记录aoi
        self.human_df['energy'] = -1  # 加入energy记录energy

        self.human_x_array = np.ones([self.episode_length + 1, ])


        # Correct the unique ids and timestamps based on user's clarification
        unique_ids = np.arange(0, self.num_sensing_targets)  # id from 0 to 91
        unique_timestamps = np.arange(self.start_timestamp, self.end_timestamp+self.step_time, self.step_time)  # timestamp from 1519894800 to 1519896600 with 15-second intervals

        # Initialize a new empty array
        id_to_index = {id: index for index, id in enumerate(unique_ids)}
        timestamp_to_index = {timestamp: index for index, timestamp in enumerate(unique_timestamps)}
        self.human_x_array_full = np.full((self.num_sensing_targets, self.episode_length + 1), np.nan)
        self.human_y_array_full = np.full((self.num_sensing_targets, self.episode_length + 1), np.nan)
        self.human_aoi_array_full = np.ones(self.num_sensing_targets, self.episode_length)
        
        # Fill the new array with data from the full DataFrame
        for _, row in self.human_df.iterrows():
            id_index = id_to_index.get(row['id'], None)
            timestamp_index = timestamp_to_index.get(row['timestamp'], None)
            if id_index is not None and timestamp_index is not None:
                self.human_x_array_full[id_index, timestamp_index] = row['x']
                self.human_y_array_full[id_index, timestamp_index] = row['y']
        
        x1 = self.human_x_array_full[:, :-1]
        y1 = self.human_y_array_full[:, :-1]
        x2 = self.human_x_array_full[:, 1:]
        y2 = self.human_y_array_full[:, 1:]
        self.human_theta_array_full = get_theta(x1, y1, x2, y2)

        # Check if there are any NaN values in the array
        assert np.isnan(self.human_x_array_full).any() is False
        assert np.isnan(self.human_y_array_full).any() is False
        assert np.isnan(self.human_theta_array_full).any() is False

        self.timestep = 0
        self.current_target_aoi_list = np.ones([self.num_sensing_targets, ])
        self.mean_aoi_timelist = np.ones([self.episode_length + 1, ])
        self.mean_aoi_timelist[self.timestep] = np.mean(self.current_target_aoi_list)
        self.agent_energy_timelist = np.zeros([self.episode_length + 1, self.num_agents])
        self.agent_x_timelist = np.zeros([self.episode_length + 1, self.num_agents])
        self.agent_y_timelist = np.zeros([self.episode_length + 1, self.num_agents])
        self.num_covered_human_timelist = np.zeros([self.episode_length, ])
        self.data_collection = 0

        # Seeding
        self.np_random = np.random
        if seed is not None:
            self.seed(seed)

        # Types and Status of vehicles
        self.agent_type = {}
        self.aerial_agents = {}
        self.ground_agents = {}
        for agent_id in range(self.num_agents):
            if agent_id < self.num_aerial_agents:
                self.agent_type[agent_id] = 1  # Drone
                self.aerial_agents[agent_id] = True
            else:
                self.agent_type[agent_id] = 0  # Car
                self.ground_agents[agent_id] = True


        self.starting_location_x = np.ones(self.num_agents) * self.nlon / 2
        self.starting_location_y = np.ones(self.num_agents) * self.nlat / 2

        # These will be set during reset (see below)
        self.timestep = None
        self.global_state = None

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.action_space = {
            agent_id: spaces.Discrete(np.int8(self.config.env.drone_action_space.shape[0]))
                if self.agent_type[agent_id] == 1 
                else spaces.Discrete(np.int8(self.config.env.car_action_space.shape[0]))
                    for agent_id in range(self.num_agents)
        }
        # Used in generate_observation()
        # When use_full_observation is True, then all the agents will have info of
        # all the other agents, otherwise, each agent will only have info of
        # its k-nearest agents (k = num_other_agents_observed)
        self.init_obs = None  # Will be set later in generate_observation()

        assert num_targets_observed <= self.num_sensing_targets
        self.num_targets_observed = num_targets_observed

        # Distance margin between agents for non-zero rewards
        # If a tagger is closer than this to a runner, the tagger
        # gets a positive reward, and the runner a negative reward
        self.drone_sensing_range = self.config.env.drone_sensing_range
        self.car_sensing_range = self.config.env.car_sensing_range
        self.drone_car_comm_range = self.config.env.drone_car_comm_range

        # Rewards and penalties
        self.energy_factor = self.config.env.energy_factor

        # These will also be set via the env_wrapper
        self.env_backend = env_backend

        # Copy drones dict for applying at reset (with limited energy reserve)
        self.drones_at_reset = copy.deepcopy(self.aerial_agents)

    
    def seed(self, seed=None):
        """
        Seeding the environment with a desired seed
        Note: this uses the code in
        https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        """
        self.np_random.seed(seed)
        return [seed]


    def compute_distance(self, entity1, entity2):
        """
        Note: 'compute_distance' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (compute_distance)
        is also part of the step() function!
        """
        return np.sqrt(
            (
                self.global_state[_LOC_X][self.timestep, entity1]
                - self.global_state[_LOC_X][self.timestep, entity2]
            )
            ** 2
            + (
                self.global_state[_LOC_Y][self.timestep, entity1]
                - self.global_state[_LOC_Y][self.timestep, entity2]
            )
            ** 2
        ).astype(self.float_dtype)

    def k_nearest_targets(self, agent_id, k):
        """
        Note: 'k_nearest_neighbors' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (k_nearest_neighbors)
        is also part of the step() function!
        """
        agent_ids_and_distances = []

        for ag_id in range(self.num_agents):
            if ag_id != agent_id:
                agent_ids_and_distances += [
                    (ag_id, self.compute_distance(agent_id, ag_id))
                ]
        k_nearest_neighbor_ids_and_distances = heapq.nsmallest(
            k, agent_ids_and_distances, key=lambda x: x[1]
        )

        return [
            item[0]
            for item in k_nearest_neighbor_ids_and_distances[
                : self.num_other_agents_observed
            ]
        ]

    def generate_observation(self):
        """
        Generate and return the observations for every agent.
        """
        obs = {}

        normalized_global_obs = None
        for feature in [
            (_LOC_X, self.grid_diagonal),
            (_LOC_Y, self.grid_diagonal),
            (_DIR, 2 * np.pi),
        ]:
            if normalized_global_obs is None:
                normalized_global_obs = (
                    self.global_state[feature[0]][self.timestep] / feature[1]
                )
            else:
                normalized_global_obs = np.vstack(
                    (
                        normalized_global_obs,
                        self.global_state[feature[0]][self.timestep] / feature[1],
                    )
                )
        agent_types = np.array(
            [self.agent_type[agent_id] for agent_id in range(self.num_agents)]
        )
        time = np.array([float(self.timestep) / self.episode_length])

        # use partial observation
        for agent_id in range(self.num_agents):
            if self.timestep == 0:
                # Set obs to all zeros
                obs_global_states = np.zeros(
                    (
                        normalized_global_obs.shape[0],
                        self.num_other_agents_observed,
                    )
                )
                obs_agent_types = np.zeros(self.num_other_agents_observed)

                # Form the observation
                self.init_obs = np.concatenate(
                    [
                        np.vstack(
                            (
                                obs_global_states,
                                obs_agent_types,
                            )
                        ).reshape(-1),
                        np.array([0.0]),  # time
                    ]
                )

            # Initialize obs to all zeros
            obs[agent_id] = self.init_obs

            nearest_neighbor_ids = self.k_nearest_neighbors(
                agent_id, k=self.num_other_agents_observed
            )
            # For the case when the number of remaining agent ids is fewer
            # than self.num_other_agents_observed (because agents have exited
            # the game), we also need to pad obs wih zeros
            obs_global_states = np.hstack(
                (
                    normalized_global_obs[:, nearest_neighbor_ids]
                    - normalized_global_obs[:, agent_id].reshape(-1, 1),
                    np.zeros(
                        (
                            normalized_global_obs.shape[0],
                            self.num_other_agents_observed
                            - len(nearest_neighbor_ids),
                        )
                    ),
                )
            )
            obs_agent_types = np.hstack(
                (
                    agent_types[nearest_neighbor_ids],
                    np.zeros(
                        (
                            self.num_other_agents_observed
                            - len(nearest_neighbor_ids)
                        )
                    ),
                )
            )

            # Form the observation
            obs[agent_id] = np.concatenate(
                [
                    np.vstack(
                        (
                            obs_global_states,
                            obs_agent_types,
                        )
                    ).reshape(-1),
                    time,
                ]
            )

        return obs

    def compute_reward(self):
        """
        Compute and return the rewards for each agent.
        """
        # Initialize rewards
        rew = {agent_id: 0.0 for agent_id in range(self.num_agents)}

        taggers_list = sorted(self.taggers)

        # At least one runner present
        if self.num_runners > 0:
            runners_list = sorted(self.runners)
            runner_locations_x = self.global_state[_LOC_X][self.timestep][runners_list]
            tagger_locations_x = self.global_state[_LOC_X][self.timestep][taggers_list]

            runner_locations_y = self.global_state[_LOC_Y][self.timestep][runners_list]
            tagger_locations_y = self.global_state[_LOC_Y][self.timestep][taggers_list]

            runners_to_taggers_distances = np.sqrt(
                (
                    np.repeat(runner_locations_x, self.num_taggers)
                    - np.tile(tagger_locations_x, self.num_runners)
                )
                ** 2
                + (
                    np.repeat(runner_locations_y, self.num_taggers)
                    - np.tile(tagger_locations_y, self.num_runners)
                )
                ** 2
            ).reshape(self.num_runners, self.num_taggers)

            min_runners_to_taggers_distances = np.min(
                runners_to_taggers_distances, axis=1
            )
            argmin_runners_to_taggers_distances = np.argmin(
                runners_to_taggers_distances, axis=1
            )
            nearest_tagger_ids = [
                taggers_list[idx] for idx in argmin_runners_to_taggers_distances
            ]

        # Rewards
        # Add edge hit reward penalty and the step rewards/ penalties
        for agent_id in range(self.num_agents):
            rew[agent_id] += self.edge_hit_reward_penalty[agent_id]
            rew[agent_id] += self.step_rewards[agent_id]

        for idx, runner_id in enumerate(runners_list):
            if min_runners_to_taggers_distances[idx] < self.distance_margin_for_reward:

                # the runner is tagged!
                rew[runner_id] += self.tag_penalty_for_runner
                rew[nearest_tagger_ids[idx]] += self.tag_reward_for_tagger


        if self.timestep == self.episode_length:
            for runner_id in self.runners:
                rew[runner_id] += self.end_of_game_reward_for_runner

        return rew

    def reset(self):
        """
        Env reset().
        """
        # Reset time to the beginning
        self.timestep = 0

        # Re-initialize the global state
        self.global_state = {}
        self.set_global_state(
            key=_LOC_X, value=self.starting_location_x, t=self.timestep
        )
        self.set_global_state(
            key=_LOC_Y, value=self.starting_location_y, t=self.timestep
        )
        self.set_global_state(key=_SP, value=self.starting_speeds, t=self.timestep)
        self.set_global_state(key=_DIR, value=self.starting_directions, t=self.timestep)
        self.set_global_state(
            key=_ACC, value=self.starting_accelerations, t=self.timestep
        )


        # Penalty for hitting the edges
        self.edge_hit_reward_penalty = np.zeros(self.num_agents, dtype=self.float_dtype)

        # Reinitialize some variables that may have changed during previous episode
        self.runners = copy.deepcopy(self.runners_at_reset)
        self.num_runners = len(self.runners)

        return self.generate_observation()

    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        self.timestep += 1
        assert self.env_backend == "cpu"
        assert isinstance(actions, dict)
        assert len(actions) == self.num_agents

        acceleration_action_ids = [
            actions[agent_id][0] for agent_id in range(self.num_agents)
        ]
        turn_action_ids = [
            actions[agent_id][1] for agent_id in range(self.num_agents)
        ]

        assert all(
            0 <= acc <= self.num_acceleration_levels
            for acc in acceleration_action_ids
        )
        assert all(0 <= turn <= self.num_turn_levels for turn in turn_action_ids)

        delta_accelerations = self.acceleration_actions[acceleration_action_ids]
        delta_turns = self.turn_actions[turn_action_ids]

        # Update state and generate observation
        self.update_state(delta_accelerations, delta_turns)
        if self.env_backend == "cpu":
            obs = self.generate_observation()

        # Compute rewards and done
        rew = self.compute_reward()
        done = {
            "__all__": (self.timestep >= self.episode_length)
            or (self.num_runners == 0)
        }
        info = {}

        result = obs, rew, done, info
        return result

    def render(self, output_file=None, plot_loop=False, moving_line=False):
        import geopandas as gpd
        import movingpandas as mpd
        from movingpandas.geometry_utils import measure_distance_geodesic
        max_distance_x = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                    Point(self.upper_right[0], self.lower_left[1]))
        max_distance_y = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                    Point(self.lower_left[0], self.upper_right[1]))

        mixed_df = self.human_df.copy()

        # 可将机器人traj，可以载入到human的dataframe中，id从-1开始递减
        for i in range(self.num_agents):
            x_list = self.agent_x_timelist[:, i]
            y_list = self.agent_y_timelist[:, i]
            id_list = np.ones_like(x_list) * (-i - 1)
            aoi_list = np.ones_like(x_list) * (-1)
            energy_list = self.robot_energy_timelist[:, i]
            timestamp_list = [self.start_timestamp + i * self.step_time for i in range(self.num_timestep + 1)]
            x_distance_list = x_list * max_distance_x / self.nlon + max_distance_x / self.nlon / 2
            y_distance_list = y_list * max_distance_y / self.nlat + max_distance_y / self.nlat / 2
            max_longitude = abs(self.lower_left[0] - self.upper_right[0])
            max_latitude = abs(self.lower_left[1] - self.upper_right[1])
            longitude_list = x_list * max_longitude / self.nlon + max_longitude / self.nlon / 2 + self.lower_left[0]
            latitude_list = y_list * max_latitude / self.nlat + max_latitude / self.nlat / 2 + self.lower_left[1]

            data = {"id": id_list, "longitude": longitude_list, "latitude": latitude_list,
                    "x": x_list, "y": y_list, "x_distance": x_distance_list, "y_distance": y_distance_list,
                    "timestamp": timestamp_list, "aoi": aoi_list, "energy": energy_list}
            robot_df = pd.DataFrame(data)
            robot_df['t'] = pd.to_datetime(robot_df['timestamp'], unit='s')  # s表示时间戳转换
            mixed_df = mixed_df.append(robot_df)

        # ------------------------------------------------------------------------------------
        # 建立moving pandas轨迹，也可以选择调用高级API继续清洗轨迹。
        mixed_gdf = gpd.GeoDataFrame(mixed_df, geometry=gpd.points_from_xy(mixed_df.longitude, mixed_df.latitude),
                                        crs=4326)
        mixed_gdf = mixed_gdf.set_index('t').tz_localize(None)  # tz=time zone, 以本地时间为准
        mixed_gdf = mixed_gdf.sort_values(by=["id", "t"], ascending=[True, True])
        trajs = mpd.TrajectoryCollection(mixed_gdf, 'id')
        # trajs = mpd.MinTimeDeltaGeneralizer(trajs).generalize(tolerance=timedelta(minutes=1))
        # for index, traj in enumerate(trajs.trajectories):
        #     print(f"id: {trajs.trajectories[index].df['id'][0]}"
        #           + f"  size:{trajs.trajectories[index].size()}")

        start_point = trajs.trajectories[0].get_start_location()

        # 经纬度反向
        m = folium.Map(location=[start_point.y, start_point.x], tiles="cartodbpositron", zoom_start=14, max_zoom=24)

        m.add_child(folium.LatLngPopup())
        minimap = folium.plugins.MiniMap()
        m.add_child(minimap)
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('Stamen Toner').add_to(m)
        folium.TileLayer('cartodbpositron').add_to(m)
        folium.TileLayer('OpenStreetMap').add_to(m)

        # 锁定范围
        grid_geo_json = get_border(self.upper_right, self.lower_left)
        color = "red"
        border = folium.GeoJson(grid_geo_json,
                                style_function=lambda feature, color=color: {
                                    'fillColor': color,
                                    'color': "black",
                                    'weight': 2,
                                    'dashArray': '5,5',
                                    'fillOpacity': 0,
                                })
        m.add_child(border)

        for index, traj in enumerate(trajs.trajectories):
            name = f"UAV {index}" if index < self.robot_num else f"Human {traj.df['id'][0]}"  # select human
            # name = f"UAV {index}" if index < robot_num else f"Human {index - robot_num}"
            randr = lambda: np.random.randint(0, 255)
            color = '#%02X%02X%02X' % (randr(), randr(), randr())  # black

            # point
            features = traj_to_timestamped_geojson(index, traj, self.robot_num, color)
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
            if index < self.robot_num:
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

        if self.config.env.tallest_locs is not None:
            # 绘制正方形
            for tallest_loc in self.config.env.tallest_locs:
                # folium.Rectangle(
                #     bounds=[(tallest_loc[0] + 0.00025, tallest_loc[1] + 0.0003),
                #             (tallest_loc[0] - 0.00025, tallest_loc[1] - 0.0003)],  # 解决经纬度在地图上的尺度不一致
                #     color="black",
                #     fill=True,
                # ).add_to(m)
                icon_square = folium.plugins.BeautifyIcon(
                    icon_shape='rectangle-dot',
                    border_color='red',
                    border_width=8,
                )
                folium.Marker(location=[tallest_loc[0], tallest_loc[1]],
                                popup=folium.Popup(html=f'<p>raw coord: ({tallest_loc[1]},{tallest_loc[0]})</p>'),
                                tooltip='High-rise building',
                                icon=icon_square).add_to(m)

        m.get_root().render()
        m.get_root().save(output_file)
        logging.info(f"{output_file} saved!")


class CUDACrowdSim(CrowdSim, CUDAEnvironmentContext):
    """
    CUDA version of the TagGridWorld environment.
    Note: this class subclasses the Python environment class TagGridWorld,
    and also the  CUDAEnvironmentContext
    """
    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        data_dict = DataFeed()
        for feature in [_LOC_X, _LOC_Y, _SP, _DIR, _ACC]:
            data_dict.add_data(
                name=feature,
                data=self.global_state[feature][0],
                save_copy_and_apply_at_reset=True,
            )
        data_dict.add_data(
            name="agent_types",
            data=[self.agent_type[agent_id] for agent_id in range(self.num_agents)],
        )
        data_dict.add_data(
            name="num_runners", data=self.num_runners, save_copy_and_apply_at_reset=True
        )
        data_dict.add_data(
            name="num_other_agents_observed", data=self.num_other_agents_observed
        )
        data_dict.add_data(name="grid_length", data=self.grid_length)
        data_dict.add_data(
            name="edge_hit_reward_penalty",
            data=self.edge_hit_reward_penalty,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="step_rewards",
            data=self.step_rewards,
        )
        data_dict.add_data(name="edge_hit_penalty", data=self.edge_hit_penalty)
        data_dict.add_data(name="max_speed", data=self.max_speed)
        data_dict.add_data(name="acceleration_actions", data=self.acceleration_actions)
        data_dict.add_data(name="turn_actions", data=self.turn_actions)
        data_dict.add_data(name="skill_levels", data=self.skill_levels)
        data_dict.add_data(
            name="distance_margin_for_reward", data=self.distance_margin_for_reward
        )
        data_dict.add_data(
            name="tag_reward_for_tagger", data=self.tag_reward_for_tagger
        )
        data_dict.add_data(
            name="tag_penalty_for_runner", data=self.tag_penalty_for_runner
        )
        data_dict.add_data(
            name="end_of_game_reward_for_runner",
            data=self.end_of_game_reward_for_runner,
        )
        data_dict.add_data(
            name="neighbor_distances",
            data=np.zeros((self.num_agents, self.num_agents - 1), dtype=np.float32),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="neighbor_ids_sorted_by_distance",
            data=np.zeros((self.num_agents, self.num_agents - 1), dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="nearest_neighbor_ids",
            data=np.zeros(
                (self.num_agents, self.num_other_agents_observed), dtype=np.int32
            ),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="runner_exits_game_after_tagged",
            data=self.agent_exits_game_after_run_out_of_energy,
        )
        return data_dict
    

    def step(self, actions=None):
        self.timestep += 1
        args = [
            _LOC_X,
            _LOC_Y,
            _SP,
            _DIR,
            _ACC,
            "agent_types",
            "edge_hit_reward_penalty",
            "edge_hit_penalty",
            "grid_length",
            "acceleration_actions",
            "turn_actions",
            "max_speed",
            "num_other_agents_observed",
            "skill_levels",
            "runner_exits_game_after_tagged",
            _OBSERVATIONS,
            _ACTIONS,
            "neighbor_distances",
            "neighbor_ids_sorted_by_distance",
            "nearest_neighbor_ids",
            _REWARDS,
            "step_rewards",
            "num_runners",
            "distance_margin_for_reward",
            "tag_reward_for_tagger",
            "tag_penalty_for_runner",
            "end_of_game_reward_for_runner",
            "_done_",
            "_timestep_",
            ("n_agents", "meta"),
            ("episode_length", "meta"),
        ]
        if self.env_backend == "pycuda":
            self.cuda_step(
                *self.cuda_step_function_feed(args),
                block=self.cuda_function_manager.block,
                grid=self.cuda_function_manager.grid,
            )
        else:
            raise Exception("CUDATagGridWorld expects env_backend = 'pycuda' ")
