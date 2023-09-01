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
        self.lower_left = self.config.env.lower_left
        self.upper_right = self.config.env.upper_right
        self.human_df = pd.read_csv(self.config.env.dataset_dir)
        logging.info("Finished reading {} rows".format(len(self.human_df)))
        self.human_df['t'] = pd.to_datetime(self.human_df['timestamp'], unit='s')  # s表示时间戳转换
        self.human_df['aoi'] = -1  # 加入aoi记录aoi
        self.human_df['energy'] = -1  # 加入energy记录energy


        # human infos
        unique_ids = np.arange(0, self.num_sensing_targets)  # id from 0 to 91
        unique_timestamps = np.arange(self.start_timestamp, self.end_timestamp+self.step_time, self.step_time)  # timestamp from 1519894800 to 1519896600 with 15-second intervals
        id_to_index = {id: index for index, id in enumerate(unique_ids)}
        timestamp_to_index = {timestamp: index for index, timestamp in enumerate(unique_timestamps)}
        self.target_x_timelist = np.full([self.episode_length + 1, self.num_sensing_targets], np.nan)
        self.target_y_timelist = np.full([self.episode_length + 1, self.num_sensing_targets], np.nan)
        self.target_aoi_timelist = np.ones([self.episode_length + 1, self.num_sensing_targets])
        self.target_aoi_current = np.ones([self.num_sensing_targets, ])
        
        # Fill the new array with data from the full DataFrame
        for _, row in self.human_df.iterrows():
            id_index = id_to_index.get(row['id'], None)
            timestamp_index = timestamp_to_index.get(row['timestamp'], None)
            if id_index is not None and timestamp_index is not None:
                self.target_x_timelist[id_index, timestamp_index] = row['x']
                self.target_y_timelist[id_index, timestamp_index] = row['y']
        
        x1 = self.target_x_timelist[:-1,:]
        y1 = self.target_y_timelist[:-1,:]
        x2 = self.target_x_timelist[1:,:]
        y2 = self.target_y_timelist[1:,:]
        self.target_theta_timelist = self.get_theta(x1, y1, x2, y2)
        self.target_theta_timelist = np.vstack([self.target_theta_timelist, self.target_theta_timelist[-1,:]])

        # Check if there are any NaN values in the array
        assert np.isnan(self.target_x_timelist).any() is False
        assert np.isnan(self.target_y_timelist).any() is False
        assert np.isnan(self.target_theta_timelist).any() is False

        # agent infos
        self.timestep = 0
        self.mean_aoi_timelist = np.ones([self.episode_length + 1,])
        self.mean_aoi_timelist[self.timestep] = np.mean(self.target_aoi_current)
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
        self.agent_types = np.ones(self.num_agents,)
        for agent_id in range(self.num_agents):
            if agent_id < self.num_ground_agents:
                self.agent_types[agent_id] = 0  # Car
            else:
                self.agent_types[agent_id] = 1  # Drone


        self.starting_location_x = np.ones(self.num_agents) * self.nlon / 2
        self.starting_location_y = np.ones(self.num_agents) * self.nlat / 2

        # These will be set during reset (see below)
        self.timestep = None
        self.global_state = None

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.drone_action_space_dx = self.config.env.drone_action_space[:,0]
        self.drone_action_space_dy = self.config.env.drone_action_space[:,1]
        self.car_action_space_dx = self.config.env.car_action_space[:,0]
        self.car_action_space_dy = self.config.env.car_action_space[:,1]
        self.action_space = {
            agent_id: spaces.Discrete(np.int8(self.drone_action_space_dx.shape[0]))
                if self.agent_types[agent_id] == 1 
                else spaces.Discrete(np.int8(self.car_action_space_dx.shape[0]))
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

    def get_theta(self, x1, y1, x2, y2):
        ang1 = np.arctan2(y1, x1)
        ang2 = np.arctan2(y2, x2)
        theta = np.rad2deg((ang1 - ang2) % (2 * np.pi))
        return theta
    
    def seed(self, seed=None):
        """
        Seeding the environment with a desired seed
        Note: this uses the code in
        https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        """
        self.np_random.seed(seed)
        return [seed]

    def reset(self):
        """
        Env reset().
        """
        # Reset time to the beginning
        self.timestep = 0

        # Re-initialize the global state
        for agent_id in range(self.num_agents):
            self.agent_x_timelist[self.timestep,agent_id] = self.starting_location_x,
            self.agent_y_timelist[self.timestep,agent_id] = self.starting_location_y,
            self.agent_energy_timelist[self.timestep,agent_id] = self.max_uav_energy,
        
        for target_id in range(self.num_sensing_targets):
            self.target_aoi_timelist[self.timestep,target_id] = 1
            self.target_aoi_current = 1

        return self.generate_observation()
    
    def k_nearest_targets(self, agent_id, k):
        """
        Note: 'k_nearest_neighbors' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (k_nearest_neighbors)
        is also part of the step() function!
        """
        target_ids_and_distances = []
        for target_id in range(self.num_sensing_targets):
            agent_target_distance = np.sqrt(np.power(self.agent_x_timelist[self.timestep, agent_id] - self.target_x_timelist[self.timestep, target_id])
                                            + np.power(self.agent_y_timelist[self.timestep, agent_id] - self.target_y_timelist[self.timestep, target_id]))
            target_ids_and_distances += [target_id, agent_target_distance]

        k_nearest_target_ids_and_distances = heapq.nsmallest(
            k, target_ids_and_distances, key=lambda x: x[1]
        )

        return [item[0] for item in k_nearest_target_ids_and_distances[:k]]
    

    def generate_observation(self):
        """
        Generate and return the observations for every agent.
        """
        # global states
        agents_state = np.zeros(self.num_agents, 4)
        for agent_id in range(self.num_agents):
            agents_state[agent_id,:] = np.array([
                self.agent_x_timelist[self.timestep,agent_id]/self.nlon,
                self.agent_y_timelist[self.timestep,agent_id]/self.nlat,
                self.agent_types[agent_id],
                self.agent_energy_timelist[self.timestep,agent_id]/self.max_uav_energy,
            ])
        
        targets_state = np.zeros(self.num_sensing_targets,4)
        for target_id in range(self.num_sensing_targets):
            targets_state[target_id,:] = np.array([
                self.target_x_timelist[self.timestep,target_id]/self.nlon,
                self.target_y_timelist[self.timestep,target_id]/self.nlat,
                self.target_theta_timelist[self.timestep, target_id] / (2 * np.pi),
                self.target_aoi_timelist[self.timestep,target_id]/self.episode_length,
            ])
        
        # generate observation
        obs = {}
        for agent_id in range(self.num_agents):
            agent_obs_part = agents_state.reshape(-1)
            nearest_neighbor_ids = self.k_nearest_targets(agent_id, self.num_targets_observed)
            nearest_target_obs = targets_state[nearest_neighbor_ids]
            target_obs_part = nearest_target_obs.reshape(-1)

            assert agent_obs_part.shape[0] == self.num_agents * 4
            assert target_obs_part.shape[0] == self.num_targets_observed * 4

            obs[agent_id] = np.hstack((agent_obs_part, target_obs_part))

        return obs

    def nearest_car_and_distance(self, drone_id):
        car_ids_and_distances = []
        for car_id in range(self.num_ground_agents):
            agent_target_distance = np.sqrt(np.power(self.agent_x_timelist[self.timestep, drone_id] - self.target_x_timelist[self.timestep, car_id])
                                            + np.power(self.agent_y_timelist[self.timestep, drone_id] - self.target_y_timelist[self.timestep, car_id]))
            car_ids_and_distances += [car_id, agent_target_distance]

        k_nearest_car_ids_and_distances = heapq.nsmallest(
            1, car_ids_and_distances, key=lambda x: x[1]
        )

        return k_nearest_car_ids_and_distances[0][0], k_nearest_car_ids_and_distances[0][1]
    
    def compute_drone_reward(self, agent_id):
        pass

    def compute_car_reward(self, agent_id):
        pass

    def compute_reward(self):
        """
        Compute and return the rewards for each agent.
        """
        # Initialize rewards
        rew = {agent_id: 0.0 for agent_id in range(self.num_agents)}

        for agent_id in range(self.num_agents):
            if self.agent_types[agent_id] == 0:  # car
                pass
            else:  # drone
                pass
            _
        

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

        return rew

    
    def consume_uav_energy(self, fly_time, hover_time):
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
        Vt = self.config.env.velocity  # velocity of the UAV,m/s

        Power_flying = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
                    P1 * np.sqrt((np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
                    0.5 * d0 * rho * s0 * A * Vt ** 3

        Power_hovering = P0 + P1

        return fly_time * Power_flying + hover_time * Power_hovering
    
    def judge_aoi_update(self, human_position, robot_position):
        # TODO：判断是否可以更新aoi，可以将空地间协同也加进来
        should_reset = False
        for robot_id in range(tmp_config.env.robot_num):
            unit_distance = np.sqrt(np.power(robot_position[robot_id][0] - human_position[0], 2)
                                    + np.power(robot_position[robot_id][1] - human_position[1], 2))
            if unit_distance <= self.drone_sensing_range:
                should_reset = True
                break

        return should_reset
    
    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        self.timestep += 1
        assert isinstance(actions, dict)
        assert len(actions) == self.num_agents


        for agent_id in range(self.num_agents):
            # agent
            is_stopping = True if actions[agent_id] == 0 else False
            if self.agent_types[agent_id] == 0:
                dx, dy = self.car_action_space_dx[actions[agent_id]], self.car_action_space_dy[actions[agent_id]]
            else:
                dx, dy = self.drone_action_space_dx[actions[agent_id]], self.drone_action_space_dy[actions[agent_id]]
            consume_energy = self.consume_uav_energy(0, self.step_time) if is_stopping else self.consume_uav_energy(self.step_time, 0)

            self.agent_x_timelist[self.timestep, agent_id] = self.agent_x_timelist[self.timestep-1,agent_id] + dx
            self.agent_y_timelist[self.timestep, agent_id] = self.agent_y_timelist[self.timestep-1,agent_id] + dy
            self.agent_energy_timelist[self.timestep, agent_id] = self.agent_energy_timelist[self.timestep-1,agent_id] - consume_energy

            # target    
            self.target_x_timelist
            self.target_y_timelist
            self.target_theta_timelist
            self.target_aoi_timelist

            # TODO: 扁平化为一维数组

        
        obs = self.generate_observation()

        # Compute rewards and done
        rew = self.compute_reward()
        done = {
            "__all__": (self.timestep >= self.episode_length)
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
