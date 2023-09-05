import numpy as np
import pandas as pd
import logging
import folium
import copy

from gym import spaces
from shapely.geometry import Point
from folium.plugins import TimestampedGeoJson, AntPath
from datasets.KAIST.env_config import BaseEnvConfig
# from datasets.Sanfrancisco.env_config import BaseEnvConfig
from .utils import *


from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS

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
        num_targets_observed = 40,
        seed=None,
        env_backend="cpu",
    ):
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)
        self.config = BaseEnvConfig()

        self.num_aerial_agents = num_aerial_agents
        self.num_ground_agents = num_ground_agents
        self.num_agents = self.num_aerial_agents + self.num_ground_agents
        self.num_sensing_targets = self.config.env.human_num

        
        self.episode_length =  self.config.env.num_timestep
        self.step_time = self.config.env.step_time
        self.start_timestamp = self.config.env.start_timestamp
        self.end_timestamp = self.config.env.end_timestamp
        

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
        unique_timestamps = np.arange(self.start_timestamp, self.end_timestamp+self.step_time, self.step_time)  
        id_to_index = {id: index for index, id in enumerate(unique_ids)}
        timestamp_to_index = {timestamp: index for index, timestamp in enumerate(unique_timestamps)}
        self.target_x_timelist = np.full([self.episode_length + 1, self.num_sensing_targets], np.nan)
        self.target_y_timelist = np.full([self.episode_length + 1, self.num_sensing_targets], np.nan)
        self.target_aoi_timelist = np.ones([self.episode_length + 1, self.num_sensing_targets])
        self.target_coveraged_timelist = np.zeros([self.episode_length, self.num_sensing_targets])
        
        # Fill the new array with data from the full DataFrame
        for _, row in self.human_df.iterrows():
            id_index = id_to_index.get(row['id'], None)
            timestamp_index = timestamp_to_index.get(row['timestamp'], None)
            if id_index is not None and timestamp_index is not None:
                self.target_x_timelist[timestamp_index, id_index] = row['x']
                self.target_y_timelist[timestamp_index, id_index] = row['y']
            else:
                print(row)
        
        x1 = self.target_x_timelist[:-1,:]
        y1 = self.target_y_timelist[:-1,:]
        x2 = self.target_x_timelist[1:,:]
        y2 = self.target_y_timelist[1:,:]
        self.target_theta_timelist = self.get_theta(x1, y1, x2, y2)
        self.target_theta_timelist = self.float_dtype(np.vstack([self.target_theta_timelist, self.target_theta_timelist[-1,:]]))

        # Check if there are any NaN values in the array
        assert not np.isnan(self.target_x_timelist).any()
        assert not np.isnan(self.target_y_timelist).any() 
        assert not np.isnan(self.target_theta_timelist).any()

        # agent infos
        self.timestep = 0
        self.starting_location_x = self.nlon / 2
        self.starting_location_y = self.nlat / 2
        self.max_uav_energy = self.config.env.max_uav_energy
        self.agent_energy_timelist = np.ones([self.episode_length + 1, self.num_agents], dtype=float) * self.max_uav_energy
        self.agent_x_timelist = np.ones([self.episode_length + 1, self.num_agents], dtype=float) * self.starting_location_x 
        self.agent_y_timelist = np.ones([self.episode_length + 1, self.num_agents], dtype=float) * self.starting_location_y
        self.data_collection = 0

        # Seeding
        self.np_random = np.random
        if seed is not None:
            self.seed(seed)

        # Types and Status of vehicles
        self.agent_types = self.int_dtype(np.ones([self.num_agents,]))
        self.cars = {}
        self.drones = {}
        for agent_id in range(self.num_agents):
            if agent_id < self.num_ground_agents:
                self.agent_types[agent_id] = 0  # Car
                self.cars[agent_id] = True
            else:
                self.agent_types[agent_id] = 1  # Drone
                self.drones[agent_id] = True


        # These will be set during reset (see below)
        self.timestep = None

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.drone_action_space_dx = self.float_dtype(self.config.env.drone_action_space[:,0])
        self.drone_action_space_dy = self.float_dtype(self.config.env.drone_action_space[:,1])
        self.car_action_space_dx = self.float_dtype(self.config.env.car_action_space[:,0])
        self.car_action_space_dy = self.float_dtype(self.config.env.car_action_space[:,1])
        self.action_space = {
            agent_id: spaces.Discrete(self.int_dtype(self.drone_action_space_dx.shape[0]))
                if self.agent_types[agent_id] == 1 
                else spaces.Discrete(self.int_dtype(self.car_action_space_dx.shape[0]))
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
        self.drone_sensing_range = self.float_dtype(self.config.env.drone_sensing_range)
        self.car_sensing_range = self.float_dtype(self.config.env.car_sensing_range)
        self.drone_car_comm_range = self.float_dtype(self.config.env.drone_car_comm_range)
        
        self.global_distance_matrix = np.full((self.num_agents,self.num_agents+self.num_sensing_targets),np.nan)

        # Rewards and penalties
        self.energy_factor = self.config.env.energy_factor

        # These will also be set via the env_wrapper
        self.env_backend = env_backend

        # [may not necessary] Copy drones dict for applying at reset (with limited energy reserve)
        # self.drones_at_reset = copy.deepcopy(self.drones)

    def get_theta(self, x1, y1, x2, y2):
        dx = x2 -x1
        dy = y2 - y1
        # 使用 arctan2 计算角度
        angles = np.arctan2(dy, dx)

        # 将角度转换为 0 到 2π 之间
        angles = np.mod(angles + 2 * np.pi, 2 * np.pi)
        return angles
    
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
            self.agent_x_timelist[self.timestep,agent_id] = self.starting_location_x
            self.agent_y_timelist[self.timestep,agent_id] = self.starting_location_y
            self.agent_energy_timelist[self.timestep,agent_id] = self.max_uav_energy
        
        for target_id in range(self.num_sensing_targets):
            self.target_aoi_timelist[self.timestep,target_id] = 1

        # reset global distance matrix
        self.calculate_global_distance_matrix()

        # for logging
        self.data_collection = 0
        self.target_coveraged_timelist = np.zeros([self.episode_length,self.num_sensing_targets])
        
        return self.generate_observation()
    
    def calculate_global_distance_matrix(self):
        for entity_id_1 in range(self.num_agents):
            for entity_id_2 in range(entity_id_1, self.num_agents+self.num_sensing_targets):
                if entity_id_1 == entity_id_2:
                    self.global_distance_matrix[entity_id_1,entity_id_2] = 0
                    continue

                if entity_id_1 < self.num_agents and entity_id_2 < self.num_agents:
                    self.global_distance_matrix[entity_id_1,entity_id_2] = np.sqrt((self.agent_x_timelist[self.timestep, entity_id_1]
                                                 - self.agent_x_timelist[self.timestep, entity_id_2])**2
                                            + (self.agent_y_timelist[self.timestep, entity_id_1]
                                                - self.agent_y_timelist[self.timestep, entity_id_2])**2)
                    self.global_distance_matrix[entity_id_2,entity_id_1] = self.global_distance_matrix[entity_id_1,entity_id_2]
                
                if entity_id_1 < self.num_agents and entity_id_2 >= self.num_agents:
                    self.global_distance_matrix[entity_id_1,entity_id_2] = np.sqrt((self.agent_x_timelist[self.timestep, entity_id_1] 
                                                - self.target_x_timelist[self.timestep, entity_id_2-self.num_agents])**2
                                            + (self.agent_y_timelist[self.timestep, entity_id_1] 
                                               - self.target_y_timelist[self.timestep, entity_id_2-self.num_agents])**2)
        return 

    def generate_observation(self):
        """
        Generate and return the observations for every agent.
        """
        # global states
        agents_state = np.zeros([self.num_agents, 4])
        for agent_id in range(self.num_agents):
            agents_state[agent_id,:] = np.array([
                # self.agent_x_timelist[self.timestep,agent_id]/self.nlon,
                # self.agent_y_timelist[self.timestep,agent_id]/self.nlat,
                # self.agent_types[agent_id],
                # self.agent_energy_timelist[self.timestep,agent_id]/self.max_uav_energy,
                # for debug
                self.agent_x_timelist[self.timestep,agent_id],
                self.agent_y_timelist[self.timestep,agent_id],
                self.agent_types[agent_id],
                self.agent_energy_timelist[self.timestep,agent_id],
            ])
        
        targets_state = np.zeros([self.num_sensing_targets,4])
        for target_id in range(self.num_sensing_targets):
            targets_state[target_id,:] = np.array([
                # self.target_x_timelist[self.timestep,target_id]/self.nlon,
                # self.target_y_timelist[self.timestep,target_id]/self.nlat,
                # self.target_theta_timelist[self.timestep, target_id] / (2 * np.pi),
                # self.target_aoi_timelist[self.timestep,target_id]/self.episode_length,
                # for debug
                self.target_x_timelist[self.timestep,target_id],
                self.target_y_timelist[self.timestep,target_id],
                self.target_theta_timelist[self.timestep, target_id],
                self.target_aoi_timelist[self.timestep,target_id],
            ])
        
        # generate observation
        obs = {}
        agent_nearest_targets_ids = np.argsort(self.global_distance_matrix[:,self.num_agents:],axis=-1,kind='stable')[:,:self.num_targets_observed]
        for agent_id in range(self.num_agents):
            agent_obs_part = agents_state.reshape(-1)

            nearest_target_obs = targets_state[agent_nearest_targets_ids[agent_id]]
            target_obs_part = nearest_target_obs.reshape(-1)

            assert agent_obs_part.shape[0] == self.num_agents * 4
            assert target_obs_part.shape[0] == self.num_targets_observed * 4

            obs[agent_id] = np.hstack((agent_obs_part, target_obs_part))

        return obs
    
    def calculate_energy_consume(self, fly_time, hover_time):  
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
    
    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        self.timestep += 1
        assert isinstance(actions, dict)
        assert len(actions) == self.num_agents

        for agent_id in range(self.num_agents):
            
            # is_stopping = True if actions[agent_id] == 0 else False
            if self.agent_types[agent_id] == 0:
                dx, dy = self.car_action_space_dx[actions[agent_id]], self.car_action_space_dy[actions[agent_id]]
            else:
                dx, dy = self.drone_action_space_dx[actions[agent_id]], self.drone_action_space_dy[actions[agent_id]]
            # TODO: 暂缺车辆的能耗公式区分,不过目前也没有加充电桩啦，本来电量就不会耗尽
            # consume_energy = self.calculate_energy_consume(0, self.step_time) if is_stopping else self.calculate_energy_consume(self.step_time, 0)
            consume_energy = 0

            self.agent_x_timelist[self.timestep, agent_id] = self.agent_x_timelist[self.timestep-1,agent_id] + dx
            self.agent_y_timelist[self.timestep, agent_id] = self.agent_y_timelist[self.timestep-1,agent_id] + dy
            self.agent_energy_timelist[self.timestep, agent_id] = self.agent_energy_timelist[self.timestep-1,agent_id] - consume_energy

        self.calculate_global_distance_matrix()

        drone_nearest_car_id = np.argsort(self.global_distance_matrix[self.num_ground_agents:self.num_agents,:self.num_ground_agents], axis=-1, kind='stable')[:,0]
        drone_car_min_distance = self.global_distance_matrix[self.num_ground_agents:self.num_agents,:self.num_ground_agents][np.arange(self.num_aerial_agents),drone_nearest_car_id]
        target_nearest_agent_ids = np.argsort(self.global_distance_matrix[:,self.num_agents:], axis=0,kind='stable')
        target_nearest_agent_distances = self.global_distance_matrix[:,self.num_agents:][target_nearest_agent_ids,np.arange(self.num_sensing_targets)]

        rew = {agent_id: 0.0 for agent_id in range(self.num_agents)}
        for target_id in range(self.num_sensing_targets):
            increase_aoi_flag = True
            for agent_id, target_agent_distance in zip(target_nearest_agent_ids[:,target_id], target_nearest_agent_distances[:, target_id]):
                if self.agent_types[agent_id] == 0 \
                        and target_agent_distance <= self.drone_sensing_range:  # TODO：目前假设car和drone的sensing range相同，便于判断
                    increase_aoi_flag = False
                    rew[agent_id] += (self.target_aoi_timelist[self.timestep-1, target_id]-1)/self.episode_length
                    self.target_aoi_timelist[self.timestep, target_id] = 1
                    break

                if self.agent_types[agent_id] == 1 \
                        and target_agent_distance <= self.drone_sensing_range \
                        and drone_car_min_distance[agent_id-self.num_ground_agents] <= self.drone_car_comm_range:
                    increase_aoi_flag = False
                    rew[agent_id] += (self.target_aoi_timelist[self.timestep-1, target_id]-1)/self.episode_length
                    rew[drone_nearest_car_id[agent_id-self.num_ground_agents]] += (self.target_aoi_timelist[self.timestep-1, target_id]-1)/self.episode_length
                    self.target_aoi_timelist[self.timestep, target_id] = 1
                    break
                    
                if increase_aoi_flag:
                    self.target_aoi_timelist[self.timestep, target_id] = self.target_aoi_timelist[self.timestep-1, target_id] + 1
                else:
                    self.target_coveraged_timelist[self.timestep-1, target_id] = 1
                    self.data_collection += (self.target_aoi_timelist[self.timestep-1, target_id] - self.target_aoi_timelist[self.timestep, target_id])
        
        obs = self.generate_observation()
        
        done = {
            "__all__": (self.timestep >= self.episode_length)
        }

        self.data_collection += np.sum(self.target_aoi_timelist[self.timestep]-self.target_aoi_timelist[self.timestep-1])
        info = {"mean_aoi": np.mean(self.target_aoi_timelist[self.timestep]),
            "mean_energy_consumption": 1.0 - (
                        np.mean(self.agent_energy_timelist[self.timestep]) / self.max_uav_energy),
            "collected_data_ratio": self.data_collection/(self.episode_length*self.num_sensing_targets),
            "target_coverage": np.sum(self.target_coveraged_timelist) / (self.episode_length*self.num_sensing_targets)
        }

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
            energy_list = self.agent_energy_timelist[:, i]
            timestamp_list = [self.start_timestamp + i * self.step_time for i in range(self.episode_length + 1)]
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
            mixed_df = pd.concat([mixed_df,robot_df])

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
            if traj.df['id'][0]<0 and traj.df['id'][0]>=(-self.num_ground_agents):
                name = f"Agent {self.num_agents - index -1} (Car)"
            elif traj.df['id'][0]<(-self.num_ground_agents):
                name = f"Agent {self.num_agents - index -1} (Drone)"
            else:
                name = f"Human {traj.df['id'][0]}"

            randr = lambda: np.random.randint(0, 255)
            color = '#%02X%02X%02X' % (randr(), randr(), randr())  # black

            # point
            features = traj_to_timestamped_geojson(index, traj, self.num_ground_agents, self.num_aerial_agents, color)
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

        # if self.config.env.tallest_locs is not None:
        #     # 绘制正方形
        #     for tallest_loc in self.config.env.tallest_locs:
        #         # folium.Rectangle(
        #         #     bounds=[(tallest_loc[0] + 0.00025, tallest_loc[1] + 0.0003),
        #         #             (tallest_loc[0] - 0.00025, tallest_loc[1] - 0.0003)],  # 解决经纬度在地图上的尺度不一致
        #         #     color="black",
        #         #     fill=True,
        #         # ).add_to(m)
        #         icon_square = folium.plugins.BeautifyIcon(
        #             icon_shape='rectangle-dot',
        #             border_color='red',
        #             border_width=8,
        #         )
        #         folium.Marker(location=[tallest_loc[0], tallest_loc[1]],
        #                         popup=folium.Popup(html=f'<p>raw coord: ({tallest_loc[1]},{tallest_loc[0]})</p>'),
        #                         tooltip='High-rise building',
        #                         icon=icon_square).add_to(m)

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
        data_dict.add_data(name="agent_types",data=self.int_dtype(self.agent_types))
        data_dict.add_data(name="car_action_space_dx", data=self.float_dtype(self.car_action_space_dx))
        data_dict.add_data(name="car_action_space_dy", data=self.float_dtype(self.car_action_space_dy))
        data_dict.add_data(name="drone_action_space_dx", data=self.float_dtype(self.drone_action_space_dx))
        data_dict.add_data(name="drone_action_space_dy", data=self.float_dtype(self.drone_action_space_dy))
        data_dict.add_data(
            name="agent_x",
            data=self.float_dtype(np.ones([self.num_agents,]) * self.starting_location_x),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(name="agent_x_range", data=self.float_dtype(self.nlon))
        data_dict.add_data(
            name="agent_y",
            data=self.float_dtype(np.ones([self.num_agents,]) * self.starting_location_y),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(name="agent_y_range", data=self.float_dtype(self.nlat))
        
        data_dict.add_data(
            name="agent_energy",
            data=self.float_dtype(np.ones([self.num_agents,]) * self.max_uav_energy),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(name="agent_energy_range", data=self.float_dtype(self.max_uav_energy))
        data_dict.add_data(name="num_targets", data=self.int_dtype(self.num_sensing_targets))
        data_dict.add_data(name="num_targets_observed", data=self.int_dtype(self.num_targets_observed))
        data_dict.add_data(name="target_x", data=self.float_dtype(self.target_x_timelist))  # [self.episode_length + 1, self.num_sensing_targets]
        data_dict.add_data(name="target_y", data=self.float_dtype(self.target_y_timelist))  # [self.episode_length + 1, self.num_sensing_targets]
        data_dict.add_data(name="target_theta", data=self.float_dtype(self.target_theta_timelist))  # [self.episode_length + 1, self.num_sensing_targets]
        data_dict.add_data(name="target_aoi", 
                           data=self.float_dtype(np.ones([self.num_sensing_targets,])),
                           save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="valid_status",  # TODO:是否暂时移出游戏，目前valid的主要原因是lost connection，引入能耗之后可能会有电量耗尽
            data=self.int_dtype(np.ones([self.num_agents,])),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="neighbor_agent_ids", 
            data=self.int_dtype(np.ones([self.num_agents,])*(-1)),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(name="car_sensing_range", data=self.float_dtype(self.car_sensing_range))
        data_dict.add_data(name="drone_sensing_range", data=self.float_dtype(self.drone_sensing_range))
        data_dict.add_data(name="drone_car_comm_range", data=self.float_dtype(self.drone_car_comm_range))
        data_dict.add_data(
            name="neighbor_target_distances",
            data=self.float_dtype(np.zeros([self.num_agents, self.num_sensing_targets])),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="neighbor_target_ids_sorted",
            data=self.int_dtype(np.zeros([self.num_agents, self.num_sensing_targets])),
            save_copy_and_apply_at_reset=True,
        )


        return data_dict
    

    def step(self, actions=None):
        self.timestep += 1
        args = [
            _OBSERVATIONS,
            _ACTIONS,
            _REWARDS,
            "agent_types",
            "car_action_space_dx",
            "car_action_space_dy",
            "drone_action_space_dx",
            "drone_action_space_dy",
            "agent_x",
            "agent_x_range",
            "agent_y",
            "agent_y_range",
            "agent_energy",
            "agent_energy_range",
            "num_targets",
            "num_targets_observed",
            "target_x",
            "target_y",
            "target_theta",
            "target_aoi",
            "valid_status",
            "neighbor_agent_ids",
            "car_sensing_range",
            "drone_sensing_range",
            "drone_car_comm_range",
            "neighbor_target_distances",
            "neighbor_target_ids_sorted",
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
            raise Exception("CUDACrowdSim expects env_backend = 'pycuda' ")
