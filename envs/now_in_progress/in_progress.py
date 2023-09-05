import os
import sys

from warp_drive.utils.common import get_project_root
from .util.config_3d import Config
from tqdm import tqdm
from .util import *
from .util.roadmap_utils import Roadmap,get_sub_graph

import numpy as np
import copy
import pickle
import math
import warnings
import osmnx as ox
import networkx as nx
import json
from gym import spaces
from collections import OrderedDict

np.seterr(all="raise")

class EnvUCS:
    def __init__(self, args=None, **kwargs):
        if args != None:
            self.args = args
        self.config = Config(args)
        self.SCALE = self.config("scale")
        self.INITIAL_ENERGY = self.config("initial_energy")
        self.EPSILON = self.config("epsilon")
        self.DEBUG_MODE = self.config("debug_mode")
        self.TEST_MODE = self.config("test_mode")
        self.MAX_EPISODE_STEP = self.config("max_episode_step")
        self._max_episode_steps = self.MAX_EPISODE_STEP
        self.TIME_SLOT = self.config("time_slot")
        self.SAVE_COUNT = self.config('seed')

        self.CONCAT_OBS = self.config("concat_obs")
        self.POI_INIT_DATA = self.config("poi_init_data")
        self.AOI_THRESHOLD = self.config("aoi_threshold")
        self.TOTAL_TIME = self.MAX_EPISODE_STEP * self.TIME_SLOT
        self.THRESHOLD_PENALTY = self.config("threshold_penalty")
        self.UAV_HEIGHT = self.config("uav_height")
        self.USER_DATA_AMOUNT = self.config("user_data_amount")
        self.CHANNEL_NUM = self.config("channel_num")

        self.UAV_TYPE = ['carrier', 'uav']
        self.NUM_UAV = OrderedDict(self.config("num_uav"))
        self.UAV_SPEED = OrderedDict(self.config("uav_speed"))
        self.RATE_THRESHOLD = OrderedDict(self.config("rate_threshold"))
        self.UPDATE_NUM = OrderedDict(self.config("update_num"))
        self.COLLECT_RANGE = OrderedDict(self.config("collect_range"))

        self.ACTION_ROOT = self.config("action_root")
        self.n_agents = sum(self.NUM_UAV.values())
        self.episode_limit = self._max_episode_steps
        self.n_actions = self.ACTION_ROOT
        self.agent_field = OrderedDict(self.config("agent_field"))
        self.reset_count = 0

        if self.config('dataset') =='KAIST':
            from ...datasets.KAIST.old_env_config import dataset_config
            self.broke_threshold = 100
            self.normal_threshold = 50
        else:
            raise NotImplementedError
        self.config.dict = {
            **self.config.dict,
            **dataset_config
        }
        
        self.MAP_X = self.config("map_x")
        self.MAP_Y = self.config("map_y")
        self.POI_NUM = self.config("poi_num")
        self._poi_position = np.array(self.config('poi'))
        self._poi_position[:, 0] *= self.MAP_X
        self._poi_position[:, 1] *= self.MAP_Y

        self._uav_energy = {this_uav: [self.config("initial_energy")[this_uav] for i in range(self.NUM_UAV[this_uav])]
                            for this_uav in
                            self.UAV_TYPE}
        self._uav_position = \
            {this_uav: [[self.config("init_position")[this_uav][i][0], self.config("init_position")[this_uav][i][1]] for
                        i in
                        range(self.NUM_UAV[this_uav])] for this_uav in self.UAV_TYPE}
        self.map = str(self.config("map"))
        print(f'selected map:{self.map}')
        
        self.rm = Roadmap(self.config("dataset"))
        with open(os.path.join(get_project_root(),"envs","mcs_data_collection",f"util/{self.config('dataset')}/road_map.json"),'r') as f:
            self.ROAD_MAP = json.load(f)
            self.ROAD_MAP = {key: set(value) for key, value in self.ROAD_MAP.items()}

        with open(os.path.join(get_project_root(),"envs","mcs_data_collection",f"util/{self.config('dataset')}/pair_dis_dict_0.json"),'r') as f:
            pairs_info = json.load(f)
            self.PAIR_DIS_DICT = pairs_info
            self.valid_nodes = set([int(item) for item in pairs_info['0'].keys()])
            self.valid_edges = {key: set() for key in self.ROAD_MAP.keys()}
            for key in tqdm(self.PAIR_DIS_DICT.keys(), desc='Constructing Edges'):
                for i in self.valid_nodes:
                    for j in self.valid_nodes:
                        if i == j:
                            continue
                        dis = pairs_info[key][str(i)][str(j)]
                        if key == '0' and dis <= self.normal_threshold:
                            self.valid_edges[key].add((i, j))
                        elif key != '0' and dis <= self.broke_threshold:
                            self.valid_edges[key].add((i, j))

            
            self.ALL_G = ox.load_graphml(os.path.join(get_project_root(),"envs","mcs_data_collection",f"util/{self.config('dataset')}/map_0.graphml")).to_undirected()

            self.node_map = {}
            for i, node in enumerate(self.ALL_G.nodes):
                self.node_map[str(node)] = i

            self.NX_G = {}
            for map_num, nodes_to_remove in self.ROAD_MAP.items():
                new_graph = nx.MultiDiGraph()
                new_graph.graph = self.ALL_G.graph
                new_graph.add_nodes_from(self.valid_nodes)
                for node in self.valid_nodes:
                    new_graph.nodes[node].update(self.ALL_G.nodes[node])
                new_graph.add_edges_from(self.valid_edges[map_num])
                new_graph = nx.convert_node_labels_to_integers(get_sub_graph(new_graph, nodes_to_remove),
                                                                first_label=0,
                                                                label_attribute='old_label')
                self.NX_G[map_num] = new_graph
                
                print(f"dataset:{self.config('dataset')},map:{map_num}, number of nodes:{len(new_graph.nodes())}, number of edges: {len(new_graph.edges())}")
                # print(len(new_graph.edges))
            self.OSMNX_TO_NX = {data['old_label']: node for node, data in
                                self.NX_G[self.map].nodes(data=True)}

            all_keys = self.ROAD_MAP.keys()
            for key in all_keys:
                for node, data in self.NX_G[key].nodes(data=True):
                    x, y = self.rm.lonlat2pygamexy(data['x'], data['y'])
                    self.NX_G[key].nodes[node]['py_x'] = x
                    self.NX_G[key].nodes[node]['py_y'] = y
                    self.NX_G[key].nodes[node]['data'] = 0
            self.EDGE_FEATURES = {key: np.array(list(self.NX_G[key].edges())).T for key in all_keys}

            poi_map = {key: self.get_node_poi_map(key, self.NX_G[key]) for key in all_keys}
            self.POI_NEAREST_DIS = {key: poi_map[key][1] for key in all_keys}
            self.NODE_TO_POI = {key: poi_map[key][0] for key in all_keys}

        self.ignore_node = []


        self.RATE_MAX = self._get_data_rate((0, 0), (0, 0))
        self.RATE_MIN = self._get_data_rate((0, 0), (self.MAP_X, self.MAP_Y))

        self.Power_flying = {}
        self.Power_hovering = {}
        self._get_energy_coefficient()

        self.noma_config = {
            'noise0_density': 5e-20,
            'bandwidth_subchannel': 40e6 / self.CHANNEL_NUM,
            'p_uav': 5,  # w, 也即34.7dbm
            'p_poi': 0.1,
            'aA': 2,
            'aG': 4,
            'nLoS': 0,  # dB, 也即1w
            'nNLoS': -20,  # dB, 也即0.01w
            'uav_init_height': self.UAV_HEIGHT,
            'psi': 9.6,
            'beta': 0.16,
        }

        self.action_space = {key: spaces.Discrete(self.ACTION_ROOT) for key in self.UAV_TYPE}
        self._poi_value = [0 for _ in range(self.POI_NUM)]

        self.poi_property_num = 2 + 1
        info = self.get_env_info()

        obs_dict = {
            'State': spaces.Box(low=-1, high=1, shape=(info['state_shape'],)),
            'available_actions': spaces.Box(low=0, high=1, shape=(self.n_agents, self.ACTION_ROOT)),
        }
        for type in self.UAV_TYPE:
            obs_dict[type + "_obs"] = spaces.Box(low=-1, high=1, shape=(self.n_agents, info['obs_shape'][type]))
        self.obs_space = spaces.Dict(obs_dict)
        self.observation_space = self.obs_space
        self.reset()
    
    def reset(self):
        # reload map
        self.nx_g = self.NX_G[self.map]
        self.node_to_poi = self.NODE_TO_POI[self.map]
        self.node_to_poi = np.array(list(self.node_to_poi.values()))
        self.edge_features = self.EDGE_FEATURES[self.map]
        self.nx_g_nodes_py_x = np.zeros(len(self.nx_g.nodes))
        self.nx_g_nodes_py_y = np.zeros(len(self.nx_g.nodes))
        self.nx_g_nodes_data = np.zeros(len(self.nx_g.nodes))
        for i,node in enumerate(self.nx_g.nodes(data=True)):
            self.nx_g_nodes_py_x[i] = node["py_x"]
            self.nx_g_nodes_py_y[i] = node["py_y"]
            self.nx_g_nodes_data[i] = node["data"]

        self.visited_nodes_count = {}
        self.poi_nearest_dis = self.POI_NEAREST_DIS[self.map]
        # reload map finished


        self.reward_co = {}
        if self.config("uav_poi_dis") > 0:
            self.reward_co['uav'] = [1 if dis > self.config("uav_poi_dis") else self.config("colla_co") for dis in
                                        self.poi_nearest_dis]
            self.reward_co['carrier'] = [self.config("colla_co") if dis > self.config("uav_poi_dis") else 1 for dis
                                            in self.poi_nearest_dis]
            print(
                f"无人机负责的poi大于路网{self.config('uav_poi_dis')}米的 共有{np.mean([1 if dis > self.config('uav_poi_dis') else 0 for dis in self.poi_nearest_dis])}")
        else:
            self.reward_co['uav'] = [1 for _ in range(self.POI_NUM)]
            self.reward_co['carrier'] = [1 for _ in range(self.POI_NUM)]

        self.reset_count += 1
        self.uav_trace = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.uav_state = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.uav_energy_consuming_list = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.uav_data_collect = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}

        self.dead_uav_list = {key: [False for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.collect_list = {key: [[] for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        self.step_count = 0
        self.last_collect = [{key: np.zeros((self.NUM_UAV[key], self.POI_NUM)) for key in self.UAV_TYPE} for _ in
                             range(self.MAX_EPISODE_STEP)]

        self.uav_energy = copy.deepcopy(self._uav_energy)
        self.poi_value = copy.deepcopy(self._poi_value)
        self.uav_position = copy.deepcopy(self._uav_position)
        self.poi_position = copy.deepcopy(self._poi_position)

        self.carrier_node = []
        for i in range(self.NUM_UAV['carrier']):
            raw_node = ox.distance.nearest_nodes(self.nx_g, *self.rm.pygamexy2lonlat(
            self.uav_position['carrier'][i][0] * self.SCALE, self.uav_position['carrier'][i][1] * self.SCALE))
            self.carrier_node.append(self.nx_g.nodes(data=True)[raw_node]['old_label'])
        self.carrier_node_history = [copy.deepcopy(self.carrier_node)]
        
        # wipe_last_things
        self.last_dst_node = np.zeros((self.NUM_UAV['carrier'], self.action_space['carrier'].n))
        self.last_length = np.zeros((self.NUM_UAV['carrier'], self.action_space['carrier'].n))
        self.last_dst_lonlat = np.zeros((self.NUM_UAV['carrier'], self.action_space['carrier'].n, 2))
        # wipe_last_things finished

        # for render
        self.poi_history = []
        self.aoi_history = [0]
        self.emergency_history = []
        self.episodic_reward_list = {key: [] for key in self.UAV_TYPE}
        self.single_uav_reward_list = {key: [] for key in self.UAV_TYPE}

        return self.get_obs()
        

    def step(self, action):
        uav_reward = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}
        uav_penalty = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}
        # uav_data_collect = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}

        distance = {key: [0 for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        for type in self.UAV_TYPE:
            for uav_index in range(self.NUM_UAV[type]):
                if type == 'carrier':
                    a = action[type][uav_index]
                    self.carrier_node[uav_index] = int(self.last_dst_node[uav_index][a])
                    self.visited_nodes_count[self.carrier_node[uav_index]] = self.visited_nodes_count.get(
                        self.carrier_node[uav_index], 0) + 1
                    assert int(self.carrier_node[uav_index]) != 0
                    new_x, new_y = self.rm.lonlat2pygamexy(self.last_dst_lonlat[uav_index][a][0],
                                                           self.last_dst_lonlat[uav_index][a][1])
                    self.uav_position[type][uav_index] = (new_x / self.SCALE, new_y / self.SCALE)
                    dis = self.last_length[uav_index][a]
                    energy_consuming = self._cal_energy_consuming(dis, type)

                    if uav_index == self.NUM_UAV[type] - 1:
                        self.carrier_node_history.append(copy.deepcopy(self.carrier_node))
                else:
                    new_x, new_y, dis, energy_consuming = self._cal_uav_next_pos(uav_index, action[type][uav_index],
                                                                                 type)
                    if (0 <=new_x <= self.MAP_X) and (0 <= new_y <= self.MAP_Y):
                        self.uav_position[type][uav_index] = (new_x, new_y)

                self.uav_trace[type][uav_index].append(self.uav_position[type][uav_index])
                self._use_energy(type, uav_index, energy_consuming)
                uav_reward[type][uav_index] -= energy_consuming * 1e-6
                distance[type][uav_index] += dis

        for type in self.UAV_TYPE:
            for uav_index in range(self.NUM_UAV[type]):
                collect_time = max(0, self.TIME_SLOT - distance[type][uav_index] / self.UAV_SPEED[type])
                r, collected_data = self._collect_data_from_poi(type, uav_index, collect_time)

                self.uav_data_collect[type][uav_index].append(collected_data)

                uav_reward[type][uav_index] += r * (10 ** -3)

                # TODO: 下面需要加一个奖励或者规则保证空地协同、例如车机距离限制？

        self.check_arrival()
        self.aoi_history.append(np.mean(self.poi_value) / self.USER_DATA_AMOUNT)
        self.emergency_history.append(
            np.mean([1 if aoi / self.USER_DATA_AMOUNT >= self.AOI_THRESHOLD else 0 for aoi in self.poi_value]))
        aoi_reward = self.aoi_history[-2] - self.aoi_history[-1]
        aoi_reward -= self.emergency_history[-1] * self.THRESHOLD_PENALTY

        for type in self.UAV_TYPE:
            for uav_index in range(self.NUM_UAV[type]):
                uav_reward[type][uav_index] -= self.emergency_history[-1] * self.THRESHOLD_PENALTY


        done = self._is_episode_done()
        self.step_count += 1
        self.poi_history.append({
            'pos': copy.deepcopy(self.poi_position).reshape(-1, 2),
            'val': copy.deepcopy(self.poi_value)})

        info = {}
        info_old = {}
        if done:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                info = self.summary_info(info)
                info_old = copy.deepcopy(info)
                info = self.save_trajectory(info)

        global_reward = {}
        for type in self.UAV_TYPE:
            global_reward[type] = np.mean(uav_reward[type]) + np.mean(uav_penalty[type])
            self.episodic_reward_list[type].append(global_reward[type])
            self.single_uav_reward_list[type].append((uav_reward[type] + uav_penalty[type]).tolist())
        obs = self.get_obs()

        return obs, uav_reward, done, info_old

    def _collect_data_from_poi(self, type, uav_index, collect_time=0):
        reward_list = []
        position_list = []
        if collect_time >= 0:
            for poi_index, (poi_position, poi_value) in enumerate(zip(self.poi_position, self.poi_value)):
                d = self._cal_distance(poi_position, self.uav_position[type][uav_index], type)
                if d < self.COLLECT_RANGE[type] and poi_value > 0:
                    position_list.append((poi_index, d))
            position_list = sorted(position_list, key=lambda x: x[1])

            update_num = min(len(position_list), self.UPDATE_NUM[type])

            for i in range(update_num):
                poi_index = position_list[i][0]
                rate = self._get_data_rate(self.uav_position[type][uav_index], self.poi_position[poi_index])
                if rate <= self.RATE_THRESHOLD[type]:
                    break
                collected_data = min(rate * collect_time, self.poi_value[poi_index])
                self.poi_value[poi_index] -= collected_data
            reward_list.append(collected_data)

        return sum(reward_list), len(reward_list)
    
    def summary_info(self, info):
        data_collection_ratio = 1 - np.sum(np.sum(self.poi_value)) / (
                self.step_count * self.USER_DATA_AMOUNT * self.POI_NUM)
        sep_collect = {'f_data_collection_ratio_' + type: np.sum(
            [sum(self.uav_data_collect[type][uav_index]) for uav_index in range(self.NUM_UAV[type])]) / (
                                                                    self.step_count * self.USER_DATA_AMOUNT * self.POI_NUM)
                        for type in self.UAV_TYPE}

        info['a_data_collection_ratio'] = data_collection_ratio.item()
        info['a_episodic_aoi'] = np.mean(self.aoi_history)

        info.update(sep_collect)

        info[f"Map: {self.map}/a_data_collection_ratio"] = data_collection_ratio.item()
        info[f"Map: {self.map}/a_episodic_aoi"] = np.mean(self.aoi_history)
        info['map'] = int(self.map)
        t_all = 0
        for type in self.UAV_TYPE:
            t_e = np.sum(np.sum(self.uav_energy_consuming_list[type]))
            t_all += t_e / 1000
            info['f_total_energy_consuming_' + type] = t_e.item()
            info['f_energy_consuming_ratio_' + type] = t_e / (self.NUM_UAV[type] * self.INITIAL_ENERGY[type])

        data_collect_all = np.sum(
            [sum(self.uav_data_collect['uav'][uav_index]) for uav_index in range(self.NUM_UAV['uav'])]) + np.sum(
            [sum(self.uav_data_collect['carrier'][uav_index]) for uav_index in range(self.NUM_UAV['carrier'])])
        aoi = np.mean(self.aoi_history)

        # print(data_collect_all,aoi,t_all)
        info['a_sensing_efficiency'] = data_collect_all.item() / (aoi * t_all)
        info['a_energy_consumption_ratio'] = t_all * 1000 / sum(
            [self.NUM_UAV[type] * self.INITIAL_ENERGY[type] for type in self.UAV_TYPE])
        return info

    def save_trajectory(self, info):
        if self.TEST_MODE:
            for type in self.UAV_TYPE:
                temp_info = {}
                temp_info['uav_trace'] = self.uav_trace[type]
                max_len = max((len(l) for l in self.uav_data_collect[type]))
                new_matrix = list(
                    map(lambda l: l + [0] * (max_len - len(l)), self.uav_data_collect[type]))
                temp_info['uav_collect'] = np.sum(new_matrix, axis=0).tolist()
                temp_info['reward_history'] = self.episodic_reward_list[type]
                temp_info['uav_reward'] = self.single_uav_reward_list[type]

                if type == 'carrier':
                    temp_info['carrier_node_history'] = self.carrier_node_history

                info[type] = temp_info

            info['map'] = self.map
            info['config'] = self.config.dict
            info['poi_history'] = self.poi_history
            path = self.args['save_path'] + '/map_{}_count_{}.txt'.format(self.map, self.SAVE_COUNT)
            self.save_variable(info, path)
            info = {}

        return info

    def save_variable(self, v, filename):
        # print('save variable to {}'.format(filename))
        f = open(filename, 'wb')
        pickle.dump(v, f)
        f.close()
        return filename


    def _cal_distance(self, pos1, pos2, type):
        height = self.UAV_HEIGHT if type == 'uav' else 0

        if isinstance(pos1, np.ndarray) and isinstance(pos2, np.ndarray):
            while pos1.ndim < 2:
                pos1 = np.expand_dims(pos1, axis=0)
            while pos2.ndim < 2:
                pos2 = np.expand_dims(pos2, axis=0)
            # expanded to 3dim
            pos1_all = np.concatenate([pos1 * self.SCALE, np.zeros((pos1.shape[0], 1))], axis=1)
            pos2_all = np.concatenate([pos2 * self.SCALE, np.ones((pos2.shape[0], 1)) * height], axis=1)
            distance = np.linalg.norm(pos1_all - pos2_all, axis=1)
        else:
            assert len(pos1) == len(
                pos2) == 2, 'cal_distance function only for 2d vector'
            distance = np.sqrt(
                np.power(pos1[0] * self.SCALE - pos2[0] * self.SCALE, 2) + np.power(pos1[1] * self.SCALE
                                                                                    - pos2[1] * self.SCALE,
                                                                                    2) + np.power(height, 2))
        return distance

    def _cal_theta(self, pos1, pos2, height=None):
        if len(pos1) == len(pos2) and len(pos2) == 2:
            r = np.sqrt(np.power(pos1[0] * self.SCALE - pos2[0] * self.SCALE, 2) + np.power(
                pos1[1] * self.SCALE - pos2[1] * self.SCALE, 2))
            h = self.UAV_HEIGHT if height is None else height
            theta = math.atan2(h, r)
        elif len(pos1) == 2:
            repeated_pos1 = np.tile(pos1, len(pos2)).reshape(-1, 2)
            r = self._cal_distance(repeated_pos1, pos2, type='carrier')
            h = self.UAV_HEIGHT if height is None else height
            theta = np.arctan2(h, r)
        return theta

    def _cal_energy_consuming(self, move_distance, type):
        moving_time = move_distance / self.UAV_SPEED[type]
        hover_time = self.TIME_SLOT - moving_time
        if type == 'carrier':
            moving_time = min(20, move_distance / 15)
            return self.Power_flying[type] * moving_time + self.Power_hovering[type] * hover_time
        else:
            return self.Power_flying[type] * moving_time + self.Power_hovering[type] * hover_time

    def _cal_uav_next_pos(self, uav_index, action, type):
        dx, dy = self._get_vector_by_action(int(action))
        distance = np.sqrt(np.power(dx * self.SCALE, 2) +
                           np.power(dy * self.SCALE, 2))
        energy_consume = self._cal_energy_consuming(distance, type)
        if self.uav_energy[type][uav_index] >= energy_consume:
            new_x, new_y = self.uav_position[type][uav_index][0] + dx, self.uav_position[type][uav_index][1] + dy
        else:
            new_x, new_y = self.uav_position[type][uav_index]

        return new_x, new_y, distance, min(self.uav_energy[type][uav_index], energy_consume)


    def _get_vector_by_action(self, action, type='uav'):
        single = 2
        base = single / math.sqrt(2)

        action_table = [
            [0, 0],
            [-base, base],
            [0, single],
            [base, base],
            [-single, 0],
            [single, 0],
            [-base, -base],
            [0, -single],
            [base, -base],
            # 额外添加3个动作，向无人车靠近，还有向剩余poi最多的靠近
            [2 * single, 0],
            [0, 2 * single],
            [-2 * single, 0],
            [0, -2 * single]
        ]
        return action_table[action]


    def _is_uav_out_of_energy(self, uav_index, type):
        return self.uav_energy[type][uav_index] < self.EPSILON

    def _is_episode_done(self):
        if (self.step_count + 1) >= self.MAX_EPISODE_STEP:
            return True
        else:
            return False


    def _use_energy(self, type, uav_index, energy_consuming):
        self.uav_energy_consuming_list[type][uav_index].append(
            min(energy_consuming, self.uav_energy[type][uav_index]))
        self.uav_energy[type][uav_index] = max(
            self.uav_energy[type][uav_index] - energy_consuming, 0)

        if self._is_uav_out_of_energy(uav_index, type):
            if self.DEBUG_MODE:
                print("Energy should not run out!")
            self.dead_uav_list[type][uav_index] = True
            self.uav_state[type][uav_index].append(0)
        else:
            self.uav_state[type][uav_index].append(1)

    def _get_energy_coefficient(self):
        P0 = 58.06  # blade profile power, W
        P1 = 79.76  # derived power, W
        U_tips = 120  # tip speed of the rotor blade of the UAV,m/s
        v0 = 4.03  # the mean rotor induced velocity in the hovering state,m/s
        d0 = 0.2  # fuselage drag ratio
        rho = 1.225  # density of air,kg/m^3
        s0 = 0.05  # the rotor solidity
        A = 0.503  # the area of the rotor disk, m^2

        for type in self.UAV_TYPE:
            Vt = self.config("uav_speed")[type]  # velocity of the UAV,m/s ???
            if type == 'uav':
                self.Power_flying[type] = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
                                          P1 * np.sqrt(
                    (np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
                                          0.5 * d0 * rho * s0 * A * Vt ** 3

                self.Power_hovering[type] = P0 + P1
            elif type == 'carrier':
                self.Power_flying[type] = 17.49 + 7.4 * 15
                self.Power_hovering[type] = 17.49

    def _get_data_rate(self, uav_position, poi_position):
        eta = 2
        alpha = 4.88
        beta = 0.43
        distance = self._cal_distance(uav_position, poi_position, 'uav')
        theta = self._cal_theta(uav_position, poi_position)
        path_loss = (54.05 + 10 * eta * np.log10(distance) + (-19.9)
                     / (1 + alpha * np.exp(-beta * (theta - alpha))))
        w_tx = 20
        w_noise = -104
        w_s_t = w_tx - path_loss - w_noise
        w_w_s_t = np.power(10, (w_s_t - 30) / 10)
        bandwidth = 20e6
        data_rate = bandwidth * np.log2(1 + w_w_s_t)
        return data_rate / 1e6

    def get_adjcent_dict(self):
        adj_dict = {
            'uav': {key: None for key in ['carrier', 'poi', 'road', 'epoi']},  # n x n, n x poi
            'carrier': {key: None for key in ['uav', 'poi', 'road', 'epoi']},  # n x n, n x poi, n x node
            'poi': {key: None for key in ['uav', 'carrier']},  # poi x n, poi x n
            'road': {key: None for key in ['carrier']}  # node x n
        }
        return_dict = {}
        for key1, s_dict in adj_dict.items():
            for key2, adj in s_dict.items():
                return_dict[f"{key1}-{key2}"] = adj
        return return_dict
    
    def get_obs(self, aoi_now=None, aoi_next=None):
        agents_obs = {key: np.vstack([self.get_obs_agent(i, type=key) for i in range(self.NUM_UAV[key])]) for key in
                        self.UAV_TYPE}
        action_mask = self.get_avail_actions()
        action_mask = {'mask_' + key: action_mask[key] for key in self.UAV_TYPE}

        obs_dict = {
            'State': self.get_state(),
            **{
                'Nodes':self.get_node_agents(0, type='uav'),
                'Edges':self.edge_features,
            },
            **action_mask,
            **agents_obs,
            **self.get_adjcent_dict()
        }

        return obs_dict

    def get_node_agents(self, agent_id, type, global_view=True):
        total_nodes_num = len(self.nx_g)
        datas = np.zeros((total_nodes_num, 3))
        for node in self.node_to_poi:
            data = self.nx_g_nodes_data[node]
            data = self.nx_g.nodes(data=True)[node]
            datas[node, :] = data['py_x'], data['py_y'], self.nx_g_nodes_data[node]
        datas[:, 0] /= self.SCALE
        datas[:, 1] /= self.SCALE

        if not global_view:
            mask = self._cal_distance([datas[:, 0], datas[:, 1]], self.uav_position[type][agent_id], 'carrier') \
                   <= self.agent_field[type]
        for i, diviend in zip(range(datas.shape[1]),
                              [self.MAP_X, self.MAP_Y, self.USER_DATA_AMOUNT * self.MAX_EPISODE_STEP]):
            if not global_view:
                datas[:, i] *= mask
            datas[:, i] /= diviend
        return datas

    def get_obs_agent(self, agent_id, global_view=False, visit_num=None, type=None):
        if visit_num is None:
            visit_num = self.POI_NUM

        if global_view:
            distance_limit = 1e10
        else:
            distance_limit = self.agent_field[type]
        target_dis = self.uav_position[type][agent_id]

        agent_pos = []
        for t in self.NUM_UAV:
            for i in range(self.NUM_UAV[t]):
                agent_pos.append(self.uav_position[t][i][0] / self.MAP_X)
                agent_pos.append(self.uav_position[t][i][1] / self.MAP_Y)

        distances = self._cal_distance(self.poi_position, np.array(target_dis), type)
        is_visible = distances < distance_limit
        dividend = self.USER_DATA_AMOUNT * self.MAX_EPISODE_STEP
        visible_poi_states = np.stack([self.poi_position[:, 0] * is_visible / self.MAP_X,
                                       self.poi_position[:, 1] * is_visible / self.MAP_Y,
                                       self.poi_value * is_visible / dividend], axis=1)

        id = agent_id if type == 'carrier' else self.NUM_UAV['carrier'] + agent_id
        one_hot = np.eye(self.n_agents)[id]
        return np.concatenate([one_hot, agent_pos, visible_poi_states.flatten()])

    def get_obs_size(self):
        obs_size = {}
        num = self.n_agents

        for type in self.UAV_TYPE:
            size = 2 * num + num
            size += self.POI_NUM * self.poi_property_num
            obs_size[type] = size
        return obs_size

    def get_state(self):
        return np.zeros(self.obs_space['State'].shape, dtype=np.float32)

    def get_concat_obs(self, agent_obs):
        state_all = {}
        for key in self.UAV_TYPE:
            state = np.zeros_like(agent_obs[key][0])
            for i in range(self.NUM_UAV[key]):
                mask = agent_obs[key][i] != 0
                np.place(state, mask, agent_obs[key][i][mask])
            state_all[key] = state
        return state_all

    def get_state_size(self):
        size = self.POI_NUM * self.poi_property_num + self.n_agents * 3
        return size

    def get_avail_actions(self):
        avail_actions_all = {}
        for type in self.UAV_TYPE:
            avail_actions = []
            for agent_id in range(self.NUM_UAV[type]):
                avail_agent = self.get_avail_agent_actions(agent_id, type)
                avail_actions.append(avail_agent)
            avail_actions_all[type] = np.vstack(avail_actions)
        return avail_actions_all

    def get_avail_agent_actions(self, agent_id, type):
        if type == 'uav':
            avail_actions = []
            temp_x, temp_y = self.uav_position[type][agent_id]
            for i in range(self.ACTION_ROOT):
                dx, dy = self._get_vector_by_action(i, type)
                if (0 <= dx + temp_x <= self.MAP_X) and (0 <= dy + temp_y <= self.MAP_Y):
                    avail_actions.append(1)
                else:
                    avail_actions.append(0)
            return np.array(avail_actions)

        elif type == 'carrier':
            num_action = self.action_space['carrier'].n
            avail_actions = np.zeros((num_action,))

            def sort_near_set_by_angle(src_node, near_set):
                thetas = []
                src_pos = self.rm.lonlat2pygamexy(self.nx_g.nodes[src_node]['x'],
                                                  self.nx_g.nodes[src_node]['y'])
                for item in near_set:
                    dst_node = self.OSMNX_TO_NX[int(item[0])]
                    if dst_node in self.ROAD_MAP[self.map]:
                        thetas.append(99999)
                        continue
                    dst_pos = self.rm.lonlat2pygamexy(self.nx_g.nodes[dst_node]['x'], self.nx_g.nodes[dst_node]['y'])
                    theta = compute_theta(dst_pos[0] - src_pos[0], dst_pos[1] - src_pos[1], 0)
                    thetas.append(theta)
                near_set = sorted(near_set, key=lambda x: thetas[near_set.index(x)])
                return near_set

            # step2. 对每辆车，根据pair_dis_dict得到sorted的10个点，如果可达则mask=1
            id = agent_id
            # 路网栅格化之后 这里可能会返回一个不在栅格化后集合中的点 因此这里直接记录一下 车现在处于哪个index
            src_node = self.carrier_node[id]
            # 验证车一定在路网的点上
            neighbor_nodes = []
            # for neighbor in self.nx_g.neighbors(src_node):
            #     distance = nx.shortest_path_length(self.nx_g, src_node, neighbor, weight='length')
            #     neighbor_nodes.append((neighbor,distance))
            distances = self.PAIR_DIS_DICT[self.map][str(src_node)]
            pairs = list(zip(distances.keys(), distances.values()))
            near_set = sorted(pairs, key=lambda x: x[1])[:max(0, num_action - len(neighbor_nodes))]
            near_set = neighbor_nodes + near_set
            near_set = sort_near_set_by_angle(self.OSMNX_TO_NX[src_node], near_set)
            valid_nodes = set([data['old_label'] for _, data in self.nx_g.nodes(data=True)])
            for act in range(num_action):
                dst_node, length = near_set[act]
                if dst_node in self.ignore_node or int(dst_node) not in valid_nodes:
                    continue
                if length != np.inf:
                    avail_actions[act] = 1
                    converted = self.OSMNX_TO_NX[int(dst_node)]
                    self.last_dst_node[id][act] = dst_node
                    self.last_length[id][act] = length
                    self.last_dst_lonlat[id][act][0], self.last_dst_lonlat[id][act][1] = (
                        self.nx_g.nodes[converted]['x'],
                        self.nx_g.nodes[converted]['y']
                    )
            return avail_actions


    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.n_actions,
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}

        return env_info


    def check_arrival(self):
        for i in range(self.POI_NUM):
            self.nx_g.nodes[self.node_to_poi[i]]['data'] = 0

        for i in range(self.POI_NUM):
            self.poi_value[i] += self.USER_DATA_AMOUNT
            self.nx_g.nodes[self.node_to_poi[i]]['data'] += self.poi_value[i]

    def get_node_poi_map(self, map, nx_g):
        result_dict = {}
        node_positions = []
        remove_map = [self.OSMNX_TO_NX[x] for x in self.ROAD_MAP[map] if x in self.valid_nodes]
        for node, data in nx_g.nodes(data=True):
            if node not in remove_map:
                node_positions.append([data['py_x'] / self.MAP_X, data['py_y'] / self.MAP_Y])
            else:
                node_positions.append([99999999, 999999])

        node_positions = np.array(node_positions)
        poi_nearest_distance = [0 for _ in range(self.POI_NUM)]
        for poi_index in range(self.POI_NUM):
            poi_position = self._poi_position[poi_index]
            distances = np.linalg.norm(poi_position - node_positions, axis=1)
            nearest_index = np.argmin(distances)
            result_dict[poi_index] = int(nearest_index)
            poi_nearest_distance[poi_index] = float(distances[nearest_index] * self.SCALE)

        return result_dict, poi_nearest_distance


def compute_theta(dpx, dpy, dpz):
    '''弧度制'''
    # 法一 无法达到
    # dpx>0 dpy>0时，theta在第一象限
    # dpx<0 dpy>0时，theta在第二象限
    # dpx<0 dpy<0时，theta在第三象限
    # dpx>0 dpy<0时，theta在第四象限
    theta = math.atan(dpy / (dpx + 1e-8))

    # 法二 2022/1/10 可以达到 不过y轴是反的 但无伤大雅~
    x1, y1 = 0, 0
    x2, y2 = dpx, dpy
    ang1 = np.arctan2(y1, x1)
    ang2 = np.arctan2(y2, x2)
    # theta = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    theta = (ang1 - ang2) % (2 * np.pi)  # theta in [0, 2*pi]

    return theta
