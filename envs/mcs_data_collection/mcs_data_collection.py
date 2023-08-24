import os
import sys

from warp_drive.utils.common import get_project_root
from .util.config_3d import Config
from .util.utils import IsIntersec
from tqdm import tqdm
from .util.noma_utils import *
from .util.roadmap_utils import Roadmap

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
from importlib import import_module

np.seterr(all="raise")

shared_feature_list = ['ROAD_MAP', 'PAIR_DIS_DICT', 'ALL_G', 'NX_G', 'OSMNX_TO_NX', 'EDGE_FEATURES',
                    'POI_NEAREST_DIS', 'NODE_TO_POI']

class EnvUCS(object):

    def __init__(self, args=None, **kwargs):
        if args != None:
            self.args = args
        self.config = Config(args)
        map_number = self.config.dict['map']
        self.load_map()
        self.DISCRIPTION = self.config('description')
        self.CENTRALIZED = self.config('centralized')
        self.SCALE = self.config("scale")
        self.INITIAL_ENERGY = self.config("initial_energy")
        self.EPSILON = self.config("epsilon")
        self.DEBUG_MODE = self.config("debug_mode")
        self.TEST_MODE = self.config("test_mode")
        self.ACTION_MODE = self.config("action_mode")
        self.COLLECT_MODE = self.config("collect_mode")
        self.MAX_EPISODE_STEP = self.config("max_episode_step")
        self._max_episode_steps = self.MAX_EPISODE_STEP
        self.TIME_SLOT = self.config("time_slot")
        self.SAVE_COUNT = self.config('seed')
        self.USE_HGCN = self.config('use_hgcn')

        self.CONCAT_OBS = self.config("concat_obs")
        self.POI_INIT_DATA = self.config("poi_init_data")
        self.AOI_THRESHOLD = self.config("aoi_threshold")
        self.TOTAL_TIME = self.MAX_EPISODE_STEP * self.TIME_SLOT
        self.THRESHOLD_PENALTY = self.config("threshold_penalty")
        self.UAV_HEIGHT = self.config("uav_height")
        self.USER_DATA_AMOUNT = self.config("user_data_amount")
        self.CHANNEL_NUM = self.config("channel_num")
        self.NOMA_MODE = self.config("noma_mode")
        self.ROADMAP_MODE = self.config("roadmap_mode")

        self.UAV_TYPE = ['carrier', 'uav']
        self.NUM_UAV = OrderedDict(self.config("num_uav"))
        self.UAV_SPEED = OrderedDict(self.config("uav_speed"))
        self.RATE_THRESHOLD = OrderedDict(self.config("rate_threshold"))
        self.UPDATE_NUM = OrderedDict(self.config("update_num"))
        self.COLLECT_RANGE = OrderedDict(self.config("collect_range"))

        self.ACTION_ROOT = self.config("action_root")
        self.n_agents = sum(self.NUM_UAV.values())
        self.episode_limit = self._max_episode_steps
        self.n_actions = 1 if self.ACTION_MODE else self.ACTION_ROOT
        self.agent_field = OrderedDict(self.config("agent_field"))
        self.reset_count = 0
        if self.config('dataset') =='KAIST':
            self.broke_threshold = 100
            self.normal_threshold = 50
        elif self.config('dataset')=='Rome':
            self.broke_threshold = 500
            self.normal_threshold = 200
        self.MAP_X = self.config("map_x")
        self.MAP_Y = self.config("map_y")
        self.POI_NUM = self.config("poi_num")
        self.OBSTACLE = self.config('obstacle')
        self._poi_position = np.array(self.config('poi'))
        self._poi_position[:, 0] *= self.MAP_X
        self._poi_position[:, 1] *= self.MAP_Y
        self.is_sub_env = self.config('is_sub_env')

        self._uav_energy = {this_uav: [self.config("initial_energy")[this_uav] for i in range(self.NUM_UAV[this_uav])]
                            for this_uav in
                            self.UAV_TYPE}
        self._uav_position = \
            {this_uav: [[self.config("init_position")[this_uav][i][0], self.config("init_position")[this_uav][i][1]] for
                        i in
                        range(self.NUM_UAV[this_uav])] for this_uav in self.UAV_TYPE}
        self.map = str(self.config("map"))
        print(f'selected map:{self.map}')
        
        if self.ROADMAP_MODE:
            self.rm = Roadmap(self.config("dataset"))
            if self.is_sub_env:
                for item in shared_feature_list:
                    setattr(self, item, self.config(item))
                    # setattr(self, item + '_mem', shared_memory.SharedMemory(self.config(item)))
                    # buffer = getattr(self, item + '_mem').buf
                    # setattr(self, item, json.loads(bytes(buffer[:]).decode('utf-8')))
            else:
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
                    new_graph = nx.convert_node_labels_to_integers(self.get_sub_graph(new_graph, nodes_to_remove),
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
                # Warning: edges and edges() output different things.
                # if not self.is_sub_env:

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

        if self.ACTION_MODE == 1:
            self.action_space = spaces.Box(min=-1, max=1, shape=(2,))
        elif self.ACTION_MODE == 0:
            self.action_space = {key: spaces.Discrete(self.ACTION_ROOT) for key in self.UAV_TYPE}
        else:
            self.action_space = spaces.Discrete(1)

        if self.COLLECT_MODE == 0:
            self._poi_value = [self.POI_INIT_DATA for _ in range(self.POI_NUM)]
        elif self.COLLECT_MODE == 1:
            self._poi_value = [0 for _ in range(self.POI_NUM)]
        elif self.COLLECT_MODE == 2:
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

    def load_map(self):
        from .util.KAIST.env_config import dataset_config
        self.config.dict = {
            **self.config.dict,
            **dataset_config
        }

    def reset(self, map_index=0):
        # if map_index != self.map:
        # print('change map!!!',map_index)
        self.reload_map(str(map_index))

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

        if self.ROADMAP_MODE:
            self.carrier_node = []
            for i in range(self.NUM_UAV['carrier']):
                raw_node = ox.distance.nearest_nodes(self.nx_g, *self.rm.pygamexy2lonlat(
                self.uav_position['carrier'][i][0] * self.SCALE, self.uav_position['carrier'][i][1] * self.SCALE))
                self.carrier_node.append(self.nx_g.nodes(data=True)[raw_node]['old_label'])
            self.carrier_node_history = [copy.deepcopy(self.carrier_node)]
            self.wipe_last_things()

        # for render
        self.poi_history = []
        self.aoi_history = [0]
        self.emergency_history = []
        self.episodic_reward_list = {key: [] for key in self.UAV_TYPE}
        self.single_uav_reward_list = {key: [] for key in self.UAV_TYPE}

        return self.get_obs()

    def get_sub_graph(self, graph: nx.Graph, sub_g: list):
        if len(sub_g) == 0:
            return graph
        remove_list = []
        for u, v, data in graph.edges(keys=False, data=True):
            if v in sub_g or u in sub_g:
                remove_list.append([u, v])
        graph.remove_edges_from(remove_list)
        return graph

    def wipe_last_things(self):
        self.last_dst_node = np.zeros((self.NUM_UAV['carrier'], self.action_space['carrier'].n))
        self.last_length = np.zeros((self.NUM_UAV['carrier'], self.action_space['carrier'].n))
        self.last_dst_lonlat = np.zeros((self.NUM_UAV['carrier'], self.action_space['carrier'].n, 2))

    def reload_map(self, map_index: str):
        if self.config("random_map"):
            self.map = map_index
        else:
            self.map = str(self.config("map"))
        

        if self.ROADMAP_MODE:
            self.nx_g = self.NX_G[self.map]
            self.node_to_poi = self.NODE_TO_POI[self.map]
            self.edge_features = self.EDGE_FEATURES[self.map]
            for node in self.node_to_poi.values():
                self.nx_g.nodes[node]['data'] = 0

            self.visited_nodes_count = {}
            self.poi_nearest_dis = self.POI_NEAREST_DIS[self.map]

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

            # print(f"加载地图{self.map}, 共有顶点{len(self.nx_g.nodes())}, 边{len(self.nx_g.edges())}")

        return

    def step(self, action):
        uav_reward = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}
        uav_penalty = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}
        # uav_data_collect = {key: np.zeros([self.NUM_UAV[key]]) for key in self.UAV_TYPE}

        distance = {key: [0 for i in range(self.NUM_UAV[key])] for key in self.UAV_TYPE}
        energy_consumption_all = 0
        for type in self.UAV_TYPE:
            for uav_index in range(self.NUM_UAV[type]):
                if self.ROADMAP_MODE and type == 'carrier':
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
                    energy_consumption_all += energy_consuming

                    if uav_index == self.NUM_UAV[type] - 1:
                        self.carrier_node_history.append(copy.deepcopy(self.carrier_node))
                else:
                    new_x, new_y, dis, energy_consuming = self._cal_uav_next_pos(uav_index, action[type][uav_index],
                                                                                 type)
                    Flag = self._judge_obstacle(self.uav_position[type][uav_index], (new_x, new_y))
                    if not Flag:
                        self.uav_position[type][uav_index] = (new_x, new_y)

                self.uav_trace[type][uav_index].append(self.uav_position[type][uav_index])
                self._use_energy(type, uav_index, energy_consuming)
                energy_consumption_all += energy_consuming
                uav_reward[type][uav_index] -= energy_consuming * 1e-6
                distance[type][uav_index] += dis

        if self.NOMA_MODE:
            relay_dict = self._relay_association()
            sorted_access = self._access_determin(self.CHANNEL_NUM)
        # if distance['carrier'][0] == 0:
        #     print(distance)
        for type in self.UAV_TYPE:
            for uav_index in range(self.NUM_UAV[type]):
                collect_time = max(0, self.TIME_SLOT - distance[type][uav_index] / self.UAV_SPEED[type])

                if self.NOMA_MODE:
                    r, collected_data = self._collect_data_by_noma(type, uav_index, relay_dict, sorted_access,
                                                                   collect_time)
                else:
                    r, collected_data = self._collect_data_from_poi(type, uav_index, collect_time)

                self.uav_data_collect[type][uav_index].append(collected_data)

                uav_reward[type][uav_index] += r * (10 ** -3)  # * (2**-4)
                # print( uav_reward[type][uav_index])

                if type == 'uav':
                    # dis_reward =  self._cal_distance(self.uav_position['carrier'][relay_dict[uav_index]],
                    # self.uav_position['uav'][uav_index])*0.0001
                    # uav_reward[type][uav_index] +
                    uav_reward['carrier'][relay_dict[uav_index]] += r * (10 ** -3) / 5

                if type == 'carrier' and self.config("carrier_explore_reward"):
                    # print(uav_reward[type][uav_index])
                    uav_reward[type][uav_index] -= math.log(
                        self.visited_nodes_count[self.carrier_node[uav_index]] + 1) * 0.1

                    # print(-math.log(self.visited_nodes_count[self.carrier_node[uav_index]]+1) * 0.05)

        if self.COLLECT_MODE == 1 or self.COLLECT_MODE == 2:
            self.check_arrival()
            self.aoi_history.append(np.mean(self.poi_value) / self.USER_DATA_AMOUNT)
            self.emergency_history.append(
                np.mean([1 if aoi / self.USER_DATA_AMOUNT >= self.AOI_THRESHOLD else 0 for aoi in self.poi_value]))
            aoi_reward = self.aoi_history[-2] - self.aoi_history[-1]
            aoi_reward -= self.emergency_history[-1] * self.THRESHOLD_PENALTY

            for type in self.UAV_TYPE:
                for uav_index in range(self.NUM_UAV[type]):
                    uav_reward[type][uav_index] -= self.emergency_history[-1] * self.THRESHOLD_PENALTY

            for type in self.UAV_TYPE:
                type_sum = sum(uav_reward[type])
                for uav_index in range(self.NUM_UAV[type]):
                    if self.CENTRALIZED:
                        uav_reward[type][uav_index] = aoi_reward - energy_consumption_all * 1e-6
                    pass

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

    def summary_info(self, info):
        if self.COLLECT_MODE == 1:
            data_collection_ratio = 1 - np.sum(np.sum(self.poi_value)) / (self.POI_INIT_DATA * self.POI_NUM)
            sep_collect = {self.map + 'f_data_collection_ratio_' + type: np.sum(
                [sum(self.uav_data_collect[type][uav_index]) for uav_index in range(self.NUM_UAV[type])]) / (
                                                                                 self.POI_INIT_DATA * self.POI_NUM)
                           for type in self.UAV_TYPE}
        else:
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

                if self.ROADMAP_MODE and type == 'carrier':
                    temp_info['carrier_node_history'] = self.carrier_node_history

                info[type] = temp_info

            info['map'] = self.map
            info['config'] = self.config.dict
            info['poi_history'] = self.poi_history
            path = self.args['save_path'] + '/map_{}_count_{}.txt'.format(self.map, self.SAVE_COUNT)
            self.save_variable(info, path)
            info = {}

        return info

    def save_trajectory_from_outside(self, info):
        for type in self.UAV_TYPE:
            temp_info = {}
            temp_info['uav_trace'] = self.uav_trace[type]
            max_len = max((len(l) for l in self.uav_data_collect[type]))
            new_matrix = list(
                map(lambda l: l + [0] * (max_len - len(l)), self.uav_data_collect[type]))
            temp_info['uav_collect'] = np.sum(new_matrix, axis=0).tolist()
            temp_info['reward_history'] = self.episodic_reward_list[type]
            temp_info['uav_reward'] = self.single_uav_reward_list[type]

            if self.ROADMAP_MODE and type == 'carrier':
                temp_info['carrier_node_history'] = self.carrier_node_history

            info[type] = temp_info

        info['map'] = self.map
        info['config'] = self.config.dict
        info['poi_history'] = self.poi_history
        path = self.args['save_path'] + \
               '/map_{}_{}.txt'.format(self.map, self.reset_count)
        for key in shared_feature_list:
            # if key == 'rm':
            #     continue
            try:
                del info['config'][key]
            except KeyError:
                continue
        self.save_variable(info, path)

    def save_variable(self, v, filename):
        # print('save variable to {}'.format(filename))
        f = open(filename, 'wb')
        pickle.dump(v, f)
        f.close()
        return filename

    def p_seed(self, seed=None):
        pass

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
                                                                                    2) + np.power(
                    height, 2))
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
        if self.ACTION_MODE == 1:
            dx, dy = self._get_vector_by_theta(action)
        else:
            dx, dy = self._get_vector_by_action(int(action))
            # dx, dy = self._get_vector_by_smart_action_(uav_index,int(action))
        distance = np.sqrt(np.power(dx * self.SCALE, 2) +
                           np.power(dy * self.SCALE, 2))
        energy_consume = self._cal_energy_consuming(distance, type)
        if self.uav_energy[type][uav_index] >= energy_consume:
            new_x, new_y = self.uav_position[type][uav_index][0] + dx, self.uav_position[type][uav_index][1] + dy
        else:
            new_x, new_y = self.uav_position[type][uav_index]

        return new_x, new_y, distance, min(self.uav_energy[type][uav_index], energy_consume)

    def _relay_association(self):
        '''
        每个无人机就近选择relay的无人车
        :return: relay_dict, 形如 {0: 1, 1: 1, 2: 1, 3: 1}
        '''
        # if self.config("fixed_relay"):
        #     return {i:i for i in range(self.NUM_UAV['uav'])}

        relay_dict = {}
        available_car = [1 for _ in range(self.NUM_UAV['carrier'])]
        for uav_index in range(self.NUM_UAV['uav']):
            dis_mat = [self._cal_distance(self.uav_position['uav'][uav_index], car_pos, 'uav') for car_pos in
                       self.uav_position['carrier']]
            for index in range(self.NUM_UAV['carrier']):
                if available_car[index] == 0 and self.config("fixed_relay"): dis_mat[index] = 999999999

            car_index = np.argmin(dis_mat)
            relay_dict[uav_index] = car_index
            available_car[car_index] = 0

        return relay_dict

    def _access_determin(self, CHANNELS):
        '''
        :param CHANNELS: 信道数量
        :return: sorted_access, 形如 {'uav': [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 'carrier': [[0, 0, 0]]}
        '''

        carrier_index = []
        sorted_access = {'uav': [], 'carrier': []}
        collect_range = self.COLLECT_RANGE if self.config("limited_collection") else {key: 99999 for key in
                                                                                      self.UAV_TYPE}
        for type in ['carrier']:
            for uav_pos in self.uav_position[type]:
                cur_poi_value = np.asarray(self.poi_value)
                distances = self._cal_distance(np.array(uav_pos), self.poi_position, type)
                distance_mask = np.logical_and(distances <= self.COLLECT_RANGE[type],
                                               cur_poi_value > self.USER_DATA_AMOUNT * 3)
                _, max_intake = compute_capacity_G2G(self.noma_config, distances)
                max_intake = max_intake / 1e6 * self.TIME_SLOT
                dis_list = -np.where(cur_poi_value < max_intake, cur_poi_value, max_intake) * distance_mask
                pois = self.get_valid_pois(CHANNELS, dis_list, distances, self.COLLECT_RANGE[type], type)
                # rate_list = []
                # for index,poi_pos in enumerate(self.poi_position):
                #     dis =  self._cal_distance(uav_pos, poi_pos,type)
                #     if dis <=  self.COLLECT_RANGE[type] and self.poi_value[index]>self.USER_DATA_AMOUNT*5:
                #         _, capacity_i = compute_capacity_G2G(self.noma_config,dis)
                #         rate_list.append(-min(self.poi_value[index],capacity_i/1e6*self.TIME_SLOT)) 
                #     else:
                #         rate_list.append(0)
                # #dis_list = np.array([max(1, self._cal_distance(uav_pos, poi_pos,0)) for poi_pos in self.poi_position])
                # dis_list = np.array(rate_list)
                # pois =  np.argsort(dis_list)[:CHANNELS]
                # #pois = [x for x in pois if self.poi_value[x]>self.USER_DATA_AMOUNT*2 and  self._cal_distance(uav_pos, self.poi_position[x])< self.COLLECT_RANGE[type]] #车一定做限制
                # pois = [x for x in pois if  self._cal_distance(uav_pos, self.poi_position[x],type) < self.COLLECT_RANGE[type]]
                # while len(pois) < CHANNELS: pois.append(-1)
                sorted_access[type].append(pois)
                carrier_index.extend(pois)

        for type in ['uav']:
            for uav_pos in self.uav_position[type]:
                distances = np.clip(self._cal_distance(self.poi_position, np.array(uav_pos), type), a_min=1,
                                    a_max=np.inf)
                np.put(distances, carrier_index, 9999999)
                pois = self.get_valid_pois(CHANNELS, distances, distances, collect_range[type], type)
                # dis_list = np.array([max(1, self._cal_distance(uav_pos, poi_pos, type)) if index not in carrier_index else 9999999 for index,poi_pos in enumerate(self.poi_position)])
                # pois =  np.argsort(dis_list)[:CHANNELS]
                # pois = [x for x in pois if self._cal_distance(uav_pos, self.poi_position[x], type)< collect_range[type]]
                # while len(pois) < CHANNELS: pois.append(-1)
                sorted_access[type].append(pois)

        return sorted_access

    def get_valid_pois(self, CHANNELS, dis_list, distances, threshold, type):
        pois = np.argsort(dis_list)[:CHANNELS]
        pois = pois[distances[pois] < threshold]
        if type == 'carrier':
            pois = pois[dis_list[pois] < 0]
        size = CHANNELS - len(pois)
        if size > 0:
            pois = np.concatenate([pois, np.full(size, fill_value=-1)])
        return pois.tolist()

    def _collect_data_by_noma(self, type, uav_index, relay_dict, sorted_access, collect_time=0):

        reward_list = []
        collect_list = []
        if self.DEBUG_MODE: print(f'====设备类型={type}，ID={uav_index}====')
        if collect_time <= 0: return 0, 0
        for channel in range(self.CHANNEL_NUM):
            poi_i_index = sorted_access[type][uav_index][channel]  # 确定从哪个poi收集数据
            if poi_i_index == -1: continue
            poi_i_pos = self.poi_position[poi_i_index]
            # 机和车以不同的方式计算capacity，遵循noma模型
            if type == 'carrier':
                # car从poi_i收集数据
                car_pos = self.uav_position[type][uav_index]
                sinr_i, capacity_i = compute_capacity_G2G(self.noma_config,
                                                          self._cal_distance(poi_i_pos, car_pos, type)
                                                          )
                capacity_i = capacity_i / 1e6

                if self.DEBUG_MODE: print(
                    f'在第{channel}个信道中，从ID={poi_i_index} poi收集数据，capacity={capacity_i}, collect_time ={collect_time}, 车PoI的距离是{self._cal_distance(poi_i_pos, car_pos, 0)},收集量={min(capacity_i * collect_time, self.poi_value[poi_i_index]) / self.USER_DATA_AMOUNT}')
            else:
                assert type == 'uav'
                relay_car_index = relay_dict[uav_index]  # 确定当前uav转发给哪个car
                uav_pos, relay_car_pos = self.uav_position[type][uav_index], self.uav_position['carrier'][
                    relay_car_index]
                # uav从poi_i收集数据，但poi_j会造成干扰
                poi_j_index = sorted_access['carrier'][relay_car_index][channel]

                if self._cal_distance(uav_pos, relay_car_pos, type) >= self.COLLECT_RANGE[type] and self.config(
                        "limited_collection"):
                    R_G2A = R_RE = 1
                    if self.DEBUG_MODE: print(
                        f"车机距离：{self._cal_distance(uav_pos, relay_car_pos, type)}, 无人机与poi i的距离{self._cal_distance(poi_i_pos, uav_pos, type)}")
                else:
                    if poi_j_index != -1:
                        poi_j_pos = self.poi_position[poi_j_index]

                        sinr_G2A, R_G2A = compute_capacity_G2A(self.noma_config,
                                                               self._cal_distance(poi_i_pos, uav_pos, type),
                                                               self._cal_distance(poi_j_pos, uav_pos, type),
                                                               )
                        sinr_RE, R_RE = compute_capacity_RE(self.noma_config,
                                                            self._cal_distance(uav_pos, relay_car_pos, type),
                                                            self._cal_distance(poi_i_pos, relay_car_pos, 'carrier'),
                                                            self._cal_distance(poi_j_pos, relay_car_pos, 'carrier'),
                                                            )
                    else:
                        uav_pos, relay_car_pos = self.uav_position[type][uav_index], self.uav_position['carrier'][
                            relay_car_index]
                        poi_j_pos = (99999, 999999)
                        sinr_G2A, R_G2A = compute_capacity_G2A(self.noma_config,
                                                               self._cal_distance(poi_i_pos, uav_pos, type),
                                                               -1,
                                                               )
                        sinr_RE, R_RE = compute_capacity_RE(self.noma_config,
                                                            self._cal_distance(uav_pos, relay_car_pos, type),
                                                            self._cal_distance(poi_i_pos, relay_car_pos, 'carrier'),
                                                            -1,
                                                            )
                    if self.DEBUG_MODE: print(
                        f"车机距离：{self._cal_distance(uav_pos, relay_car_pos, type)}, 无人机与poi i的距离{self._cal_distance(poi_i_pos, uav_pos, type)},无人机与poi_j的距离{self._cal_distance(poi_j_pos, uav_pos, type)}")

                # TODO 这里注意由于poi的功率远小于UAV的功率，所以更多情况下R_G2A比R_RE小，也即前者是瓶颈。可以统计一下前者更小的频率会不会太高
                capacity_i = min(R_G2A, R_RE) / 1e6  # 取两段信道的较小值

                if self.DEBUG_MODE: print(
                    f"在第{channel}个信道中，R_G2A={R_G2A / 1e6}，R_RE={R_RE / 1e6}，前者是后者的{'%.3f' % (R_G2A / R_RE * 100)}%")
                if self.DEBUG_MODE: print(
                    f'在第{channel}个信道中，从ID={poi_i_index} poi收集数据，并转发给ID={relay_car_index} carrier，受到ID={poi_j_index} poi的干扰，capacity={capacity_i}, collect_time ={collect_time}, 收集量={capacity_i * collect_time / self.USER_DATA_AMOUNT}')
            # 根据capacity进行数据收集
            collected_data = min(capacity_i * collect_time, self.poi_value[poi_i_index])

            if self.COLLECT_MODE == 0:
                self.poi_value[poi_i_index] -= collected_data
                reward_list.append(collected_data * self.reward_co[type][poi_i_index])
                collect_list.append(collected_data)

            elif self.COLLECT_MODE == 1:
                reward_list.append(self.poi_value[poi_i_index] * self.reward_co[type][poi_i_index])
                collect_list.append(self.poi_value[poi_i_index])
                self.poi_value[poi_i_index] = 0

            elif self.COLLECT_MODE == 2:
                self.poi_value[poi_i_index] -= collected_data
                reward_list.append(collected_data * self.reward_co[type][poi_i_index])
                collect_list.append(collected_data)

            self.last_collect[self.step_count][type][uav_index][poi_i_index] = 1

        return sum(reward_list), sum(collect_list)

    def _collect_data_from_poi(self, type, uav_index, collect_time=0):
        raise AssertionError
        reward_list = []
        if type == 'uav':
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
                    if self.COLLECT_MODE == 0:
                        collected_data = min(rate * collect_time / update_num, self.poi_value[poi_index])
                        self.poi_value[poi_index] -= collected_data
                        reward_list.append(collected_data)
                    elif self.COLLECT_MODE == 1:
                        reward_list.append(self.poi_value[poi_index])
                        self.poi_value[poi_index] = 0
                    elif self.COLLECT_MODE == 2:
                        collected_data = min(rate * collect_time, self.poi_value[poi_index])
                        self.poi_value[poi_index] -= collected_data
                        reward_list.append(collected_data)

        elif type == 'carrier':
            pass
        return sum(reward_list), len(reward_list)

    def _get_vector_by_theta(self, action):
        theta = action[0] * np.pi
        l = action[1] + 1
        dx = l * np.cos(theta)
        dy = l * np.sin(theta)
        return dx, dy

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

    def _get_vector_by_smart_action_(self, uav_index, action, type='uav'):
        single = 3
        base = single / math.sqrt(2)
        if action < 9:
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
                # [2*single,0],
                # [0,2*single],
                # [-2*single,0],
                # [0,-2*single]
            ]
            return action_table[action]
        elif action == 9:  # 向剩余数据量最大的poi 移动
            data_list = []
            for i in range(self.POI_NUM):
                dis = self._cal_distance(self.uav_position['uav'][uav_index], self.poi_position[i], type)
                data = self.poi_value[i] if dis < self.COLLECT_RANGE['uav'] else 0
                data_list.append(data)
            if len(data_list) > 0:
                move_target = np.argmax(data_list)
                dx = self.poi_position[move_target][0] - self.uav_position['uav'][uav_index][0]
                dy = self.poi_position[move_target][1] - self.uav_position['uav'][uav_index][1]
            else:
                dx = dy = 0
            return [dx, dy]
        else:  # 向无人车移动
            action = action - 10
            target_position = self.uav_position['carrier'][action]
            dx = target_position[0] - self.uav_position['uav'][uav_index][0]
            dy = target_position[1] - self.uav_position['uav'][uav_index][1]
            if math.sqrt(dx ** 2 + dy ** 2) > 4:
                dx = min(3, np.abs(dx)) * math.copysign(1, dx)
                dy = min(3, np.abs(dy)) * math.copysign(1, dy)
            return [dx, dy]

    def _is_uav_out_of_energy(self, uav_index, type):
        return self.uav_energy[type][uav_index] < self.EPSILON

    def _is_episode_done(self):
        if (self.step_count + 1) >= self.MAX_EPISODE_STEP:
            return True
        else:
            for type in self.UAV_TYPE:
                if type == 'carrier':
                    continue
                for i in range(self.NUM_UAV[type]):
                    if self._judge_obstacle(None, self.uav_position[type][i]):
                        print('cross the border!')
                        return True
            # return np.bool(np.all(self.dead_uav_list))
        return False

    def _judge_obstacle(self, cur_pos, next_pos):
        if self.ACTION_MODE == 2 or self.ACTION_MODE == 3: return False
        if cur_pos is not None:
            for o in self.OBSTACLE:
                vec = [[o[0], o[1]],
                       [o[2], o[3]],
                       [o[4], o[5]],
                       [o[6], o[7]]]
                if IsIntersec(cur_pos, next_pos, vec[0], vec[1]):
                    return True
                if IsIntersec(cur_pos, next_pos, vec[1], vec[2]):
                    return True
                if IsIntersec(cur_pos, next_pos, vec[2], vec[3]):
                    return True
                if IsIntersec(cur_pos, next_pos, vec[3], vec[0]):
                    return True

        if (0 <= next_pos[0] <= self.MAP_X) and (0 <= next_pos[1] <= self.MAP_Y):
            return False
        else:
            return True

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
        if not self.USE_HGCN:
            return_dict = {}
            for key1, s_dict in adj_dict.items():
                for key2, adj in s_dict.items():
                    return_dict[f"{key1}-{key2}"] = adj
            return return_dict

        uav_field = self.agent_field['uav']
        carrier_field = self.agent_field['carrier']
        # uav-carrier
        uav_carrier = np.zeros((self.NUM_UAV['uav'], self.NUM_UAV['carrier']))
        # relay_dict = self._relay_association()
        # for k, v in relay_dict.items():
        #    uav_carrier[k][v] = 1

        for uav_id in range(self.NUM_UAV['uav']):
            for carrier_id in range(self.NUM_UAV['carrier']):
                uav_carrier[uav_id][carrier_id] = self._cal_distance(self.uav_position['uav'][uav_id],
                                                                     self.uav_position['carrier'][carrier_id],
                                                                     type='uav') / (self.SCALE * self.MAP_X * 1.414) \
                                                  < uav_field

        # for uav_id in range(self.NUM_UAV['uav']):
        #     for carrier_id in range(self.NUM_UAV['carrier']):
        #         uav_carrier[uav_id][carrier_id] = self._get_data_rate(self.uav_position['uav'][uav_id],
        #                                                              self.uav_position['carrier'][carrier_id])

        record_times = 10  # 记录过去5步收集的poi index
        start_t = max(0, self.step_count - record_times)
        uav_collect = np.zeros((self.NUM_UAV['uav'], self.POI_NUM))
        carrier_collect = np.zeros((self.NUM_UAV['carrier'], self.POI_NUM))
        for t in range(start_t, self.step_count):
            uav_collect = np.logical_or(uav_collect, self.last_collect[t]['uav'])
            carrier_collect = np.logical_or(carrier_collect, self.last_collect[t]['carrier'])
        uav_collect = uav_collect.astype(np.float32)
        carrier_collect = carrier_collect.astype(np.float32)

        poi_visible = np.zeros(((self.n_agents, self.POI_NUM)))
        # uav-poi
        uav_poi = np.zeros(((self.NUM_UAV['uav'], self.POI_NUM)))
        for uav_id in range(self.NUM_UAV['uav']):
            uav_poi[uav_id] = uav_collect[uav_id]
            # poi_v = self._get_data_rate(np.array(self.uav_position['uav'][uav_id]), self.poi_position)
            # poi_visible[self.NUM_UAV['carrier'] + uav_id, :] = poi_v / np.linalg.norm(poi_v)

            poi_v = self._cal_distance(np.array(self.uav_position['uav'][uav_id]), self.poi_position, type='uav') < uav_field
            poi_visible[self.NUM_UAV['carrier']+uav_id,:] = poi_v

            # poi_v = self._get_data_rate(np.array(self.uav_position['uav'][uav_id]), self.poi_position)
            # poi_v /= np.linalg.norm(poi_v)
            # poi_visible[self.NUM_UAV['carrier'] + uav_id, :] = poi_v * (self._cal_distance(np.array(self.uav_position['uav'][uav_id]),
            #                                     self.poi_position, type='uav') < uav_field)

        # carrier-poi
        carrier_poi = np.zeros(((self.NUM_UAV['carrier'], self.POI_NUM)))
        for carrier_id in range(self.NUM_UAV['carrier']):
            carrier_poi[carrier_id] = carrier_collect[carrier_id]
            # carrier_v = self._get_data_rate(np.array(self.uav_position['carrier'][carrier_id]), self.poi_position)
            # poi_visible[carrier_id, :] = carrier_v / np.linalg.norm(carrier_v)

            carrier_v = self._cal_distance(np.array(self.uav_position['carrier'][carrier_id]),self.poi_position, type='carrier') < carrier_field
            poi_visible[carrier_id,:] = carrier_v

            # carrier_v = self._get_data_rate(np.array(self.uav_position['carrier'][carrier_id]), self.poi_position)
            # carrier_v /= np.linalg.norm(carrier_v)
            # poi_visible[carrier_id,:] = carrier_v * (self._cal_distance(np.array(self.uav_position['carrier'][carrier_id]),
            #                                            self.poi_position, type='carrier') < carrier_field)

        # carrier-roadmap
        carrier_road = np.zeros(((self.NUM_UAV['carrier'], len(self.nx_g))))
        num_action = self.action_space['carrier'].n
        for carrier_id in range(self.NUM_UAV['carrier']):
            for i in range(num_action):
                last_node = self.OSMNX_TO_NX[self.last_dst_node[carrier_id][i]]
                carrier_road[carrier_id][last_node] = 1

        # for carrier_id in self.NUM_UAV['carrier']:
        #     carrier_road[carrier_id] = 1 - self._cal_distance(self.uav_position['carrier'][carrier_id],self.poi_position)/(self.MAP_X*1.414)
  
        
        adj_dict['uav']['carrier'] = row_normalize(uav_carrier)
        adj_dict['uav']['poi'] = row_normalize(uav_poi)
        adj_dict['uav']['epoi'] = row_normalize(np.dot(uav_carrier, carrier_poi))
        adj_dict['uav']['road'] = row_normalize(np.dot(uav_carrier, carrier_road))

        adj_dict['carrier']['uav'] = row_normalize(uav_carrier.T)
        adj_dict['carrier']['poi'] = row_normalize(carrier_poi)
        adj_dict['carrier']['road'] = row_normalize(carrier_road)
        adj_dict['carrier']['epoi'] = row_normalize(np.dot(uav_carrier.T, uav_poi))

        adj_dict['poi']['uav'] = row_normalize(uav_poi.T)
        adj_dict['poi']['carrier'] = row_normalize(carrier_poi.T)

        adj_dict['road']['carrier'] = row_normalize(carrier_road.T)

        return_dict = {}
        for key1, s_dict in adj_dict.items():
            for key2, adj in s_dict.items():
                return_dict[f"{key1}-{key2}"] = adj

        # -----------------------------------------------------------------
        one_hot = np.eye(self.n_agents)
        agent_pos = []
        for t in self.NUM_UAV:
            for i in range(self.NUM_UAV[t]):
                agent_pos.append(self.uav_position[t][i][0] / self.MAP_X)
                agent_pos.append(self.uav_position[t][i][1] / self.MAP_Y)
        agent_pos = np.array([agent_pos for _ in range(self.n_agents)])

        poi_value = np.array([self.poi_value for _ in range(self.n_agents)])
        poi_position = np.array([self.poi_position for _ in range(self.n_agents)])
        dividend = self.get_poi_dividend()
        visible_poi_states = np.concatenate([poi_position[:, :, 0] * poi_visible / self.MAP_X,
                                             poi_position[:, :, 1] * poi_visible / self.MAP_Y,
                                             poi_value * poi_visible / dividend], axis=1)

        obs = np.concatenate([one_hot, agent_pos, visible_poi_states], axis=1)
        # -----------------------------------------------------------------
        return_dict.update({
            'carrier': obs[:self.NUM_UAV['carrier'], :],
            'uav': obs[self.NUM_UAV['carrier']:, :],
        })
        return return_dict

    
    def get_obs(self, aoi_now=None, aoi_next=None):

        if self.USE_HGCN:
            agents_obs = {}
        else:
            agents_obs = {key: np.vstack([self.get_obs_agent(i, type=key) for i in range(self.NUM_UAV[key])]) for key in
                          self.UAV_TYPE}
        action_mask = self.get_avail_actions()
        action_mask = {'mask_' + key: action_mask[key] for key in self.UAV_TYPE}

        # node_features = np.array([[data['py_x'] / self.MAP_X, data['py_y'] / self.MAP_Y,
        #                            data['data'] / (self.USER_DATA_AMOUNT * self.MAX_EPISODE_STEP)] for _, data in
        #                           self.nx_g.nodes(data=True)])
        # edge_features = np.array(self.nx_g.edges())
        # edge_features = self.edge_features
        obs_dict = {
            'State': self.get_state(),
            # 'State':[],
            **{
                'Nodes':self.get_node_agents(0, type='uav'),
                'Edges':self.edge_features,
            },
            # **{'Nodes_' + key: np.vstack([[self.get_node_agents(0, type=key)] for _ in range(self.NUM_UAV[key])]) for
            #    key in self.UAV_TYPE},
            # **{'Edges_' + key: np.vstack([[self.edge_features] for i in range(self.NUM_UAV[key])]) for key in
            #    self.UAV_TYPE},
            **action_mask,
            **agents_obs,
            **self.get_adjcent_dict()
        }

        return obs_dict

    def get_node_agents(self, agent_id, type, global_view=True):
        total_nodes_num = len(self.nx_g)
        datas = np.zeros((total_nodes_num, 3))
        for node in self.node_to_poi.values():
            data = self.nx_g.nodes(data=True)[node]
            datas[node, :] = data['py_x'], data['py_y'], data['data']
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

        if self.USE_HGCN:
            return np.zeros(self.obs_space['State'].shape, dtype=np.float32)

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
        dividend = self.get_poi_dividend()
        visible_poi_states = np.stack([self.poi_position[:, 0] * is_visible / self.MAP_X,
                                       self.poi_position[:, 1] * is_visible / self.MAP_Y,
                                       self.poi_value * is_visible / dividend], axis=1)

        id = agent_id if type == 'carrier' else self.NUM_UAV['carrier'] + agent_id
        one_hot = np.eye(self.n_agents)[id]
        return np.concatenate([one_hot, agent_pos, visible_poi_states.flatten()])

    def get_poi_dividend(self):
        if self.COLLECT_MODE:
            dividend = self.USER_DATA_AMOUNT * self.MAX_EPISODE_STEP
        else:
            dividend = self.POI_INIT_DATA
        return dividend

    def get_obs_size(self, visit_num=None):
        obs_size = {}
        num = self.n_agents

        for type in self.UAV_TYPE:
            if visit_num is None:
                visit_num = self.POI_NUM

            size = 2 * num + num
            if visit_num is None:
                size += self.POI_NUM * self.poi_property_num
            else:
                size += visit_num * self.poi_property_num
            obs_size[type] = size
        return obs_size

    def get_state(self):

        if not self.USE_HGCN:
            return np.zeros(self.obs_space['State'].shape, dtype=np.float32)

        obs = []
        for t in self.NUM_UAV:
            for i in range(self.NUM_UAV[t]):
                obs.append(self.uav_position[t][i][0] / self.MAP_X)
                obs.append(self.uav_position[t][i][1] / self.MAP_Y)

        dividend = self.get_poi_dividend()
        poi_value = np.array(self.poi_value)
        visible_poi_states = np.stack([self.poi_position[:, 0] / self.MAP_X,
                                       self.poi_position[:, 1] / self.MAP_Y,
                                       poi_value / dividend], axis=1)

        return np.concatenate([np.ones((self.n_agents,)), obs, visible_poi_states.flatten()])

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
                # dx, dy = self._get_vector_by_smart_action_(agent_id,i,type)
                dx, dy = self._get_vector_by_action(i, type)
                if not self._judge_obstacle((temp_x, temp_y), (dx + temp_x, dy + temp_y)):
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

    def get_total_actions(self):
        return self.n_actions

    def get_num_of_agents(self):
        return self.n_agents

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}

        return env_info

    def get_obs_from_outside(self):
        return self.get_obs()

    def check_arrival(self):
        for i in range(self.POI_NUM):
            self.nx_g.nodes[self.node_to_poi[i]]['data'] = 0

        for i in range(self.POI_NUM):
            if self.COLLECT_MODE == 1:
                self.poi_value[i] += 1
            elif self.COLLECT_MODE == 2:
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


def myfloor(x):
    a = x.astype(np.int)
    return a


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

def row_normalize(matrix):
    # 计算每行的和
    row_sums = matrix.sum(axis=1)[:, np.newaxis]

    # 避免除以0
    row_sums[row_sums == 0] = 1

    # 将每行除以其和
    normalized = matrix / row_sums

    return normalized