import json5
import numpy as np
import pandas as pd
import random


'''环境配置类'''


class Config(object):

    def __init__(self,
                 args=None):
        self.default_config()
        if args is not None:
            for key in args.keys():
                self.dict[key] = args[key]

        if self('debug_mode'):
            print("Init config")
            print("Config:")
            print(self.dict)

    def __call__(self, attr):
        assert attr in self.dict.keys(), print('key error[', attr, ']')
        return self.dict[attr]

    def save_config(self, outfile=None):
        if outfile is None:
            outfile = 'default_save.json5'
        json_str = json5.dumps(self.dict, indent=4)
        with open(outfile, 'w') as f:
            f.write(json_str)

    def default_config(self):
        self.dict = {
            "map": 0,
            "description": "default",
            # Env
            "centralized": False,
            "task_id": 0,
            "action_mode": 0,  # 1 for continuous,  0 for discrete, 
            "collect_mode": 2,  # 0 for collection, 1 for aoi 2 for continuous aoi
            "noma_mode": True,
            "roadmap_mode": True,
            "seed": 0,
            "debug_mode": False,
            "action_root": 13,
            "max_episode_step": 120,
            "dataset": "KAIST",
            "test_mode": False,
            "scale": 50,
            "time_slot": 20,
            "use_hgcn": False,

            # Setting
            "concat_obs": False,

            # Energy
            "initial_energy": {'carrier': 311040.0, 'uav': 359640.0},  # 12 V*7.2Ah = 86.4Wh, 99.9WH  *3600 j
            "obstacle_penalty": -1,
            "normalize": 0.1,
            "epsilon": 1e-3,

            # UAV
            "num_uav": {'carrier': 2, 'uav': 2},
            "agent_field": {'carrier': 500, 'uav': 500},
            "uav_speed": {'carrier': 100000, 'uav': 20},  # 假设车可以一边移动一边收集
            "uav_height": 100,
            "channel_num": 5,

            # PoI
            "update_num": {'carrier': 5, 'uav': 10},
            "collect_range": {'carrier': 250, 'uav': 500},
            "rate_threshold": {'carrier': 1, 'uav': 1},
            "user_data_amount": 40,  # 1080P 2Mbps * 20s = 40
            "poi_init_data": 20,

            "aoi_threshold": 60,
            "threshold_penalty": 0.1,

            # Manager
            "log_path": './log_result',

            # Running:
            'limited_collection': False,
            'random_map': False,
            "fixed_relay": False,
            "uav_poi_dis": -1,
            "colla_co": 0.2,
            "carrier_explore_reward": False,
        }

    def generate_task(self):
        location = [[0, 0, 20, 20], [20, 20, 40, 40], [40, 40, 60, 60], [0, 20, 20, 40], [0, 40, 20, 60],
                    [20, 0, 40, 20], [40, 0, 60, 20], [40, 20, 60, 40], [20, 40, 40, 60]]
        task_num = 5
        for i in range(task_num):
            l = location[np.random.randint(0, 9)]
            x_min, y_min, x_max, y_max = l
            poi_num = 35
            poi_list = []
            for i in range(poi_num):
                x = random.random() * (x_max - x_min) + x_min
                y = random.random() * (y_max - y_min) + y_min
                poi_list.append((x, y))
            print('[', end='')
            for index, p in enumerate(poi_list):
                if index != poi_num - 1:
                    print('[{:.2f},{:.2f}],'.format(p[0], p[1]))
                else:
                    print('[{:.2f},{:.2f}]'.format(p[0], p[1]))
            print('],', end='')

    def generate_speed(self):
        poi_num = 360
        poi_list = []
        for i in range(poi_num):
            poi_list.append(np.random.poisson(2) * 0.1)
        print(poi_list)


def get_poi():
    poi = pd.read_csv('/home/liuchi/fangchen/AirDropMCS/source_code/LaunchMCS/util/KAIST/human.csv')
    for row in poi.iterrows():
        print(f"[{row[1]['px'] / 2100.207579392558},{row[1]['py'] / 2174.930950809533}],")


if __name__ == '__main__':
    generate_poi()
