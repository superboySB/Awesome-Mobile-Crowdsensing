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
            # Env
            "seed": 0,
            "debug_mode": False,
            "action_root": 13,
            "max_episode_step": 120,
            "dataset": "KAIST",
            "test_mode": False,
            "scale": 50,
            "time_slot": 20,

            # Setting
            "concat_obs": False,

            # Energy
            "initial_energy": {'carrier': 311040.0, 'uav': 359640.0},  # 12 V*7.2Ah = 86.4Wh, 99.9WH  *3600 j
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
            "fixed_relay": False,
            "uav_poi_dis": -1,
            "colla_co": 0.2,
            "carrier_explore_reward": False,
        }


