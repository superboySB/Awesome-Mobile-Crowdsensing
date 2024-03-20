import os

import math
import numpy as np

from datasets.env_test import try_sensing_range
from warp_drive.utils.common import get_project_root


class Config(object):
    def __init__(self):
        pass


class BaseEnvConfig(object):
    env = Config()
    env.dataset_name = 'Chengdu'
    env.aoi_threshold = 30
    env.emergency_threshold = 10
    env.num_timestep = 120  # num_timestep x step_time = 1800s = 30min
    env.step_time = 15  # second per step
    env.max_uav_energy = 467856
    # 359640 J <-- 359.64 kJ (4500mAh, 22.2v) 大疆经纬
    env.rotation_limit = 360
    env.diameter_of_human_blockers = 0.5  # m
    env.h_rx = 1.3  # m, height of RX
    env.h_b = 1.7  # m, height of a human blocker
    env.car_velocity = 8
    env.drone_velocity = 20  # https://www.dji.com/hk/matrice600
    env.frequence_band = 28  # GHz
    env.h_d = 120  # m, height of drone-BS
    env.alpha_nlos = 66.25
    env.beta_nlos = 3.3
    env.zeta_nlos = 4.48  # Frequency 28GHz, high-rise building
    env.alpha_los = 88.76
    env.beta_los = 1.68
    env.zeta_los = 2.47
    env.g_tx = 10  # dB
    env.g_rx = 5  # dB
    env.start_timestamp = 1479258000
    env.end_timestamp = 1479259800

    env.lower_left = [104.04319927137526, 30.657065702618834]
    env.upper_right = [104.09715856773038, 30.711024998973958]
    env.dataset_dir = os.path.join(get_project_root(), 'datasets', 'Chengdu', 'ground_trajs_0900_0930.csv')
    hypo = 300
    leg = hypo / math.sqrt(2)

    env.drone_action_space = np.array([[0, 0], [hypo, 0], [-hypo, 0],
                                       [0, hypo], [0, -hypo], [leg, leg],
                                       [leg, -leg], [-leg, leg], [-leg, -leg]])
    # env.drone_action_space = np.array([[0, 0],
    #                                    [300, 0], [-300, 0],
    #                                    [0, 300], [0, -300],
    #                                    [210, 210], [210, -210], [-210, 210], [-210, -210],
    #                                    [150, 0], [-150, 0],
    #                                    [0, 150], [0, -150],
    #                                    [60, 60], [60, -60], [-60, 60], [-60, -60]])
    env.drone_sensing_range = 500  # unit
    env.car_action_space = env.drone_action_space / 3
    env.car_sensing_range = 250
    env.drone_car_comm_range = 1000

    env.max_x_distance = 6000  # m
    env.max_y_distance = 6000  # m
    env.density_of_human_blockers = 1057 / env.max_x_distance / env.max_y_distance  # block/m2

    def __init__(self, debug=False):
        pass


# r:meters, 2d distance
# threshold: dB

# Maximum Coupling Loss (110dB is recommended)
# san:
# 123dB -> 600m -> 600 range
# 121dB -> 450m -> 450 range
# 119dB -> 330m -> 330 range
# 117dB -> 240m -> 240 range √
# 115dB -> 165m -> 165 range


if __name__ == "__main__":
    try_sensing_range(220, BaseEnvConfig)
