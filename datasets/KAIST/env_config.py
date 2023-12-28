import os

import numpy as np
from datasets.env_test import try_sensing_range
from warp_drive.utils.common import get_project_root


class Config(object):
    def __init__(self):
        pass


class BaseEnvConfig(object):
    env = Config()
    env.aoi_threshold = 10
    env.num_timestep = 120  # 120x15=1800s=30min
    env.step_time = 15  # second per step
    env.max_uav_energy = 359640  # 359640 J <-- 359.64 kJ (4500mah, 22.2v) 大疆经纬
    env.rotation_limit = 360
    env.diameter_of_human_blockers = 0.5  # m
    env.h_rx = 1.3  # m, height of RX
    env.h_b = 1.7  # m, height of a human blocker
    env.drone_velocity = 18
    env.car_velocity = 8
    env.frequence_band = 28  # GHz
    env.h_d = 120  # m, height of drone-BS
    env.alpha_nlos = 113.63
    env.beta_nlos = 1.16
    env.zeta_nlos = 2.58  # Frequency 28GHz, sub-urban. channel model
    env.alpha_los = 84.64
    env.beta_los = 1.55
    env.zeta_los = 0.12
    env.g_tx = 0  # dB
    env.g_rx = 5  # dB
    env.start_timestamp = 1519894800
    env.end_timestamp = 1519896600
    env.energy_factor = 3  # TODO: energy factor in reward function

    # KAIST
    env.lower_left = [127.3475, 36.3597]
    env.upper_right = [127.3709, 36.3793]
    env.nlon = 2340
    env.nlat = 1960
    env.human_num = 92
    env.dataset_dir = os.path.join(get_project_root(), 'datasets', 'KAIST', 'ground_trajs.csv')
    
    env.drone_action_space = np.array([[0, 0], [300, 0], [-300, 0], [0, 300], [0, -300], [210, 210], [210, -210], [-210, 210],
                                [-210, -210]])
    env.drone_sensing_range = 200  # unit
    env.car_action_space = env.drone_action_space / 3
    env.car_sensing_range = env.drone_sensing_range  # TODO: 为方便判断，暂时设置成空地同一个sensing range
    env.drone_car_comm_range = 500
    
    env.max_x_distance = 2100.207579392558
    env.max_y_distance = 2174.930950809533
    env.density_of_human_blockers = 30000 / env.max_x_distance / env.max_y_distance  # block/m2


    def __init__(self, debug=False):
        pass


# r:meters, 2d distance
# threshold: dB


# Maximum Coupling Loss (110dB is recommended)
# kaist:
# 123dB -> 600m -> 600 range
# 121dB -> 435m -> 435 range
# 119dB -> 315m -> 315 range
# 117dB -> 220m -> 220 range √
# 115dB -> 145m -> 145 range


if __name__ == "__main__":
    try_sensing_range(220, BaseEnvConfig)
