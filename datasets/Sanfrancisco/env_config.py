import numpy as np

from datasets.env_test import try_sensing_range

class Config(object):
    def __init__(self):
        pass


class BaseEnvConfig(object):
    env = Config()
    env.num_timestep = 120  # 120x15=1800s=30min
    env.step_time = 15  # second per step
    env.max_uav_energy = 359640  # 359640 J <-- 359.64 kJ (4500mAh, 22.2v) 大疆经纬
    env.rotation_limit = 360
    env.diameter_of_human_blockers = 0.5  # m
    env.h_rx = 1.3  # m, height of RX
    env.h_b = 1.7  # m, height of a human blocker
    env.car_velocity = 8
    env.drone_velocity = 18
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
    env.tallest_locs = None  # obstacle
    env.no_fly_zone = None  # obstacle
    env.start_timestamp = 1519894800
    env.end_timestamp = 1519896600
    env.energy_factor = 3  # TODO: energy factor in reward function

    # TODO: San Francisco datasets
    env.lower_left = [-122.4620, 37.7441]
    env.upper_right = [-122.3829, 37.8137]
    env.nlon = 7910
    env.nlat = 6960
    env.human_num = 536
    env.dataset_dir = 'datasets/Sanfrancisco/ground_trajs.csv'
    env.drone_action_space = np.array([[0, 0], [300, 0], [-300, 0], [0, 300], [0, -300], [210, 210], [210, -210], [-210, 210],
                                [-210, -210]])
    env.drone_sensing_range = 200  # unit
    env.car_action_space = env.drone_action_space / 3
    env.car_sensing_range = env.drone_sensing_range
    env.drone_car_comm_range = 500
    
    env.max_x_distance = 6951  # m
    env.max_y_distance = 7734  # m
    env.density_of_human_blockers = 1057 / env.max_x_distance / env.max_y_distance  # block/m2
    env.tallest_locs = [(37.7899, -122.3969), (37.7952, -122.4028), (37.7897, -122.3953), (37.7919, -122.4038),
                        (37.7925, -122.4005), (37.7904, -122.3961), (37.7858, -122.3921), (37.7878, -122.3942),
                        (37.7903, -122.3942), (37.7905, -122.3972), (37.7929, -122.3979), (37.7895, -122.4003),
                        (37.7952, -122.3961), (37.7945, -122.3997), (37.7898, -122.4018), (37.7933, -122.3945),
                        (37.7904, -122.4013), (37.7864, -122.3921), (37.7918, -122.3988), (37.7905, -122.3991),
                        (37.7887, -122.4026), (37.7911, -122.3981), (37.7861, -122.4025), (37.7891, -122.4033),
                        (37.7906, -122.403), (37.7853, -122.4109), (37.7916, -122.3958), (37.794, -122.3974),
                        (37.7885, -122.3986), (37.7863, -122.4013), (37.7926, -122.3989), (37.7912, -122.3971),
                        (37.7919, -122.3975), (37.7928, -122.4052), (37.7887, -122.3922), (37.7892, -122.3975),
                        (37.787, -122.3927), (37.7872, -122.392), (37.7872, -122.3953), (37.7932, -122.3972),
                        (37.7849, -122.4043), (37.7912, -122.4028), (37.787, -122.4), (37.7859, -122.3938),
                        (37.79, -122.3917), (37.7894, -122.3907), (37.7888, -122.3994), (37.7867, -122.4019),
                        (37.7912, -122.395), (37.7951, -122.3974), (37.7949, -122.3985), (37.7909, -122.3967),
                        (37.7893, -122.4008), (37.7919, -122.3945), (37.7904, -122.4024), (37.7939, -122.4004),
                        (37.7767, -122.4192), (37.7887, -122.3915), (37.7737, -122.4183)]
    env.no_fly_zone = [[(4370.0, 3370.0), (4370.0, 3180.0), (4180.0, 3180.0), (4180.0, 3370.0)],
                       [(4470.0, 3020.0), (4470.0, 2880.0), (4280.0, 2880.0), (4280.0, 3020.0)],
                       [(5200.0, 4200.0), (5200.0, 4050.0), (5020.0, 4050.0), (5020.0, 4200.0)],
                       [(7100.0, 4450.0), (7100.0, 4090.0), (6580.0, 4090.0), (6580.0, 4450.0)],
                       [(7240.0, 4680.0), (7240.0, 4380.0), (6880.0, 4380.0), (6880.0, 4680.0)],
                       [(6300.0, 4360.0), (6300.0, 4000.0), (5660.0, 4000.0), (5660.0, 4360.0)],
                       [(6880.0, 5200.0), (6880.0, 4350.0), (6060.0, 4350.0), (6060.0, 5200.0)],
                       [(6130.0, 4960.0), (6130.0, 4360.0), (5540.0, 4360.0), (5540.0, 4960.0)],
                       [(6010.0, 5190.0), (6010.0, 5020.0), (5830.0, 5020.0), (5830.0, 5190.0)]]


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
