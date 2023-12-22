import os
import numpy as np
from warp_drive.utils.common import get_project_root
from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from warp_drive.utils.env_registrar import EnvironmentRegistrar
from envs.crowd_sim.env_wrapper import CUDAEnvWrapper
from envs.crowd_sim.crowd_sim import CrowdSim, CUDACrowdSim


env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(CrowdSim.name, os.path.join(get_project_root(), "envs", "crowd_sim", "crowd_sim_step.cu"))
env_configs = {
    "test0": {
        "num_cars": 4,
        "num_drones": 0,
        "seed": 274880,
    },
    "test1": {
        "num_cars": 2,
        "num_drones": 2,
        "seed": 274880,
    },
    "test2": {
        "num_cars": 1,
        "num_drones": 4,
        "seed": 274880,
    },
    "test3": {
        "num_cars": 4,
        "num_drones": 1,
        "seed": 274880,
    },
    "test4": {
        "num_cars": 5,
        "num_drones": 5,
        "seed": 274880,
    },
    "test5": {
        "num_cars": 10,
        "num_drones": 10,
        "seed": 121024,
    },
    "test6": {
        "num_cars": 50,
        "num_drones": 50,
        "seed": 121024,
    },
}


testing_class = EnvironmentCPUvsGPU(
    cpu_env_class=CrowdSim,
    cuda_env_class=CUDACrowdSim,
    env_configs=env_configs,
    num_envs=2,
    num_episodes=2,
    env_wrapper=CUDAEnvWrapper,
    env_registrar=env_registrar,
)

testing_class.test_env_reset_and_step(consistency_threshold_pct=1, seed=17)