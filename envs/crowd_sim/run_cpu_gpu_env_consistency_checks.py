import os
import numpy as np
from warp_drive.utils.common import get_project_root
from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from warp_drive.utils.env_registrar import EnvironmentRegistrar
from envs.crowd_sim.env_wrapper import CrowdSimEnvWrapper
from envs.crowd_sim.crowd_sim import CrowdSim, CUDACrowdSim


env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(CrowdSim.name, os.path.join(get_project_root(), "envs", "crowd_sim", "crowd_sim_step.cu"))
env_configs = {
    # "test0": {
    #     "num_ground_agents": 1,
    #     "num_aerial_agents": 1,
    #     "seed": 274880,
    # },
    # "test1": {
    #     "num_ground_agents": 2,
    #     "num_aerial_agents": 2,
    #     "seed": 274880,
    # },
    # "test2": {
    #     "num_ground_agents": 1,
    #     "num_aerial_agents": 4,
    #     "seed": 274880,
    # },
    "test3": {
        "num_ground_agents": 4,
        "num_aerial_agents": 1,
        "seed": 274880,
    },
    "test4": {
        "num_ground_agents": 10,
        "num_aerial_agents": 10,
        "seed": 654208,
    },
    # "test5": {
    #     "num_ground_agents": 20,
    #     "num_aerial_agents": 20,
    #     "seed": 121024,
    # },
}


testing_class = EnvironmentCPUvsGPU(
    cpu_env_class=CrowdSim,
    cuda_env_class=CUDACrowdSim,
    env_configs=env_configs,
    num_envs=1,
    num_episodes=2,
    env_wrapper=CrowdSimEnvWrapper,
    env_registrar=env_registrar,
)

testing_class.test_env_reset_and_step(consistency_threshold_pct=1, seed=17)