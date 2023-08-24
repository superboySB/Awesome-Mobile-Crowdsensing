import os
import numpy as np
from warp_drive.utils.common import get_project_root
from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from warp_drive.utils.env_registrar import EnvironmentRegistrar
from envs.tag_gridworld.tag_gridworld import TagGridWorld,CUDATagGridWorld


env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(TagGridWorld.name, os.path.join(get_project_root(), "envs", "tag_gridworld", "tag_gridworld_step_pycuda.cu"))
env_configs = {
    "test1": {
        "num_taggers": 4,
        "grid_length": 4,
        "episode_length": 20,
        "seed": 27,
        "wall_hit_penalty": 0.1,
        "tag_reward_for_tagger": 10.0,
        "tag_penalty_for_runner": 2.0,
        "step_cost_for_tagger": 0.01,
        "use_full_observation": True,
    },
    "test2": {
        "num_taggers": 4,
        "grid_length": 4,
        "episode_length": 20,
        "seed": 27,
        "wall_hit_penalty": 0.1,
        "tag_reward_for_tagger": 10.0,
        "tag_penalty_for_runner": 2.0,
        "step_cost_for_tagger": 0.01,
        "use_full_observation": False,
    },
}
testing_class = EnvironmentCPUvsGPU(
    cpu_env_class=TagGridWorld,
    cuda_env_class=CUDATagGridWorld,
    env_configs=env_configs,
    num_envs=2,
    num_episodes=2,
    env_registrar=env_registrar,
)

testing_class.test_env_reset_and_step(consistency_threshold_pct=1, seed=17)