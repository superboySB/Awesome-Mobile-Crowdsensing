import os
import numpy as np
from warp_drive.utils.common import get_project_root
from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from warp_drive.utils.env_registrar import EnvironmentRegistrar
from envs.tag_continuous.tag_continuous import TagContinuous


env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(TagContinuous.name, os.path.join(get_project_root(), "envs", "tag_continuous", "tag_continuous_step_pycuda.cu"))
env_configs = {
    "test1": {
        "num_taggers": 20,
        "num_runners": 10,
        "max_acceleration": 1,
        "max_turn": np.pi / 4,
        "num_acceleration_levels": 3,
        "num_turn_levels": 3,
        "grid_length": 10,
        "episode_length": 100,
        "seed": 274880,
        "skill_level_runner": 1,
        "skill_level_tagger": 1,
        "use_full_observation": True,
        "runner_exits_game_after_tagged": True,
        "tagging_distance": 0.0,
    },
    "test2": {
        "num_taggers": 30,
        "num_runners": 30,
        "max_acceleration": 0.05,
        "max_turn": np.pi / 4,
        "num_acceleration_levels": 3,
        "num_turn_levels": 3,
        "grid_length": 10,
        "episode_length": 100,
        "step_penalty_for_tagger": -0.1,
        "seed": 428096,
        "skill_level_runner": 1,
        "skill_level_tagger": 2,
        "use_full_observation": False,
        "runner_exits_game_after_tagged": False,
        "tagging_distance": 0.25,
    },
    "test3": {
        "num_taggers": 1,
        "num_runners": 4,
        "max_acceleration": 2,
        "max_turn": np.pi / 2,
        "num_acceleration_levels": 3,
        "num_turn_levels": 3,
        "grid_length": 10,
        "episode_length": 100,
        "step_reward_for_runner": 0.1,
        "seed": 654208,
        "skill_level_runner": 1,
        "skill_level_tagger": 0.5,
        "use_full_observation": False,
        "runner_exits_game_after_tagged": True,
    },
    "test4": {
        "num_taggers": 3,
        "num_runners": 2,
        "max_acceleration": 0.05,
        "max_turn": np.pi,
        "num_acceleration_levels": 3,
        "num_turn_levels": 3,
        "grid_length": 10,
        "episode_length": 100,
        "seed": 121024,
        "skill_level_runner": 0.5,
        "skill_level_tagger": 1,
        "use_full_observation": True,
        "runner_exits_game_after_tagged": False,
    },
}
testing_class = EnvironmentCPUvsGPU(
    dual_mode_env_class=TagContinuous,
    env_configs=env_configs,
    num_envs=2,
    num_episodes=2,
    env_registrar=env_registrar,
)

testing_class.test_env_reset_and_step(consistency_threshold_pct=1, seed=17)