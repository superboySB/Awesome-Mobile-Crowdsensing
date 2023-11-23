"""
Helper file for generating an environment rollout
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import art3d
from warp_drive.trainer_lightning import WarpDriveModule
from envs.crowd_sim.crowd_sim import CUDACrowdSim


def generate_crowd_sim_animations(
        trainer: WarpDriveModule
):
    assert trainer is not None
    episode_states = trainer.fetch_episode_states(
        ["agent_x", "agent_y"]
    )
    assert isinstance(episode_states, dict)
    env: CUDACrowdSim = trainer.cuda_envs.env
    # TODO: Hack at here, env is filled with the last timestep or something else, need to figure out why
    env.agent_x_timelist = episode_states['agent_x']
    env.agent_y_timelist = episode_states['agent_y']
    env.render(args.output_dir, args.plot_loop, args.moving_line)


def setup_cuda_crowd_sim_env():
    """
    common setup for cuda_crowd_sim_env
    """
    import os
    torch.set_float32_matmul_precision('medium')
    run_config = dict(
        name="crowd_sim",
        # Environment settings.
        env=dict(
            num_cars=4,  # number of drones in the environment
            num_drones=4,  # number of runners in the environment
        ),
        # 框架重要参数，当前两个环境均为 episode_length = 120
        # Trainer settings.
        trainer=dict(
            num_envs=500,  # number of environment replicas (number of GPU blocks used)
            train_batch_size=1000,  # total batch size used for training per iteration (across all the environments)
            num_episodes=5000000,
            # total number of episodes to run the training for (can be arbitrarily high!)   # 120 x 50000 = 6M
        ),
        # Policy network settings.
        policy=dict(
            car=dict(  # 无人车
                to_train=True,  # flag indicating whether the model needs to be trained
                algorithm="PPO",  # algorithm used to train the policy
                vf_loss_coeff=1,  # loss coefficient for the value function loss
                entropy_coeff=0.01,  # coefficient for the entropy component of the loss
                clip_grad_norm=True,  # flag indicating whether to clip the gradient norm or not
                max_grad_norm=10,  # when clip_grad_norm is True, the clip level
                normalize_advantage=True,  # flag indicating whether to normalize advantage or not
                normalize_return=False,  # flag indicating whether to normalize return or not
                gamma=0.99,  # discount rate
                lr=1e-5,  # learning rate
                model=dict(
                    type="fully_connected",
                    fc_dims=[512, 512],
                    model_ckpt_filepath=""
                ),  # policy model settings
            ),
            drone=dict(  # 无人机
                to_train=True,
                algorithm="PPO",
                vf_loss_coeff=1,
                entropy_coeff=0.01,  # [[0, 0.5],[3000000, 0.01]]
                clip_grad_norm=True,
                max_grad_norm=0.5,
                normalize_advantage=True,
                normalize_return=False,
                gamma=0.99,
                lr=1e-5,
                model=dict(
                    type="fully_connected",
                    fc_dims=[512, 512],
                    model_ckpt_filepath=""
                ),
            ),
        ),
        # Checkpoint saving setting.
        saving=dict(
            metrics_log_freq=100,  # how often (in iterations) to print the metrics
            model_params_save_freq=5000,  # how often (in iterations) to save the model parameters
            basedir="./saved_data",  # base folder used for saving
            name="crowd_sim",  # experiment name
            tag="infocom2022",  # experiment tag
        ),
    )
    env_registrar = EnvironmentRegistrar()
    env_registrar.add_cuda_env_src_path(CUDACrowdSim.name,
                                        os.path.join(get_project_root(), "envs", "crowd_sim", "crowd_sim_step.cu"))
    env_wrapper = CrowdSimEnvWrapper(
        CUDACrowdSim(**run_config["env"]),
        num_envs=run_config["trainer"]["num_envs"],
        env_backend="pycuda",
        env_registrar=env_registrar
    )
    # Agents can share policy models: this dictionary maps policy model names to agent ids.
    policy_tag_to_agent_id_map = {
        "car": list(env_wrapper.env.cars),
        "drone": list(env_wrapper.env.drones),
    }
    new_wd_module = WarpDriveModule(
        env_wrapper=env_wrapper,
        config=run_config,
        policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
        create_separate_placeholders_for_each_policy=False,  # TODO: True -> IPPO?
        obs_dim_corresponding_to_num_agents="first",
        verbose=True,
    )
    return new_wd_module


if __name__ == "__main__":
    import torch
    import argparse
    from envs.crowd_sim.crowd_sim import CUDACrowdSim
    from envs.crowd_sim.env_wrapper import CrowdSimEnvWrapper
    from warp_drive.utils.env_registrar import EnvironmentRegistrar
    from warp_drive.utils.common import get_project_root

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--ckpts_path', type=str
    # )
    parser.add_argument('--output_dir', type=str, default='./logs.html')
    parser.add_argument('--plot_loop', action='store_true')
    parser.add_argument('--moving_line', action='store_true')
    args = parser.parse_args()
    wd_module = setup_cuda_crowd_sim_env()
    wd_module.load_model_checkpoint({"car":
                                         "./saved_data/crowd_sim/infocom2022/1700472331/car_50000000.state_dict",
                                     "drone":
                                         "./saved_data/crowd_sim/infocom2022/1700472331/drone_50000000.state_dict"})
    generate_crowd_sim_animations(wd_module)
