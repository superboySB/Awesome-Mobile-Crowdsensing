from typing import Union
import os
algo_name: str = "PPO"
num_episodes: int = 5000000
learning_rate: Union[float, list] = [[0, 4e-5], [num_episodes * 120, 0]]
RUN_NAME: str = "kdd2024"
ENV_NAME: str = "crowd_sim"
checkpoint_dir = os.path.join("/workspace", "saved_data", "checkpoints")
run_config = dict(
    model_ckpt_filepath=None,
    name=ENV_NAME,
    # Environment settings.
    env=dict(
        num_cars=3,  # number of drones in the environment
        num_drones=3,  # number of runners in the environment
    ),
    # 框架重要参数，当前两个环境均为 episode_length = 120
    # Trainer settings.
    trainer=dict(
        num_envs=500,  # number of environment replicas (number of GPU blocks used)
        train_batch_size=4000,  # total batch size used for training per iteration (across all the environments)
        num_episodes=num_episodes,
        # total number of episodes to run the training for (can be arbitrarily high!) # 120 x 5000000 = 600M
        num_mini_batches=4,  # number of mini-batches to split the training batch into
        seed=2023,

    ),
    # Policy network settings.
    policy=dict(
        car=dict(  # 无人车
            to_train=True,  # flag indicating whether the model needs to be trained
            algorithm=algo_name,  # algorithm used to train the policy
            vf_loss_coeff=1,  # loss coefficient for the value function loss
            entropy_coeff=0.01,  # coefficient for the entropy component of the loss
            clip_grad_norm=True,  # flag indicating whether to clip the gradient norm or not
            max_grad_norm=1,  # when clip_grad_norm is True, the clip level
            normalize_advantage=True,  # flag indicating whether to normalize advantage or not
            normalize_return=True,  # flag indicating whether to normalize return or not
            gamma=0.99,  # discount rate
            lr=learning_rate,  # learning rate
            use_gae=False,
            model=dict(
                type="fully_connected",
                fc_dims=[512, 512],
                model_ckpt_filepath=""
            ),  # policy model settings
        ),
        drone=dict(  # 无人机
            to_train=True,
            algorithm=algo_name,
            vf_loss_coeff=1,
            entropy_coeff=0.01,  # [[0, 0.5],[3000000, 0.01]]
            clip_grad_norm=True,
            max_grad_norm=0.5,
            normalize_advantage=True,
            normalize_return=True,
            gamma=0.99,
            lr=learning_rate,
            use_gae=False,
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
        basedir=os.path.join("/workspace", "saved_data", "checkpoints"),
        # base folder used for saving, do not change (because of docker)
        name=ENV_NAME,  # experiment name
        tag=RUN_NAME,  # experiment tag
    ),
)
PROJECT_NAME = "awesome-mcs"
