import torch

assert torch.cuda.device_count() > 0, "This notebook needs a GPU to run!"

# %%
import logging

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from envs.tag_continuous.generate_rollout_animation import (
    generate_tag_env_rollout_animation,
)
from envs.tag_continuous.tag_continuous import TagContinuous
from warp_drive.env_wrapper import EnvWrapper
from warp_drive.trainer_lightning import (
    CUDACallback,
    PerfStatsCallback,
    WarpDriveModule,
)

# %%
# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)


# %%
run_config = dict(
    name="tag_continuous",
    # Environment settings.
    env=dict(
        num_taggers=5,  # number of taggers in the environment
        num_runners=100,  # number of runners in the environment
        grid_length=20.0,  # length of the (square) grid on which the game is played
        episode_length=400,  # episode length in timesteps
        max_acceleration=0.1,  # maximum acceleration
        min_acceleration=-0.1,  # minimum acceleration
        max_turn=2.35,  # 3*pi/4 radians
        min_turn=-2.35,  # -3*pi/4 radians
        num_acceleration_levels=10,  # number of discretized accelerate actions
        num_turn_levels=10,  # number of discretized turn actions
        skill_level_tagger=1.0,  # skill level for the tagger
        skill_level_runner=1.0,  # skill level for the runner
        use_full_observation=False,  # each agent only sees full or partial information
        runner_exits_game_after_tagged=True,  # flag to indicate if a runner stays in the game after getting tagged
        num_other_agents_observed=10,  # number of other agents each agent can see
        tag_reward_for_tagger=10.0,  # positive reward for the tagger upon tagging a runner
        tag_penalty_for_runner=-10.0,  # negative reward for the runner upon getting tagged
        end_of_game_reward_for_runner=1.0,  # reward at the end of the game for a runner that isn't tagged
        tagging_distance=0.02,  # margin between a tagger and runner to consider the runner as 'tagged'.
    ),
    # Trainer settings.
    trainer=dict(
        num_envs=200,  # number of environment replicas (number of GPU blocks used)
        train_batch_size=10000,  # total batch size used for training per iteration (across all the environments)
        num_episodes=2000000,  # total number of episodes to run the training for (can be arbitrarily high!)
    ),
    # Policy network settings.
    policy=dict(
        runner=dict(    # 或许可以给无人机
            to_train=True,  # flag indicating whether the model needs to be trained
            algorithm="A2C",  # algorithm used to train the policy
            gamma=0.98,  # discount rate
            lr=0.005,  # learning rate
            model=dict(
                type="fully_connected", fc_dims=[512, 512], model_ckpt_filepath=""
            ),  # policy model settings
        ),
        tagger=dict(    # 或许可以给无人车
            to_train=True,
            algorithm="A2C",
            gamma=0.98,
            lr=0.002,
            model=dict(
                type="fully_connected", fc_dims=[512, 512], model_ckpt_filepath=""
            ),
        ),
    ),
    # Checkpoint saving setting.
    saving=dict(
        metrics_log_freq=10,  # how often (in iterations) to print the metrics
        model_params_save_freq=5000,  # how often (in iterations) to save the model parameters
        basedir="./saved_data",  # base folder used for saving
        name="continuous_tag",  # experiment name
        tag="example",  # experiment tag
    ),
)


env_wrapper = EnvWrapper(
    TagContinuous(**run_config["env"]),
    num_envs=run_config["trainer"]["num_envs"],
    env_backend="pycuda",
)

# Agents can share policy models: this dictionary maps policy model names to agent ids.
policy_tag_to_agent_id_map = {
    "tagger": list(env_wrapper.env.taggers),
    "runner": list(env_wrapper.env.runners),
}

wd_module = WarpDriveModule(
    env_wrapper=env_wrapper,
    config=run_config,
    policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
    verbose=True,
)

# anim = generate_tag_env_rollout_animation(wd_module, fps=25)
# anim.to_html5_video()


log_freq = run_config["saving"]["metrics_log_freq"]

# Define callbacks.
cuda_callback = CUDACallback(module=wd_module)
perf_stats_callback = PerfStatsCallback(
    batch_size=wd_module.training_batch_size,
    num_iters=wd_module.num_iters,
    log_freq=log_freq,
)

# Instantiate the PytorchLightning trainer with the callbacks.
# # Also, set the number of gpus to 1, since this notebook uses just a single GPU.
num_gpus = 1
num_episodes = run_config["trainer"]["num_episodes"]
episode_length = run_config["env"]["episode_length"]
training_batch_size = run_config["trainer"]["train_batch_size"]
num_epochs = int(num_episodes * episode_length / training_batch_size)

# Set reload_dataloaders_every_n_epochs=1 to invoke
# train_dataloader() each epoch.
wandb_logger = WandbLogger(project="awesome-mcs",name="data-collection")
trainer = Trainer(
    accelerator="gpu",
    devices=num_gpus,
    callbacks=[cuda_callback, perf_stats_callback],
    max_epochs=num_epochs,
    reload_dataloaders_every_n_epochs=1,
    logger = wandb_logger
)

trainer.fit(wd_module)


anim = generate_tag_env_rollout_animation(wd_module, fps=25)
anim.save("./mymovie.mp4")


wd_module.graceful_close()
