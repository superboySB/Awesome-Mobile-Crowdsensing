import os
import logging
from argparse import ArgumentParser
from datetime import datetime
import torch
import argparse

assert torch.cuda.device_count() > 0, "This code needs at least a GPU to run!"

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from envs.crowd_sim.crowd_sim import CUDACrowdSim
from envs.crowd_sim.env_wrapper import CrowdSimEnvWrapper
from envs.crowd_sim.crowd_sim import AOI_METRIC_NAME, DATA_METRIC_NAME, ENERGY_METRIC_NAME, COVERAGE_METRIC_NAME
from warp_drive.utils.env_registrar import EnvironmentRegistrar
from warp_drive.trainer_lightning import CUDACallback, PerfStatsCallback, WarpDriveModule
from warp_drive.utils.common import get_project_root
import multiprocessing as mp
import setproctitle

logging.getLogger().setLevel(logging.ERROR)

PROJECT_NAME = "awesome-mcs"
algo_name: str = "PPO"
learning_rate: float = 4e-5

parser: ArgumentParser = argparse.ArgumentParser()
parser.add_argument(
    '--track', action='store_true'
)
# check point contains two digits, one is the initialization time of the experiment,
# the other is the training timestep
parser.add_argument(
    '--ckpt', nargs=2, type=str, default=None
)
parser.add_argument(
    "--use_gae", action='store_true'
)
args = parser.parse_args()
mp.set_start_method("spawn", force=True)
torch.set_float32_matmul_precision('medium')
RUN_NAME = "kdd2024"
ENV_NAME = "crowd_sim"
run_config = dict(
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
        num_episodes=5000000,
        num_mini_batches=4,  # number of mini-batches to split the training batch into
        # total number of episodes to run the training for (can be arbitrarily high!)   # 120 x 50000 = 6M
    ),
    # Policy network settings.
    policy=dict(
        car=dict(  # 无人车
            to_train=True,  # flag indicating whether the model needs to be trained
            algorithm=algo_name,  # algorithm used to train the policy
            vf_loss_coeff=1,  # loss coefficient for the value function loss
            entropy_coeff=0.01,  # coefficient for the entropy component of the loss
            clip_grad_norm=True,  # flag indicating whether to clip the gradient norm or not
            max_grad_norm=10,  # when clip_grad_norm is True, the clip level
            normalize_advantage=True,  # flag indicating whether to normalize advantage or not
            normalize_return=False,  # flag indicating whether to normalize return or not
            gamma=0.99,  # discount rate
            lr=learning_rate,  # learning rate
            use_gae=args.use_gae,
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
            normalize_return=False,
            gamma=0.99,
            lr=learning_rate,
            use_gae=args.use_gae,
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
        name=ENV_NAME,  # experiment name
        tag=RUN_NAME,  # experiment tag
    ),
)

current_datetime = datetime.now()
# Format the date and time as a string
datetime_string = current_datetime.strftime("%m%d-%H%M%S")
expr_name = f"{datetime_string}_car{run_config['env']['num_cars']}_drone{run_config['env']['num_drones']}_kdd2024"
if args.ckpt is not None:
    expr_start_time, train_timestep = args.ckpt
    parent_path = os.path.join(get_project_root(), "saved_data", ENV_NAME, RUN_NAME, expr_start_time)
    run_config['policy']['car']['model']['model_ckpt_filepath'] = f"{parent_path}/car_{train_timestep}.state_dict"
    run_config['policy']['drone']['model']['model_ckpt_filepath'] = f"{parent_path}/drone_{train_timestep}.state_dict"
    expr_name += '_ckpt'
setproctitle.setproctitle(expr_name)
env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(CUDACrowdSim.name,
                                    os.path.join(get_project_root(), "envs", ENV_NAME, "crowd_sim_step.cu"))

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

wd_module = WarpDriveModule(
    env_wrapper=env_wrapper,
    config=run_config,
    policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
    create_separate_placeholders_for_each_policy=False,  # TODO: True -> IPPO?
    obs_dim_corresponding_to_num_agents="first",
    verbose=True,
    results_dir=expr_name
)

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
episode_length = env_wrapper.env.episode_length
training_batch_size = run_config["trainer"]["train_batch_size"]
num_epochs = int(num_episodes * episode_length / training_batch_size)

# Set reload_dataloaders_every_n_epochs=1 to invoke
# train_dataloader() each epoch.
if args.track:
    import wandb

    # Initialize a wandb run
    wandb_logger = WandbLogger(project=PROJECT_NAME, name=expr_name)
    wandb.init(project=PROJECT_NAME, name=expr_name)
    prefix = 'env/'
    wandb.define_metric(prefix + COVERAGE_METRIC_NAME, summary="max")
    wandb.define_metric(prefix + ENERGY_METRIC_NAME, summary="min")
    wandb.define_metric(prefix + DATA_METRIC_NAME, summary="max")
    wandb.define_metric(prefix + AOI_METRIC_NAME, summary="min")
else:
    wandb_logger = None
trainer = Trainer(
    accelerator="gpu",
    devices=num_gpus,
    callbacks=[cuda_callback, perf_stats_callback],
    max_epochs=num_epochs,
    reload_dataloaders_every_n_epochs=1,
    logger=wandb_logger,
)

trainer.fit(wd_module)

# anim = generate_tag_env_rollout_animation(wd_module, fps=25)
# anim.save("./mymovie.mp4")
wd_module.graceful_close()

# example shell
# CUDA_VISIBLE_DEVICES=1 PATH=/usr/local/cuda/bin:$PATH python train_rl_policy.py