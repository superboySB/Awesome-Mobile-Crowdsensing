import argparse
import logging
import os
from argparse import ArgumentParser
import wandb
import types
import traceback
import torch

from common import add_common_arguments, logging_dir, customize_experiment, closest_multiple
from run_configs.mcs_configs_python import run_config, PROJECT_NAME, RUN_NAME, ENV_NAME, checkpoint_dir
from envs.crowd_sim.crowd_sim import (AOI_METRIC_NAME, DATA_METRIC_NAME,
                                      ENERGY_METRIC_NAME, COVERAGE_METRIC_NAME, MAIN_METRIC_NAME, LARGE_DATASET_NAME)
from warp_drive.utils.common import get_project_root
from pytorch_lightning.callbacks import ModelCheckpoint
import multiprocessing as mp
import setproctitle

total_logging_limit = 100000

my_root = get_project_root()


def run_experiment():
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import Trainer
    from envs.crowd_sim.crowd_sim import CUDACrowdSim
    from envs.crowd_sim.env_wrapper import CUDAEnvWrapper
    from warp_drive.utils.env_registrar import EnvironmentRegistrar
    from warp_drive.trainer_lightning import CUDACallback, PerfStatsCallback, WarpDriveModule
    """
    run a single experiment
    """
    global GLOBAL_ARGS

    def update_dot_config(config):
        """
        automatically supports dot notation for nested parameters.
        """
        # Iterate through all keys in the dictionary
        for key in list(config.keys()):
            # Check if the key contains a dot, indicating a nested structure
            if '.' in key:
                # Split the key into outer and inner parts
                outer_key, inner_key = key.split('.', 1)

                # Ensure the outer key exists in the config, if not, initialize it
                if outer_key not in config:
                    config[outer_key] = {}

                # Check if inner key contains further nesting
                if '.' in inner_key:
                    # Recursive call to handle further nested keys
                    update_dot_config({inner_key: config[key]})
                    # Merge the updated inner dictionary with the corresponding outer dictionary
                    config[outer_key].update(config[inner_key])
                else:
                    # Update the nested value
                    config[outer_key][inner_key] = config[key]

    try:
        new_args = types.SimpleNamespace(**GLOBAL_ARGS)
        if new_args.track or new_args.sweep_config is not None:
            logging_option = WandbLogger(project=PROJECT_NAME, name=new_args.expr_name, save_dir=logging_dir)
            wandb.init(project=PROJECT_NAME, name=new_args.expr_name, group=new_args.group,
                       tags=[new_args.dataset] + new_args.tag if new_args.tag is not None else [],
                       config=run_config, dir=logging_dir)
            setproctitle.setproctitle(new_args.expr_name)
            # prefix = 'env/'
            wandb.define_metric(COVERAGE_METRIC_NAME, summary="max")
            wandb.define_metric(ENERGY_METRIC_NAME, summary="min")
            wandb.define_metric(DATA_METRIC_NAME, summary="max")
            wandb.define_metric(MAIN_METRIC_NAME, summary="max")
            wandb.define_metric(AOI_METRIC_NAME, summary="min")
            actual_config = wandb.config
        else:
            logging_option = False
            actual_config = run_config
        actual_config['env_args']['env_config'] = new_args.env_config
        update_dot_config(actual_config)
        train_batch_size = actual_config['trainer']['train_batch_size']
        num_envs = actual_config['trainer']['num_envs']
        num_mini_batches = actual_config['trainer']['num_mini_batches']
        if (train_batch_size // num_envs) < num_mini_batches:
            print("Batch per env must be larger than num_mini_batches, Exiting...")
            return
        env_registrar = EnvironmentRegistrar()
        env_registrar.add_cuda_env_src_path(CUDACrowdSim.name,
                                            os.path.join(my_root, "envs", ENV_NAME, "crowd_sim_step.cu"))
        actual_config["env_args"]['dynamic_zero_shot'] = new_args.dynamic_zero_shot
        env_wrapper = CUDAEnvWrapper(
            CUDACrowdSim(**actual_config["env_args"]),
            num_envs=num_envs,
            env_backend="pycuda",
            env_registrar=env_registrar
        )
        # Agents can share policy models: this dictionary maps policy model names to agent ids.
        policy_tag_to_agent_id_map = {
            "car": list(env_wrapper.env.cars),
            "drone": list(env_wrapper.env.drones),
        }
        episode_length = env_wrapper.env.episode_length
        rollout_count = episode_length // (train_batch_size // num_envs)
        print_freq = closest_multiple(rollout_count, actual_config["saving"]["metrics_print_freq"])
        actual_config["saving"]["metrics_print_freq"] = print_freq
        # logging_limit = 100000
        # num_iters = actual_config['trainer']['num_episodes'] * episode_length / train_batch_size
        # log_freq = 1 if num_iters > logging_limit else ceil(num_iters / logging_limit)
        # Define callbacks.
        wd_module = WarpDriveModule(
            env_wrapper=env_wrapper,
            config=dict(actual_config),
            policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
            create_separate_placeholders_for_each_policy=False,  # TODO: True -> IPPO?
            obs_dim_corresponding_to_num_agents="first",
            verbose=True,
            results_dir=expr_name
        )
        cuda_callback = CUDACallback(module=wd_module)
        perf_stats_callback = PerfStatsCallback(
            batch_size=wd_module.training_batch_size,
            num_iters=wd_module.num_iters,
            log_freq=print_freq,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(os.path.join(checkpoint_dir, ENV_NAME, RUN_NAME, expr_name)),
            # Specify the path to save the checkpoints
            filename='checkpoint_{epoch}',  # Naming convention for checkpoint files
            every_n_epochs=actual_config['saving']['model_params_save_freq'],
            every_n_train_steps=None,  # Set this to a number if you want checkpointing by steps
            save_top_k=-1,  # Set how many of the latest checkpoints you want to keep
            save_weights_only=False  # Set to False if you want to save the whole model
        )
        # Instantiate the PytorchLightning trainer with the callbacks.
        # # Also, set the number of gpus to 1, since this notebook uses just a single GPU.
        num_gpus = 1
        num_episodes = actual_config["trainer"]["num_episodes"]
        training_batch_size = actual_config["trainer"]["train_batch_size"]
        num_epochs = int(num_episodes * episode_length / training_batch_size)
        # Set reload_dataloaders_every_n_epochs=1 to invoke
        # train_dataloader() each epoch.
        trainer = Trainer(
            accelerator="gpu",
            devices=num_gpus,
            callbacks=[cuda_callback, perf_stats_callback, checkpoint_callback],
            max_epochs=num_epochs,
            reload_dataloaders_every_n_epochs=1,
            logger=logging_option,
            enable_checkpointing=True,
        )
        trainer.fit(wd_module, ckpt_path=run_config["model_ckpt_filepath"])
        # anim = generate_tag_env_rollout_animation(wd_module, fps=25)
        # anim.save("./my_movie.mp4")
        wd_module.graceful_close()
    except Exception as e:
        # Capture the traceback
        error_trace = traceback.format_exc()
        # Log the error and its traceback to W&B, failing
        # if GLOBAL_ARGS['track']:
        #     wandb.alert(
        #         title='Error in Training Run',
        #         text=f'Error: {str(e)}\nTraceback:\n{error_trace}',
        #         level=wandb.AlertLevel.ERROR
        #     )
        # print error trace
        print(f'Error: {str(e)}\nTraceback:\n{error_trace}')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    mp.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision('medium')
    parser: ArgumentParser = argparse.ArgumentParser()
    GLOBAL_ARGS = None
    num_of_gpu_available = torch.cuda.device_count()
    assert num_of_gpu_available > 0, "This code needs at least a GPU to run!"
    add_common_arguments(parser)

    parser.add_argument(
        "--auto_scale",
        "-a",
        action="store_true",
        help="perform auto scaling.",
    )
    parser.add_argument(
        '--sweep-config',
        type=str,
        default=None,
        help="sweep config for wandb"
    )
    parser.add_argument(
        '--ckpt', nargs=2, type=str, default=None, help="checkpoint to load, the first is the experiment name,"
                                                        " the second is the timestamp."
    )
    args = parser.parse_args()
    expr_name = customize_experiment(args, run_config=run_config)
    if args.ckpt is not None:
        expr_start_time, train_timestep = args.ckpt
        parent_path = os.path.join(checkpoint_dir, ENV_NAME, RUN_NAME, expr_start_time)

        # Check if separate state dict exists for 'car'
        car_state_dict_path = os.path.join(str(parent_path), f"car_{train_timestep}.state_dict")
        if os.path.exists(car_state_dict_path):
            run_config['policy']['car']['model']['model_ckpt_filepath'] = car_state_dict_path

            # Check for 'drone' state dict
            drone_state_dict_path = os.path.join(str(parent_path), f"drone_{train_timestep}.state_dict")
            run_config['policy']['drone']['model']['model_ckpt_filepath'] = drone_state_dict_path
        else:
            # Use the newer version, full state dict
            full_ckpt_path = os.path.join(str(parent_path), f"checkpoint_epoch={train_timestep}.ckpt")
            run_config['model_ckpt_filepath'] = full_ckpt_path

        expr_name += '_ckpt'
    if args.dataset == LARGE_DATASET_NAME:
        from datasets.Sanfrancisco.env_config import BaseEnvConfig
    else:
        from datasets.KAIST.env_config import BaseEnvConfig
    args = parser.parse_args()
    if args.track and args.sweep_config is not None:
        raise ValueError("Cannot enable both 'track' and 'sweep_config'.")
    if args.sweep_config is not None and not args.track:
        # Run a sweep with W&B, currently failing
        import yaml

        with open(args.sweep_config) as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        GLOBAL_ARGS = vars(args)  # Set the global variable for W&B config
        GLOBAL_ARGS.update({'env_config': BaseEnvConfig, 'expr_name': expr_name})
        sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
        wandb.agent(sweep_id, function=run_experiment)
    else:
        # Run the experiment without W&B
        config_dict = vars(args)
        config_dict.update({'env_config': BaseEnvConfig, 'expr_name': expr_name})
        GLOBAL_ARGS = config_dict  # Update GLOBAL_ARGS without W&B config
        run_experiment()
# example shell
# CUDA_VISIBLE_DEVICES=1 PATH=/usr/local/cuda/bin:$PATH python train_rl_policy.py
# --tag lr=4e-5 batch_size=8000
# --sweep-config run_configs/sweep_test.yaml
