"""
Helper file for generating an environment rollout
"""

from warp_drive.trainer_lightning import WarpDriveModule
from warp_drive.training.trainer import Metrics
from envs.crowd_sim.crowd_sim import COVERAGE_METRIC_NAME, LARGE_DATASET_NAME
from run_configs.mcs_configs_python import run_config, checkpoint_dir
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import re
import subprocess


def generate_crowd_sim_animations(
        trainer: WarpDriveModule,
        animate: bool = False,
        verbose: bool = False,
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
    metrics = env.collect_info()
    if verbose:
        printer = Metrics()
        printer.pretty_print(metrics)
    if animate:
        env.render(args.output_dir, args.plot_loop, args.moving_line)
    return metrics


def setup_cuda_crowd_sim_env(dynamic_zero_shot: bool = False, dataset: str = None):
    """
    common setup for cuda_crowd_sim_env
    """
    assert dataset is not None
    import os
    torch.set_float32_matmul_precision('medium')
    env_registrar = EnvironmentRegistrar()
    env_registrar.add_cuda_env_src_path(CUDACrowdSim.name,
                                        os.path.join(get_project_root(), "envs", "crowd_sim", "crowd_sim_step.cu"))
    if dataset == LARGE_DATASET_NAME:
        from datasets.Sanfrancisco.env_config import BaseEnvConfig
    elif dataset == "KAIST":
        from datasets.KAIST.env_config import BaseEnvConfig
    else:
        raise NotImplementedError(f"dataset {dataset} not supported")
    run_config["env"]['dynamic_zero_shot'] = dynamic_zero_shot
    run_config["env"]['env_config'] = BaseEnvConfig
    env_wrapper = CUDAEnvWrapper(
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


def extract_timestamps(directory):
    timestamps = set()
    pattern = re.compile(r'_(\d+).state_dict')

    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            timestamps.add(int(match.group(1)))

    return sorted(list(timestamps))


def benchmark_coverage(directory: str):
    """
    benchmark coverage for all checkpoints under a given directory
    """
    coverage_list = []
    best_coverage = 0
    best_timestep = -1
    timestamps = extract_timestamps(directory)
    if not timestamps:
        print("No valid checkpoints found.")
        return None
    for checkpoint_timestep in tqdm(timestamps):
        car_path = f"{directory}/car_{checkpoint_timestep}.state_dict"
        drone_path = f"{directory}/drone_{checkpoint_timestep}.state_dict"

        if os.path.exists(car_path) and os.path.exists(drone_path):
            wd_module.load_model_checkpoint_separate({"car": car_path, "drone": drone_path})
            metrics = generate_crowd_sim_animations(wd_module)
            coverage_list.append(metrics[COVERAGE_METRIC_NAME])

            if metrics[COVERAGE_METRIC_NAME] > best_coverage:
                best_coverage = metrics[COVERAGE_METRIC_NAME]
                best_timestep = checkpoint_timestep

    print(f"best zero shot coverage: {best_coverage}, timestep: {best_timestep}")
    # plot coverage and timestep
    # x label: timestep
    plt.xlabel('timestep')
    # y label: target coverage
    plt.ylabel('target coverage')
    # title: change of target coverage vs timestep
    plt.title('change of target coverage vs timestep')
    # x ticks change scale to 10^6
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.plot(timestamps, coverage_list)
    # get parent_path last dir name
    expr_name = directory.split('/')[-1]
    # save figure as image
    plt.savefig(f'./{expr_name}_coverage.png')
    return best_timestep


if __name__ == "__main__":
    import torch
    import argparse
    from envs.crowd_sim.crowd_sim import CUDACrowdSim
    from envs.crowd_sim.env_wrapper import CUDAEnvWrapper
    from warp_drive.utils.env_registrar import EnvironmentRegistrar
    from warp_drive.utils.common import get_project_root

    parser = argparse.ArgumentParser()
    default_output_dir = os.path.join('/workspace', 'saved_data', 'trajectories', 'logs.html')
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    parser.add_argument('--dataset', type=str, default=LARGE_DATASET_NAME)
    parser.add_argument('--plot_loop', action='store_true')
    parser.add_argument('--moving_line', action='store_true')
    parser.add_argument("--dyn_zero_shot", action="store_true")
    args = parser.parse_args()
    wd_module = setup_cuda_crowd_sim_env(args.dyn_zero_shot, args.dataset)
    # generalized from KAIST to San Francisco
    parent_path = checkpoint_dir
    # generalized from San Francisco to KAIST
    # parent_path = "./saved_data/crowd_sim/kdd2024/1202-160001_car3_drone3_kdd2024"
    # 1701321935 San Fran from scratch
    # 1701264833 KAIST from scratch, the best generalization at 260000000
    # best generalization for new points at
    timestep = 241999
    if timestep is not None:
        car_path = os.path.join(parent_path, f"car_{timestep}.state_dict")
        drone_path = os.path.join(parent_path, f"drone_{timestep}.state_dict")
        if os.path.exists(car_path) and os.path.exists(drone_path):
            print(f"loading {timestep} checkpoint")
            wd_module.load_model_checkpoint_separate({"car": car_path, "drone": drone_path})
        else:
            # full_ckpt_name = os.path.join(parent_path, f"checkpoint_epoch={timestep}.ckpt")
            full_ckpt_name = os.path.join(checkpoint_dir, f"checkpoint_epoch={timestep}.ckpt")
            # check if full checkpoint exists
            if os.path.exists(full_ckpt_name):
                # extract car and drone checkpoints from full checkpoint
                wd_module.load_model_checkpoint(full_ckpt_name)
            else:
                print("no valid checkpoint found")
        generate_crowd_sim_animations(wd_module, animate=True, verbose=True)
        # # send html to local
        # result_file = args.output_dir
        # destination = "Charlie@10.108.17.19:~/Downloads"
        # rsync_command = f"rsync -avz {result_file} {destination}"
        #
        # # Execute the rsync command
        # subprocess.run(rsync_command, shell=True)
