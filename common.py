import os
import re
from argparse import ArgumentParser
from typing import Union

import yaml
from datetime import datetime
import argparse
from envs.crowd_sim.crowd_sim import LARGE_DATASET_NAME

# 'encoder_layer', 'core_arch', 'cut_points', 'fix_target', 'num_drones', 'num_cars', 'share_policy',
# 'gen_interval', 'intrinsic_mode', 'emergency_reward'
display_tags = {'selector_type', 'emergency_queue_length', 'dataset', 'alpha'}
ignore_tags = {'dynamic_zero_shot', 'fix_target', 'use_2d_state', 'reward_mode', 'emergency_threshold',
               'with_programming_optimization', 'use_random'}

logging_dir = os.path.join("/workspace", "saved_data")


def add_common_arguments(parser: ArgumentParser):
    """
    Fill in the common arguments for all experiments
    """
    parser.add_argument(
        '--track', action='store_true'
    )
    # check point contains two digits, one is the initialization time of the experiment,
    # the other is the training timestep
    parser.add_argument(
        '--group', type=str, default='debug', help="group name for wandb project"
    )
    parser.add_argument(
        '--dataset', type=str, default=LARGE_DATASET_NAME, help="The map chosen to deploy"
    )
    parser.add_argument(
        '--tag',
        nargs='+',
        type=str,
        default=None,
        help="special comments for each experiment"
    )
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="gpu id to use, will be setted to CUDA_VISIBLE_DEVICES")
    # add argument for resume
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument(
        '--algo',
        type=str,
        default='ippo',
        help='algorithm to choose (warpdrive does not support algorithms other than ippo)',
    )
    parser.add_argument(
        '--dynamic_zero_shot',
        action='store_true',
        help='enable dynamic zero shot'
    )
    parser.add_argument('--env', type=str, default='crowdsim', help='select environment')
    parser.add_argument('--num_drones', type=int, default=2, help='number of drones')
    parser.add_argument('--num_cars', type=int, default=2, help='number of cars')
    # parser.add_argument("--num_envs", type=int, default=500, help='number of environments to sample')
    parser.add_argument('--fix_target', action='store_true', default=True, help='fix target')
    parser.add_argument('--gen_interval', type=int, default=10, help='time interval between '
                                                                     'two generations of emergencies')
    parser.add_argument('--evaluation_interval', type=int, default=1000, help='evaluation interval')
    parser.add_argument("--cut_points", type=int, default=-1, help='number of points allowed')
    parser.add_argument('--emergency_threshold', type=int, default=10, help='emergency threshold')

def customize_experiment(args: argparse.Namespace, run_config: dict = None, yaml_config_path: str = None, ):
    """
    Setup tags to update hyperparameters based on argparse arguments and optionally a YAML configuration file.
    The function appends attributes set to True in both argparse and the YAML file as tags.
    """
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%m%d-%H%M%S")

    expr_name = f"{datetime_string}_{args.algo}"
    tags = args.tag if args.tag is not None else []
    # Process argparse arguments
    for arg in list(vars(args)):
        if arg != "track":
            attr_val = getattr(args, arg)
            if attr_val is True and arg not in ignore_tags:
                tags.append(arg)
            elif attr_val is not False and arg in display_tags:
                if isinstance(attr_val, list):
                    attr_val = "_".join(map(str, attr_val))
                else:
                    attr_val = str(attr_val)
                expr_name += f"_{arg}_{attr_val}"
    # Process YAML configuration if provided
    if yaml_config_path:
        with open(yaml_config_path, 'r') as file:
            yaml_config = yaml.safe_load(file)

        for key, value in yaml_config.items():
            if isinstance(value, bool) and value:
                tags.append(key)

    # Append tags to expr_name if any
    if tags:
        full_tag = '_'.join(tags)
        expr_name += f"_{full_tag}"

    if run_config is not None:
        update_config_from_tags(tags, run_config)
    args.tag = tags
    return expr_name


def update_config_from_tags(tags, config):
    """
    automatically update config from tags, following the format:
    tag1=value1 tag2=value2 ...
    when inputting a schedule, use the format:
    tag1=[[epoch1,value1],[epoch2,value2], ...], without space
    """
    # Define the keys and their corresponding paths in the config
    config_paths = {
        "lr": ('policy', 'lr'),  # Both car and drone
        "use_gae": ('policy', 'use_gae'),
        "batch_size": ('trainer', 'train_batch_size'),
        "num_envs": ('trainer', 'num_envs'),
        "num_episodes": ('trainer', 'num_episodes'),
        "minibatch": ('trainer', 'num_mini_batches'),
    }

    # Convert tags into a dictionary for easy lookup
    tag_dict: dict[str, Union[str, None]] = {}
    for tag in tags:
        key, value = tag.split('=') if '=' in tag else (tag, None)
        tag_dict[key] = value

    # Extract 'num_episodes' first if it exists
    num_episodes = int(tag_dict['num_episodes']) if 'num_episodes' in tag_dict else 0

    # Process each config key
    for key, path in config_paths.items():
        if key in tag_dict:
            value = tag_dict[key]
            if isinstance(value, str):  # Update the config if value is provided
                if key == "lr" or key == "use_gae":  # Special case for lr as it might be a schedule
                    if '[' in value:  # Check if it's a schedule
                        # Replace 'num_episodes' token in the lr value, if present
                        if num_episodes != 0:
                            value = value.replace('num_episodes', str(num_episodes))
                    new_value = eval(value)
                    config[path[0]]['car'][path[1]] = new_value
                    config[path[0]]['drone'][path[1]] = new_value
                else:
                    new_value = eval(value)
                    config[path[0]][path[1]] = new_value


def closest_multiple(a, b):
    """
    Calculate the multiple of number 'a' which is closest to number 'b'.

    Parameters:
    a (int): The number whose multiple is to be found.
    b (int): The target number to approach.

    Returns:
    int: The multiple of 'a' closest to 'b'.
    """
    # Finding the nearest multiple of 'a' to 'b' can be done by dividing 'b' by 'a',
    # rounding the result to the nearest whole number, and then multiplying by 'a'.
    if b % a == 0:
        return b
    multiple = round(b / a) * a
    return multiple


def is_valid_format(input_str):
    # Define a regular expression pattern to match "X-X," "X-X-X," and "X-X-X-X" formats
    pattern = r'^(\d+-)+\d+$'

    # Use re.match to check if the input matches the pattern
    if re.match(pattern, input_str):
        return True
    else:
        return False


def get_restore_dict(args: argparse.Namespace, uuid: str,
                     checkpoint_num: int, time_str: str, backup_str: str = ""):
    """
    Get the restore dict for a given experiment.

    Parameters:
    uuid (str): The unique identifier of the experiment.
    time_str (str): The time string of the experiment.
    checkpoint_num (int): The checkpoint number to restore.

    Returns:
    dict: The restore dict for the experiment.
    """
    parent_result_name = os.path.join("/workspace", "saved_data", "marllib_results")
    sub_folder_name = "_".join([args.algo, args.core_arch, args.dataset])
    restore_dict = {
        'model_path': os.path.join(parent_result_name, sub_folder_name,
                                   f"{str(args.algo).upper()}Trainer_{args.env}_"
                                   f"{args.dataset}_{uuid}_00000_0_{time_str}",
                                   f"checkpoint_{str(checkpoint_num).zfill(6)}", f"checkpoint-{checkpoint_num}"),
        'params_path': os.path.join(parent_result_name, sub_folder_name,
                                    f"experiment_state-{time_str}.json"),
        'render': args.render
    }
    if len(backup_str) > 0:
        restore_dict['params_path'] = os.path.join(parent_result_name, sub_folder_name,
                                                   f"experiment_state-{backup_str}.json")
    return restore_dict
