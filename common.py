import os
from argparse import ArgumentParser
from typing import Union

import yaml
from datetime import datetime
import argparse
from envs.crowd_sim.crowd_sim import LARGE_DATASET_NAME


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
        '--ckpt', nargs=2, type=str, default=None, help="checkpoint to load, the first is the experiment name,"
                                                        " the second is the timestamp."
    )
    parser.add_argument(
        '--group', type=str, default='debug', help="group name for wandb project"
    )
    parser.add_argument(
        '--dataset', type=str, default=LARGE_DATASET_NAME, help="KAIST or SanFrancisco"
    )
    parser.add_argument(
        '--tag',
        nargs='+',
        type=str,
        default=None,
        help="special comments for each experiment"
    )


logging_dir = os.path.join("/workspace", "saved_data")


def customize_experiment(args: argparse.Namespace, run_config: dict = None, yaml_config_path: str = None, ):
    """
    Setup tags to update hyperparameters based on argparse arguments and optionally a YAML configuration file.
    The function appends attributes set to True in both argparse and the YAML file as tags.
    """
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime("%m%d-%H%M%S")

    expr_name = f"{datetime_string}"
    tags = args.tag if args.tag is not None else []

    # Process argparse arguments
    for arg in vars(args):
        if arg != 'track' and getattr(args, arg) is True:
            tags.append(arg)

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
