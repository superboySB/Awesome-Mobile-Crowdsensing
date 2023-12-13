import os
from argparse import ArgumentParser

from envs.crowd_sim.crowd_sim import LARGE_DATASET_NAME


def add_common_arguments(parser: ArgumentParser):
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
