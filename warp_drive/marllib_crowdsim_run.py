import argparse
import logging
import os

from envs.crowd_sim.crowd_sim import (RLlibCUDACrowdSim, LARGE_DATASET_NAME, RLlibCUDACrowdSimWrapper)
from warp_drive.utils.common import get_project_root
from common import add_common_arguments, logging_dir
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from datetime import datetime
from common import logging_dir

# import warnings

# Suppress UserWarning
# Get rid of this as soon as you can!!!
# warnings.filterwarnings('ignore', category=UserWarning)

# register all scenario with env class
REGISTRY = {}
# add nvcc path to os environment
os.environ["PATH"] += os.pathsep + '/usr/local/cuda/bin'

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARN)
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args()

    current_datetime = datetime.now()
    # Format the date and time as a string
    datetime_string = current_datetime.strftime("%m%d-%H%M%S")
    expr_name = f"{datetime_string}_kdd2024"
    logging_config = {'log_level': 'INFO', 'logging_dir': logging_dir, 'group': "debug", 'dataset': "SanFrancisco",
                      'tag': None, 'expr_name': expr_name}
    # register new env
    ENV_REGISTRY["crowdsim"] = RLlibCUDACrowdSim
    # initialize env
    if args.dataset == LARGE_DATASET_NAME:
        from datasets.Sanfrancisco.env_config import BaseEnvConfig
    else:
        from datasets.KAIST.env_config import BaseEnvConfig

    env_params = {'env_setup': BaseEnvConfig}
    # env_params['env_registrar'] = env_registrar
    if args.track:
        env_params['logging_config'] = logging_config
    # this is a mocking env not used in actual run.
    env = marl.make_env(environment_name="crowdsim", map_name=LARGE_DATASET_NAME,
                        abs_path=os.path.join(get_project_root(), "run_configs", "mcs_data_collection.yaml"),
                        env_params=env_params)
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="common")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "512-512"})
    # start learning
    mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 40000000}, local_mode=False,
              num_workers=1, sshare_policy='all', checkpoint_freq=100,
              ckpt_path=os.path.join("/workspace", "checkpoints", "marllib"), resume=False, evaluation_interval=False,
              logging_config=logging_config if args.track else None, remote_worker_envs=False,
              custom_vector_env=RLlibCUDACrowdSimWrapper
              )
'''
  restore_path={'model_path': "/workspace/saved_data/marllib_results/mappo_mlp_SanFrancisco/"
                              "MAPPOTrainer_crowdsim_SanFrancisco_559f3_00000_0_2023-12-12_11-27-10",
                'params_path': "/workspace/saved_data/marllib_results/mappo_mlp_SanFrancisco/"
                               "experiment_state-2023-12-12_11-27-10.json"}
'''
