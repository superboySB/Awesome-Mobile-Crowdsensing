import argparse
import logging
import os

from marllib import marl
from marllib.marl.common import algo_type_dict
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl import _Algo
import setproctitle
from common import add_common_arguments, logging_dir, customize_experiment
from envs.crowd_sim.crowd_sim import (RLlibCUDACrowdSim, LARGE_DATASET_NAME, RLlibCUDACrowdSimWrapper)

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
    parser.add_argument('--centralized', action='store_true', help='use centralized reward function')
    # select algorithm
    parser.add_argument("--num_workers", type=int, default=0, help='number of workers to sample environment.')
    parser.add_argument("--render", action='store_true', help='render the environment')
    args = parser.parse_args()
    expr_name = customize_experiment(args)
    setproctitle.setproctitle(expr_name)
    # register new env
    if args.env == 'crowdsim':
        ENV_REGISTRY[args.env] = RLlibCUDACrowdSim
        COOP_ENV_REGISTRY[args.env] = RLlibCUDACrowdSim
    # initialize crowdsim configuration
    if args.env == 'crowdsim':
        share_policy = 'all'
        if args.dataset == LARGE_DATASET_NAME:
            from datasets.Sanfrancisco.env_config import BaseEnvConfig
        else:
            from datasets.KAIST.env_config import BaseEnvConfig
        env_params = {'env_setup': BaseEnvConfig}
        # env_params['env_registrar'] = env_registrar
        if args.track:
            # link logging config with tag
            logging_config = {'log_level': 'INFO', 'logging_dir': logging_dir, 'expr_name': expr_name}
            for item in ['group', 'dataset', 'tag', 'resume']:
                assert item in args, f"missing {item} in logging_args, please check"
                logging_config[item] = getattr(args, item)
            env_params['logging_config'] = logging_config
        else:
            logging_config = None
        for item in ['centralized', 'gpu_id', 'dynamic_zero_shot']:
            env_params[item] = getattr(args, item)
        logging.debug(env_params)
        env = marl.make_env(environment_name=args.env, map_name=args.dataset, env_params=env_params)
    else:
        # this is a mocking env not used in actual run.
        if args.env == 'crowdsim' or args.algo in algo_type_dict['VD']:
            share_policy = 'all'
        else:
            share_policy = 'group'
        env = marl.make_env(environment_name=args.env, map_name=args.dataset)
        logging_config = None

    # filter all string in tags not in format "key=value"
    custom_algo_params = dict(filter(lambda x: "=" in x, args.tag if args.tag is not None else []))
    if args.render:
        custom_algo_params['lr'] = 0
    algorithm_list = dir(marl.algos)
    # filter all strings in the list with prefix "_"
    algorithm_list = list(filter(lambda x: not x.startswith("_"), algorithm_list))
    assert args.algo in algorithm_list, f"algorithm {args.algo} not supported, please implement your custom algorithm"
    algorithm_object: _Algo = getattr(marl.algos, args.algo)(hyperparam_source="common", **custom_algo_params)
    # customize model
    model = marl.build_model(env, algorithm_object, {"core_arch": "mlp", "encode_layer": "512-512"})
    # start learning
    # passing loggign_config to fit is for trainer Initialization
    # (in remote mode, env and learner are on different processes)
    # 'share_policy': share_policy
    if args.render:
        restore_dict = {
            'model_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                          "IPPOTrainer_crowdsim_SanFrancisco_dbd29_00000_0_2023-12-21_11-05-56/"
                          "checkpoint_029000/checkpoint-29000",
            'params_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                           "experiment_state-2023-12-21_11-05-56.json",
            'render': True
        }
        kwargs = {
            'restore_path': restore_dict, 'local_mode': True, 'share_policy': "all", 'checkpoint_end': False,
            'num_workers': 0, 'rollout_fragment_length': BaseEnvConfig.env.num_timestep
        }
        if args.env == 'crowdsim':
            kwargs['custom_vector_env'] = RLlibCUDACrowdSimWrapper
            kwargs['env_args'] = dict(trainer=dict(num_envs=1))
        algorithm_object.render(env, model, **kwargs)
        # figure out how to let evaluation program call "render", set lr=0
    else:
        kwargs = {'local_mode': False, 'num_gpus': 1, 'num_workers': args.num_workers, 'share_policy': share_policy,
                  'checkpoint_end': False, 'resume': False, 'checkpoint_freq': 500,
                  'stop': {"timesteps_total": 40000000},
                  'ckpt_path': os.path.join("/workspace", "checkpoints", "marllib"), 'evaluation_interval': False,
                  'logging_config': logging_config if args.track else None, 'remote_worker_envs': False}
        if args.env == 'crowdsim':
            kwargs['custom_vector_env'] = RLlibCUDACrowdSimWrapper
        algorithm_object.fit(env, model, **kwargs)
'''
           --algo qmix --env mpe --dataset simple_spread --num_workers 1
'''
