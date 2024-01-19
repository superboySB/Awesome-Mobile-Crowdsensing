import argparse
import logging
import os
import warnings
from typing import List
import numpy as np
from marllib import marl
from marllib.marl.common import algo_type_dict
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl import _Algo
import setproctitle
from common import add_common_arguments, logging_dir, customize_experiment, is_valid_format, get_restore_dict
from envs.crowd_sim.crowd_sim import (RLlibCUDACrowdSim, LARGE_DATASET_NAME,
                                      RLlibCUDACrowdSimWrapper, user_override_params)
from ray.tune import Callback


class MyCallback(Callback):
    def on_step_end(self, iteration: int, trials: List["Trial"], **info):
        print("test")
# register all scenario with env class
REGISTRY = {}
# add nvcc path to os environment
os.environ["PATH"] += os.pathsep + '/usr/local/cuda/bin'

# ray custom callback



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--centralized', action='store_true', help='use centralized reward function')
    # select algorithm
    parser.add_argument("--num_workers", type=int, default=0, help='number of workers to sample environment.')
    parser.add_argument("--render", action='store_true', help='render the environment')
    parser.add_argument("--render_file_name", type=str, default='trajectory',
                        help='file name for resulting render html file')
    parser.add_argument("--use_2d_state", action='store_true', help='use 2d state representation')
    parser.add_argument("--encoder_layer", type=str, help='encoder layer config, input in format X-X-X',
                        default='128-128-128')
    parser.add_argument("--core_arch", type=str, help='core architecture, mlp, gru or lstm',
                        choices=['mlp', 'gru', 'lstm', 'crowdsim_net'], default='mlp')
    parser.add_argument('--local_mode', action='store_true', help='run in local mode')
    parser.add_argument('--all_random', action='store_true', help='PoIs in the environment '
                                                                  'are completely random')
    parser.add_argument("--cut_points", type=int, default=200, help='number of points allowed')
    parser.add_argument("--ckpt", action='store_true', help='load checkpoint')
    parser.add_argument("--share_policy", choices=['all', 'group', 'individual'], default='all')
    parser.add_argument("--separate_render", action='store_true', help='render file will be stored separately')
    parser.add_argument('--no_refresh', action='store_true', help='do not reset randomly generated emergency points')
    parser.add_argument("--selector_type", type=str, default='NN', choices=['NN', 'greedy', 'oracle', 'random'])
    # parser.add_argument("--ckpt", nargs=3, type=str, help='uuid, time_str, checkpoint_num to restore')
    args = parser.parse_args()

    assert args.encoder_layer is not None and is_valid_format(args.encoder_layer), \
        f"encoder_layer should be in format X-X-X, got {args.encoder_layer}"
    expr_name = customize_experiment(args)
    this_expr_dir = os.path.join(logging_dir, 'trajectories', '_'.join([args.algo, args.core_arch, args.dataset]),
                                 expr_name)
    # make dir
    if args.core_arch == 'crowdsim_net':
        assert args.env == 'crowdsim', f"crowdsim_net only supports crowdsim env, got {args.env}"
        assert args.use_2d_state and args.dynamic_zero_shot, \
            f"crowdsim_net only supports 2d state and dynamic zero shot"
    if not os.path.exists(this_expr_dir):
        os.makedirs(this_expr_dir)
    logging.debug("experiment name: %s", expr_name)
    if args.dynamic_zero_shot and args.all_random:
        raise ValueError("dynamic_zero_shot and all_random cannot be both true")
    if args.render:
        logging.getLogger().setLevel(logging.INFO)
    else:
        if args.local_mode:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.WARN)
    setproctitle.setproctitle(expr_name)
    # initialize crowdsim configuration
    if args.env == 'crowdsim':
        # register new env
        ENV_REGISTRY[args.env] = RLlibCUDACrowdSim
        COOP_ENV_REGISTRY[args.env] = RLlibCUDACrowdSim
        share_policy = args.share_policy
        if args.dataset == LARGE_DATASET_NAME:
            from datasets.Sanfrancisco.env_config import BaseEnvConfig
        else:
            from datasets.KAIST.env_config import BaseEnvConfig
        env_params = {'env_config': BaseEnvConfig}
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
        for item in ['centralized', 'gpu_id', 'render_file_name', 'render', 'local_mode'] + user_override_params:
            if item != 'env_config':
                if item == 'render_file_name':
                    original = getattr(args, item)
                    if args.separate_render:
                        env_params[item] = os.path.join("/workspace", "saved_data", "trajectories", original)
                    else:
                        env_params[item] = os.path.join(str(this_expr_dir), original)
                else:
                    env_params[item] = getattr(args, item)
        logging.debug(env_params)
        env = marl.make_env(environment_name=args.env, map_name=args.dataset, env_params=env_params)
    else:
        # this is a mocking env not used in actual run.
        if args.env == 'crowdsim' or args.algo in algo_type_dict['VD']:
            warnings.warn("VD Method must use share_policy='all'")
            share_policy = 'all'
        else:
            share_policy = 'group'
        env = marl.make_env(environment_name=args.env, map_name=args.dataset)
        logging_config = None

    # filter all string in tags not in format "key=value"
    custom_algo_params = dict(filter(lambda x: "=" in x, args.tag if args.tag is not None else []))
    algorithm_list = dir(marl.algos)
    # filter all strings in the list with prefix "_"
    algorithm_list = list(filter(lambda x: not x.startswith("_"), algorithm_list))
    assert args.algo in algorithm_list, f"algorithm {args.algo} not supported, please implement your custom algorithm"
    algorithm_object: _Algo = getattr(marl.algos, args.algo)(hyperparam_source="common", **custom_algo_params)
    # customize model
    model_preference = {"core_arch": args.core_arch, "encode_layer": args.encoder_layer}
    if args.env == 'crowdsim':
        for item in ['selector_type']:
            model_preference[item] = getattr(args, item)
    model = marl.build_model(env, algorithm_object, model_preference)
    # start learning
    # passing logging_config to fit is for trainer Initialization
    # (in remote mode, env and learner are on different processes)
    # 'share_policy': share_policy
    if args.render or args.ckpt:
        uuid = "7523b"
        time_str = "2024-01-15_18-40-10"
        checkpoint_num = 1000
        backup_str = ""
        restore_dict = get_restore_dict(args, uuid, time_str, checkpoint_num, backup_str)
        for info in [uuid, str(checkpoint_num)]:
            if info not in env_params['render_file_name']:
                env_params['render_file_name'] += f"_{info}"
    else:
        restore_dict = {}
    if args.render:
        # adjust to latest update!
        kwargs = {
            'restore_path': restore_dict, 'local_mode': True, 'share_policy': share_policy,
            'checkpoint_end': False,
            'num_workers': 0, 'rollout_fragment_length': BaseEnvConfig.env.num_timestep,
            'algo_args': {'resume': False}
        }
        if args.env == 'crowdsim':
            kwargs['custom_vector_env'] = RLlibCUDACrowdSimWrapper
        # figure out how to let evaluation program call "render", set lr=0
        algorithm_object.render(env, model, **kwargs)
    else:
        kwargs = {'local_mode': args.local_mode, 'num_gpus': 1, 'num_workers': args.num_workers,
                  'share_policy': share_policy,
                  'checkpoint_end': False, 'algo_args': {'resume': args.resume},
                  'checkpoint_freq': args.evaluation_interval,
                  'stop': {"timesteps_total": 60000000}, 'restore_path': restore_dict,
                  'evaluation_interval': False,
                  'logging_config': logging_config if args.track else None, 'remote_worker_envs': False}
        # 1 if args.local_mode else args.evaluation_interval
        if args.env == 'crowdsim':
            kwargs['custom_vector_env'] = RLlibCUDACrowdSimWrapper
        algorithm_object.fit(env, model, **kwargs)
'''
           --algo qmix --env mpe --dataset simple_spread --num_workers 1
'''
