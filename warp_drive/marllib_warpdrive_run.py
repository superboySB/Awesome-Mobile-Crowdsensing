import argparse
import logging
import os
import pprint
from marllib import marl
from marllib.marl.common import algo_type_dict
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl import _Algo
import setproctitle
from common import add_common_arguments, logging_dir, customize_experiment, is_valid_format, get_restore_dict
from envs.crowd_sim.crowd_sim import (RLlibCUDACrowdSim, LARGE_DATASET_NAME,
                                      RLlibCUDACrowdSimWrapper, user_override_params)

# register all scenario with env class
REGISTRY = {}
# add nvcc path to os environment
os.environ["PATH"] += os.pathsep + '/usr/local/cuda/bin'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--centralized', action='store_true', help='use centralized reward function')
    # select algorithm
    parser.add_argument("--num_workers", type=int, default=0, help='number of workers to sample environment.')
    parser.add_argument("--render", action='store_true', help='render the environment')
    parser.add_argument("--render_file_name", type=str, default='trajectory.html',
                        help='file name for resulting render html file')
    parser.add_argument("--use_2d_state", action='store_true', help='use 2d state representation')
    parser.add_argument("--encoder_layer", type=str, help='encoder layer config, input in format X-X-X',
                        default='128-128-128')
    parser.add_argument("--core_arch", type=str, help='core architecture, mlp, gru or lstm',
                        choices=['mlp', 'gru', 'lstm'], default='mlp')
    parser.add_argument('--local_mode', action='store_true', help='run in local mode')
    parser.add_argument('--all_random', action='store_true', help='PoIs in the environment '
                                                                  'are completely random')
    parser.add_argument("--cut_points", type=int, default=200, help='number of points allowed')
    parser.add_argument("--ckpt", action='store_true', help='load checkpoint')
    # parser.add_argument("--ckpt", nargs=3, type=str, help='uuid, time_str, checkpoint_num to restore')
    args = parser.parse_args()

    assert args.encoder_layer is not None and is_valid_format(args.encoder_layer), \
        f"encoder_layer should be in format X-X-X, got {args.encoder_layer}"
    expr_name = customize_experiment(args)
    logging.debug("experiment name: %s", expr_name)
    if args.dynamic_zero_shot and args.all_random:
        raise ValueError("dynamic_zero_shot and all_random cannot be both true")
    if args.render:
        logging.getLogger().setLevel(logging.DEBUG)
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
        share_policy = 'all'
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
                    for key_feature in [args.algo, args.core_arch, args.dataset]:
                        if key_feature not in original:
                            original += f"_{key_feature}"
                    env_params[item] = original
                else:
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
    algorithm_list = dir(marl.algos)
    # filter all strings in the list with prefix "_"
    algorithm_list = list(filter(lambda x: not x.startswith("_"), algorithm_list))
    assert args.algo in algorithm_list, f"algorithm {args.algo} not supported, please implement your custom algorithm"
    algorithm_object: _Algo = getattr(marl.algos, args.algo)(hyperparam_source="common", **custom_algo_params)
    # customize model
    model = marl.build_model(env, algorithm_object, {"core_arch": args.core_arch,
                                                     "encode_layer": args.encoder_layer})
    # start learning
    # passing logging_config to fit is for trainer Initialization
    # (in remote mode, env and learner are on different processes)
    # 'share_policy': share_policy
    if args.render or args.ckpt:
        uuid = "c4a4a"
        time_str = "2024-01-08_22-03-58"
        checkpoint_num = 1000
        restore_dict = get_restore_dict(args, uuid, time_str, checkpoint_num)
    else:
        restore_dict = {}
    if args.render:
        # adjust to latest update!
        kwargs = {
            'restore_path': restore_dict, 'local_mode': True, 'share_policy': "all", 'checkpoint_end': False,
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
                  'checkpoint_end': False, 'algo_args': {'resume': args.resume}, 'checkpoint_freq': 500,
                  'stop': {"timesteps_total": 60001200}, 'restore_path': restore_dict,
                  'ckpt_path': os.path.join("/workspace", "checkpoints", "marllib"), 'evaluation_interval': False,
                  'logging_config': logging_config if args.track else None, 'remote_worker_envs': False}
        if args.env == 'crowdsim':
            kwargs['custom_vector_env'] = RLlibCUDACrowdSimWrapper
        algorithm_object.fit(env, model, **kwargs)
'''
           --algo qmix --env mpe --dataset simple_spread --num_workers 1
'''
