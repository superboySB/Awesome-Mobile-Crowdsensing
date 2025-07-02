import argparse
import csv
import logging
import os
import pprint
import warnings
from marllib import marl
from marllib.marl.common import algo_type_dict
from marllib.marl.algos.scripts.coma import restore_ignore_params
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl import _Algo
from util_misc import set_freest_gpu
import setproctitle

from common import add_common_arguments, logging_dir, customize_experiment, is_valid_format, get_restore_dict
from envs.crowd_sim.crowd_sim import (RLlibCUDACrowdSim, LARGE_DATASET_NAME, CUDACrowdSim,
                                      RLlibCUDACrowdSimWrapper, SendAllocationCallback,
                                      OUTPACECallback, user_override_params)
from warp_drive.utils.common import get_project_root


def load_preferences(item, custom_preference: dict, args: argparse.Namespace, this_expr_dir: str):
    if item == 'render_file_name':
        original = getattr(args, item)
        if args.render:
            custom_preference[item] = os.path.join("/workspace", "saved_data", "trajectories", original)
        else:
            custom_preference[item] = os.path.join(str(this_expr_dir), original)
    elif item == 'checkpoint_path':
        pass
    else:
        custom_preference[item] = getattr(args, item)


if __name__ == '__main__':
    import numpy as np

    np.seterr(all="raise")
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
    parser.add_argument("--core_arch", type=str, help='core architecture, mlp, gru, lstm or attention',
                        choices=['mlp', 'gru', 'lstm', 'crowdsim_net', 'attention', 'pred_loc'], default='mlp')
    parser.add_argument('--local_mode', action='store_true', help='run in local mode')
    parser.add_argument('--all_random', action='store_true', help='PoIs in the environment '
                                                                  'are completely random')
    parser.add_argument('--multi_type', action='store_true', help='use multi-type emergency poi from csv file.')
    parser.add_argument("--ckpt", nargs=4, type=str, help='uuid, checkpoint_num, time_str '
                                                          'and backup_str to restore')
    parser.add_argument("--share_policy", choices=['all', 'group', 'individual'], default='all')
    parser.add_argument("--separate_encoder", action='store_true', help='use separate head for emergency')
    parser.add_argument("--with_programming_optimization", action='store_true')
    parser.add_argument('--no_refresh', action='store_true', help='do not reset randomly generated emergency points')
    parser.add_argument("--selector_type", type=str, default=['NN'],
                        choices=['NN', 'greedy', 'oracle', 'random', 'RL'], nargs='+')
    parser.add_argument("--switch_step", type=int, default=600000, help='switch step for NN selector')
    parser.add_argument("--one_agent_multi_task", action='store_true', help='allocate multiple task for a single agent')
    parser.add_argument("--look_ahead", action='store_true',
                        help='greedily assign according to agent final destination')
    parser.add_argument("--emergency_queue_length", type=int, default=1, help='emergency queue length')
    parser.add_argument("--tolerance", type=float, default=1e-2, help='tolerance for choosing multiple emergencies')
    parser.add_argument("--rl_gamma", type=float, default=0.5, help='gamma for RL selector')
    parser.add_argument("--force_allocate", action='store_true', help='force emergencies to be allocated, agent'
                                                                      'will receive no reward if it is not allocated '
                                                                      'to cover.')
    # parser.add_argument("--with_end_time", action='store_true', help='use end time for emergency')
    parser.add_argument("--sibling_rivalry", action='store_true', help='enable anti-goal distance reward')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha for anti-goal distance reward')
    parser.add_argument("--buffer_in_obs", action='store_true', help='display entire buffer in the observation')
    parser.add_argument("--reward_mode", type=str, default='greedy',
                        choices=['mix', 'original', 'intrinsic', 'none', 'greedy'],
                        help='reward type for emergency')
    parser.add_argument('--fail_hint', action='store_true', help='use fail hint for emergency allocation reward')
    parser.add_argument('--prioritized_buffer', action='store_true', help='use prioritized buffer for RL selector')
    parser.add_argument('--NN_buffer', action='store_true', help='use NN generated weights for RL selector')
    parser.add_argument('--rl_use_cnn', action='store_true', help='use CNN for RL selector')
    parser.add_argument('--intrinsic_mode', type=str, default='scaled_dis_aoi', choices=['none', 'dis', 'aoi',
                                                                                         'scaled_dis_aoi', 'dis_aoi',
                                                                                         'aim'])
    parser.add_argument('--use_random', action='store_true', help='use random emergency generation')
    parser.add_argument('--use_attention', action='store_true', help='use attention mechanism in high level assign')
    parser.add_argument('--attention_dim', type=int, default=128, help='attention dimension (single head)')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads for attention')
    parser.add_argument('--speed_action', action='store_true', help='enable speed action')
    parser.add_argument('--blur_requirement', type=float, default=5,
                        help='blur requirement for image, proportional to speed')
    parser.add_argument('--horizon', type=int, default=5, help='train timstep for pred loc')
    parser.add_argument('--emergency_reward', type=float, default=10, help='reward for covering emergency')
    parser.add_argument('--refill_emergency', action='store_true', help='fill in uncovered surveillance as emergency')
    parser.add_argument('--encoder_core_arch', type=str, default='mlp',
                        choices=['mlp', 'mlp_residual', 'pred_loc', 'greedy'], help='core architecture for encoder')
    parser.add_argument('--use_action_mask', action='store_true', help='use action mask for emergency')
    parser.add_argument('--no_task_allocation', action='store_true', help='disable task allocation high-level agent')
    parser.add_argument('--use_neural_ucb', action='store_true', help='use neural ucb for upper-level assignment.')
    parser.add_argument('--use_pcgrad', action='store_true', help='use conflicting gradient projection')
    parser.add_argument('--use_bvn', action='store_true', help='use bilinear value network for lower-level agent')
    # parser.add_argument('--scale_size', type=float, default=10, help='scaling factor for surveillance reward')
    parser.add_argument('--use_relabeling', type=str, default='none',
                        choices=['none', 'agent', 'emergency'], help='enable relabeling for high level agent')
    parser.add_argument('--relabel_threshold', type=float, default=0, help='threshold reward for relabeling')
    parser.add_argument('--gdan_eta', type=float, default=1, help='eta for GDAN')
    parser.add_argument('--use_action_label', action='store_true',
                        help='use action label for GDAN (only valid when using GDAN)')
    # add a group for variants of GDAN (LSTM and non-LSTM)
    gdan_group = parser.add_mutually_exclusive_group(required=False)
    gdan_group.add_argument('--use_gdan_no_loss', action='store_true', help='use GDAN without cross entropy loss')
    gdan_group.add_argument('--use_gdan', action='store_true', help='use Goal Discriminative Attention Network')
    gdan_group.add_argument('--use_gdan_lstm', action='store_true',
                            help='use Goal Discriminative Attention Network with LSTM')
    parser.add_argument('--display_tags', nargs='+', type=str, default=None,
                        help='special comments for each experiment')
    args = parser.parse_args()

    assert args.encoder_layer is not None and is_valid_format(args.encoder_layer), \
        f"encoder_layer should be in format X-X-X, got {args.encoder_layer}"
    if args.display_tags is not None:
        expr_name = customize_experiment(args, display_tags=set(args.display_tags))
    else:
        expr_name = customize_experiment(args)
    this_expr_dir = os.path.join(logging_dir, 'trajectories', '_'.join([args.algo, args.core_arch, args.dataset]),
                                 expr_name)
    if args.selector_type == 'oracle':
        assert args.share_policy != 'individual' and args.env == 'crowdsim', \
            f"selector_type {args.selector_type} only works with crowdsim env and share_policy != individual"
    if args.core_arch == 'crowdsim_net':
        warnings.warn("encoder_layer is ignored for crowdsim_net separate encoder")

    # initialize crowdsim configuration
    if args.env == 'crowdsim':
        # register new env
        ENV_REGISTRY[args.env] = RLlibCUDACrowdSim
        COOP_ENV_REGISTRY[args.env] = RLlibCUDACrowdSim
        share_policy = args.share_policy
        print('=============> args.dataset: ', args.dataset)
        if args.dataset == LARGE_DATASET_NAME:
            from datasets.Sanfrancisco.env_config import BaseEnvConfig
        elif args.dataset == 'KAIST':
            from datasets.KAIST.env_config import BaseEnvConfig
        elif args.dataset == 'Chengdu':
            from datasets.Chengdu.env_config import BaseEnvConfig
        elif args.dataset == 'Beijing':
            from datasets.Beijing.env_config import BaseEnvConfig
        else:
            raise NotImplementedError(f"dataset {args.dataset} not supported")
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
                load_preferences(item, custom_preference=env_params, args=args, this_expr_dir=this_expr_dir)
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
        env_params = {}

    if not os.path.exists(this_expr_dir):
        os.makedirs(this_expr_dir)
    logging.debug("experiment name: %s", expr_name)
    if args.algo == 'trafficppo' or args.algo == 'traffica2c' or args.algo == 'traffictrpo':
        assert args.env == 'crowdsim', \
            f"traffic series algorithm only supports crowdsim env, got {args.env}"
    elif args.algo in ['random', 'tsp']:
        new_env = marl.make_env(environment_name=args.env, map_name=args.dataset,
                                env_params=env_params, mock=False)
        env, env_config = new_env
        raw_env: CUDACrowdSim = env.env
        # construct a random action with num_agents keys
        if args.algo == 'random':
            # generate a list of dict, each dict with num_agents keys + 'Drones_' prefix
            routes = []
            for _ in range(raw_env.episode_length):
                routes.append({f'Drones_{i}': np.random.randint(0, raw_env.action_space[0].n)
                               for i in range(raw_env.num_agents)})
            env.reset()
        else:
            from warp_drive.tsp import CrowdSimTSPSolver

            raw_env.emergency_threshold = raw_env.episode_length
            raw_env.aoi_schedule = np.zeros_like(raw_env.aoi_schedule)
            raw_env.drone_sensing_range *= 1.5
            env.reset()
            tsp_solver = CrowdSimTSPSolver(env, add_surveillance=False)
            routes = tsp_solver.get_solution(this_expr_dir=this_expr_dir)

        for action in routes:
            # step environment
            env.step(action)
            env.render()
        # collect metrics from env
        env_metrics = raw_env.collect_info()
        env_metrics['experiment_name'] = expr_name
        # pprint the env_metrics
        pprint.pprint(env_metrics)
        # write to a common file, with experiment_name and all metrics
        csv_file = os.path.join('/workspace', 'saved_data', 'trajectories', 'misc_results.csv')
        info = env_metrics
        if not os.path.exists(csv_file):
            # write info to the csv
            with open(csv_file, 'w', newline='') as file:
                # serialize the info dict
                writer = csv.DictWriter(file, fieldnames=info.keys())
                writer.writeheader()
                writer.writerow(info)
        else:
            # open the csv file and write
            with open(csv_file, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=info.keys())
                writer.writerow(info)
        exit(0)

    if args.dynamic_zero_shot and args.all_random:
        raise ValueError("dynamic_zero_shot and all_random cannot be both true")
    if args.render:
        logging.getLogger().setLevel(logging.INFO)
    else:
        if args.local_mode:
            logging.getLogger().setLevel(logging.DEBUG)
            # os.environ['NUMBA_DISABLE_JIT'] = '1'
        else:
            logging.getLogger().setLevel(logging.DEBUG)
    setproctitle.setproctitle(expr_name)

    # filter all string in tags not in format "key=value"
    custom_algo_params = dict(filter(lambda x: "=" in x, args.tag if args.tag is not None else []))
    algorithm_list = dir(marl.algos)
    # filter all strings in the list with prefix "_"
    algorithm_list = list(filter(lambda x: not x.startswith("_"), algorithm_list))
    algorithm_list.remove('register_algo')
    assert args.algo in algorithm_list, f"algorithm {args.algo} not supported, please implement your custom algorithm"
    my_algorithm: _Algo = getattr(marl.algos, args.algo)(hyperparam_source="common", **custom_algo_params)
    model_preference = {"core_arch": args.core_arch, "encode_layer": args.encoder_layer}
    if args.render or args.ckpt:
        uuid, checkpoint_num, time_str, backup_str = args.ckpt
        restore_dict = get_restore_dict(args, uuid, checkpoint_num, time_str, backup_str)
        model_preference['checkpoint_path'] = restore_dict
        if not args.render:
            restore_dict = {}
        for info in [uuid, str(checkpoint_num)]:
            if info not in env_params['render_file_name']:
                env_params['render_file_name'] += f"_{info}"
        for path in restore_dict.values():
            assert os.path.exists(path), f"checkpoint path {path} does not exist"
    else:
        restore_dict = {}
    # customize model
    if args.env == 'crowdsim':
        for item in (['gen_interval', 'with_programming_optimization',
                      'dataset', 'emergency_threshold', 'switch_step', 'one_agent_multi_task',
                      'emergency_queue_length', 'tolerance', 'look_ahead', 'local_mode',
                      'render_file_name', 'buffer_in_obs', 'separate_encoder', 'prioritized_buffer',
                      'rl_use_cnn', 'intrinsic_mode', 'dynamic_zero_shot', 'use_random',
                      'attention_dim', 'num_heads', 'NN_buffer', 'encoder_core_arch',
                      'use_action_mask', 'use_attention', 'use_neural_ucb', 'use_pcgrad', 'use_bvn',
                      'use_relabeling', 'relabel_threshold', 'use_gdan', 'use_gdan_lstm',
                      'use_gdan_no_loss', 'use_action_label', 'gdan_eta', 'num_drones',
                      'points_per_gen', 'no_task_allocation', 'horizon', 'multi_type'] +
                     restore_ignore_params):
            load_preferences(item, custom_preference=model_preference, args=args, this_expr_dir=this_expr_dir)
    model = marl.build_model(env, my_algorithm, model_preference)
    # start learning
    # passing logging_config to fit is for trainer Initialization
    # (in remote mode, env and learner are on different processes)
    # 'share_policy': share_policy
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
            kwargs['callbacks'] = SendAllocationCallback
        # figure out how to let evaluation program call "render", set lr=0
        my_algorithm.render(env, model, **kwargs)
    else:

        kwargs = {'local_mode': args.local_mode, 'num_gpus': 1, 'num_workers': args.num_workers,
                  'share_policy': share_policy,
                  'checkpoint_end': False, 'algo_args': {'resume': args.resume},
                  'checkpoint_freq': args.evaluation_interval,
                  # 'stop': {"timesteps_total": 2400},
                  'stop': {"timesteps_total": 10000000},
                  'restore_path': restore_dict,
                  'evaluation_interval': False,
                  'logging_config': logging_config if args.track else None, 'remote_worker_envs': False}
        # 1 if args.local_mode else args.evaluation_interval
        if args.env == 'crowdsim':
            kwargs['custom_vector_env'] = RLlibCUDACrowdSimWrapper
            kwargs['callbacks'] = SendAllocationCallback
        if args.algo == 'outpace':
            kwargs['callbacks'] = OUTPACECallback

        my_algorithm.fit(env, model, **kwargs)
'''
           --algo qmix --env mpe --dataset simple_spread --num_workers 1
'''
