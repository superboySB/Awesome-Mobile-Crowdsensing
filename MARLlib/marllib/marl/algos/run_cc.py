# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ray
import gym
from ray import tune
from envs.crowd_sim.crowd_sim import teams_name
from typing import Dict
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.env.base_env import dummy_agent_id
from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.marl.common import recursive_dict_update, dict_update
from marllib.marl.algos.utils.log_dir_util import path_to_temp

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


def setup_running(env, exp_info):
    ray.init(local_mode=exp_info["local_mode"], num_gpus=exp_info["num_gpus"], _temp_dir=path_to_temp)
    ########################
    ### environment info ###
    ########################
    env_info = env.get_env_info()
    map_name = exp_info['env_args']['map_name']
    agent_name_ls = env.agents
    env_info["agent_name_ls"] = agent_name_ls
    env.close()
    return agent_name_ls, env_info, map_name


def restore_config_update(exp_info, run_config, stop_config):
    if exp_info['restore_path']['model_path'] == '':
        restore_config = None
    else:
        restore_config = exp_info['restore_path']
        if 'render' in exp_info['restore_path']:
            render_config = {
                "evaluation_interval": 1,
                "evaluation_num_episodes": 100,
                "evaluation_num_workers": 1,
                "evaluation_config": {
                    "record_env": False,
                    "render_env": True,
                }
            }

            run_config = recursive_dict_update(run_config, render_config)

            render_stop_config = {
                "training_iteration": 0,
            }

            stop_config = recursive_dict_update(stop_config, render_stop_config)

    return exp_info, run_config, stop_config, restore_config

def run_cc(exp_info: Dict, env, model, stop=None):
    agent_name_ls, env_info, map_name = setup_running(env, exp_info)

    ######################
    ### space checking ###
    ######################

    action_discrete = isinstance(env_info["space_act"], gym.spaces.Discrete) or isinstance(env_info["space_act"],
                                                                                           gym.spaces.MultiDiscrete)
    if action_discrete:
        if exp_info["algorithm"] in ["maddpg"]:
            raise ValueError(
                "Algo -maddpg- only supports continuous action space, Env -{}- requires Discrete action space".format(
                    exp_info["env"]))
    else:  # continuous
        if exp_info["algorithm"] in ["coma"]:
            raise ValueError(
                "Algo -coma- only supports discrete action space, Env -{}- requires continuous action space".format(
                    exp_info["env"]))

    ######################
    ### policy sharing ###
    ######################

    policy_mapping_info = env_info["policy_mapping_info"]

    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    # whether to agent level batch update when shared model parameter:
    # True -> default_policy | False -> shared_policy
    shared_policy_name = "default_policy" if exp_info["agent_level_batch_update"] else "shared_policy"
    if exp_info["share_policy"] == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError("in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))
        policies = {shared_policy_name}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: shared_policy_name)

    elif exp_info["share_policy"] == "group":
        groups = policy_mapping_info["team_prefix"]

        if len(groups) == 1:
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError(
                    "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))
            policies = {shared_policy_name}
            policy_mapping_fn = (
                lambda agent_id, episode, **kwargs: shared_policy_name)

        else:
            policies = {
                "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
                groups
            }
            policy_ids = list(policies.keys())
            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                if agent_id != dummy_agent_id():
                    return "policy_" + agent_id.split("_")[0] + "_"
                else:
                    return "policy_" + teams_name[0]


    elif exp_info["share_policy"] == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
            range(env_info["num_agents"])
        }

        policy_ids = list(policies.keys())

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id != dummy_agent_id():
                return policy_ids[agent_name_ls.index(agent_id)]
            else:
                return "policy_" + teams_name[0]
    else:
        raise ValueError("wrong share_policy {}".format(exp_info["share_policy"]))

    # if happo or hatrpo, force individual
    if exp_info["algorithm"] in ["happo", "hatrpo"]:
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
            range(env_info["num_agents"])
        }
        policy_ids = list(policies.keys())

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id != dummy_agent_id():
                return policy_ids[agent_name_ls.index(agent_id)]
            else:
                return "policy_" + teams_name[0]

    #########################
    ### experiment config ###
    #########################

    run_config = {
        "seed": int(exp_info["seed"]),
        "env": exp_info["env"] + "_" + exp_info["env_args"]["map_name"],
        "num_gpus_per_worker": exp_info.get("num_gpus_per_worker", 0),
        "num_cpus_per_worker": exp_info.get("num_cpus_per_worker", 1),
        "num_gpus": exp_info.get("num_gpus", 0),
        "num_workers": exp_info["num_workers"],
        "num_envs_per_worker": exp_info.get("num_envs_per_worker", 1),
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": exp_info["framework"],
        "evaluation_interval": exp_info["evaluation_interval"],
        "simple_optimizer": False,  # force using better optimizer
        "track": exp_info.get("track", False),
        "logging_config": exp_info.get("logging_config", {}),
        "remote_worker_envs": exp_info.get("remote_worker_envs", False),
    }
    if "custom_vector_env" in exp_info:
        run_config["custom_vector_env"] = exp_info["custom_vector_env"]

    stop_config = {
        "episode_reward_mean": exp_info["stop_reward"],
        "timesteps_total": exp_info["stop_timesteps"],
        "training_iteration": exp_info["stop_iters"],
    }

    stop_config = dict_update(stop_config, stop)

    exp_info, run_config, stop_config, restore_config = restore_config_update(exp_info, run_config, stop_config)

    ##################
    ### run script ###
    ##################

    results = POlICY_REGISTRY[exp_info["algorithm"]](model, exp_info, run_config, env_info, stop_config,
                                                     restore_config)
    ray.shutdown()

    return results
