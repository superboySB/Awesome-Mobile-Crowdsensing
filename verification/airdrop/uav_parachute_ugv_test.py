import argparse
import logging
import os
# import swanlab
import pandas as pd
from collections import namedtuple
from util_misc import file_to_string

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import MultiDiscrete, Discrete, Box
from torch.distributions import Categorical
from tqdm import trange
from util_misc import set_freest_gpu, logHandler

from ppo_algo_verify import PPOVecPolicy, PPOCNNPolicy, Policy, MultiPPORollout
from verification.airdrop.parachute_env_test import (logger, DEPLOYED, EPISODE_LENGTH, SURVEILLANCE_AOI, EMERGENCY_AOI, \
                                                     BIG_AGENT_RANGE, SMALL_AGENT_RANGE, NUM_BIG_AGENTS, NUM_SMALL_AGENTS,
                                             NUM_MOVEMENTS, STOP,
                                             random_act, MultiAgentGridWorld)

NUM_MINIBATCHES = 4

APPEND_TAGS = ['group_factor']

# set up logger
logging.basicConfig(level=logging.INFO)
logger.addHandler(logHandler)

EMERGENCY_NUMBER = 15
LEARNING_RATE = 1e-4

# Constants
prompt_dir = os.path.join(os.getcwd(), 'verification/airdrop/prompts')
USERNAME = 'aequatio'
PROJECT_NAME = 'uav-parachute-ugv'
LOG_ACTION = False
LOG_TABLE = True
GAMMA = 0
RANDOM_ACT = False
FIX_SMALL_AGENT = True
HIDDEN_SIZE = 128
EVAL_INTERVAL = 100
PLOT_NAME = "trajectory"
BIG_AGENT_METRIC = "big_reward"
BIG_AGENT_TRAIN = "big_train"
BIG_AGENT_ACTION = "big_action"
SMALL_AGENT_TRAIN = "small_train"
SMALL_AGENT_METRIC = "small_reward"
SMALL_AGENT_ACTION = "small_action"
BIG_AGENT_MODEL = "big_model"
SMALL_AGENT_MODEL = "small_model"
BIG_AGENT_DEPLOY_METRIC = "big_reward_deploy"
ENV_INFO = 'env_info'
TIMESTEP_DEPLOY = [20, 40]

# Actions
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    Base class for policy, includes a method to finish the episode and update the policy.
    """

    def __init__(self):
        super(Policy, self).__init__()
        # Initialize action and reward buffers
        self.saved_actions = []
        self.rewards = []

    def finish_episode(self, optimizer, gamma=0.99, eps=1e-8, max_grad_norm=1.0):
        """
        Perform backpropagation to update policy and value function with gradient clipping.
        """
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # List to store actor (policy) loss
        value_losses = []  # List to store critic (value) loss
        returns = []  # List to store the discounted rewards

        # Calculate the discounted rewards
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            # Calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)
            # Calculate critic (value) loss using smooth L1 loss (Huber loss)
            value_losses.append(F.smooth_l1_loss(value, R))

        # Reset gradients
        optimizer.zero_grad()

        # Sum up all the policy losses and value losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # Perform backpropagation
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        # Update parameters
        optimizer.step()

        # Reset rewards and saved actions
        del self.rewards[:]
        del self.saved_actions[:]


def select_actions(agent_type: str, policy, num_agents, obs: dict, actions, env, device,
                   rollout: MultiPPORollout = None):
    for i in range(num_agents):
        if agent_type == 'big':
            current_obs = obs[f'big_{i}']
        elif agent_type == 'small':
            if not env.small_agents[i][DEPLOYED]:
                actions[f'small_{i}'] = STOP
                continue
            current_obs = obs[f'small_{i}']
        else:
            raise NotImplementedError("Invalid agent type: {agent_type}")

        # Convert observation to tensor and ensure correct dimensions
        if len(current_obs.shape) == 1 or len(current_obs.shape) == 3:
            state = torch.from_numpy(current_obs).float().unsqueeze(0).to(device)
        else:
            state = torch.from_numpy(current_obs).float().unsqueeze(0).unsqueeze(0).to(device)

        # Get action, log probability, and state value from policy
        action, log_prob, _, state_value = policy.get_action_and_value(state)

        if rollout is not None:
            rollout.saved_actions[i].append(action)
            rollout.log_probs[i].append(log_prob)
            rollout.values[i].append(state_value.squeeze())
            rollout.saved_obs[i].append(state)
        else:
            # Save the action, log probability, and state value for training later
            policy.saved_actions.append(action)
            policy.log_probs.append(log_prob)
            policy.values.append(state_value.squeeze())
            policy.saved_obs.append(state)

        # Record the selected action
        actions[f'{agent_type}_{i}'] = action.cpu().numpy()


def actor_critic_joint_act(env, obs: dict[np.ndarray], big_agent_policy: Policy,
                           small_agent_policy: Policy) -> dict[int]:
    # Select actions for big agents using the actor-critic model
    actions = {}
    for i in range(NUM_BIG_AGENTS):
        state = torch.from_numpy(obs[f'big_{i}']).float().unsqueeze(0).unsqueeze(0).to(device)
        action_probs, state_value = big_agent_policy(state)
        # Categorical, probs or logits.
        m = Categorical(probs=action_probs)
        action = m.sample()

        # Save the action and state value for training later
        big_agent_policy.saved_actions.append(SavedAction(m.log_prob(action), state_value.squeeze()))
        actions[f'big_{i}'] = action.item()
    # Select actions for small agents (only for deployed agents)
    for i in range(NUM_SMALL_AGENTS):
        if env.small_agents[i][DEPLOYED]:
            # if not deployed, observation is invalid, no input.
            state = torch.from_numpy(obs[f'small_{i}']).float().unsqueeze(0).unsqueeze(0).to(device)
            action_probs, state_value = small_agent_policy(state)
            m = Categorical(probs=action_probs)
            action = m.sample()
            # Save the action and state value for training later
            small_agent_policy.saved_actions.append(SavedAction(m.log_prob(action), state_value.squeeze()))
            actions[f'small_{i}'] = action.item()
        else:
            actions[f'small_{i}'] = STOP
    return actions


def get_action_shape(my_space: gym.Space):
    if isinstance(my_space, Discrete):
        return my_space.n
    elif isinstance(my_space, Box):
        return my_space.shape[0]
    elif isinstance(my_space, MultiDiscrete):
        return my_space.nvec
    else:
        raise ValueError("Invalid action space for big agent")


if __name__ == '__main__':
    import wandb

    set_freest_gpu()
    os.environ['NUMEXPR_MAX_THREADS'] = "4"
    # setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', action='store_true', help='track the experiment')
    parser.add_argument('--name', type=str, default='test', help='name of the experiment')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'debug'], default='train',
                        help='mode of the experiment')
    parser.add_argument('--num-big-agents', type=int, default=2, help='number of big agents')
    parser.add_argument('--num-small-agents', type=int, default=4, help='number of big agents')
    parser.add_argument('--num-episodes', type=int, default=5000, help='number of episodes')
    parser.add_argument('--self-factor', type=float, default=0, help='self factor')
    parser.add_argument('--group-factor', type=float, default=1, help='group factor')
    parser.add_argument('--model-path', type=str, default='', help='path to saved model')
    args = parser.parse_args()
    num_episodes: int = args.num_episodes
    gamma = 0.99
    max_grad_norm = 0.5
    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    gae_lambda = 0.95
    NUM_ENVS = 1
    seed = 1
    GPT_LOG_INTERVAL = max(int(num_episodes // 10), 1)
    anneal_lr = False
    track = args.track
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # create a config file including all hyperparameters and agent info
    config = {
        "anneal_lr": anneal_lr,
        "gamma": gamma,
        "max_grad_norm": max_grad_norm,
        "clip_coef": clip_coef,
        "vf_coef": vf_coef,
        "ent_coef": ent_coef,
        "gae_lambda": gae_lambda,
        "num_envs": NUM_ENVS,
        "seed": seed,
        "device": device,
        "num_big_agents": NUM_BIG_AGENTS,
        "num_small_agents": NUM_SMALL_AGENTS,
        "episode_length": EPISODE_LENGTH,
        "random_act": RANDOM_ACT,
        "big_agent_range": BIG_AGENT_RANGE,
        "small_agent_range": SMALL_AGENT_RANGE,
        "learning_rate": LEARNING_RATE,
        **vars(args),
    }
    import datetime

    env_info_statistics: dict[str, list] = {}
    # name = current date + customed name
    best_mean_aoi = 200
    additional_tags = []
    for item in ['anneal_lr']:
        if config[item]:
            additional_tags.append(item)
    # add suffix for name at here.
    for item in APPEND_TAGS:
        if item in config:
            additional_tags.append(item + '_' + str(config[item]))
    # add date time to name
    expr_name = datetime.datetime.today().strftime("%m%d-%H%M%S") + '-' + args.name

    # add additional_tags to name
    if len(additional_tags) > 0:
        expr_name += ('-' + "-".join(additional_tags))

    checkpoint_path = os.path.join('/workspace', 'saved_data', 'checkpoints', expr_name)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    env = MultiAgentGridWorld(num_big_agents=NUM_BIG_AGENTS, num_small_agents=NUM_SMALL_AGENTS,
                              self_factor=args.self_factor, group_factor=args.group_factor,
                              log_action=LOG_ACTION, num_emergencies=EMERGENCY_NUMBER,
                              max_timesteps=EPISODE_LENGTH)
    # Initialize actor-critic models for big and small agents
    big_action_shape = get_action_shape(env.big_action_space)
    small_action_shape = get_action_shape(env.small_action_space)
    if env.big_vec_mode:
        big_agent_policy = PPOVecPolicy(input_dim=env.big_agent_shape,
                                        num_actions=big_action_shape,
                                        fc_size=HIDDEN_SIZE).to(device)
    else:
        big_agent_policy = PPOCNNPolicy(input_shape=env.big_agent_shape,
                                        num_actions=big_action_shape,
                                        fc_size=HIDDEN_SIZE).to(device)
    if env.small_vec_mode:
        small_agent_policy = PPOVecPolicy(input_dim=env.small_agent_shape).to(device)
    else:
        small_agent_policy = PPOCNNPolicy(input_shape=env.small_agent_shape,
                                          num_actions=small_action_shape,
                                          fc_size=HIDDEN_SIZE).to(device)

    small_agent_rollout = MultiPPORollout(num_agents=NUM_SMALL_AGENTS)
    big_agent_rollout = MultiPPORollout(num_agents=NUM_BIG_AGENTS)

    if track:
        # note big_XXX indicates the neural network architecture, XXX is the architecture
        # which may be cnn, mlp, etc.
        # swanlab.sync_wandb(wandb_run=False)
        wandb.init(project=PROJECT_NAME, name=expr_name, group='emergency_mvp',
                   tags=['ppo', 'big_cnn', 'small_cnn'],
                   config=config, dir=os.path.join('/workspace', 'saved_data'))
        wandb.define_metric(BIG_AGENT_METRIC, summary="max")
        wandb.define_metric(SMALL_AGENT_METRIC, summary="max")
        wandb.define_metric(f'{ENV_INFO}/{SURVEILLANCE_AOI}', summary='min')
        wandb.define_metric(f'{ENV_INFO}/EMERGENCY_AOI', summary='min')
        wandb.watch([big_agent_policy, small_agent_policy], log="all", log_graph=False)

    next_obs = env.reset()
    next_done = torch.zeros(NUM_ENVS).to(device)
    # Optimizers for big and small agents
    big_agent_optimizer = optim.Adam(big_agent_policy.parameters(), lr=LEARNING_RATE)
    small_agent_optimizer = optim.Adam(small_agent_policy.parameters(), lr=LEARNING_RATE)

    if args.mode == 'train':
        progress = trange(num_episodes)
    else:
        progress = trange(1)
    if args.mode == 'debug':
        progress = trange(100)
        logger.setLevel(logging.DEBUG)
        EVAL_INTERVAL = 1
    info = {}
    if args.mode == 'test' and args.model_path != '':
        # load model from saved checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)
        big_agent_policy.load_state_dict(checkpoint[BIG_AGENT_MODEL])
        small_agent_policy.load_state_dict(checkpoint[SMALL_AGENT_MODEL])
    # Main loop for environment interaction
    for episode in progress:  # 200 episodes for demonstration
        if anneal_lr:
            frac = 1.0 - (episode - 1.0) / num_episodes
            lrnow = frac * LEARNING_RATE
            big_agent_optimizer.param_groups[0]["lr"] = lrnow
            small_agent_optimizer.param_groups[0]["lr"] = lrnow
        next_obs = env.reset()
        # Initialize reward tracking for the episode
        total_big_agent_rewards = [0 for _ in range(NUM_BIG_AGENTS)]
        total_small_agent_rewards = [0 for _ in range(NUM_SMALL_AGENTS)]
        deploy_statistic = {}
        # create keys that stores "deploy_reward" and "deploy_time" for each big agent
        for i in range(NUM_BIG_AGENTS):
            metrics = ['deploy_reward', 'deploy_time', 'assign_reward', 'targets']
            for metric in metrics:
                deploy_statistic[f'big_{i}_{metric}'] = []
        small_agent_statistic = {}
        big_agent_statistic = {}
        for t in range(EPISODE_LENGTH):
            if RANDOM_ACT:
                actions = random_act(env)
            else:
                with torch.no_grad():
                    actions = {}
                    if not env.simple_mode:
                        # Usage for big agents
                        select_actions('big', big_agent_policy, NUM_BIG_AGENTS,
                                       next_obs, actions, env, device, big_agent_rollout)
                    # Usage for small agents
                    select_actions('small', small_agent_policy, NUM_SMALL_AGENTS,
                                   next_obs, actions, env, device, small_agent_rollout)
            # Step the environment with the selected actions
            next_obs, rewards, next_done, info = env.step(actions)

            # Accumulate rewards for training and for average reward calculation
            if not env.simple_mode:
                for i in range(NUM_BIG_AGENTS):
                    big_agent_rollout.rewards[i].append(rewards[f'big_{i}'])
                    big_agent_rollout.dones[i].append(next_done)
                    total_big_agent_rewards[i] += rewards[f'big_{i}']
                    for metric in deploy_statistic.keys():
                        if metric in info:
                            deploy_statistic[metric].append(info[metric])

            for i in range(NUM_SMALL_AGENTS):
                if env.small_agents[i]['last_deploy_status']:
                    small_agent_rollout.rewards[i].append(rewards[f'small_{i}'])
                    small_agent_rollout.dones[i].append(next_done)
                    total_small_agent_rewards[i] += rewards[f'small_{i}']

            if next_done:
                break

        # After the episode, update the parameters for big agents and small agents
        if (not RANDOM_ACT) or args.mode == 'train':
            if not env.simple_mode:
                big_agent_rollout.concatenate_rollouts(big_agent_policy)
                # select small agent obs and big obs, concat them into together, respectively.
                big_agent_obs = torch.cat([torch.from_numpy(next_obs[f'big_{i}']).float().unsqueeze(0)
                                           for i in range(NUM_BIG_AGENTS)]).to(device)
                # Update the big agent policy using the saved actions and rewards
                big_agent_statistic.update(
                    big_agent_policy.finish_episode(big_agent_optimizer, max_grad_norm=max_grad_norm,
                                                    clip_coef=clip_coef, vf_coef=vf_coef,
                                                    ent_coef=ent_coef,
                                                    gae_lambda=gae_lambda, num_minibatches=NUM_MINIBATCHES,
                                                    num_envs=NUM_BIG_AGENTS,
                                                    next_state=big_agent_obs, next_dones=next_done,
                                                    device=device,
                                                    num_steps=EPISODE_LENGTH))

            small_agent_rollout.concatenate_rollouts(small_agent_policy)
            small_agent_obs = torch.cat([torch.from_numpy(next_obs[f'small_{i}']).float().unsqueeze(0)
                                         for i in range(NUM_SMALL_AGENTS)]).to(device)
            if len(small_agent_policy.rewards) >= NUM_SMALL_AGENTS:
                # Update the small agent policy using the saved actions and rewards
                small_agent_statistic.update(
                    small_agent_policy.finish_episode(small_agent_optimizer, max_grad_norm=max_grad_norm,
                                                      clip_coef=clip_coef, vf_coef=vf_coef, ent_coef=ent_coef,
                                                      gae_lambda=gae_lambda, num_minibatches=NUM_MINIBATCHES,
                                                      num_envs=NUM_SMALL_AGENTS,
                                                      next_state=small_agent_obs, next_dones=next_done, device=device,
                                                      num_steps=len(small_agent_policy.rewards) // NUM_SMALL_AGENTS))

        # Calculate average rewards
        avg_big_agent_reward = sum(total_big_agent_rewards) / NUM_BIG_AGENTS
        avg_small_agent_reward = sum(total_small_agent_rewards) / max(1, len([r for r in total_small_agent_rewards if
                                                                              r != 0]))
        # average reward with NUM_SMALL_AGENTS in deploy_statistic dict
        average_deploy_statistic = {}
        for key in deploy_statistic:
            length = len(deploy_statistic[key])
            if length > 0:
                average_deploy_statistic[key] = sum(deploy_statistic[key]) / length
        log_dict = {}
        if args.mode == 'test' or episode % EVAL_INTERVAL == 0:
            figure = env.render()
            log_dict[f'{PLOT_NAME}'] = figure
            if LOG_TABLE:
                # convert small agent position and target to dataframe
                for small_agent_id, small_agent in enumerate(env.small_agents):
                    # Convert small_agent_trajectories and small_agent_targets into numpy arrays for efficiency
                    if len(env.small_agent_trajectories[small_agent_id]) > 0:
                        trajectories = np.array(env.small_agent_trajectories[small_agent_id])  # Shape: (num_steps, 2)
                        targets = np.array(env.small_agent_targets[small_agent_id])  # Shape: (num_steps, 2)
                        # Create DataFrame directly from the numpy arrays
                        df = pd.DataFrame({
                            "x": trajectories[:, 0],  # First column from trajectories
                            "y": trajectories[:, 1],  # Second column from trajectories
                            "target_x": targets[:, 0],  # First column from targets
                            "target_y": targets[:, 1],  # Second column from targets
                        }, index=pd.Index(np.arange(trajectories.shape[0]), name="step"))  # Set the index to "step"
                        log_dict[f"small_{small_agent_id}_history"] = wandb.Table(data=df,
                                                                                  columns=["x", "y", "target_x",
                                                                                           "target_y"])
        # GPT Logging Interval
        # if episode % GPT_LOG_INTERVAL == 0:
        #     for metric in info:
        #         if metric in env_info_statistics:
        #             env_info_statistics[metric].append(info[metric])
        #         else:
        #             env_info_statistics[metric] = [info[metric]]
        # add prefix for big agent dict and small agent dict
        for k, v in small_agent_statistic.items():
            log_dict[f'{SMALL_AGENT_TRAIN}/{k}'] = v
        for k, v in big_agent_statistic.items():
            log_dict[f'{BIG_AGENT_TRAIN}/{k}'] = v
        for k, v in info.items():
            log_dict[f'{ENV_INFO}/{k}'] = v
        if LOG_ACTION:
            for i in range(NUM_BIG_AGENTS):
                log_dict[f'{BIG_AGENT_ACTION}/Agent{i}'] = wandb.Histogram(env.big_agent_actions[i],
                                                                           num_bins=NUM_MOVEMENTS)
        log_dict.update(
            {
                BIG_AGENT_METRIC: avg_big_agent_reward,
                SMALL_AGENT_METRIC: avg_small_agent_reward,
                **average_deploy_statistic,
            }
        )
        assert EMERGENCY_AOI in info
        progress.set_postfix(EMERGENCY_AOI=info[EMERGENCY_AOI])
        # save best model according min mean aoi
        if track:
            if info[EMERGENCY_AOI] < best_mean_aoi:
                best_mean_aoi = info[EMERGENCY_AOI]
                # save big and small agent as a single state_dict
                state_dicts = {
                    BIG_AGENT_MODEL: big_agent_policy.state_dict(),
                    SMALL_AGENT_MODEL: small_agent_policy.state_dict(),
                }
                torch.save(state_dicts,
                           os.path.join(checkpoint_path, 'best_model.pt')
                           )
        if track and wandb.log is not None:
            wandb.log(log_dict)
    current_run_id = wandb.run.id if wandb.run is not None else None
    wandb.finish()
    # Generate GPT feedback prompt
    feedback_prompt = ''
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    feedback_prompt += policy_feedback.format(epoch_freq=GPT_LOG_INTERVAL)
    log_api = wandb.Api()
    if current_run_id is not None:
        run = log_api.run(f'{USERNAME}/{PROJECT_NAME}/{current_run_id}')
        history = run.history()
        for metric in list(history.columns):
            if metric.startswith(f"{ENV_INFO}/"):
                # sample every GPT_LOG_INTERVAL steps, calculate max, min, mean for entire history
                sampled_history = history[metric].iloc[::GPT_LOG_INTERVAL]
                # round to 2 decimal places
                sampled_history = sampled_history.round(decimals=2)
                feedback_prompt += (f"{metric}: {sampled_history.to_list()} "
                                    f"Max: {history[metric].max():.2f}, "
                                    f"Min: {history[metric].min():.2f}, "
                                    f"Mean: {history[metric].mean():.2f}\n")
        feedback_prompt += (code_feedback + code_output_tip)
        # save feedback prompt to file
        with open(os.path.join(checkpoint_path, "feedback_prompt.txt"), "w") as f:
            f.write(feedback_prompt)
