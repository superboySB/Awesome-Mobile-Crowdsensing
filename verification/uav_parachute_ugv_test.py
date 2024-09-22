import os
import random
from collections import namedtuple
import argparse
import gym
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import MultiDiscrete
from torch.distributions import Categorical
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ppo_algo_verify import PPOVecPolicy, PPOCNNPolicy

LEARNING_RATE = 1e-4

# Constants
PROJECT_NAME = 'uav-parachute-ugv'
GAMMA = 0
RANDOM_ACT = False
GRID_SIZE = 20
EPISODE_LENGTH = 60
BIG_AGENT_METRIC = "big_reward"
BIG_AGENT_TRAIN = "big_train"
SMALL_AGENT_TRAIN = "small_train"
SMALL_AGENT_METRIC = "small_reward"
BIG_AGENT_DEPLOY_METRIC = "big_reward_deploy"
MEAN_AOI = "mean_aoi"
BIG_AGENT_RANGE = 8
SMALL_AGENT_RANGE = 4
NUM_BIG_AGENTS = 2
NUM_SMALL_AGENTS = 4
NUM_ACTIONS = 5  # up, down, left, right, stop
TIMESTEP_DEPLOY = [20, 40]
NUM_CLUSTERS = 5
CLUSTER_RADIUS = 3
MAX_VALUE = 10

# Actions
UP, DOWN, LEFT, RIGHT, STOP = 0, 1, 2, 3, 4

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def generate_clusters(grid_size, num_clusters, cluster_radius, max_value=10):
    grid = np.zeros((grid_size, grid_size))

    for _ in range(num_clusters):
        # Randomly choose a cluster center
        center_x = random.randint(0, grid_size - 1)
        center_y = random.randint(0, grid_size - 1)

        # Populate the cluster area with values
        for x in range(max(0, center_x - cluster_radius), min(grid_size, center_x + cluster_radius + 1)):
            for y in range(max(0, center_y - cluster_radius), min(grid_size, center_y + cluster_radius + 1)):
                # The value decreases with distance from the center
                distance = np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
                if distance <= cluster_radius:
                    grid[x, y] += max_value - int(distance)

    return grid


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


# Environment class

class MultiAgentGridWorld(gym.Env):
    def __init__(self, num_big_agents=NUM_BIG_AGENTS, num_small_agents=NUM_SMALL_AGENTS):
        super(MultiAgentGridWorld, self).__init__()
        self.grid_size = GRID_SIZE
        self.num_big_agents = num_big_agents
        self.num_small_agents = num_small_agents
        self.timestep = 0
        self.max_timesteps = EPISODE_LENGTH

        self.seed = 1
        self.small_vec_mode = False
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Update action space for big agents
        self.big_action_space = MultiDiscrete([5, 2])  # [movement (5), deploy (2)]
        self.small_action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # Combine action spaces for compatibility
        self.action_space = {
            'big': self.big_action_space,
            'small': self.small_action_space
        }

        # Observation space remains the same
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Initialize agents' positions and state
        self.big_agents = [
            {'position': [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)], 'carried_agents': 2} for
            _ in range(NUM_BIG_AGENTS)]
        self.small_agents = [{'position': None, 'deployed': False, 'last_deploy_status': False} for _ in
                             range(NUM_SMALL_AGENTS)]

        # Initialize the PoI grid with certain clustered PoI values
        self.poi_grid = generate_clusters(GRID_SIZE, NUM_CLUSTERS, CLUSTER_RADIUS, MAX_VALUE)

        # Initialize AoI grid (starts at 0 for all PoIs)
        self.aoi_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.aoi_grid_by_time = np.zeros((EPISODE_LENGTH, GRID_SIZE, GRID_SIZE))
        self.max_reward = self.poi_grid.max()
        self.min_reward = 0
        # self.max_deploy_reward = self.max_reward
        # self.min_deploy_reward = 0

        # Initialize rewards
        self.big_agent_rewards = [0 for _ in range(NUM_BIG_AGENTS)]
        self.small_agent_rewards = [0 for _ in range(NUM_SMALL_AGENTS)]

    def step(self, actions) -> [dict, dict, bool, dict]:
        info = {}
        rewards = {f'big_{i}': 0 for i in range(self.num_big_agents)}
        rewards.update({f'small_{i}': 0 for i in range(self.num_small_agents)})
        # Process small agent actions
        for small_agent_id, small_agent in enumerate(self.small_agents):
            small_agent['last_deploy_status'] = small_agent['deployed']

            if small_agent['deployed']:
                action = actions[f'small_{small_agent_id}']
                self._move_agent(small_agent, action)
                x, y = small_agent['position']
                rewards[f'small_{small_agent_id}'] = self.aoi_grid[x, y] * self.poi_grid[x, y] / self.max_timesteps
                self.max_reward = max(self.max_reward, rewards[f'small_{small_agent_id}'])
                self.min_reward = min(self.min_reward, rewards[f'small_{small_agent_id}'])
                rewards[f'small_{small_agent_id}'] = (
                        (rewards[f'small_{small_agent_id}'] - self.min_reward) / (self.max_reward - self.min_reward))
                self.aoi_grid[x, y] = 0
            else:
                rewards[f'small_{small_agent_id}'] = 0

        # Process big agent actions
        for big_agent_id, big_agent in enumerate(self.big_agents):
            my_action = actions[f'big_{big_agent_id}']
            if isinstance(my_action, np.ndarray):
                movement_action, deploy_action = my_action[0]
            else:
                raise NotImplementedError("Action must be a numpy array of shape (2,)")
            self._move_agent(big_agent, movement_action)

            # Handle deployment action
            if deploy_action == 1 and big_agent['carried_agents'] > 0:
                deploy_x, deploy_y = big_agent['position']
                # big agent is rewarded with AoI sum of PoIs around deployment area
                deploy_reward = self.aoi_grid[deploy_x, deploy_y] * self.poi_grid[
                    deploy_x, deploy_y] / self.max_timesteps
                info[f'big_{big_agent_id}_deploy_reward'] = rewards[f'big_{big_agent_id}'] = deploy_reward
                # self.max_deploy_reward = max(self.max_deploy_reward, rewards[f'big_{big_agent_id}'])
                # self.min_deploy_reward = min(self.min_deploy_reward, rewards[f'big_{big_agent_id}'])
                # rewards[f'big_{big_agent_id}'] = (
                #         (rewards[f'big_{big_agent_id}'] - self.min_deploy_reward) / (
                #                 self.max_deploy_reward - self.min_deploy_reward))
                info[f'big_{big_agent_id}_deploy_time'] = self.timestep
                self._deploy_small_agent(big_agent_id)
            elif deploy_action == 1 and big_agent['carried_agents'] == 0:
                # Penalize big agent for not deploying when there are no small agents to carry
                # Warn: Penalty make to policy unable to train or not deploying at all.
                pass
            else:
                rewards[f'big_{big_agent_id}'] = 0

        # Update AoI for all grid cells
        self.aoi_grid += 1  # Increment AoI for all PoIs at each timeste

        # Big agents get rewards based on the total rewards of their small agents for this timestep
        for big_agent_id, big_agent in enumerate(self.big_agents):
            small_agent_ids = range(big_agent_id * 2, big_agent_id * 2 + 2)
            rewards[f'big_{big_agent_id}'] += sum(rewards[f'small_{i}'] for i in small_agent_ids)

        self.aoi_grid_by_time[self.timestep] = self.aoi_grid * self.poi_grid
        self.timestep += 1
        done = self.timestep >= self.max_timesteps

        if done:
            info[MEAN_AOI] = np.mean(self.aoi_grid_by_time)

        # Return the observations, rewards for this timestep, done flag, and additional info
        return self._get_observation(), rewards, done, info

    def reset(self):
        self.timestep = 0
        self.big_agents = [
            {'position': [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)], 'carried_agents': 2} for
            _ in range(self.num_big_agents)]
        self.small_agents = [{'position': None, 'deployed': False, 'last_deploy_status': False} for _ in
                             range(self.num_small_agents)]
        self.aoi_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        return self._get_observation()

    def _get_vec_observation(self, agent):
        """
        Generates a vector observation for a UGV agent based on its position
        and the AoI grid.
        """
        # Get agent position
        x, y = agent['position']

        # Initialize the reward vector for the 5 possible actions
        action_rewards = np.zeros(NUM_ACTIONS)

        # Define action directions
        directions = {
            UP: (-1, 0),
            DOWN: (1, 0),
            LEFT: (0, -1),
            RIGHT: (0, 1),
            STOP: (0, 0)
        }

        for action in range(NUM_ACTIONS):
            dx, dy = directions[action]
            new_x = max(0, min(self.grid_size - 1, x + dx))
            new_y = max(0, min(self.grid_size - 1, y + dy))
            if action == STOP:
                # When stopping, reward is based on current cell AoI
                action_rewards[action] = self.aoi_grid[x, y] * self.poi_grid[x, y] / self.max_timesteps
            else:
                # When moving, reward is based on the AoI of the new cell
                action_rewards[action] = self.aoi_grid[new_x, new_y] * self.poi_grid[new_x, new_y] / self.max_timesteps

        self.max_reward = max(self.max_reward, np.max(action_rewards))
        self.min_reward = min(self.min_reward, np.min(action_rewards))
        # scale rewards to [0, 1]
        action_rewards = 2 * (action_rewards - self.min_reward) / (self.max_reward - self.min_reward) - 1

        # combine action_rewards with current agent location
        # bug 1: when actions are masked, small agent should be 0.
        # bug 2: invalid data should be removed from buffer.
        return np.concatenate([np.array(agent['position']) / self.grid_size, action_rewards])

    def _get_grid_observation(self, agent, observation_range):
        """
        Generates a grid observation for a UAV agent based on its position
        and the PoI grid.
        """
        x, y = agent['position']

        # Extracting the grid portion
        obs = self.poi_grid[max(0, x - observation_range // 2):min(self.grid_size, x + observation_range // 2),
              max(0, y - observation_range // 2):min(self.grid_size, y + observation_range // 2)]

        # Padding to make the observation square of size observation_range * observation_range
        padded_obs = np.pad(obs,
                            ((max(0, observation_range // 2 - x),
                              max(0, x + observation_range // 2 - self.grid_size)),
                             (max(0, observation_range // 2 - y),
                              max(0, y + observation_range // 2 - self.grid_size))),
                            mode='constant', constant_values=0)

        return padded_obs

    def _get_observation(self):
        """
        Combines observations for all agents. Small agents get vector observations,
        and big agents get grid observations.
        """
        observations = {}

        for big_agent_id, big_agent in enumerate(self.big_agents):
            observations[f'big_{big_agent_id}'] = self._get_grid_observation(big_agent, BIG_AGENT_RANGE)

        for small_agent_id, small_agent in enumerate(self.small_agents):
            if small_agent['deployed']:
                if self.small_vec_mode:
                    observations[f'small_{small_agent_id}'] = self._get_vec_observation(small_agent)
                else:
                    observations[f'small_{small_agent_id}'] = self._get_grid_observation(small_agent, SMALL_AGENT_RANGE)
            else:
                # Small agents not deployed will have masked observation.
                if self.small_vec_mode:
                    observations[f'small_{small_agent_id}'] = np.zeros(NUM_ACTIONS)
                else:
                    observations[f'small_{small_agent_id}'] = np.zeros((SMALL_AGENT_RANGE, SMALL_AGENT_RANGE))

        return observations

    def render(self, mode='human'):
        """
        Renders the grid environment with big agents, small agents, and the AoI * PoI product in each grid cell.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=float)

        # Calculate AoI * PoI product for each cell
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                grid[i, j] = self.aoi_grid[i, j] * self.poi_grid[i, j]

        # Set up the plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw grid with AoI * PoI product as text
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                ax.text(j, i, f'{grid[i, j]:.1f}', ha='center', va='center', fontsize=8, color='black')

        # Draw big agents (as stars)
        for big_agent in self.big_agents:
            x, y = big_agent['position']
            ax.scatter(y, x, marker='*', color='blue', s=200, edgecolor='black')

        # Draw small agents (as circles)
        for small_agent in self.small_agents:
            if small_agent['deployed']:
                x, y = small_agent['position']
                ax.scatter(y, x, marker='o', color='red', s=100, edgecolor='black')

        # Set title and show the plot
        ax.set_title(f'Timestep: {self.timestep}')
        plt.show()
        plt.close()
        time.sleep(0.1)



    def _move_agent(self, agent, action):
        """
        Move the agent based on the action (up, down, left, right, stop).
        Big agents move in larger steps compared to small agents.
        """
        # Determine step size based on agent type
        if agent in self.big_agents:
            step_size = 4  # Larger step size for big agents
        else:
            step_size = 1  # Default step size for small agents

        if action == UP:
            agent['position'][0] = max(0, agent['position'][0] - step_size)
        elif action == DOWN:
            agent['position'][0] = min(self.grid_size - 1, agent['position'][0] + step_size)
        elif action == LEFT:
            agent['position'][1] = max(0, agent['position'][1] - step_size)
        elif action == RIGHT:
            agent['position'][1] = min(self.grid_size - 1, agent['position'][1] + step_size)
        elif action == STOP:
            pass  # Do nothing

    def _move_agent_unified(self, agent, action):
        # Move agent based on the action (up, down, left, right, stop)
        if action == UP:
            agent['position'][0] = max(0, agent['position'][0] - 1)
        elif action == DOWN:
            agent['position'][0] = min(self.grid_size - 1, agent['position'][0] + 1)
        elif action == LEFT:
            agent['position'][1] = max(0, agent['position'][1] - 1)
        elif action == RIGHT:
            agent['position'][1] = min(self.grid_size - 1, agent['position'][1] + 1)
        elif action == STOP:
            pass  # Do nothing

    def _deploy_small_agent(self, big_agent_id):
        big_agent = self.big_agents[big_agent_id]
        if big_agent['carried_agents'] > 0:
            small_agent_id = big_agent_id * 2 + (2 - big_agent['carried_agents'])
            small_agent = self.small_agents[small_agent_id]
            small_agent['position'] = big_agent['position'][:]
            small_agent['deployed'] = True
            big_agent['carried_agents'] -= 1


def test_MultiAgentGridWorld():
    global env
    # Demo of environment interaction
    env = MultiAgentGridWorld()
    obs = env.reset()
    for _ in range(EPISODE_LENGTH):
        # Random actions for big agents
        actions = random_act(env)

        obs, rewards, done, _ = env.step(actions)
        print(f"Step {_}: Rewards: {rewards}")

        if done:
            break


def random_act(env):
    actions = {}
    for i in range(NUM_BIG_AGENTS):
        actions[f'big_{i}'] = env.action_space.sample()
    # Random actions for deployed small agents (all initially STOP)
    for i in range(NUM_SMALL_AGENTS):
        if env.small_agents[i]['deployed']:
            actions[f'small_{i}'] = env.action_space.sample()
        else:
            actions[f'small_{i}'] = STOP
    return actions


def select_actions(agent_type: str, policy, num_agents, obs: dict, actions, env, device):
    for i in range(num_agents):
        if agent_type == 'big':
            current_obs = obs[f'big_{i}']
        elif agent_type == 'small':
            if not env.small_agents[i]['deployed']:
                actions[f'small_{i}'] = STOP
                continue
            current_obs = obs[f'small_{i}']
        else:
            raise NotImplementedError("Invalid agent type: {agent_type}")

        # Convert observation to tensor and ensure correct dimensions
        if len(current_obs.shape) == 1:
            state = torch.from_numpy(current_obs).float().unsqueeze(0).to(device)
        else:
            state = torch.from_numpy(current_obs).float().unsqueeze(0).unsqueeze(0).to(device)

        # Get action, log probability, and state value from policy
        action, log_prob, _, state_value = policy.get_action_and_value(state)

        # Save the action, log probability, and state value for training later
        policy.saved_actions.append(action)
        policy.log_probs.append(log_prob)
        policy.values.append(state_value.squeeze())
        policy.saved_obs.append(state)

        # Record the selected action
        actions[f'{agent_type}_{i}'] = action.cpu().numpy()


def ppo_joint_act(
        obs: dict[np.ndarray],
        big_agent_policy: PPOVecPolicy,
        small_agent_policy: PPOVecPolicy,
) -> dict[int]:
    # Dictionary to hold actions for each agent
    actions = {}
    # Usage for big agents
    select_actions('big', big_agent_policy, NUM_BIG_AGENTS, obs, actions, env, device)
    # Usage for small agents
    select_actions('small', small_agent_policy, NUM_SMALL_AGENTS, obs, actions, env, device)
    return actions


def actor_critic_joint_act(obs: dict[np.ndarray], big_agent_policy: Policy,
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
        if env.small_agents[i]['deployed']:
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


if __name__ == '__main__':
    # setup wandb
    import wandb

    # setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', action='store_true', help='track the experiment')
    parser.add_argument('--name', type=str, default='test', help='name of the experiment')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='mode of the experiment')
    args = parser.parse_args()
    num_episodes: int = 5000
    gamma = 0.99
    max_grad_norm = 0.5
    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    gae_lambda = 0.95
    NUM_ENVS = 1
    seed = 1
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
    }
    import datetime

    # name = current date + customed name
    additional_tags = []
    for item in ['anneal_lr']:
        if config[item]:
            additional_tags.append(item)
    # add date time to name
    expr_name = datetime.datetime.today().strftime("%m%d-%H%M") + '-' + args.name
    # add additional_tags to name
    if len(additional_tags) > 0:
        expr_name += ('-' + "-".join(additional_tags))
    if track:
        # note big_XXX indicates the neural network architecture, XXX is the architecture
        # which may be cnn, mlp, etc.
        wandb.init(project=PROJECT_NAME, name=expr_name, group='mvp',
                   tags=['ppo', 'big_cnn', 'small_cnn'],
                   config=config, dir=os.path.join('/workspace', 'saved_data'))
        wandb.define_metric(BIG_AGENT_METRIC, summary="max")
        wandb.define_metric(SMALL_AGENT_METRIC, summary="max")
        wandb.define_metric(MEAN_AOI, summary='min')

    env = MultiAgentGridWorld()

    # Initialize actor-critic models for big and small agents
    big_agent_policy = PPOCNNPolicy(input_shape=(1, BIG_AGENT_RANGE, BIG_AGENT_RANGE),
                                    num_actions=[5, 2],
                                    fc_size=64).to(device)
    # small_agent_policy = PPOVecPolicy(input_dim=5 + 2).to(device)
    small_agent_policy = PPOCNNPolicy(input_shape=(1, SMALL_AGENT_RANGE, SMALL_AGENT_RANGE),
                                      num_actions=[NUM_ACTIONS],
                                      fc_size=64).to(device)

    next_obs = env.reset()
    next_done = torch.zeros(NUM_ENVS).to(device)
    # Optimizers for big and small agents
    big_agent_optimizer = optim.Adam(big_agent_policy.parameters(), lr=LEARNING_RATE)
    small_agent_optimizer = optim.Adam(small_agent_policy.parameters(), lr=LEARNING_RATE)

    if args.mode == 'train':
        progress = trange(num_episodes)
    else:
        progress = range(1)
    info = {}
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
            metrics = ['deploy_reward', 'deploy_time']
            for metric in metrics:
                deploy_statistic[f'big_{i}_{metric}'] = 0
        small_agent_statistic = {}
        big_agent_statistic = {}
        for t in range(EPISODE_LENGTH):
            if RANDOM_ACT:
                actions = random_act(env)
            else:
                with torch.no_grad():
                    actions = ppo_joint_act(next_obs, big_agent_policy, small_agent_policy)
            # Step the environment with the selected actions
            next_obs, rewards, next_done, info = env.step(actions)
            if args.mode == 'test':
                env.render()

            # Accumulate rewards for training and for average reward calculation
            for i in range(NUM_BIG_AGENTS):
                big_agent_policy.rewards.append(rewards[f'big_{i}'])
                big_agent_policy.dones.append(next_done)
                total_big_agent_rewards[i] += rewards[f'big_{i}']
                for metric in deploy_statistic.keys():
                    if metric in info:
                        deploy_statistic[metric] += info[metric]

            for i in range(NUM_SMALL_AGENTS):
                if env.small_agents[i]['last_deploy_status']:
                    small_agent_policy.rewards.append(rewards[f'small_{i}'])
                    small_agent_policy.dones.append(next_done)
                    total_small_agent_rewards[i] += rewards[f'small_{i}']

            if next_done:
                break

        # After the episode, update the parameters for big agents and small agents
        if (not RANDOM_ACT) or args.mode == 'train':
            # select small agent obs and big obs, concat them into together, respectively.
            big_agent_obs = torch.cat([torch.from_numpy(next_obs[f'big_{i}']).float().unsqueeze(0)
                                       for i in range(NUM_BIG_AGENTS)]).to(device)
            small_agent_obs = torch.cat([torch.from_numpy(next_obs[f'small_{i}']).float().unsqueeze(0)
                                         for i in range(NUM_SMALL_AGENTS)]).to(device)
            # Update the big agent policy using the saved actions and rewards
            big_agent_statistic.update(big_agent_policy.finish_episode(big_agent_optimizer, max_grad_norm=max_grad_norm,
                                            clip_coef=clip_coef, vf_coef=vf_coef, ent_coef=ent_coef,
                                            gae_lambda=gae_lambda, num_minibatches=4, num_envs=NUM_BIG_AGENTS,
                                            next_state=big_agent_obs, next_dones=next_done, device=device,
                                                                       num_steps=EPISODE_LENGTH))
            if len(small_agent_policy.rewards) >= NUM_SMALL_AGENTS:
                # Update the small agent policy using the saved actions and rewards
                small_agent_statistic.update(
                    small_agent_policy.finish_episode(small_agent_optimizer, max_grad_norm=max_grad_norm,
                                                      clip_coef=clip_coef, vf_coef=vf_coef, ent_coef=ent_coef,
                                                      gae_lambda=gae_lambda, num_minibatches=4,
                                                      num_envs=NUM_SMALL_AGENTS,
                                                      next_state=small_agent_obs, next_dones=next_done, device=device,
                                                      num_steps=len(small_agent_policy.rewards) // NUM_SMALL_AGENTS))

        # Calculate average rewards
        avg_big_agent_reward = sum(total_big_agent_rewards) / NUM_BIG_AGENTS
        avg_small_agent_reward = sum(total_small_agent_rewards) / max(1, len([r for r in total_small_agent_rewards if
                                                                              r != 0]))
        # average reward with NUM_SMALL_AGENTS in deploy_statistic dict
        for key in deploy_statistic:
            if 'reward' in key:
                deploy_statistic[key] /= NUM_SMALL_AGENTS

        # add prefix for big agent dict and small agent dict
        log_dict = {}
        for k, v in small_agent_statistic.items():
            log_dict[f'{SMALL_AGENT_TRAIN}/{k}'] = v
        for k, v in big_agent_statistic.items():
            log_dict[f'{BIG_AGENT_TRAIN}/{k}'] = v
        log_dict.update({
            BIG_AGENT_METRIC: avg_big_agent_reward,
            SMALL_AGENT_METRIC: avg_small_agent_reward,
            **info,
            **deploy_statistic,
        })
        assert MEAN_AOI in info
        progress.set_postfix(MEAN_AOI=info[MEAN_AOI])
        if track and wandb.log is not None:
            wandb.log(log_dict)
    wandb.finish()
