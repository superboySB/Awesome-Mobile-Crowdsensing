import random
from collections import namedtuple

import gym
import torch.optim as optim
from torch.distributions import Categorical

# Constants
GAMMA = 0
RANDOM_ACT = False
GRID_SIZE = 20
EPISODE_LENGTH = 60
BIG_AGENT_RANGE = 8
SMALL_AGENT_RANGE = 4
NUM_BIG_AGENTS = 2
NUM_SMALL_AGENTS = 4
NUM_ACTIONS = 5  # up, down, left, right, stop
TIMESTEP_DEPLOY = [20, 40]

# Actions
UP, DOWN, LEFT, RIGHT, STOP = 0, 1, 2, 3, 4

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class CNNPolicy(Policy):
    """
    Implements both actor and critic in one model using CNN for 2D grid input.
    """

    def __init__(self, input_shape, num_actions=5, conv_channels=[4, 8, 16], kernel_sizes=[3, 3, 3], fc_size=32):
        super(CNNPolicy, self).__init__()

        # Assert the lengths of conv_channels and kernel_sizes match the number of layers
        assert len(conv_channels) == 3, "Please provide 3 values for conv_channels"
        assert len(kernel_sizes) == 3, "Please provide 3 values for kernel_sizes"

        # CNN layers for 2D input with adjustable channels and kernel sizes
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=conv_channels[0],
                               kernel_size=kernel_sizes[0], stride=1, padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1],
                               kernel_size=kernel_sizes[1], stride=1, padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2],
                               kernel_size=kernel_sizes[2], stride=1, padding=kernel_sizes[2] // 2)

        # Calculate the output size of the convolution layers to determine input size for fc1
        conv_output_size = self._get_conv_output_size(input_shape)

        # Fully connected layer after flattening
        self.fc1 = nn.Linear(conv_output_size, fc_size)

        # Actor's layer (outputs probabilities over actions)
        self.action_head = nn.Linear(fc_size, num_actions)

        # Critic's layer (outputs state value)
        self.value_head = nn.Linear(fc_size, 1)

    def _get_conv_output_size(self, input_shape):
        """Calculate the size of the output after the convolution layers"""
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            sample_output = self.conv3(self.conv2(self.conv1(sample_input)))
            return int(np.prod(sample_output.size()))

    def forward(self, x):
        """
        Forward pass of both actor and critic.
        """
        # Apply CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from the convolution layers
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = F.relu(self.fc1(x))

        # Actor: chooses action to take from state s_t
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # Critic: evaluates the value of the state
        state_value = self.value_head(x)

        # Return both actor and critic values
        return action_prob, state_value


class VecPolicy(Policy):
    """
    Implements both actor and critic in one model for vector input.
    """

    def __init__(self, input_dim, num_actions=5, fc_size=32):
        super(VecPolicy, self).__init__()

        # Define fully connected layers
        self.fc1 = nn.Linear(input_dim, fc_size)

        self.fc2 = nn.Linear(fc_size, fc_size)

        # Actor's layer (outputs probabilities over actions)
        self.action_head = nn.Linear(fc_size, num_actions)

        # Critic's layer (outputs state value)
        self.value_head = nn.Linear(fc_size, 1)

    def forward(self, x):
        """
        Forward pass of both actor and critic.
        """
        # Apply fully connected layer
        x = F.relu(self.fc2(F.relu(self.fc1(x))))

        # Actor: chooses action to take from state s_t
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # Critic: evaluates the value of the state
        state_value = self.value_head(x)

        # Return both actor and critic values
        return action_prob, state_value


# Environment class
class MultiAgentGridWorld(gym.Env):
    def __init__(self):
        super(MultiAgentGridWorld, self).__init__()
        self.grid_size = GRID_SIZE
        self.num_big_agents = NUM_BIG_AGENTS
        self.num_small_agents = NUM_SMALL_AGENTS
        self.timestep = 0
        self.max_timesteps = EPISODE_LENGTH

        # Create action and observation space
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Initialize agents' positions and state
        self.big_agents = [
            {'position': [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)], 'carried_agents': 2} for
            _ in range(NUM_BIG_AGENTS)]
        self.small_agents = [{'position': None, 'deployed': False} for _ in range(NUM_SMALL_AGENTS)]

        # Initialize the PoI grid with random values
        self.poi_grid = np.random.poisson(1, (GRID_SIZE, GRID_SIZE))  # PoI generation rate

        # Initialize AoI grid (starts at 0 for all PoIs)
        self.aoi_grid = np.zeros((GRID_SIZE, GRID_SIZE))

        # Initialize rewards
        self.big_agent_rewards = [0 for _ in range(NUM_BIG_AGENTS)]
        self.small_agent_rewards = [0 for _ in range(NUM_SMALL_AGENTS)]

    def reset(self):
        # Reset environment for a new episode
        self.timestep = 0
        self.big_agents = [
            {'position': [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)], 'carried_agents': 2} for
            _ in range(NUM_BIG_AGENTS)]
        self.small_agents = [{'position': None, 'deployed': False} for _ in range(NUM_SMALL_AGENTS)]
        self.big_agent_rewards = [0 for _ in range(NUM_BIG_AGENTS)]
        self.small_agent_rewards = [0 for _ in range(NUM_SMALL_AGENTS)]

        # Reset AoI grid
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
                action_rewards[action] = self.aoi_grid[x, y] / self.max_timesteps
            else:
                # When moving, reward is based on the AoI of the new cell
                action_rewards[action] = self.aoi_grid[new_x, new_y] / self.max_timesteps

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
                observations[f'small_{small_agent_id}'] = self._get_vec_observation(small_agent)
            else:
                # Small agents not deployed will have masked observation.
                observations[f'small_{small_agent_id}'] = np.zeros(NUM_ACTIONS)

        return observations

    def step(self, actions) -> [dict, dict, bool, dict]:
        # Initialize rewards for this timestep only
        rewards = {f'big_{i}': 0 for i in range(self.num_big_agents)}
        rewards.update({f'small_{i}': 0 for i in range(self.num_small_agents)})

        # Move big agents
        for big_agent_id, big_agent in enumerate(self.big_agents):
            action = actions[f'big_{big_agent_id}']
            self._move_agent(big_agent, action)

        # Deploy small agents at specific timesteps
        if self.timestep in TIMESTEP_DEPLOY:
            self._deploy_small_agents()

        # Update AoI for all grid cells
        self.aoi_grid += 1  # Increment AoI for all PoIs at each timestep

        # Move small agents if they are deployed
        for small_agent_id, small_agent in enumerate(self.small_agents):
            if small_agent['deployed']:
                action = actions[f'small_{small_agent_id}']
                self._move_agent(small_agent, action)
                # Get the position of the small agent
                x, y = small_agent['position']
                # Collect reward for the small agent based on the AoI of the grid cell it enters
                rewards[f'small_{small_agent_id}'] = self.aoi_grid[x, y] / self.max_timesteps
                # Reset AoI for the PoIs in the grid cell to 0
                self.aoi_grid[x, y] = 0
            else:
                rewards[f'small_{small_agent_id}'] = 0

        # Big agents get rewards based on the total rewards of their small agents for this timestep
        for big_agent_id, big_agent in enumerate(self.big_agents):
            small_agent_ids = range(big_agent_id * 2, big_agent_id * 2 + 2)
            # Big agent gets reward based on the sum of its small agents' rewards for this timestep
            rewards[f'big_{big_agent_id}'] = sum(rewards[f'small_{i}'] for i in small_agent_ids)

        # Advance timestep
        self.timestep += 1
        done = self.timestep >= self.max_timesteps

        # Return the observations, rewards for this timestep, done flag, and additional info
        return self._get_observation(), rewards, done, {}

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

    def _deploy_small_agents(self):
        # Deploy small agents from the big agents at timesteps 20 and 40
        for big_agent_id, big_agent in enumerate(self.big_agents):
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


env = MultiAgentGridWorld()

# Initialize actor-critic models for big and small agents
big_agent_policy = CNNPolicy(input_shape=(1, BIG_AGENT_RANGE, BIG_AGENT_RANGE), num_actions=NUM_ACTIONS)
small_agent_policy = VecPolicy(input_dim=5 + 2)

# Optimizers for big and small agents
big_agent_optimizer = optim.Adam(big_agent_policy.parameters(), lr=1e-4)
small_agent_optimizer = optim.Adam(small_agent_policy.parameters(), lr=1e-4)
device = 'cpu'


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


# Main loop for environment interaction
for episode in range(200):  # 200 episodes for demonstration
    obs = env.reset()

    # Initialize reward tracking for the episode
    total_big_agent_rewards = [0 for _ in range(NUM_BIG_AGENTS)]
    total_small_agent_rewards = [0 for _ in range(NUM_SMALL_AGENTS)]

    for t in range(EPISODE_LENGTH):
        if RANDOM_ACT:
            actions = random_act(env)
        else:
            actions = actor_critic_joint_act(obs, big_agent_policy, small_agent_policy)
        # Step the environment with the selected actions
        obs, rewards, done, _ = env.step(actions)

        # Accumulate rewards for training and for average reward calculation
        for i in range(NUM_BIG_AGENTS):
            big_agent_policy.rewards.append(rewards[f'big_{i}'])
            total_big_agent_rewards[i] += rewards[f'big_{i}']

        for i in range(NUM_SMALL_AGENTS):
            # if env.small_agents[i]['deployed']:
            small_agent_policy.rewards.append(rewards[f'small_{i}'])
            total_small_agent_rewards[i] += rewards[f'small_{i}']

        if done:
            break

        # Print out the observations for debugging purposes
        # print(f'Episode {episode}, Step {t}:')
        # for key, value in obs.items():
        #     print(f'{key}: {value}')

    # After the episode, update the parameters for big agents and small agents
    if not RANDOM_ACT:
        # Update the big agent policy using the saved actions and rewards
        big_agent_policy.finish_episode(big_agent_optimizer, gamma=GAMMA)
        # Update the small agent policy using the saved actions and rewards
        small_agent_policy.finish_episode(small_agent_optimizer, gamma=GAMMA)

    # Calculate average rewards
    avg_big_agent_reward = sum(total_big_agent_rewards) / NUM_BIG_AGENTS
    avg_small_agent_reward = sum(total_small_agent_rewards) / max(1,
                                                                  len([r for r in total_small_agent_rewards if r != 0]))

    print(f"Episode {episode} completed: Average Big Agent Reward = {avg_big_agent_reward:.2f}, "
          f"Average Small Agent Reward = {avg_small_agent_reward:.2f}")
