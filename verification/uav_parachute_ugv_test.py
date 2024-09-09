import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import gym
import torch.optim as optim
from torch.distributions import Categorical
from ppo_algo_verify import PPOVecPolicy, PPOCNNPolicy

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
        self.max_reward = -1000
        self.min_reward = 1000
        self.seed = 1
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Create action and observation space
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Initialize agents' positions and state
        self.big_agents = [
            {'position': [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)], 'carried_agents': 2} for
            _ in range(NUM_BIG_AGENTS)]
        self.small_agents = [{'position': None, 'deployed': False, 'last_deploy_status': False} for _ in
                             range(NUM_SMALL_AGENTS)]

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
        self.small_agents = [{'position': None, 'deployed': False, 'last_deploy_status': False} for _ in
                             range(NUM_SMALL_AGENTS)]
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
                observations[f'small_{small_agent_id}'] = self._get_vec_observation(small_agent)
            else:
                # Small agents not deployed will have masked observation.
                observations[f'small_{small_agent_id}'] = np.zeros(NUM_ACTIONS)

        return observations

    def step(self, actions) -> [dict, dict, bool, dict]:
        # Initialize rewards for this timestep only
        rewards = {f'big_{i}': 0 for i in range(self.num_big_agents)}
        rewards.update({f'small_{i}': 0 for i in range(self.num_small_agents)})
        # if self.timestep % 10 == 0:
        #     print(self.aoi_grid)
        # Move big agents
        for big_agent_id, big_agent in enumerate(self.big_agents):
            action = actions[f'big_{big_agent_id}']
            self._move_agent(big_agent, action)

        # Update AoI for all grid cells
        self.aoi_grid += 1  # Increment AoI for all PoIs at each timestep

        # Move small agents if they are deployed
        for small_agent_id, small_agent in enumerate(self.small_agents):
            # Update last_deploy_status
            small_agent['last_deploy_status'] = small_agent['deployed']

            if small_agent['deployed']:
                action = actions[f'small_{small_agent_id}']
                self._move_agent(small_agent, action)
                # Get the position of the small agent
                x, y = small_agent['position']
                # Collect reward for the small agent based on the AoI of the grid cell it enters
                rewards[f'small_{small_agent_id}'] = self.aoi_grid[x, y] * self.poi_grid[x, y] / self.max_timesteps
                self.max_reward = max(self.max_reward, rewards[f'small_{small_agent_id}'])
                self.min_reward = min(self.min_reward, rewards[f'small_{small_agent_id}'])
                # scale the reward to be between 0 and 1
                rewards[f'small_{small_agent_id}'] = (
                        (rewards[f'small_{small_agent_id}'] - self.min_reward) / (self.max_reward - self.min_reward))
                # Reset AoI for the PoIs in the grid cell to 0
                self.aoi_grid[x, y] = 0
            else:
                rewards[f'small_{small_agent_id}'] = 0

        # Big agents get rewards based on the total rewards of their small agents for this timestep
        for big_agent_id, big_agent in enumerate(self.big_agents):
            small_agent_ids = range(big_agent_id * 2, big_agent_id * 2 + 2)
            # Big agent gets reward based on the sum of its small agents' rewards for this timestep
            rewards[f'big_{big_agent_id}'] = sum(rewards[f'small_{i}'] for i in small_agent_ids)

        # Deploy small agents at specific timesteps
        if self.timestep in TIMESTEP_DEPLOY:
            self._deploy_small_agents()

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


def ppo_joint_act(
        obs: dict[np.ndarray],
        big_agent_policy: PPOVecPolicy,
        small_agent_policy: PPOVecPolicy,
) -> dict[int]:
    # Dictionary to hold actions for each agent
    actions = {}

    # Select actions for big agents using the actor-critic model
    for i in range(NUM_BIG_AGENTS):
        state = torch.from_numpy(obs[f'big_{i}']).float().unsqueeze(0).unsqueeze(0).to(device)
        action, log_prob, _, state_value = big_agent_policy.get_action_and_value(state)

        # Save the action, log probability, entropy, and value for training later
        big_agent_policy.saved_actions.append(action)
        big_agent_policy.log_probs.append(log_prob)
        big_agent_policy.values.append(state_value.squeeze())
        big_agent_policy.saved_obs.append(state)

        # Record the selected action
        actions[f'big_{i}'] = action.item()

    # Select actions for small agents (only for deployed agents)
    for i in range(NUM_SMALL_AGENTS):
        if env.small_agents[i]['deployed']:
            # If deployed, select actions using the small agent policy
            state = torch.from_numpy(obs[f'small_{i}']).float().unsqueeze(0).to(device)
            action, log_probs, _, state_value = small_agent_policy.get_action_and_value(state)

            # Save the action, log probability, entropy, and value for training later
            small_agent_policy.saved_actions.append(action)
            small_agent_policy.log_probs.append(log_prob)
            small_agent_policy.values.append(state_value.squeeze())
            small_agent_policy.saved_obs.append(state)

            # Record the selected action
            actions[f'small_{i}'] = action.item()
        else:
            # If not deployed, the small agent takes no action
            actions[f'small_{i}'] = STOP

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
    num_episodes: int = 1000

    gamma = 0.99
    max_grad_norm = 0.5
    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    gae_lambda = 0.95
    NUM_ENVS = 1
    seed = 1
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    env = MultiAgentGridWorld()

    # Initialize actor-critic models for big and small agents
    big_agent_policy = PPOCNNPolicy(input_shape=(1, BIG_AGENT_RANGE, BIG_AGENT_RANGE),
                                    num_actions=NUM_ACTIONS,
                                    fc_size=64).to(device)
    small_agent_policy = PPOVecPolicy(input_dim=5 + 2).to(device)

    # Optimizers for big and small agents
    big_agent_optimizer = optim.Adam(big_agent_policy.parameters(), lr=1e-4)
    small_agent_optimizer = optim.Adam(small_agent_policy.parameters(), lr=1e-4)

    next_obs = env.reset()
    next_done = torch.zeros(NUM_ENVS).to(device)
    progress = trange(num_episodes)
    # Main loop for environment interaction
    for episode in progress:  # 200 episodes for demonstration
        next_obs = env.reset()
        # Initialize reward tracking for the episode
        total_big_agent_rewards = [0 for _ in range(NUM_BIG_AGENTS)]
        total_small_agent_rewards = [0 for _ in range(NUM_SMALL_AGENTS)]

        for t in range(EPISODE_LENGTH):
            if RANDOM_ACT:
                actions = random_act(env)
            else:
                actions = ppo_joint_act(next_obs, big_agent_policy, small_agent_policy)
            # Step the environment with the selected actions
            next_obs, rewards, next_done, info = env.step(actions)


            # Accumulate rewards for training and for average reward calculation
            for i in range(NUM_BIG_AGENTS):
                big_agent_policy.rewards.append(rewards[f'big_{i}'])
                big_agent_policy.dones.append(next_done)
                total_big_agent_rewards[i] += rewards[f'big_{i}']

            for i in range(NUM_SMALL_AGENTS):
                if env.small_agents[i]['last_deploy_status']:
                    small_agent_policy.rewards.append(rewards[f'small_{i}'])
                    small_agent_policy.dones.append(next_done)
                    total_small_agent_rewards[i] += rewards[f'small_{i}']

            if next_done:
                break

        # After the episode, update the parameters for big agents and small agents
        if not RANDOM_ACT:
            # select small agent obs and big obs, concat them into together, respectively.
            big_agent_obs = torch.cat([torch.from_numpy(next_obs[f'big_{i}']).float().unsqueeze(0)
                                       for i in range(NUM_BIG_AGENTS)]).to(device)
            small_agent_obs = torch.cat([torch.from_numpy(next_obs[f'small_{i}']).float().unsqueeze(0)
                                         for i in range(NUM_SMALL_AGENTS)]).to(device)
            # Update the big agent policy using the saved actions and rewards
            big_agent_policy.finish_episode(big_agent_optimizer, max_grad_norm=max_grad_norm,
                                            clip_coef=clip_coef, vf_coef=vf_coef, ent_coef=ent_coef,
                                            gae_lambda=gae_lambda, num_minibatches=4, num_envs=NUM_BIG_AGENTS,
                                            next_state=big_agent_obs, next_dones=next_done, device=device,
                                            num_steps=EPISODE_LENGTH)
            # Update the small agent policy using the saved actions and rewards
            small_agent_policy.finish_episode(small_agent_optimizer, max_grad_norm=max_grad_norm,
                                              clip_coef=clip_coef, vf_coef=vf_coef, ent_coef=ent_coef,
                                              gae_lambda=gae_lambda, num_minibatches=4, num_envs=NUM_SMALL_AGENTS,
                                              next_state=small_agent_obs, next_dones=next_done, device=device,
                                              num_steps=len(small_agent_policy.rewards) // NUM_SMALL_AGENTS)

        # Calculate average rewards
        avg_big_agent_reward = sum(total_big_agent_rewards) / NUM_BIG_AGENTS
        avg_small_agent_reward = sum(total_small_agent_rewards) / max(1, len([r for r in total_small_agent_rewards if
                                                                              r != 0]))

        progress.set_postfix(big_agent=avg_big_agent_reward, small_agent=avg_small_agent_reward)

        # print(f"Episode {episode} completed: Average Big Agent Reward = {avg_big_agent_reward:.2f}, "
        #       f"Average Small Agent Reward = {avg_small_agent_reward:.2f}")
