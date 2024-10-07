import argparse
import logging
import os
from typing import List, Any, Union

import pandas as pd
import random
from collections import namedtuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import MultiDiscrete, Discrete, Box
from matplotlib import cm
from matplotlib.colors import to_hex
from torch.distributions import Categorical
from tqdm import trange

from ppo_algo_verify import PPOVecPolicy, PPOCNNPolicy, Policy, MultiPPORollout

# set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

ID = 'id'
EMERGENCY_NUMBER = 15
POSITION = 'position'
CARRIED_AGENTS = 'carried_agents'
DEPLOYED = 'deployed'
LEARNING_RATE = 1e-4

# Constants
PROJECT_NAME = 'uav-parachute-ugv'
LOG_ACTION = False
LOG_TABLE = False
GAMMA = 0
RANDOM_ACT = False
FIX_SMALL_AGENT = True
GRID_SIZE = 20
HIDDEN_SIZE = 64
EPISODE_LENGTH = 60
EVAL_INTERVAL = 100
PLOT_NAME = "trajectory"
BIG_AGENT_METRIC = "big_reward"
BIG_AGENT_TRAIN = "big_train"
BIG_AGENT_ACTION = "big_action"
SMALL_AGENT_TRAIN = "small_train"
SMALL_AGENT_METRIC = "small_reward"
SMALL_AGENT_ACTION = "small_action"
BIG_AGENT_MODEL = "big_model"
TARGETS = "big_agent_targets"
TARGET = "small_agent_target"
SMALL_AGENT_MODEL = "small_model"
BIG_AGENT_DEPLOY_METRIC = "big_reward_deploy"
SURVEILLANCE_AOI = "surveillance_aoi"
EMERGENCY_AOI = "emergency_aoi"
BIG_AGENT_RANGE = 8
SMALL_AGENT_RANGE = 4
NUM_BIG_AGENTS = 2
NUM_SMALL_AGENTS = 4
NUM_MOVEMENTS = 5  # up, down, left, right, stop
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


class MultiAgentGridWorld(gym.Env):
    emergency_aoi: np.ndarray
    aoi_grid_by_time: np.ndarray
    emergency_state_time: np.ndarray
    emergency_assign_status: np.ndarray
    timestep: int
    small_agent_targets: list[list[Any]]
    small_agent_trajectories: list[list[Any]]
    big_agent_trajectories: list[list[Any]]
    emergency_poi_grid: np.ndarray

    def __init__(self, num_big_agents, num_small_agents, group_factor=1, self_factor=0.1,
                 log_action=LOG_ACTION, num_emergencies=EMERGENCY_NUMBER):
        super(MultiAgentGridWorld, self).__init__()
        self.big_agent_shape = self.small_agent_shape = None
        self.grid_size = GRID_SIZE
        self.group_factor = group_factor
        self.self_factor = self_factor
        self.num_big_agents = num_big_agents
        self.num_emergencies = num_emergencies
        self.log_action = log_action
        self.num_small_agents = num_small_agents
        self.carried_small_agents = self.num_small_agents // self.num_big_agents

        self.max_timesteps = EPISODE_LENGTH
        self.figure = None

        self.seed = 1
        self.big_vec_mode = True
        self.small_vec_mode = True
        self.big_agent_buffer_size = 4
        self.coordinate_dim = 2

        if self.big_vec_mode:
            # self-position, position of carried agents, and emergency (x,y) buffer.
            self.big_agent_shape = (self.coordinate_dim + self.coordinate_dim * self.carried_small_agents
                                    + self.big_agent_buffer_size * self.coordinate_dim)
        else:
            self.big_agent_shape = (2, BIG_AGENT_RANGE, BIG_AGENT_RANGE)
        if self.small_vec_mode:
            # self-position and emergency (x,y) target.
            self.small_agent_shape = self.coordinate_dim + self.coordinate_dim
        else:
            self.small_agent_shape = (2, SMALL_AGENT_RANGE, SMALL_AGENT_RANGE)
        self.observation_space = {
            f'big_{i}': Box(low=np.zeros(self.big_agent_shape, dtype=np.float32),
                            high=np.ones(self.big_agent_shape, dtype=np.float32),
                            dtype=np.float32)
            for i in range(self.num_big_agents)
        }
        self.observation_space.update({
            f'small_{i}': Box(low=np.zeros(self.small_agent_shape, dtype=np.float32),
                              high=np.ones(self.small_agent_shape, dtype=np.float32),
                              dtype=np.float32)
            for i in range(self.num_small_agents)
        })
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Update action space for big agents
        self.big_action_space = MultiDiscrete([NUM_MOVEMENTS, 2])  # [movement (5), deploy (2), assign (2)]
        # note assign is temporarily disabled.
        self.small_action_space = Discrete(NUM_MOVEMENTS)
        # Combine action spaces for compatibility
        self.action_space = {
            'big': self.big_action_space,
            'small': self.small_action_space
        }
        self.reset_states()
        self.emergency_poi_grid = np.full((self.grid_size, self.grid_size), fill_value=-1)
        # Initialize the PoI grid with certain clustered PoI values
        self.poi_grid = generate_clusters(self.grid_size, NUM_CLUSTERS, CLUSTER_RADIUS, MAX_VALUE)
        # randomly generate 15 distinct poi with numpy indexing
        emergency_x = np.random.choice(self.grid_size, size=self.num_emergencies, replace=False)
        emergency_y = np.random.choice(self.grid_size, size=self.num_emergencies, replace=False)
        self.emergency_positions = np.stack((emergency_x, emergency_y), axis=1)
        self.emer_ids = np.arange(self.num_emergencies)
        self.emergency_mapping = {
            (x, y): i for i, (x, y) in enumerate(zip(emergency_x, emergency_y))
        }
        self.emergency_start_time = np.random.randint(1, self.max_timesteps, size=self.num_emergencies)
        self.emergency_poi_grid[emergency_x, emergency_y] = self.emergency_start_time
        self.num_poi = np.sum(self.poi_grid) + self.num_emergencies


    def step(self, actions) -> [dict, dict, bool, dict]:
        info = {}
        unassigned_mask = ~self.emergency_assign_status
        emergency_unhandle_mask = (self.emergency_start_time <= self.timestep) & (self.emergency_end_time == -1)
        rewards = {f'big_{i}': 0 for i in range(self.num_big_agents)}
        rewards.update({f'small_{i}': 0 for i in range(self.num_small_agents)})
        # Process small agent actions
        for small_agent_id, small_agent in enumerate(self.small_agents):
            small_agent['last_deploy_status'] = small_agent[DEPLOYED]
            if small_agent[DEPLOYED]:
                action = actions[f'small_{small_agent_id}']
                self._move_agent(small_agent, action)
                x, y = small_agent[POSITION]
                # Record small agent position
                self.small_agent_trajectories[small_agent_id].append(small_agent[POSITION].copy())
                self.small_agent_targets[small_agent_id].append(small_agent[TARGET] if
                                                                small_agent[TARGET] is not None
                                                                else np.array([-1, -1]))
                if 0 <= self.emergency_poi_grid[x, y] < self.timestep:
                    emergency_id = self.emergency_mapping[(x, y)]
                    rewards[f'small_{small_agent_id}'] = self.emergency_aoi[emergency_id] / self.max_timesteps
                    self.emergency_end_time[emergency_id] = self.timestep
                    if np.all(small_agent[POSITION] == small_agent[TARGET]):
                        small_agent[TARGET] = None
                        rewards[f'small_{small_agent_id}'] = 1

                if small_agent[TARGET] is not None:
                    # target_aoi = self.emergency_aoi[self.emergency_mapping[tuple(small_agent[TARGET])]]
                    rewards[f'small_{small_agent_id}'] -= np.linalg.norm(
                        (small_agent[TARGET] - small_agent[POSITION]) / self.grid_size
                    )

                rewards[f'small_{small_agent_id}'] += self.group_factor * self.aoi_grid[x, y] * self.poi_grid[
                    x, y] / self.max_timesteps
                #
                # self.max_reward = max(self.max_reward, rewards[f'small_{small_agent_id}'])
                # self.min_reward = min(self.min_reward, rewards[f'small_{small_agent_id}'])
                # rewards[f'small_{small_agent_id}'] = (
                #         (rewards[f'small_{small_agent_id}'] - self.min_reward) / (self.max_reward - self.min_reward))
                self.aoi_grid[x, y] = 0
            else:
                rewards[f'small_{small_agent_id}'] = 0

        # Process big agent actions
        for big_agent_id, big_agent in enumerate(self.big_agents):
            my_action = actions[f'big_{big_agent_id}']
            if isinstance(my_action, np.ndarray):
                movement_action, deploy_action = my_action[0]
            else:
                raise NotImplementedError("Action must be a numpy array")
            self._move_agent(big_agent, movement_action)
            self.big_agent_trajectories[big_agent_id].append(big_agent[POSITION].copy())
            if self.log_action:
                self.big_agent_actions[big_agent_id].append(movement_action)
            deploy_x, deploy_y = big_agent[POSITION]
            # reward big agent for its own movement
            rewards[f'big_{big_agent_id}'] = self.self_factor * self.aoi_grid[deploy_x, deploy_y] * self.poi_grid[
                deploy_x, deploy_y] / self.max_timesteps
            # calculate the average distance between this big agent and other big agent
            total_dist = 0
            big_agent_count = 0
            for other_agent in self.big_agents:
                if other_agent != big_agent:
                    dist = np.linalg.norm(other_agent[POSITION] - big_agent[POSITION])
                    total_dist += dist
                    big_agent_count += 1
            if big_agent_count > 0:
                avg_dist = total_dist / big_agent_count
                info[f'big_{big_agent_id}_avg_dist'] = avg_dist
            # normalize distance and add as reward
            # if avg_dist > 0:
            #     rewards[f'big_{big_agent_id}'] += 0.5 * (avg_dist / self.grid_size)

            # Calculate distance between all emergencies and the deployment point (deploy_x, deploy_y)
            deploy_position = big_agent[POSITION]  # Shape: (2,)
            distances = np.abs(self.emergency_positions - deploy_position)  # Shape: (num_emergencies, 2)
            within_range_mask = (distances[:, 0] <= BIG_AGENT_RANGE) & (distances[:, 1] <= BIG_AGENT_RANGE)
            # Combine the two masks to get the valid emergencies
            valid_emergencies_mask = unassigned_mask & emergency_unhandle_mask & within_range_mask
            # Get valid emergency IDs and their AoI values
            valid_emer_ids = self.emer_ids[valid_emergencies_mask]
            valid_emergency_positions = self.emergency_positions[valid_emergencies_mask]
            valid_aoi = self.emergency_aoi[valid_emer_ids]
            # Apply discovery reward for valid emergencies
            rewards[f'big_{big_agent_id}'] += np.sum(valid_aoi) / self.max_timesteps
            info[f'big_{big_agent_id}_targets'] = len(big_agent[TARGETS])
            # Update targets for the big agent and mark emergencies as assigned
            big_agent[TARGETS].extend(valid_emergency_positions)
            self.emergency_assign_status[valid_emer_ids] = True

            # Handle deployment action
            if deploy_action == 1 and len(big_agent[CARRIED_AGENTS]) > 0:
                # big agent is rewarded with AoI sum of PoIs around the deployment area
                self.small_agent_deploy_position.append(deploy_position.copy())
                deployed_small_agent = self._deploy_small_agent(big_agent_id)
                if deployed_small_agent[TARGET] is None:
                    deploy_reward = 0
                else:
                    deploy_reward = - np.linalg.norm(
                        (deploy_position - deployed_small_agent[TARGET]) / self.grid_size)
                info[f'big_{big_agent_id}_deploy_reward'] = deploy_reward
                rewards[f'big_{big_agent_id}'] += deploy_reward
                info[f'big_{big_agent_id}_deploy_time'] = self.timestep

            elif deploy_action == 1 and len(big_agent[CARRIED_AGENTS]) == 0:
                # assign emergency in buffer to the closest available UGV.
                for small_agent in self.small_agents:
                    if small_agent[DEPLOYED]:
                        small_x, small_y = small_agent[POSITION]
                        # confirm the big agent can see this small agent
                        if np.abs(small_x - deploy_position[0]) < BIG_AGENT_RANGE and \
                                np.abs(small_y - deploy_position[1]) < BIG_AGENT_RANGE:
                            if small_agent[TARGET] is None and len(big_agent[TARGETS]) > 0:
                                small_agent[TARGET] = big_agent[TARGETS].pop()
                                logger.debug(f"Assignment Operation Successful")
                                assign_reward = - np.linalg.norm(
                                    (small_agent[POSITION] - small_agent[TARGET]) / self.grid_size)
                                info[f'big_{big_agent_id}_assign_reward'] = assign_reward
                                rewards[f'big_{big_agent_id}'] += assign_reward
                                break
            else:
                pass

        # Update AoI for all grid cells
        self.aoi_grid += 1
        # increment emergency only when it starts and it is not handled.
        self.emergency_aoi[emergency_unhandle_mask] += 1

        self.aoi_grid_by_time[self.timestep] = self.aoi_grid * self.poi_grid
        self.timestep += 1
        done = self.timestep >= self.max_timesteps

        if done:
            info[SURVEILLANCE_AOI] = np.mean(self.aoi_grid_by_time)
            info[EMERGENCY_AOI] = np.mean(self.emergency_aoi)

        # Return the observations, rewards for this timestep, done flag, and additional info
        return self._get_observation(), rewards, done, info

    def compute_density_reward(self, big_agent):
        x, y = big_agent[POSITION]
        # Define a neighborhood range to compute local PoI density
        density_radius = 3  # Adjust this radius as needed
        local_poi_density = np.sum(self.poi_grid[max(0, x - density_radius):min(self.grid_size, x + density_radius + 1),
                                   max(0, y - density_radius):min(self.grid_size, y + density_radius + 1)])
        return local_poi_density / self.num_poi  # Normalize by the area

    def reset(self):
        if self.figure is not None:
            plt.close(self.figure)
        self.reset_states()
        return self._get_observation()

    def reset_states(self):
        self.timestep = 0
        self.emergency_aoi = np.zeros(self.num_emergencies)
        self.emergency_assign_status = np.zeros(self.num_emergencies, dtype=np.bool8)
        self.emergency_end_time = np.full(self.num_emergencies, -1)
        self.big_agents = [
            {
                ID: big_id,
                POSITION: np.array([self.grid_size // 2, self.grid_size // 2]),
                CARRIED_AGENTS: [small_id for small_id in range(self.carried_small_agents * big_id,
                                                                self.carried_small_agents * (big_id + 1))],
                TARGETS: [],
            } for
            big_id in range(self.num_big_agents)
        ]
        self.small_agents = [{
            ID: id,
            POSITION: None, DEPLOYED: False, TARGET: None,
            'last_deploy_status': False} for id in
            range(self.num_small_agents)]
        self.aoi_grid = np.zeros((self.grid_size, self.grid_size))
        self.small_agent_trajectories = [[] for _ in range(self.num_small_agents)]
        self.small_agent_targets = [[] for _ in range(self.num_small_agents)]
        self.small_agent_deploy_position = []
        self.big_agent_trajectories = [[] for _ in range(self.num_big_agents)]
        self.big_agent_actions = [[] for _ in range(self.num_big_agents)]
        self.small_agent_actions = [[] for _ in range(self.num_small_agents)]
        self.aoi_grid_by_time = np.zeros((self.max_timesteps, self.grid_size, self.grid_size))

    def _get_vec_observation(self, agent):
        if CARRIED_AGENTS in agent:
            # big agent, self-position, position of carried agents, and emergency (x,y) buffer.
            self_position = agent[POSITION] / self.grid_size
            carried_agents_pos = np.zeros((self.carried_small_agents * self.coordinate_dim))
            big_agent_id = agent[ID]
            my_small_agent = self.small_agents[big_agent_id * self.coordinate_dim:
                                               (big_agent_id + 1) * self.coordinate_dim]
            for i, small_agent in enumerate(my_small_agent):
                if small_agent[DEPLOYED]:
                    delta_pos = (small_agent[POSITION] - agent[POSITION]) / self.grid_size
                    carried_agents_pos[
                    i * self.coordinate_dim:(i + 1) * self.coordinate_dim
                    ] = delta_pos
                else:
                    carried_agents_pos[i * self.coordinate_dim:(i + 1) * self.coordinate_dim] = self_position
            target_obs = np.zeros((self.big_agent_buffer_size * self.coordinate_dim))
            for i, target in enumerate(agent[TARGETS][:self.big_agent_buffer_size]):
                target_obs[i * self.coordinate_dim:(i + 1) * self.coordinate_dim] = target / self.grid_size
            return np.concatenate([self_position, carried_agents_pos, target_obs])
        else:
            self_position = agent[POSITION] / self.grid_size
            if agent[TARGET] is not None:
                target_position = agent[TARGET] / self.grid_size
            else:
                target_position = np.zeros(self.coordinate_dim)
            return np.concatenate([self_position, target_position])

    def _get_vec_action_reward_observation(self, agent):
        """
        Generates a vector observation for a UGV agent based on its position
        and the AoI grid.
        """
        # Get agent position
        x, y = agent[POSITION]

        # Initialize the reward vector for the 5 possible actions
        action_rewards = np.zeros(NUM_MOVEMENTS)

        # Define action directions
        directions = {
            UP: (-1, 0),
            DOWN: (1, 0),
            LEFT: (0, -1),
            RIGHT: (0, 1),
            STOP: (0, 0)
        }

        for action in range(NUM_MOVEMENTS):
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

        return np.concatenate([agent[POSITION] / self.grid_size, action_rewards])

    def _get_grid_observation(self, agent, agents, observation_range):
        """
        Generates a grid observation for a UAV agent based on its position
        and the PoI grid.
        """
        x, y = agent[POSITION]

        # Extracting the grid portion
        half_range = observation_range // 2
        obs = self.poi_grid[max(0, x - half_range):min(self.grid_size, x + half_range),
              max(0, y - half_range):min(self.grid_size, y + half_range)]

        # Padding to make the observation square of size observation_range * observation_range
        padded_obs = np.pad(obs,
                            ((max(0, half_range - x),
                              max(0, x + half_range - self.grid_size)),
                             (max(0, half_range - y),
                              max(0, y + half_range - self.grid_size))),
                            mode='constant', constant_values=0)

        # generate a grid that represents agents in the map, including its sensing range.
        agent_grid = np.zeros((observation_range, observation_range))
        # map agent position to grid
        for single_agent in agents:
            if DEPLOYED in single_agent and single_agent[DEPLOYED]:
                sx, sy = single_agent[POSITION]
                if abs(sx - x) <= half_range and abs(sy - y) <= half_range:
                    agent_grid[sx - x + half_range, sy - y + half_range] = 1

        return np.stack([agent_grid, padded_obs], axis=0)

    def _get_observation(self):
        """
        Combines observations for all agents. Small agents get vector observations,
        and big agents get grid observations.
        """
        observations = {}

        for big_agent_id, big_agent in enumerate(self.big_agents):
            if self.big_vec_mode:
                observations[f'big_{big_agent_id}'] = self._get_vec_observation(big_agent)
            else:
                observations[f'big_{big_agent_id}'] = self._get_grid_observation(big_agent, self.big_agents,
                                                                                 BIG_AGENT_RANGE)

        for small_agent_id, small_agent in enumerate(self.small_agents):
            if small_agent[DEPLOYED]:
                if self.small_vec_mode:
                    observations[f'small_{small_agent_id}'] = self._get_vec_observation(small_agent)
                else:
                    observations[f'small_{small_agent_id}'] = self._get_grid_observation(small_agent, self.small_agents,
                                                                                         SMALL_AGENT_RANGE)
            else:
                # Small agents not deployed will have masked observation.
                observations[f'small_{small_agent_id}'] = np.zeros(self.small_agent_shape)

        return observations

    def render(self, mode='human', return_plot=True):
        """
        Renders the entire grid environment at the end of the episode, showing agent trajectories and PoI grid.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot the PoI grid as the background
        ax.imshow(self.emergency_poi_grid, cmap='Greens', origin='upper', alpha=0.5)

        # Plot big agent trajectories
        # Generate a set of blue colors according to len(self.big_agents)
        colors = cm.Blues(np.linspace(0.5, 1, len(self.big_agents)))
        for i, (color, trajectory) in enumerate(zip(colors, self.big_agent_trajectories)):
            if len(trajectory) > 1:
                combined_traj = np.array(trajectory).reshape(-1, 2)
                traj_x, traj_y = combined_traj[:, 0], combined_traj[:, 1]
                ax.plot(traj_x, traj_y, color=to_hex(color), marker='*', markersize=10, label=f'Big Agent {i}')

        # Plot small agent trajectories
        # Generate a set of red colors according to len(self.small_agents)
        colors = cm.Reds(np.linspace(0.5, 1, len(self.small_agents)))
        for i, (color, trajectory) in enumerate(zip(colors, self.small_agent_trajectories)):
            if len(trajectory) > 1:
                combined_traj = np.array(trajectory).reshape(-1, 2)
                traj_x, traj_y = combined_traj[:, 0], combined_traj[:, 1]
                ax.plot(traj_x, traj_y, color=to_hex(color), marker='o', markersize=5, label=f'Small Agent {i}')

        ax.set_title(f'Trajectories of Agents')
        ax.legend()
        if return_plot:
            self.figure = fig
            return fig

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

        if action == DOWN:
            agent[POSITION][1] = max(0, agent[POSITION][1] - step_size)
        elif action == UP:
            agent[POSITION][1] = min(self.grid_size - 1, agent[POSITION][1] + step_size)
        elif action == LEFT:
            agent[POSITION][0] = max(0, agent[POSITION][0] - step_size)
        elif action == RIGHT:
            agent[POSITION][0] = min(self.grid_size - 1, agent[POSITION][0] + step_size)
        elif action == STOP:
            pass  # Do nothing

    def _move_agent_unified(self, agent, action):
        # Move agent based on the action (up, down, left, right, stop)
        if action == UP:
            agent[POSITION][0] = max(0, agent[POSITION][0] - 1)
        elif action == DOWN:
            agent[POSITION][0] = min(self.grid_size - 1, agent[POSITION][0] + 1)
        elif action == LEFT:
            agent[POSITION][1] = max(0, agent[POSITION][1] - 1)
        elif action == RIGHT:
            agent[POSITION][1] = min(self.grid_size - 1, agent[POSITION][1] + 1)
        elif action == STOP:
            pass  # Do nothing

    def _deploy_small_agent(self, big_agent_id):
        big_agent = self.big_agents[big_agent_id]
        if len(big_agent[CARRIED_AGENTS]) > 0:
            small_agent_id = big_agent[CARRIED_AGENTS].pop()
            small_agent = self.small_agents[small_agent_id]
            small_agent[POSITION] = big_agent[POSITION].copy()
            small_agent[DEPLOYED] = True
            if len(big_agent[TARGETS]) > 0:
                small_agent[TARGET] = big_agent[TARGETS].pop()
            return small_agent


def test_MultiAgentGridWorld():
    global env
    # Demo of environment interaction
    env = MultiAgentGridWorld(num_big_agents=NUM_BIG_AGENTS, num_small_agents=NUM_SMALL_AGENTS)
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
        if env.small_agents[i][DEPLOYED]:
            actions[f'small_{i}'] = env.action_space.sample()
        else:
            actions[f'small_{i}'] = STOP
    return actions


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
    # setup wandb
    import wandb

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

    # name = current date + customed name
    best_mean_aoi = 200
    additional_tags = []
    for item in ['anneal_lr']:
        if config[item]:
            additional_tags.append(item)
    # add suffix for name at here.
    for item in []:
        if item in config:
            additional_tags.append(item + '_' + str(config[item]))
    # add date time to name
    expr_name = datetime.datetime.today().strftime("%m%d-%H%M") + '-' + args.name

    # add additional_tags to name
    if len(additional_tags) > 0:
        expr_name += ('-' + "-".join(additional_tags))

    checkpoint_path = os.path.join('/workspace', 'saved_data', 'checkpoints', expr_name)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    env = MultiAgentGridWorld(num_big_agents=NUM_BIG_AGENTS, num_small_agents=NUM_SMALL_AGENTS,
                              self_factor=args.self_factor, group_factor=args.group_factor)

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
        wandb.init(project=PROJECT_NAME, name=expr_name, group='emergency_mvp',
                   tags=['ppo', 'big_cnn', 'small_cnn'],
                   config=config, dir=os.path.join('/workspace', 'saved_data'))
        wandb.define_metric(BIG_AGENT_METRIC, summary="max")
        wandb.define_metric(SMALL_AGENT_METRIC, summary="max")
        wandb.define_metric(SURVEILLANCE_AOI, summary='min')
        wandb.define_metric(EMERGENCY_AOI, summary='min')
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
                    # Usage for big agents
                    select_actions('big', big_agent_policy, NUM_BIG_AGENTS,
                                   next_obs, actions, env, device, big_agent_rollout)
                    # Usage for small agents
                    select_actions('small', small_agent_policy, NUM_SMALL_AGENTS,
                                   next_obs, actions, env, device, small_agent_rollout)
            # Step the environment with the selected actions
            next_obs, rewards, next_done, info = env.step(actions)

            # Accumulate rewards for training and for average reward calculation
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
            small_agent_rollout.concatenate_rollouts(small_agent_policy)
            big_agent_rollout.concatenate_rollouts(big_agent_policy)
            # select small agent obs and big obs, concat them into together, respectively.
            big_agent_obs = torch.cat([torch.from_numpy(next_obs[f'big_{i}']).float().unsqueeze(0)
                                       for i in range(NUM_BIG_AGENTS)]).to(device)
            small_agent_obs = torch.cat([torch.from_numpy(next_obs[f'small_{i}']).float().unsqueeze(0)
                                         for i in range(NUM_SMALL_AGENTS)]).to(device)
            # Update the big agent policy using the saved actions and rewards
            big_agent_statistic.update(big_agent_policy.finish_episode(big_agent_optimizer, max_grad_norm=max_grad_norm,
                                                                       clip_coef=clip_coef, vf_coef=vf_coef,
                                                                       ent_coef=ent_coef,
                                                                       gae_lambda=gae_lambda, num_minibatches=4,
                                                                       num_envs=NUM_BIG_AGENTS,
                                                                       next_state=big_agent_obs, next_dones=next_done,
                                                                       device=device,
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
        # add prefix for big agent dict and small agent dict
        for k, v in small_agent_statistic.items():
            log_dict[f'{SMALL_AGENT_TRAIN}/{k}'] = v
        for k, v in big_agent_statistic.items():
            log_dict[f'{BIG_AGENT_TRAIN}/{k}'] = v
        if LOG_ACTION:
            for i in range(NUM_BIG_AGENTS):
                log_dict[f'{BIG_AGENT_ACTION}/Agent{i}'] = wandb.Histogram(env.big_agent_actions[i],
                                                                           num_bins=NUM_MOVEMENTS)
        log_dict.update(
            {
                BIG_AGENT_METRIC: avg_big_agent_reward,
                SMALL_AGENT_METRIC: avg_small_agent_reward,
                **info,
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
    wandb.finish()
