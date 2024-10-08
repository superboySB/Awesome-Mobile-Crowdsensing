import logging
import random
from typing import Any, List

import gym
import torch
import numpy as np
from gym.spaces import Box, MultiDiscrete, Discrete
from matplotlib import pyplot as plt, cm
from typing import Tuple, Dict
from matplotlib.colors import to_hex

UP, DOWN, LEFT, RIGHT, STOP = 0, 1, 2, 3, 4
logger = logging.getLogger(__name__)
ID = 'id'
POSITION = 'position'
CARRIED_AGENTS = 'carried_agents'
DEPLOYED = 'deployed'
GRID_SIZE = 20
EPISODE_LENGTH = 60
TARGETS = "big_agent_targets"
TARGET = "small_agent_target"
SURVEILLANCE_AOI = "surveillance_aoi"
EMERGENCY_AOI = "emergency_aoi"
BIG_AGENT_RANGE = 8
SMALL_AGENT_RANGE = 4
NUM_BIG_AGENTS = 2
NUM_SMALL_AGENTS = 4
NUM_MOVEMENTS = 5  # up, down, left, right, stop
NUM_CLUSTERS = 5
CLUSTER_RADIUS = 3
MAX_VALUE = 10


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


class MultiAgentGridWorld(gym.Env):
    device: torch.device
    big_agent_shape: tuple
    small_agent_shape: tuple
    grid_size: int
    group_factor: float
    self_factor: float
    num_big_agents: int
    num_emergencies: int
    log_action: bool
    num_small_agents: int
    carried_small_agents: int
    max_timesteps: int
    figure: Any
    seed: int
    big_vec_mode: bool
    small_vec_mode: bool
    big_agent_buffer_size: int
    coordinate_dim: int
    observation_space: dict
    big_action_space: MultiDiscrete
    small_action_space: Discrete
    action_space: dict
    emergency_poi_grid: np.ndarray  # shape: (grid_size, grid_size)
    poi_grid: np.ndarray  # shape: (grid_size, grid_size)
    emergency_positions: np.ndarray  # shape: (num_emergencies, 2)
    emer_ids: np.ndarray  # shape: (num_emergencies,)
    emergency_mapping: dict
    emergency_start_time: np.ndarray  # shape: (num_emergencies,)
    num_poi: int
    big_agents: List[dict]
    small_agents: List[dict]
    emergency_aoi: np.ndarray  # shape: (grid_size, grid_size)
    aoi_grid_by_time: np.ndarray  # shape: (max_timesteps, grid_size, grid_size)
    emergency_state_time: np.ndarray  # shape: (num_emergencies,)
    emergency_assign_status: np.ndarray  # shape: (num_emergencies,)
    timestep: int
    small_agent_targets: List[List[Any]]
    small_agent_trajectories: List[List[Any]]
    big_agent_trajectories: List[List[Any]]

    def __init__(self, num_big_agents, num_small_agents, group_factor, self_factor,
                 log_action, num_emergencies, max_timesteps):
        super(MultiAgentGridWorld, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.big_agent_shape = self.small_agent_shape = None
        self.grid_size = GRID_SIZE
        self.group_factor = group_factor
        self.self_factor = self_factor
        self.num_big_agents = num_big_agents
        self.num_emergencies = num_emergencies
        self.log_action = log_action
        self.num_small_agents = num_small_agents
        self.carried_small_agents = self.num_small_agents // self.num_big_agents
        self.max_timesteps = max_timesteps

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

    def compute_reward(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # Initialize reward dictionary and lists for storing reward components
        reward_dict = {}
        surveillance_reward = []
        emergency_reward = []
        assignment_proximity_reward = []

        # Process rewards for big agents
        for big_agent_id, big_agent in enumerate(self.big_agents):
            big_agent_pos = torch.tensor(big_agent[POSITION], device=self.device, dtype=torch.float32)

            # Compute proximity reward: closest distance to deployed small agents
            big_to_small_distances = []
            for small_agent in self.small_agents:
                if small_agent[DEPLOYED]:
                    small_agent_pos = torch.tensor(small_agent[POSITION], device=self.device, dtype=torch.float32)
                    distance = torch.norm(big_agent_pos - small_agent_pos, p=2)
                    big_to_small_distances.append(distance)

            if big_to_small_distances:
                min_big_to_small_distance = torch.stack(big_to_small_distances).min()
                proximity_reward = 1.0 / (min_big_to_small_distance + 1e-6)
            else:
                proximity_reward = torch.tensor(0.0, device=self.device, dtype=torch.float32)

            # Apply temperature to proximity reward
            temperature_assignment = 0.05
            transformed_proximity_reward = torch.exp(-temperature_assignment * proximity_reward)

            # Store the reward for the big agent (convert to NumPy)
            reward_dict[f"big_{big_agent_id}"] = transformed_proximity_reward.cpu().item()
            assignment_proximity_reward.append(transformed_proximity_reward)

        # Process rewards for small agents
        for small_agent_id, small_agent in enumerate(self.small_agents):
            if small_agent[DEPLOYED]:  # Only compute reward for deployed small agents
                small_agent_pos = torch.tensor(small_agent[POSITION], device=self.device, dtype=torch.float32)

                # Surveillance task reward: proximity to surveillance points of interest (poi)
                poi_grid = torch.tensor(self.poi_grid, device=self.device, dtype=torch.float32)
                surveillance_aoi_pos = torch.nonzero(poi_grid)  # Assuming non-zero grid points are AoI
                surveillance_distances = torch.norm(small_agent_pos - surveillance_aoi_pos.float(), dim=-1, p=2)
                surveillance_r = 1.0 / (torch.min(surveillance_distances) + 1e-6)

                # Emergency task reward: proximity to emergency points of interest (emergency_poi)
                emergency_poi_grid = torch.tensor(self.emergency_poi_grid, device=self.device, dtype=torch.float32)
                emergency_aoi_pos = torch.nonzero(emergency_poi_grid)  # Non-zero grid points are emergency AoI
                emergency_distances = torch.norm(small_agent_pos - emergency_aoi_pos.float(), dim=-1, p=2)
                emergency_r = 1.0 / (torch.min(emergency_distances) + 1e-6)

                # Apply temperature to surveillance and emergency rewards
                temperature_surveillance = 0.1
                temperature_emergency = 0.2
                transformed_surveillance_r = torch.exp(-temperature_surveillance * surveillance_r)
                transformed_emergency_r = torch.exp(-temperature_emergency * emergency_r)

                # Store the reward for the small agent (convert to NumPy)
                reward_dict[f"small_{small_agent_id}"] = (
                        transformed_surveillance_r + transformed_emergency_r).cpu().item()
                surveillance_reward.append(transformed_surveillance_r)
                emergency_reward.append(transformed_emergency_r)
            else:
                # Not deployed, no reward (convert to NumPy)
                reward_dict[f"small_{small_agent_id}"] = torch.tensor(0.0, device=self.device,
                                                                      dtype=torch.float32).cpu().item()

        # Calculate average reward components across agents
        avg_surveillance_reward = torch.stack(surveillance_reward).mean() if surveillance_reward else torch.tensor(0.0,
                                                                                                                   device=self.device,
                                                                                                                   dtype=torch.float32)
        avg_emergency_reward = torch.stack(emergency_reward).mean() if emergency_reward else torch.tensor(0.0,
                                                                                                          device=self.device,
                                                                                                          dtype=torch.float32)
        avg_assignment_proximity_reward = torch.stack(
            assignment_proximity_reward).mean() if assignment_proximity_reward else torch.tensor(0.0,
                                                                                                 device=self.device,
                                                                                                 dtype=torch.float32)

        # Convert the averaged reward components to NumPy arrays
        reward_components = {
            "avg_surveillance_reward": avg_surveillance_reward.cpu().item(),
            "avg_emergency_reward": avg_emergency_reward.cpu().item(),
            "avg_assignment_proximity_reward": avg_assignment_proximity_reward.cpu().item()
        }

        return reward_dict, reward_components

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
                    # rewards[f'small_{small_agent_id}'] = self.emergency_aoi[emergency_id] / self.max_timesteps
                    self.emergency_end_time[emergency_id] = self.timestep
                    if np.all(small_agent[POSITION] == small_agent[TARGET]):
                        small_agent[TARGET] = None
                        rewards[f'small_{small_agent_id}'] = 1

                if small_agent[TARGET] is not None:
                    pass
                    # target_aoi = self.emergency_aoi[self.emergency_mapping[tuple(small_agent[TARGET])]]
                    # rewards[f'small_{small_agent_id}'] -= np.linalg.norm(
                    #     (small_agent[TARGET] - small_agent[POSITION]) / self.grid_size
                    # )

                # rewards[f'small_{small_agent_id}'] += self.group_factor * self.aoi_grid[x, y] * self.poi_grid[
                #     x, y] / self.max_timesteps
                #
                # self.max_reward = max(self.max_reward, rewards[f'small_{small_agent_id}'])
                # self.min_reward = min(self.min_reward, rewards[f'small_{small_agent_id}'])
                # rewards[f'small_{small_agent_id}'] = (
                #         (rewards[f'small_{small_agent_id}'] - self.min_reward) / (self.max_reward - self.min_reward))
                self.aoi_grid[x, y] = 0
            else:
                pass
                # rewards[f'small_{small_agent_id}'] = 0

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
            # rewards[f'big_{big_agent_id}'] = self.self_factor * self.aoi_grid[deploy_x, deploy_y] * self.poi_grid[
            #     deploy_x, deploy_y] / self.max_timesteps
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
            # rewards[f'big_{big_agent_id}'] += np.sum(valid_aoi) / self.max_timesteps
            info[f'big_{big_agent_id}_buffer_length'] = len(big_agent[TARGETS])
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
                    deploy_reward = 1 - np.linalg.norm(
                        (deploy_position - deployed_small_agent[TARGET]) / self.grid_size)
                # info[f'big_{big_agent_id}_deploy_reward'] = deploy_reward
                # rewards[f'big_{big_agent_id}'] += deploy_reward
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
                                assign_reward = 1 - np.linalg.norm(
                                    (small_agent[POSITION] - small_agent[TARGET]) / self.grid_size)
                                # info[f'big_{big_agent_id}_assign_reward'] = assign_reward
                                # rewards[f'big_{big_agent_id}'] += assign_reward
                                break
            else:
                pass
                # rewards[f'big_{big_agent_id}'] -= len(big_agent[TARGETS]) * 0.5

        rewards, reward_component = self.compute_reward()
        info.update(**reward_component)
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
