import argparse
from typing import Optional, Tuple

import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.core import ObsType, ActType
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


class DistanceSelectEnv(gym.Env):
    def __init__(self, select_range=4):
        super(DistanceSelectEnv, self).__init__()
        self.select_range = select_range
        self.action_space = gym.spaces.Discrete(select_range)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(select_range + 1,))
        self.emergency_queue_length = 3
        self.state = None
        self.timestep = 0

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        np.random.seed(seed)
        # self.state = np.random.uniform(0, 1, size=(self.select_range,))
        # state is the permututaion of 0 to select_range
        self.update_state()
        self.timestep = 0
        self.time_limit = 100
        return self.state, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # calculate the distance between selected state dim and final_dim
        # greedy_action = np.argmin(np.abs(self.state[:-1] - self.state[-1]))
        # print(f"Actual Action: {action}, Greedy Action: {greedy_action}, Distance: {distance}")
        # distance reward is proved to fail
        # reward = -distance
        # sparse reward does not work
        reward = self.mock_assign_reward(action)
        # state is an array from 0 to
        self.update_state()
        self.timestep += 1
        if self.timestep == self.time_limit:
            return self.state, reward, True, False, {}
        else:
            return self.state, reward, False, False, {}

    def l1_diff_reward(self, action):
        rewards = np.zeros(self.select_range)
        rewards[np.argsort(np.abs(self.state[:-1] - self.state[-1]))] = np.arange(self.select_range)
        reward = rewards[action]
        return reward

    def l2_diff_reward(self, action):
        # calculate the L2 norm for each 2 dim in state
        rewards = np.zeros(self.select_range)
        coord_x = self.state[::2]
        all_x, target_x = coord_x[:-1], coord_x[-1]
        coord_y = self.state[1::2]
        all_y, target_y = coord_y[:-1], coord_y[-1]
        distances = np.sqrt((all_x - target_x) ** 2 + (all_y - target_y) ** 2)
        rewards[np.argsort(distances)] = -np.arange(self.select_range) / self.select_range
        reward = rewards[action]
        return reward

    def mock_assign_reward(self, action):
        agents_info = self.state[:-2].reshape(-1, 2 + self.emergency_queue_length * 2)
        agents_x, agents_y = agents_info[..., 0], agents_info[..., 1]
        target_x, target_y = self.state[-2], self.state[-1]
        distances = np.sqrt((agents_x - target_x) ** 2 + (agents_y - target_y) ** 2)
        rewards = np.zeros(self.select_range)
        rewards[np.argsort(distances)] = np.arange(self.select_range) / self.select_range
        reward = rewards[action]
        return reward
        
    def update_state_simple(self):
        self.state = np.linspace(0, 1, self.select_range)
        # add noise to the state
        self.state += np.random.normal(0, 0.1, size=(self.select_range,))

    def update_state(self):
        self.state = np.random.uniform(-1, 1, size=((2 + self.emergency_queue_length * 2) *
                                                    (self.select_range) + 2,))

    def render(self, mode='human'):
        pass


env = DistanceSelectEnv()
# env = gym.make('CartPole-v1')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.separate = False
        if self.separate:
            self.affine1 = nn.Linear(10, 20)
            self.side_affine1 = nn.Linear(1, 12)
        else:
            self.affine1 = nn.Linear(34, 128)
        # self.dropout = nn.Dropout(p=0.3)
        self.affine2 = nn.Linear(128, 4)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        if self.separate:
            x = torch.cat((self.affine1(x[:, :-1]), self.side_affine1(x[:, -1].unsqueeze(-1))), dim=-1)
        else:
            x = self.affine1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(iter: int, log_interval: int):
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    if iter % log_interval == 0:
        print(f"Policy Loss: {policy_loss}")
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 0
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(i_episode, args.log_interval)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        if running_reward > 60:
            # if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
