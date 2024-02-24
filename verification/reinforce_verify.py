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
        self.state = None
        self.timestep = 0

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        np.random.seed(seed)
        self.state = np.random.uniform(-1, 1, size=(self.select_range + 1,))
        self.timestep = 0
        self.time_limit = 100
        return self.state, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # calculate the distance between selected state dim and final_dim
        distance = np.abs(self.state[-1] - self.state[action])
        reward = -distance
        self.state = np.random.uniform(-1, 1, size=(self.select_range + 1,))
        self.tiemstep += 1
        if self.timestep == self.time_limit:
            return self.state, reward, True, False, {}
        else:
            return self.state, reward, False, False, {}

    def render(self, mode='human'):
        pass


env = DistanceSelectEnv()
# env = gym.make('CartPole-v1')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(5, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 4)
        self.fc = nn.Sequential(
            self.affine1,
            self.dropout,
            nn.ReLU(),
            self.affine2
        )

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # x = self.affine1(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        # action_scores = self.affine2(x)
        action_scores = self.fc(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
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
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
