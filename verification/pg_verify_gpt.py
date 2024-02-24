import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from torch.distributions import Categorical


# Define the environment
class PointSelectionEnv(gym.Env):
    def __init__(self, n_points):
        self.n_points = n_points
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2 * n_points,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n_points)

    def reset(self):
        self.points = np.random.rand(self.n_points, 2) * 2 - 1  # Random points in the range [-1, 1]
        self.current_step = 0
        return self.points.flatten()

    def step(self, action):
        selected_point = self.points[action]
        reward = -np.linalg.norm(selected_point, ord=2)  # Negative L2 norm as we want to maximize
        done = self.current_step == self.n_points - 1
        self.current_step += 1
        return self.points.flatten(), reward, done, {}


# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, action_size)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


# Training parameters
n_points = 5
input_size = 2 * n_points
action_size = n_points
learning_rate = 0.001
gamma = 0.99
num_episodes = 10000

# Initialize environment and policy network
env = PointSelectionEnv(n_points)
policy = PolicyNetwork(input_size, action_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    log_probs = []
    rewards = []

    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy(state_tensor)
        m = Categorical(action_probs)
        action = m.sample()

        log_probs.append(m.log_prob(action))
        state, reward, done, _ = env.step(action.item())
        rewards.append(reward)

        if done:
            discounted_rewards = []
            running_add = 0
            for r in reversed(rewards):
                running_add = running_add * gamma + r
                discounted_rewards.insert(0, running_add)

            policy_loss = []
            for log_prob, r in zip(log_probs, discounted_rewards):
                policy_loss.append(-log_prob * r)

            optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()
            if episode % 100 == 0:
                print(f"Episode: {episode + 1}, Total Reward: {sum(rewards)}")
            break
