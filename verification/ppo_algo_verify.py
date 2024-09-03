import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym


class VecPolicy(nn.Module):
    def __init__(self, input_dim, num_actions=5, fc_size=64):
        super(VecPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.action_head = nn.Linear(fc_size, num_actions)
        self.value_head = nn.Linear(fc_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value

    def get_value(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_value = self.value_head(x)
        return state_value


class PPOVecPolicy(VecPolicy):
    def __init__(self, input_dim, num_actions=5, fc_size=64):
        super(PPOVecPolicy, self).__init__(input_dim, num_actions, fc_size)
        # Initialize action and reward buffers
        self.saved_actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []

    def finish_episode(self, optimizer, gamma=0.99, eps=1e-8, max_grad_norm=1.0,
                       clip_coef=0.2, vf_coef=0.5, ent_coef=0.01, gae_lambda=0.95, num_minibatches=4, next_state=None,
                       done=False):
        """
        Perform backpropagation to update the policy and value function using PPO with gradient clipping.
        """
        R = 0
        policy_losses = []
        value_losses = []
        log_probs = self.log_probs
        entropies = self.entropies
        values = torch.cat(self.values).squeeze(-1)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)

        # If the episode is done, we set the next value to 0.0 as there's no future reward to be expected
        with torch.no_grad():
            if done:
                next_value = torch.tensor([0.0])
            else:
                next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
                _, next_value = self(next_state_tensor)

            # Compute GAE (Generalized Advantage Estimation)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - done
                else:
                    nextnonterminal = 1.0
                    next_value = values[t + 1]

                delta = rewards[t] + gamma * next_value * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        # Flatten tensors
        old_log_probs = torch.cat(log_probs)
        advantages = advantages.view(-1)
        returns = returns.view(-1)

        # Prepare for minibatch update
        batch_size = len(self.rewards)
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        minibatch_size = batch_size // num_minibatches

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = indices[start:end]

            # Slice minibatch data
            mb_log_probs = old_log_probs[mb_inds]
            mb_advantages = advantages[mb_inds]
            mb_returns = returns[mb_inds]
            mb_values = values[mb_inds]

            # PPO Loss computation
            ratio = (mb_log_probs - old_log_probs[mb_inds]).exp()
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages
            policy_loss = -torch.max(surr1, surr2).mean()
            value_loss = F.mse_loss(mb_values, mb_returns).to(torch.float32)

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

        # Compute entropy loss
        entropy_loss = -torch.cat(entropies).mean()

        # Perform backpropagation
        optimizer.zero_grad()
        loss = sum(policy_losses) + vf_coef * sum(value_losses) - ent_coef * entropy_loss
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

        # Update parameters
        optimizer.step()

        # Reset action, reward, and value buffers
        del self.saved_actions[:]
        del self.rewards[:]
        del self.log_probs[:]
        del self.values[:]
        del self.entropies[:]


def train_cartpole():
    env = gym.make('CartPole-v1')
    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    policy = PPOVecPolicy(input_dim=input_dim, num_actions=num_actions).float()
    optimizer = optim.Adam(policy.parameters(), lr=2.5e-4)

    num_episodes = 1000
    gamma = 0.99
    max_grad_norm = 0.5
    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    gae_lambda = 0.95

    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = 0
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action_probs, state_value = policy(state_tensor)
            m = Categorical(action_probs)
            action = m.sample()

            policy.saved_actions.append((m.log_prob(action), state_value))
            policy.log_probs.append(m.log_prob(action))
            policy.values.append(state_value)
            policy.entropies.append(m.entropy())

            next_state, reward, done, _ = env.step(action.item())
            policy.rewards.append(reward)
            episode_rewards += reward

            state = next_state

        # Prepare the final next_value for bootstrapping if the episode is not terminal
        policy.finish_episode(optimizer, gamma=gamma, max_grad_norm=max_grad_norm,
                              clip_coef=clip_coef, vf_coef=vf_coef, ent_coef=ent_coef,
                              gae_lambda=gae_lambda, num_minibatches=4,
                              next_state=next_state if not done else None, done=done)

        print(f"Episode {episode + 1}: Total Reward: {episode_rewards}")

        if episode_rewards >= 475:
            print(f"Solved after {episode + 1} episodes!")
            break


if __name__ == "__main__":
    train_cartpole()
