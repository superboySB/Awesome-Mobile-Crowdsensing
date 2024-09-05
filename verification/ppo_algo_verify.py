import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from cleanrl_ppo import Agent
from tqdm import trange

DEBUG = True
num_steps: int = 128
num_envs = 4  # Number of parallel environments
learning_rate = 2.5e-4

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

    def get_action_and_value(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        m = Categorical(action_probs)
        # return action, log_prob and entropy
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        return action, log_prob, entropy, self.value_head(x)


class PPOVecPolicy(Agent):
    def __init__(self, envs):
        super(PPOVecPolicy, self).__init__(envs)
        # input_dim, num_actions=5, fc_size=64
        # super(PPOVecPolicy, self).__init__(input_dim, num_actions, fc_size)
        # Initialize action and reward buffers
        self.saved_actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.saved_obs = []
        self.values = []
        # self.entropies = []

    def finish_episode(self, optimizer, gamma=0.99, eps=1e-8, max_grad_norm=0.5,
                       clip_coef=0.2, vf_coef=0.5, ent_coef=0.01, gae_lambda=0.95,
                       num_minibatches=4, next_state=None, update_epochs=1,
                       next_dones=None, device: str = 'cpu'):
        """
        Perform backpropagation to update the policy and value function using PPO with gradient clipping.
        """
        log_probs = self.log_probs
        saved_actions = self.saved_actions
        all_obs = torch.cat(self.saved_obs).squeeze(-1).reshape(num_steps, num_envs, -1)
        values = torch.cat(self.values).squeeze(-1).reshape(num_steps, num_envs)
        dones = torch.tensor(self.dones).reshape(num_steps, num_envs)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device).reshape(num_steps, num_envs)

        # If the episode is done, we set the next value to 0.0 as there's no future reward to be expected
        with torch.no_grad():
            # if done:
            #     next_value = torch.tensor([0.0]).to(device)
            # else:
            next_state_tensor = torch.from_numpy(next_state).float().to(device).squeeze(-1)
            agent_next_value = self.get_value(next_state_tensor).reshape(1, -1)

            # Compute GAE (Generalized Advantage Estimation)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_dones
                    next_value = agent_next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]

                delta = rewards[t] + gamma * next_value * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values

        # advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        # Flatten tensors
        old_log_probs = torch.tensor(log_probs, device=device).view(-1)
        saved_actions = torch.tensor(saved_actions, device=device).view(-1)
        advantages = advantages.view(-1)
        returns = returns.view(-1)
        values = values.view(-1)
        all_obs = all_obs.view(-1, all_obs.shape[-1])

        # Prepare for minibatch update
        batch_size = len(self.rewards)
        indices = np.arange(batch_size)

        minibatch_size = batch_size // num_minibatches
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = indices[start:end]

                # Slice minibatch data
                mb_obs = all_obs[mb_inds]
                mb_actions = saved_actions[mb_inds].long()
                _, mb_log_probs, mb_entropies, new_values = self.get_action_and_value(mb_obs, mb_actions)
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                mb_values = values[mb_inds]
                new_values = new_values.squeeze(-1)

                # PPO Loss computation
                logratio = (mb_log_probs - old_log_probs[mb_inds])
                ratio = logratio.exp()
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps).to(torch.float32)
                surr1 = -ratio * mb_advantages
                surr2 = -torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                policy_loss = torch.max(surr1, surr2).mean()
                # Unclipped value loss
                value_loss_unclipped = F.mse_loss(new_values, mb_returns).to(torch.float32)

                # Clipped value estimate
                value_pred_clipped = mb_values + torch.clamp(
                    new_values - mb_returns,
                    -clip_coef,
                    clip_coef
                )

                # Clipped value loss
                value_loss_clipped = F.mse_loss(value_pred_clipped, mb_returns).to(torch.float32)

                # Final value loss is the maximum of the two
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Combine losses
                entropy_loss = -mb_entropies.mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

                # Perform backpropagation for the current mini-batch
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()

        # Reset action, reward, and value buffers
        del self.saved_actions[:]
        del self.rewards[:]
        del self.log_probs[:]
        del self.values[:]
        del self.saved_obs[:]
        del self.dones[:]


def make_env(env_id, idx, capture_video=False, run_name=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def train_cartpole():
    env_id = 'CartPole-v1'

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i) for i in range(num_envs)]
    )

    input_dim = envs.single_observation_space.shape[0]
    num_actions = envs.single_action_space.n
    device = 'cuda:0' if (torch.cuda.is_available() and not DEBUG) else 'cpu'
    policy = PPOVecPolicy(envs).to(device)
    # policy = PPOVecPolicy(input_dim=input_dim, num_actions=num_actions).float().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)

    num_episodes: int = 1000

    gamma = 0.99
    max_grad_norm = 0.5
    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    gae_lambda = 0.95

    next_obs, _ = envs.reset(seed=seed)
    next_dones = torch.zeros(num_envs).to(device)
    episode_rewards = np.zeros(num_envs)
    progress = trange(num_episodes)
    anneal_lr = False
    for episode in progress:
        if anneal_lr:
            frac = 1.0 - (episode - 1.0) / num_episodes
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        policy.rewards = []
        policy.log_probs = []
        policy.values = []
        policy.entropies = []
        policy.dones = []

        for _ in range(num_steps):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(next_obs).float().to(device).to(torch.float32)
                actions, log_probs, entropies, state_values = policy.get_action_and_value(obs_tensor)

            policy.saved_actions.extend(actions)
            policy.log_probs.extend(log_probs)
            policy.values.extend(state_values)
            policy.saved_obs.extend(obs_tensor)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions.cpu().numpy())
            next_dones = torch.from_numpy(np.logical_or(terminations, truncations)).to(torch.float32)
            policy.rewards.extend(rewards)
            policy.dones.extend(next_dones)
            episode_rewards += rewards

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        progress.set_description(f"Return: {info['episode']['r']}")
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        # Prepare final bootstrapped value
        policy.finish_episode(optimizer, gamma=gamma, max_grad_norm=max_grad_norm,
                              clip_coef=clip_coef, vf_coef=vf_coef, ent_coef=ent_coef,
                              gae_lambda=gae_lambda, num_minibatches=4,
                              next_state=next_obs, next_dones=next_dones, device=device)

        # print(f"Episode {episode + 1}: Total Reward: {episode_rewards.mean()}")

        # if episode_rewards.mean() >= 475:
        #     print(f"Solved after {episode + 1} episodes!")
        #     break

        # Reset rewards for the next episode
        episode_rewards.fill(0)

        # Reset environment (No reset, then rollout length can continue)
        # obs, _ = envs.reset()
    envs.close()


if __name__ == "__main__":
    # fix random seed
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    train_cartpole()
