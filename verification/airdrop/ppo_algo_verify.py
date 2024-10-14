import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange

DEBUG = True
num_steps: int = 128
NUM_ENVS = 4  # Number of parallel environments
learning_rate = 2.5e-4


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        pass

    def forward(self, x):
        raise NotImplementedError

    def get_value(self, x):
        """
        Returns the state value from the critic's output.
        """
        return self.value_head(self._get_embedding(x))

    def _get_embedding(self, x):
        raise NotImplementedError

    def get_action_and_value(self, x, actions: list = None):
        """
        Returns actions, log probabilities, entropy, and state value, handling MultiDiscrete action space.
        """
        emb = self._get_embedding(x)
        action_probs = [F.softmax(action_head(emb), dim=-1) for action_head in self.action_heads]
        # detect NaN in action_probs
        for i, probs in enumerate(action_probs):
            if torch.isnan(probs).any():
                raise ValueError('NaN detected in action_probs')
        distributions = [Categorical(probs=probs) for probs in action_probs]

        # Sample actions if not provided
        if actions is None:
            actions = [dist.sample() for dist in distributions]

        log_probs = torch.stack([dist.log_prob(action) for dist, action in zip(distributions, actions)], dim=-1)
        entropies = torch.stack([dist.entropy() for dist in distributions], dim=-1)

        # Convert actions to a single tensor for compatibility with the rest of the code
        actions = torch.stack(actions, dim=-1)

        return actions, log_probs, entropies, self.value_head(emb)


class VecPolicy(Policy):
    def __init__(self, input_dim, num_actions=5, fc_size=64):
        super(VecPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.fc3 = nn.Linear(fc_size, fc_size)
        if isinstance(num_actions, int):
            num_actions = [num_actions]
        self.action_heads = nn.ModuleList([nn.Linear(fc_size, num_action) for num_action in num_actions])
        self.value_head = nn.Linear(fc_size, 1)

    def forward(self, x):
        action_probs = F.softmax(self.action_head(self._get_embedding(x)), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value

    def _get_embedding(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class CNNPolicy(Policy):
    """
    Implements both actor and critic in one model using CNN for 2D grid input, supporting MultiDiscrete action space.
    """

    def __init__(self, input_shape, num_actions=[5, 2], conv_channels=[4, 8, 16], kernel_sizes=[3, 3, 3], fc_size=32):
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
        if isinstance(num_actions, int):
            num_actions = [num_actions]
        # Actor's layers for each dimension of the MultiDiscrete action space
        self.action_heads = nn.ModuleList([nn.Linear(fc_size, num_action) for num_action in num_actions])

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
        x = self._get_embedding(x)
        # Actor: chooses action probabilities for each dimension in the MultiDiscrete action space
        action_probs = [F.softmax(action_head(x), dim=-1) for action_head in self.action_heads]
        # Critic: evaluates the value of the state
        state_value = self.value_head(x)
        # Return both actor probabilities and critic values
        return action_probs, state_value

    def _get_embedding(self, x):
        # Apply CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten the output from the convolution layers
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = F.relu(self.fc1(x))
        return x


class PPO(Policy):
    def __init__(self):
        # super(PPOVecPolicy, self).__init__(envs)
        # input_dim, num_actions=5, fc_size=64
        # Initialize action and reward buffers
        self.saved_actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.saved_obs = []
        self.values = []

    def finish_episode(self, optimizer, gamma=0.99, eps=1e-8, max_grad_norm=0.5,
                       clip_coef=0.2, vf_coef=0.5, ent_coef=0.01, gae_lambda=0.95,
                       num_minibatches=4, next_state=None, update_epochs=4, num_envs=4,
                       next_dones=None, device: str = 'cpu', num_steps=num_steps):
        """
        Perform backpropagation to update the policy and value function using PPO with gradient clipping.
        """
        # print("batch length: {}, fixed length: {}".format( len(self.rewards), num_envs * num_steps))
        if len(self.values) == 0:
            # Skip Training if no data is available
            return
        log_probs = self.log_probs[:num_envs * num_steps]
        saved_actions = self.saved_actions[:num_envs * num_steps]

        if len(self.values[0].shape) != 0:
            values = torch.cat(self.values[:num_envs * num_steps]).squeeze(-1).reshape(num_steps, -1).to(device)
        else:
            values = torch.Tensor(self.values[:num_envs * num_steps]).reshape(num_steps, num_envs).to(device)
        dones = torch.tensor(self.dones[:num_envs * num_steps]).reshape(num_steps, -1).to(torch.float32).to(device)
        rewards = torch.tensor(self.rewards[:num_envs * num_steps], dtype=torch.float32, device=device).reshape(
            num_steps, -1)

        # If the episode is done, we set the next value to 0.0 as there's no future reward to be expected
        with torch.no_grad():
            if isinstance(next_state, torch.Tensor):
                if next_state.shape[0] == 1:
                    # 1D tensor
                    next_state_tensor = next_state.unsqueeze(0).to(device)
                else:
                    # 2D tensor
                    next_state_tensor = next_state.to(device)
            else:
                next_state_tensor = torch.from_numpy(next_state).float().to(device)

            agent_next_value = self.get_value(next_state_tensor).reshape(-1, num_envs)

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

        # Flatten tensors
        if len(log_probs[0].shape) > 1:
            old_log_probs = torch.cat(log_probs).squeeze(-1)
            saved_actions = torch.cat(saved_actions).squeeze(-1)
        else:
            old_log_probs = torch.tensor(log_probs, device=device).view(-1)
            saved_actions = torch.tensor(saved_actions, device=device).view(-1)

        advantages = advantages.view(-1)
        returns = returns.view(-1)
        values = values.view(-1)
        all_obs = torch.cat(self.saved_obs[:num_envs * num_steps])

        # Prepare for minibatch update
        batch_size = len(returns)
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
                if len(mb_actions.shape) > 1:
                    mb_actions = [t.squeeze(-1) for t in torch.split(mb_actions, 1, dim=-1)]
                else:
                    mb_actions = [mb_actions]
                _, mb_log_probs, mb_entropies, new_values = self.get_action_and_value(mb_obs, mb_actions)
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                mb_values = values[mb_inds]
                new_values = new_values.squeeze(-1)

                # PPO Loss computation
                logratio = (mb_log_probs - old_log_probs[mb_inds])
                ratio = logratio.exp()
                if len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps).to(
                        torch.float32)
                if len(ratio.shape) == 1:
                    ratio = ratio.unsqueeze(-1)
                mb_advantages = mb_advantages.unsqueeze(-1)
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
                if torch.isnan(loss).any():
                    raise ValueError("Loss is nan")
                optimizer.step()

        # Reset action, reward, and value buffers
        del self.saved_actions[:]
        del self.rewards[:]
        del self.log_probs[:]
        del self.values[:]
        del self.saved_obs[:]
        del self.dones[:]
        # Construct a dict of statistics (training)
        return {
            "old_approx_kl": old_approx_kl,
            "approx_kl": approx_kl,
            "clipfrac": np.mean(np.array(clipfracs)),
            "policy_loss": policy_loss.detach().cpu().numpy(),
            "value_loss": value_loss.detach().cpu().numpy(),
            "entropy_loss": entropy_loss.detach().cpu().numpy(),
        }


class PPOVecPolicy(PPO, VecPolicy):
    def __init__(self, input_dim, num_actions=5, fc_size=64):
        PPO.__init__(self)
        VecPolicy.__init__(self, input_dim, num_actions, fc_size)


class PPOCNNPolicy(PPO, CNNPolicy):
    def __init__(self, input_shape, fc_size, num_actions=5):
        PPO.__init__(self)
        CNNPolicy.__init__(self, input_shape=input_shape, num_actions=num_actions, fc_size=fc_size)


class MultiPPORollout:
    """
    Support MultiAgent for PPO.
    """

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.rollout_items = ['saved_actions', 'rewards', 'dones', 'log_probs', 'saved_obs', 'values']
        for item in self.rollout_items:
            setattr(self, item, [[] for _ in range(num_agents)])

    def concatenate_rollouts(self, my_policy: PPO):
        """
        Concatenate all the rollouts into one list
        """

        for item in self.rollout_items:
            list(map(getattr(my_policy, item).extend, getattr(self, item)))
        # reset all the rollouts
        for item in self.rollout_items:
            getattr(self, item)[:] = [[] for _ in range(self.num_agents)]


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
        [make_env(env_id, i) for i in range(NUM_ENVS)]
    )

    input_dim = envs.single_observation_space.shape[0]
    num_actions = envs.single_action_space.n
    device = 'cuda:0' if (torch.cuda.is_available() and not DEBUG) else 'cpu'
    # policy = PPOVecPolicy(envs).to(device)
    policy = PPOVecPolicy(input_dim=input_dim, num_actions=num_actions).float().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)

    num_episodes: int = 1000

    gamma = 0.99
    max_grad_norm = 0.5
    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    gae_lambda = 0.95

    next_obs, _ = envs.reset(seed=seed)
    next_dones = torch.zeros(NUM_ENVS).to(device)
    episode_rewards = torch.zeros(NUM_ENVS).to(device)
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
                              gae_lambda=gae_lambda, num_minibatches=4, num_envs=NUM_ENVS,
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
