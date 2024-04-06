import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder

np.seterr(all="raise")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Model(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        return self.affine2(x)


class ReplayBuffer:

    def __init__(self, d, capacity):
        self.buffer = {'context': np.zeros((capacity, d)), 'reward': np.zeros((capacity, 1))}
        self.capacity = capacity
        self.size = 0
        self.pointer = 0

    def add(self, context, reward):
        # logging.debug("add in replay buffer is called")
        self.buffer['context'][self.pointer] = context
        self.buffer['reward'][self.pointer] = reward
        self.size = min(self.size + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, n):
        idx = np.random.randint(0, self.size, size=n)
        return self.buffer['context'][idx], self.buffer['reward'][idx]


class NeuralUCB:

    def __init__(self, model_config: dict, d, num_of_arms, beta=1, lamb=1,
                 hidden_size=32, lr=1e-4, reg=0.000625):
        self.num_arms = num_of_arms
        self.T = 0
        self.reg = reg
        self.beta = beta
        self.custom_config = model_config['custom_model_config']
        self.model_arch_args = self.custom_config['model_arch_args']
        self.use_2d_state = 'conv_layer' in self.model_arch_args
        self.net = Model(d, hidden_size, 1)
        self.hidden_size = hidden_size
        self.last_loss = 0.0
        self.net.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.numel = sum(w.numel() for w in self.net.parameters() if w.requires_grad)
        self.sigma_inv = lamb * np.eye(self.numel, dtype=np.float32)
        self.device = device
        self.count = 0

        self.theta0 = torch.cat(
            [w.detach().flatten() for w in self.net.parameters() if w.requires_grad]
        )
        # must detach theta0 at here, otherwise cause
        # Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.
        self.replay_buffer = ReplayBuffer(d, 10000)

    def take_action(self, context, invalid_mask=None, grid=None):
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=torch.float32)
        context = context.to(self.device)
        g = np.zeros((self.num_arms, self.numel), dtype=np.float32)

        for k in range(self.num_arms):
            if self.use_2d_state:
                g[k] = self.grad(context[k], grid[k]).detach().cpu().numpy()
            else:
                g[k] = self.grad(context[k]).detach().cpu().numpy()

        with torch.no_grad():
            if grid is not None:
                y_hat = self.net(context, grid).cpu().numpy()
            else:
                y_hat = self.net(context).cpu().numpy()
            # logging.debug("calculating matmul of gradient")
            p = y_hat + self.beta * np.sqrt(
                np.matmul(np.matmul(g[:, None, :], self.sigma_inv), g[:, :, None])[:, 0, :])
            # logging.debug("matmul of gradient is done")

        if invalid_mask is not None:
            p[invalid_mask] = -np.inf
        action = np.argmax(p)
        return action

    def grad(self, x, grid=None):
        # logging.debug("grad in neural ucb is called")
        if grid is not None:
            input_dict = {
                "obs": x,
                "grid": grid
            }
            y = self.net(input_dict)
        else:
            y = self.net(x)
        self.optimizer.zero_grad()
        y.backward()
        # logging.debug("grad in neural ucb is exiting")
        return torch.cat(
            [w.grad.detach().flatten() / np.sqrt(self.hidden_size) for w in self.net.parameters() if w.requires_grad]
        ).to(self.device)

    def update(self, context, action, reward, grid=None):
        context = torch.tensor(context, dtype=torch.float32)
        context = context.to(self.device)
        # logging.debug(f"context added {context} with action {action} and reward {reward}")
        self.sherman_morrison_update(self.grad(context[action, None]).detach().cpu().numpy()[:, None])
        if self.use_2d_state:
            context_dict = {
                "obs": context[action],
                "grid": grid[action]
            }
            self.replay_buffer.add(context_dict, reward)
        else:
            self.replay_buffer.add(context[action].detach().cpu().numpy(), reward)
        self.T += 1
        # self.train()

    def update_batch(self, contexts, actions, rewards, grid=None):
        # call self.update with a for loop
        logging.debug("update_batch is called")
        for context, action, reward in zip(contexts, actions, rewards):
            self.update(context, action, reward, grid)
        logging.debug("exiting update_batch")

    def sherman_morrison_update(self, v):
        """
        Sherman-Morrison formula for updating the inverse of a matrix.
        """
        # logging.debug("sherman_morrison_update is called")
        A_inv = self.sigma_inv  # Get the inverse of matrix A
        # logging.debug(f"sigma_inv before: {self.sigma_inv}")
        # Compute A^{-1}uv^TA^{-1}
        # logging.debug(f"v: {v}")
        # dump A_inv and v as txt
        # np.savetxt("A_inv.txt", A_inv)
        # np.savetxt("v.txt", v)
        # print pwd
        term = np.matmul(np.matmul(np.matmul(A_inv, v), v.T), A_inv)
        # logging.debug(f"term: {term}")
        # Compute the denominator 1 + v^TA^{-1}u
        denominator = 1 + np.matmul(np.matmul(v.T, A_inv), v)
        # Update the inverse of the matrix using the Sherman-Morrison formula
        # assert np.all(term / denominator == (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (1 + v.T @ self.sigma_inv @ v)), \
        #     "Sherman-Morrison formula is not working"
        self.sigma_inv -= term / denominator
        # logging.debug("exiting sherman_morrison_update")

    def train(self):
        if self.T > self.num_arms and self.T % 1 == 0:
            for _ in range(2):
                x, y = self.replay_buffer.sample(64)
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
                y = torch.tensor(y, dtype=torch.float32).to(self.device).view(-1, 1)
                y_hat = self.net(x)
                loss = F.mse_loss(y_hat, y)
                theta = torch.cat(
                    [w.flatten() for w in self.net.parameters() if w.requires_grad]
                )
                loss += self.reg * torch.norm(theta - self.theta0) ** 2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.last_loss = loss.detach().item()

    def eval(self):
        pass
