import numpy as np
import torch


class GridMDP:
    """
    Grid world MDP
    """

    def __init__(self):
        self.x_dim = 10
        self.y_dim = 10
        self.num_states = self.x_dim * self.y_dim
        self.start_state = np.asarray([0, 0])
        self._state = self.start_state
        self.action_space = [0, 1, 2, 3]
        self.target_state = [np.asarray([9, 9])]
        self.time_limit = 15
        self.t = 0
        self._done = False
        self._rewards = np.zeros(shape=(self.x_dim, self.y_dim), dtype=np.float)
        for t in self.target_state:
            self._rewards[t[0], t[1]] = 1.
        self.dims = 2
        self.min_state = np.asarray([0, 0])
        self.max_state = np.asarray([9, 9])

    def target_distribution(self, weight=3.) -> np.ndarray:
        """
        Return the target distribution
        :return:
        """
        # dist = np.exp(self._rewards * weight)
        dist = self._rewards * weight
        dist /= np.sum(dist)
        return dist

    def reset(self) -> np.ndarray:
        """
        reset state to 0
        :return:
        """
        self._state = np.copy(self.start_state)
        self.t = 0
        self.done()
        return self._state

    def step(self, act):
        """
        take action and return next state
        :param act:
        :return:
        """
        act = int(act)
        if self.t >= self.time_limit:
            print("Time limit exceeded. Reset for new episode")
            raise Exception()

        if self._done:
            print("Episode ended. Reset for new episode")
            raise Exception()

        assert act in self.action_space

        if np.random.random() < 0.:
            return self._state, self.reward(), self._done

        if act < 2:
            self._state[0] += act * 2 - 1
            self._state[0] = np.clip(self._state[0], 0, self.x_dim - 1)
        else:
            act = (act - 2) * 2 - 1
            self._state[1] += act
            self._state[1] = np.clip(self._state[1], 0, self.y_dim - 1)
        self.t += 1
        self.done()
        return self._state, self.reward(), self._done, {}

    def done(self):
        """
        check if episode over
        :return:
        """
        if self.t >= self.time_limit:
            self._done = True
        elif any(np.array_equal(self._state, x) for x in self.target_state):
            self._done = True
        else:
            self._done = False
        return

    def reward(self):
        """
        Reward function
        :return:
        """
        return self._rewards[self._state[0], self._state[1]]


test_env = GridMDP()
num_samples = 100
target_dist = np.reshape(test_env.target_distribution(), (-1,))
target_distribution = np.random.choice(target_dist.shape[0], num_samples, p=target_dist)
target_distribution = target_distribution.reshape([-1, 1])
if test_env.dims > 1:
    target_distribution = np.concatenate([target_distribution, target_distribution], axis=-1)
    target_distribution[:, 0] = target_distribution[:, 0] // test_env.y_dim
    target_distribution[:, 1] = target_distribution[:, 1] % test_env.y_dim
ones = torch.tensor(target_distribution).type(torch.float32).reshape([-1, test_env.dims])
print(ones)
