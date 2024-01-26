import numpy as np
from scipy.optimize import linear_sum_assignment


class Function:
    def __init__(self, matrix):
        self.matrix = matrix
        self.agent_number = self.matrix.shape[0]

    def __call__(self, x):
        assert x.shape == (self.agent_number,)
        return np.sum(self.matrix[np.arange(self.agent_number), x])


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(0)
    # Generate random distances matrix
    # distances = np.array([[3, 1, 2, 0], [1, 0, 3, 2], [2, 3, 0, 1]])
    distances = np.array([[3, 1, 2, ], [1, 0, 3, ], [2, 3, 0, ], [0, 1, 2]])
    row_ind, col_ind = linear_sum_assignment(distances)
    print(row_ind, col_ind)
