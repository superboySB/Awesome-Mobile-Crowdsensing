import numpy as np
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint
from scipy import optimize


class Function:
    def __init__(self, matrix):
        self.matrix = matrix
        self.agent_number = self.matrix.shape[0]

    def __call__(self, x):
        assert x.shape == (self.agent_number,)
        return np.sum(self.matrix[np.arange(self.agent_number), x])


def generate_identity_matrices(n, m):
    # Generate n identity matrices of size m
    identity_matrices = [np.eye(m) for _ in range(n)]
    # Stack the identity matrices horizontally
    result = np.hstack(identity_matrices)
    return result

if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(0)
    # Generate random distances matrix
    # distances = np.array([[3, 1, 2, 0], [1, 0, 3, 2], [2, 3, 0, 1]])
    n_agents = 4
    tasks = 3
    # distances = np.array([[3, 1, 2, ], [1, 0, 3, ], [2, 3, 0, ], [0, 1, 2]])
    distances = np.array([[0, 0, 2, ], [1, 2, 3, ], [1, 1, 2, ], [1, 3, 0]])
    row_ind, col_ind = linear_sum_assignment(distances)
    print(row_ind, col_ind)
    distances_flat = distances.ravel()
    A = generate_identity_matrices(n_agents, tasks)
    example = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    temp = A.dot(example)
    lb = ub = np.ones(tasks)
    task_constraint = LinearConstraint(A, lb, ub)
    integrality = np.ones(n_agents * tasks)
    bounds = optimize.Bounds(0, 1)
    result = milp(distances_flat, integrality=integrality, bounds=bounds, constraints=task_constraint)
    print(result)
