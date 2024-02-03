import numpy as np
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint
from scipy import optimize
from MARLlib.marllib.marl.models.zoo.mlp.base_mlp import (
    generate_identity_matrices, separate_first_task, generate_agent_matrix)


class Function:
    def __init__(self, matrix):
        self.matrix = matrix
        self.agent_number = self.matrix.shape[0]

    def __call__(self, x):
        assert x.shape == (self.agent_number,)
        return np.sum(self.matrix[np.arange(self.agent_number), x])


def separate_first_task_verify(assignments):
    first_task, remaining_tasks = separate_first_task(assignments)
    print("First Task:", first_task)
    print("Remaining Tasks:", remaining_tasks)


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(0)
    # Generate random distances matrix
    # distances = np.array([[3, 1, 2, 0], [1, 0, 3, 2], [2, 3, 0, 1]])
    num_of_agents = 4
    num_of_tasks = 3
    # distances = np.array([[3, 1, 2, ], [1, 0, 3, ], [2, 3, 0, ], [0, 1, 2]])
    distances = np.array([[0, 0, 2, ], [1, 2, 3, ], [1, 1, 2, ], [1, 3, 0]])
    # row_ind, col_ind = linear_sum_assignment(distances)
    # print(row_ind, col_ind)
    distances_flat = distances.ravel()
    # A = generate_identity_matrices(num_of_agents, num_of_tasks)
    # example = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    # temp = A.dot(example)
    # lb = ub = np.ones(num_of_tasks)
    # task_constraint = LinearConstraint(A, lb, ub)
    # integrality = np.ones(num_of_agents * num_of_tasks)
    # bounds = optimize.Bounds(0, 1)
    # result = milp(distances_flat, integrality=integrality, bounds=bounds, constraints=task_constraint)
    # print(result)
    # allocation_result = result['x'].reshape(num_of_agents, num_of_tasks)
    assignments = (np.array([0, 0, 0, 1, 1, 2, 2, 2]), np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    separate_first_task_verify(assignments)
    assignments = (np.array([0, 1, 2]), np.array([0, 1, 2]))
    separate_first_task_verify(assignments)
    assignments = (np.array([0, 0, 0]), np.array([0, 1, 2]))
    separate_first_task_verify(assignments)
    example = generate_agent_matrix(num_of_agents, num_of_tasks)
    print(example)
