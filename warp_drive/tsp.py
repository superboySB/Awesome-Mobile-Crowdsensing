import pandas as pd
import numpy as np
import cvxpy as cp
from envs.crowd_sim.crowd_sim import RLlibCUDACrowdSim
from sklearn.cluster import KMeans


def calculate_route_costs(route, cost_matrix):
    segments = []
    for i in range(len(route) - 1):
        start = route[i]
        end = route[i + 1]
        segment_cost = cost_matrix[start][end]
        segments.append(((start, end), segment_cost))
    return segments


def max_distance_in_cluster(points, center):
    """Calculate the maximum distance from the center to any point in the cluster."""
    distances = np.linalg.norm(points - center, axis=1)
    return np.max(distances)


def valid_cluster_config(coordinates, k, radius):
    """Perform K-Means clustering and check if all clusters have all points within the radius.
    Return cluster centers if valid, None otherwise."""
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(coordinates)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    for i in range(k):
        cluster_points = coordinates[labels == i]
        center = centers[i]
        if max_distance_in_cluster(cluster_points, center) > radius:
            return None
    return centers  #


def find_optimal_k_and_centers(coordinates, min_k, max_k, r):
    """Use binary search to find the minimal k such that all clusters meet the radius condition,
    and return the cluster centers."""
    optimal_k = None
    optimal_centers = None
    while min_k <= max_k:
        mid_k = (min_k + max_k) // 2
        print(f"Trying k = {mid_k}")
        centers = valid_cluster_config(coordinates, mid_k, r)
        if centers is not None:
            optimal_k = mid_k
            optimal_centers = centers
            max_k = mid_k - 1  # Try smaller k to see if still valid
        else:
            min_k = mid_k + 1  # Increase k, since mid_k does not satisfy condition
    return optimal_k, optimal_centers


class CrowdSimTSPSolver:
    '''
    class instance initiaton
    '''

    def __init__(self, env: RLlibCUDACrowdSim):
        self.env = env.env
        # Apply PoI Clustering
        coordinates = np.stack(
            [
                self.env.target_x_time_list[0, :-self.env.emergency_count],
                self.env.target_y_time_list[0, :-self.env.emergency_count]
            ],
            axis=-1,
        )
        # Maximum radius for clusters
        r = self.env.config.env.drone_sensing_range
        min_k = 1
        max_k = len(coordinates) // 2  # No need to go beyond half the number of points

        optimal_k, optimal_centers = find_optimal_k_and_centers(coordinates, min_k, max_k, r)
        # calculate the distance matrix between all emergency PoIs and starting point

        all_positions = np.stack(
            [
                np.concatenate(
                    [self.env.emergency_centers_x, np.array([self.env.starting_location_x])]
                ),
                np.concatenate(
                    [self.env.emergency_centers_y, np.array([self.env.starting_location_y])]
                )
            ],
            axis=-1
        )
        # calculate distance matrix with all_positions and numpy vectorization in batch
        self.cost_matrix = np.zeros((len(all_positions), len(all_positions)))
        for i in range(len(all_positions)):
            for j in range(len(all_positions)):
                self.cost_matrix[i, j] = np.linalg.norm(
                    all_positions[i] - all_positions[j]
                ) / (self.env.config.env.drone_velocity * self.env.config.env.step_time)
        self.t_start = np.concatenate([np.array([0]), self.env.aoi_schedule])
        self.t_end = np.concatenate([np.array([self.env.episode_length]),
                                     self.env.aoi_schedule + self.env.emergency_threshold])
        # clip t_end to episode_length
        self.t_end = np.minimum(self.t_end, self.env.episode_length)
        self.num_agents = self.env.num_agents

    def get_solution(self, print_path=False):
        C = self.cost_matrix
        n = C.shape[0]  # Assuming C is the cost matrix and its size is defined
        X = cp.Variable(C.shape, boolean=True)
        u = cp.Variable(n, integer=True)
        t = cp.Variable(n)  # Time variables for each city visit
        m = self.num_agents
        ones = np.ones((n, 1))
        t_start, t_end = self.t_start, self.t_end

        # Penalty coefficients
        P_early = 1
        P_late = 1

        # Defining the objective function
        objective = cp.Minimize(cp.sum(cp.multiply(C, X)) +
                                cp.sum(P_early * cp.pos(t_start - t)) +
                                cp.sum(P_late * cp.pos(t - t_end)))

        # Defining the constraints
        constraints = []
        constraints += [0 <= X, X <= 1]
        constraints += [X[0, :] @ ones == m]
        constraints += [X[:, 0] @ ones == m]
        constraints += [X[1:, :] @ ones == 1]
        constraints += [X[:, 1:].T @ ones == 1]
        constraints += [cp.diag(X) == 0]
        constraints += [u[1:] >= 2]
        constraints += [u[1:] <= n]
        constraints += [u[0] == 1]
        constraints += [t >= t_start]
        constraints += [t <= t_end]

        # Adding time feasibility constraints for sequential city visits
        M = 1000  # A large constant for big-M method
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    constraints += [u[i] - u[j] + 1 <= (n - 1) * (1 - X[i, j])]
                    constraints += [t[j] >= t[i] + C[i, j] - M * (1 - X[i, j])]

        # Solving the problem
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.GLPK_MI, verbose=True, miptol=1e-8)

        # Transforming the solution to a path
        X_sol = np.argwhere(X.value == 1)
        print(np.unique(X.value))
        routes = {}
        for i in range(0, m):
            routes['Drone_' + str(i + 1)] = [0]
            j = i
            a = 10e10
            while a != 0:
                a = X_sol[j, 1]
                # print(a)
                routes['Drone_' + str(i + 1)].append(a)
                j = np.where(X_sol[:, 0] == a)
                # print(j)
                j = j[0][0]
                a = j
        if print_path:
            # Showing the paths
            for agent in routes.keys():
                print('The path of ' + agent + ' is:\n')
                print(' => '.join(map(str, routes[agent])))
                print('')
        # get the cost for each route segment and store them as a dict
        detailed_routes = {drone: calculate_route_costs(route, C) for drone, route in routes.items()}
        return routes
