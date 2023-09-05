import numpy as np

# 修改格子的宽度为 100
def generate_grid_with_width_100(center_x, center_y, num_points=93, grid_width=100):
    # 生成 num_points 个点的模拟数据
    np.random.seed(0)  # 为了可重复性
    points = np.random.rand(num_points, 2) * 1200  # x, y 坐标在 [0, 1200) 范围内

    # 每个点有一个随机的 "信号强度"，范围在 [0, 1)
    signal_strengths = np.random.rand(num_points)

    # 设置固定的最小值和最大值（中心点为基础）
    fixed_min = np.array([center_x - 5 * grid_width, center_y - 5 * grid_width])
    fixed_max = np.array([center_x + 5 * grid_width, center_y + 5 * grid_width])

    # 离散化坐标
    discrete_points = np.floor((points - fixed_min) / (fixed_max - fixed_min) * 10).astype(int)

    # 初始化一个 10x10 的网格来存储点的平均 "信号强度"
    grid_signal_strength = np.zeros((10, 10))
    grid_point_count = np.zeros((10, 10), dtype=int)

    # 遍历所有离散点，更新网格
    for i in range(num_points):
        x, y = discrete_points[i]

        # 检查点是否在 10x10 网格内
        if 0 <= x < 10 and 0 <= y < 10:
            # 更新点的数量
            grid_point_count[x, y] += 1

            # 累加点的 "信号强度"
            grid_signal_strength[x, y] += signal_strengths[i]

    # 修正：在进行除法之前，先检查是否有格子的点数量为 0，以避免除以 0 的情况
    grid_point_count_nonzero = np.where(grid_point_count > 0, grid_point_count, 1)

    # 计算每个格子的平均 "信号强度"
    grid_signal_strength_avg = grid_signal_strength / grid_point_count_nonzero

    # 再次将点数量为 0 的格子的平均 "信号强度" 设置为 0
    grid_signal_strength_avg[grid_point_count == 0] = 0

    # 展平 10x10 的网格为一个向量
    flattened_grid_signal_strength = grid_signal_strength_avg.ravel()

    return grid_signal_strength_avg, flattened_grid_signal_strength

# 已知一个中心点的坐标
center_x, center_y = 500, 500

# 生成网格和展平的向量
grid_signal_strength_avg_100, flattened_grid_signal_strength_100 = generate_grid_with_width_100(center_x, center_y)

grid_signal_strength_avg_100, flattened_grid_signal_strength_100