import timeit

import numpy as np
import time
example = np.random.rand(10, 10)
example = -1
print(example)
mock_obs = np.random.rand(2000, 50)
mock_other = np.random.rand(2000, 50)
my_code = """
for this_obs, this_other in zip(mock_obs, mock_other):
    mask = this_obs > 0
    mock_select = this_other[mask]
"""
print(timeit.timeit(stmt=my_code, globals=globals(), number=100))
# 3x faster
my_code_2 = """
full_mask = mock_obs > 0
length = len(full_mask)
for i in range(length):
    mask = full_mask[i]
    mock_select = mock_other[i][mask]
"""
print(timeit.timeit(stmt=my_code_2, globals=globals(), number=100))

from collections import deque
# Queue verify
# all_queues = [deque() for _ in range(2000)]
# 2d_queues = [[deque() for _ in range(4)] for _ in range(500)]
# for
