import collections
import random
from numba import njit
import numpy as np
from numba.typed import List


@njit
def return_numpy_array(test_array: np.ndarray):
    for i, array in enumerate(test_array):
        print(test_array[i])
    array_list = []
    for i in range(5):
        # test_tile = np.tile(test_array, (1, 1))
        length = np.random.randint(1, 10)
        array_list.append(np.random.rand(3, length))
    if len(array_list) > 0:
        result = []
        for arr in array_list:
            result.extend(arr.ravel())
        return np.array(result)
    else:
        return None


@njit
def modify(array):
    array[0] = 2
    array[..., 0] = 0
    return array


@njit
def try_queue(c: collections.deque):
    c.append('a')
    c.popleft()
    return c

@njit
def return_dict():
    return {"test": 2}


@njit
def nested_list(a):
    a[0].append(8)
    return a


@njit
def try_dict(b):
    b[0].pop(1)
    b[1].append(2)
    return b

if __name__ == '__main__':
    test_array = np.random.rand(5, 9)
    in_place = np.ones((5, 9))
    modify(in_place)
    print(in_place)
    try_queue(collections.deque(['a', 'b', 'c']))
    a = return_numpy_array(np.array([])).reshape(-1, 3)
    list_of_list = [[1, 2, 3], [2, 3], [4, 5, 6, 7]]
    dict_of_list = {0: [1, 2, 3], 1: [2, 3], 2: [4, 5, 6, 7]}
    try_dict(dict_of_list)
    print(a)
    b = return_dict()
    print(b)
