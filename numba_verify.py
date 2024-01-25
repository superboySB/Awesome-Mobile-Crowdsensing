import random
from numba import njit
import numpy as np


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
    return array


@njit
def return_dict():
    return {"test": 2}


if __name__ == '__main__':
    test_array = np.random.rand(5, 9)
    in_place = np.ones((5, 9))
    modify(in_place)
    print(in_place)
    a = return_numpy_array(test_array).reshape(-1, 3)
    print(a)
    b = return_dict()
    print(b)
