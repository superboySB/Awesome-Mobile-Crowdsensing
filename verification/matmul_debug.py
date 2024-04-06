import numpy as np
import os

parent_path = "/workspace/saved_data"
# read A_inv and v from txt
A_inv = np.loadtxt(os.path.join(parent_path, "A_inv.txt"))
v = np.loadtxt(os.path.join(parent_path, "v.txt"))[:, None]
term = np.matmul(np.matmul(np.matmul(A_inv, v), v.T), A_inv)
print(term)
