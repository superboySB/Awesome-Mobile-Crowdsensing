# generate a series of grid points, plot it on matplotlib
import matplotlib.pyplot as plt
import numpy as np

trajectory = np.random.randint(0, 100, (100, 2))
plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.show()
