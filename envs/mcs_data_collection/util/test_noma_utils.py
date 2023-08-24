from unittest import TestCase
from noma_utils import compute_capacity_G2A
import numpy as np
import matplotlib.pyplot as plt

size = 2000


class TestCommModel(TestCase):

    noma_config = {
        'noise0_density': 5e-20,
        'bandwidth_subchannel': 20e6 / 5,
        'p_uav': 4,  # w, 也即34.7dbm
        'p_poi': 0.1,
        'aA': 2,
        'aG': 3,
        'nLoS': 0,  # dB, 也即1w
        'nNLoS': -20,  # dB, 也即0.01w
        'uav_init_height': 50,
        'psi': 9.6,
        'beta': 0.16,
    }

    def test_compute_capacity_g2a(self):
        x, y = np.meshgrid(np.arange(0, size, dtype='f'), np.arange(0, size, dtype='f'))
        z = np.arange(0, size * size, dtype='f')
        for index, (main_poi_dis, interfere) in enumerate(zip(x.flat, y.flat)):
            sinr_G2A, R_G2A = compute_capacity_G2A(self.noma_config, main_poi_dis, interfere)
            z[index] = R_G2A
        ax = plt.axes(projection="3d")
        ax.scatter3D(x.flat, y.flat, z, c=z, cmap='cividis')

