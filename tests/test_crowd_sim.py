import unittest

import numpy as np

from envs.crowd_sim.crowd_sim import CrowdSim
from run_configs.mcs_configs_python import run_config
from datasets.Sanfrancisco.env_config import BaseEnvConfig


class CrowdSimTest(unittest.TestCase):
    def setUp(self):
        run_config['env_args']['env_config'] = BaseEnvConfig
        self.mock_env = CrowdSim(**run_config['env_args'])

    def test_generate_observation_and_update_state(self):
        self.mock_env.use_2d_state = False
        obs = self.mock_env.generate_observation_and_update_state()
        assert isinstance(obs, dict)
        for item in obs.values():
            assert isinstance(item, np.ndarray) and len(item.shape) == 1
