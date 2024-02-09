from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np

a = SampleBatch({"obs": np.random.rand(10, 4), "new_obs": np.random.rand(10, 4)})
a['actions'] = np.random.rand(6, 2)
