import torch

assert torch.cuda.device_count() > 0, "This notebook needs a GPU to run!"

import numpy as np
from warp_drive.cuda_managers.pycuda_function_manager import PyCUDAFunctionManager, PyCUDASampler
from warp_drive.cuda_managers.pycuda_data_manager import PyCUDADataManager
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.common import get_project_root

_MAIN_FILEPATH = f"{get_project_root()}/warp_drive/cuda_includes"
_CUBIN_FILEPATH = f"{get_project_root()}/warp_drive/cuda_bin"
_ACTIONS = Constants.ACTIONS

# Set logger level e.g., DEBUG, INFO, WARNING, ERROR
import logging

logging.getLogger().setLevel(logging.INFO)

# Initialize PyCUDASampler
cuda_data_manager = PyCUDADataManager(num_agents=5, episode_length=10, num_envs=2)
cuda_function_manager = PyCUDAFunctionManager(
    num_agents=cuda_data_manager.meta_info("n_agents"),
    num_envs=cuda_data_manager.meta_info("n_envs"),
)

main_example_file = f"{_MAIN_FILEPATH}/test_build.cu"
bin_example_file = f"{_CUBIN_FILEPATH}/test_build.fatbin"

cuda_function_manager._compile(main_file=main_example_file, 
                               cubin_file=bin_example_file)

cuda_function_manager.load_cuda_from_binary_file(
    bin_example_file, default_functions_included=True
)
cuda_sampler = PyCUDASampler(function_manager=cuda_function_manager)
cuda_sampler.init_random(seed=None)


data_feed = DataFeed()
data_feed.add_data(name=f"{_ACTIONS}_a", data=[[[0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]]])
cuda_data_manager.push_data_to_device(data_feed, torch_accessible=True)
assert cuda_data_manager.is_data_on_device_via_torch(f"{_ACTIONS}_a")

cuda_sampler.register_actions(
    cuda_data_manager, action_name=f"{_ACTIONS}_a", num_actions=3
)

distribution = np.array(
    [
        [
            [0.333, 0.333, 0.333],
            [0.2, 0.5, 0.3],
            [0.95, 0.02, 0.03],
            [0.02, 0.95, 0.03],
            [0.02, 0.03, 0.95],
        ],
        [
            [0.1, 0.7, 0.2],
            [0.7, 0.2, 0.1],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ],
    ]
)
distribution = torch.from_numpy(distribution).float().cuda()


# Run 10000 times to collect statistics
actions_batch = torch.from_numpy(np.empty((10000, 2, 5), dtype=np.int32)).cuda()

for i in range(10000):
    cuda_sampler.sample(cuda_data_manager, distribution, action_name=f"{_ACTIONS}_a")
    actions_batch[i] = cuda_data_manager.data_on_device_via_torch(f"{_ACTIONS}_a")[:, :, 0]
actions_batch_host = actions_batch.cpu().numpy()
actions_env_0 = actions_batch_host[:, 0]
actions_env_1 = actions_batch_host[:, 1]
print(
    "Sampled actions distribution versus the given distribution (in bracket) for env 0: \n"
)
for agent_id in range(5):
    print(
        f"Sampled action distribution for agent_id: {agent_id}:\n"
        f"{(actions_env_0[:, agent_id] == 0).sum() / 10000.0}({distribution[0, agent_id, 0]}), \n"
        f"{(actions_env_0[:, agent_id] == 1).sum() / 10000.0}({distribution[0, agent_id, 1]}), \n"
        f"{(actions_env_0[:, agent_id] == 2).sum() / 10000.0}({distribution[0, agent_id, 2]})  \n"
    )
print(
    "Sampled actions distribution versus the given distribution (in bracket) for env 1: "
)

for agent_id in range(5):
    print(
        f"Sampled action distribution for agent_id: {agent_id}:\n"
        f"{(actions_env_1[:, agent_id] == 0).sum() / 10000.0}({distribution[1, agent_id, 0]}), \n"
        f"{(actions_env_1[:, agent_id] == 1).sum() / 10000.0}({distribution[1, agent_id, 1]}), \n"
        f"{(actions_env_1[:, agent_id] == 2).sum() / 10000.0}({distribution[1, agent_id, 2]})  \n"
    )


data_feed = DataFeed()
data_feed.add_data(name=f"{_ACTIONS}_b", data=[[[0], [0], [0], [0], [0]], [[0], [0], [0], [0], [0]]])
cuda_data_manager.push_data_to_device(data_feed, torch_accessible=True)
assert cuda_data_manager.is_data_on_device_via_torch(f"{_ACTIONS}_b")
cuda_sampler.register_actions(
    cuda_data_manager, action_name=f"{_ACTIONS}_b", num_actions=4
)
distribution = np.array(
    [
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ],
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ],
    ]
)
distribution = torch.from_numpy(distribution).float().cuda()
# Run 10000 times to collect statistics.
actions_batch = torch.from_numpy(np.empty((10000, 2, 5), dtype=np.int32)).cuda()

for i in range(10000):
    cuda_sampler.sample(cuda_data_manager, distribution, action_name=f"{_ACTIONS}_b")
    actions_batch[i] = cuda_data_manager.data_on_device_via_torch(f"{_ACTIONS}_b")[:, :, 0]
actions_batch_host = actions_batch.cpu().numpy()
print(actions_batch_host)
print(actions_batch_host.std(axis=2).mean(axis=0))


actions_batch_numpy = np.empty((10000, 2, 5), dtype=np.int32)
for i in range(10000):
    actions_batch_numpy[i, 0, :] = np.random.choice(4, 5)
    actions_batch_numpy[i, 1, :] = np.random.choice(4, 5)
print(actions_batch_numpy.std(axis=2).mean(axis=0))


from torch.distributions import Categorical
distribution = np.array(
    [
        [
            [0.333, 0.333, 0.333],
            [0.2, 0.5, 0.3],
            [0.95, 0.02, 0.03],
            [0.02, 0.95, 0.03],
            [0.02, 0.03, 0.95],
        ],
        [
            [0.1, 0.7, 0.2],
            [0.7, 0.2, 0.1],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ],
    ]
)
distribution = torch.from_numpy(distribution).float().cuda()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(1000):
    cuda_sampler.sample(cuda_data_manager, distribution, action_name=f"{_ACTIONS}_a")
end_event.record()
torch.cuda.synchronize()
print(f"time elapsed: {start_event.elapsed_time(end_event)} ms")
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(1000):
    Categorical(distribution).sample()
end_event.record()
torch.cuda.synchronize()
print(f"time elapsed: {start_event.elapsed_time(end_event)} ms")
