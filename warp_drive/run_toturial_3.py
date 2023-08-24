import torch

assert torch.cuda.device_count() > 0, "This notebook needs a GPU to run!"


import numpy as np
from warp_drive.cuda_managers.pycuda_data_manager import PyCUDADataManager
from warp_drive.cuda_managers.pycuda_function_manager import (
    PyCUDAFunctionManager,
    PyCUDALogController,
    PyCUDAEnvironmentReset,
)
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.common import get_project_root

_MAIN_FILEPATH = f"{get_project_root()}/warp_drive/cuda_includes"
_CUBIN_FILEPATH = f"{get_project_root()}/warp_drive/cuda_bin"
_ACTIONS = Constants.ACTIONS

# Set logger level e.g., DEBUG, INFO, WARNING, ERROR
import logging

logging.getLogger().setLevel(logging.INFO)


cuda_data_manager = PyCUDADataManager(num_agents=5, num_envs=2, episode_length=2)
cuda_function_manager = PyCUDAFunctionManager(
    num_agents=cuda_data_manager.meta_info("n_agents"),
    num_envs=cuda_data_manager.meta_info("n_envs"),
)

main_example_file = f"{_MAIN_FILEPATH}/test_build.cu"
bin_example_file = f"{_CUBIN_FILEPATH}/test_build.fatbin"

cuda_function_manager._compile(main_file=main_example_file, 
                               cubin_file=bin_example_file)


cuda_function_manager.load_cuda_from_binary_file(bin_example_file)
cuda_env_resetter = PyCUDAEnvironmentReset(function_manager=cuda_function_manager)
cuda_env_logger = PyCUDALogController(function_manager=cuda_function_manager)


cuda_function_manager.initialize_functions(["testkernel"])


def cuda_dummy_step(
    function_manager: PyCUDAFunctionManager,
    data_manager: PyCUDADataManager,
    env_resetter: PyCUDAEnvironmentReset,
    target: int,
    step: int,
):

    env_resetter.reset_when_done(data_manager)

    step = np.int32(step)
    target = np.int32(target)
    test_step = function_manager.get_function("testkernel")
    test_step(
        data_manager.device_data("X"),
        data_manager.device_data("Y"),
        data_manager.device_data("_done_"),
        data_manager.device_data(f"{_ACTIONS}"),
        data_manager.device_data("multiplier"),
        target,
        step,
        data_manager.meta_info("episode_length"),
        block=function_manager.block,
        grid=function_manager.grid,
    )


data = DataFeed()
data.add_data(
    name="X",
    data=[[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
    save_copy_and_apply_at_reset=True,
    log_data_across_episode=True,
)

data.add_data(
    name="Y",
    data=np.array([[6, 7, 8, 9, 10], [1, 2, 3, 4, 5]]),
    save_copy_and_apply_at_reset=True,
    log_data_across_episode=True,
)
data.add_data(name="multiplier", data=2.0)

tensor = DataFeed()
tensor.add_data(
    name=f"{_ACTIONS}",
    data=[
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ],
)

cuda_data_manager.push_data_to_device(data)
cuda_data_manager.push_data_to_device(tensor, torch_accessible=True)

assert cuda_data_manager.is_data_on_device("X")
assert cuda_data_manager.is_data_on_device("Y")
assert cuda_data_manager.is_data_on_device_via_torch(f"{_ACTIONS}")

# t = 0 is reserved for the initial state.
cuda_env_logger.reset_log(data_manager=cuda_data_manager, env_id=0) # TODO: env id可以修改，默认跟踪env 0

for t in range(1, cuda_data_manager.meta_info("episode_length") + 1):
    cuda_dummy_step(
        function_manager=cuda_function_manager,
        data_manager=cuda_data_manager,
        env_resetter=cuda_env_resetter,
        target=100,
        step=t,
    )
    cuda_env_logger.update_log(data_manager=cuda_data_manager, step=t)

dense_log = cuda_env_logger.fetch_log(data_manager=cuda_data_manager, names=["X", "Y"])
print("fetch_log:",dense_log)

# Test after two steps that the log buffers for X and Y log are updating.
X_update = dense_log["X_for_log"]
Y_update = dense_log["Y_for_log"]

assert abs(X_update[1].mean() - 0.15) < 1e-5
assert abs(X_update[2].mean() - 0.075) < 1e-5
assert Y_update[1].mean() == 16
assert Y_update[2].mean() == 32

# Right now, the reset functions have not been activated.
# The done flags should be all True now.

done = cuda_data_manager.pull_data_from_device("_done_")
print(f"The done array = {done}")


cuda_env_resetter.reset_when_done(data_manager=cuda_data_manager)

done = cuda_data_manager.pull_data_from_device("_done_")
assert done[0] == 0
assert done[1] == 0

X_after_reset = cuda_data_manager.pull_data_from_device("X")
Y_after_reset = cuda_data_manager.pull_data_from_device("Y")
# the 0th dim is env
assert abs(X_after_reset[0].mean() - 0.3) < 1e-5
assert abs(X_after_reset[1].mean() - 0.8) < 1e-5
assert Y_after_reset[0].mean() == 8
assert Y_after_reset[1].mean() == 3