# 参考教程：https://github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-1.a-warp_drive_basics.ipynb

import numpy as np
from timeit import Timer

from warp_drive.cuda_managers.pycuda_data_manager import PyCUDADataManager
from warp_drive.cuda_managers.pycuda_function_manager import PyCUDAFunctionManager
from warp_drive.utils.data_feed import DataFeed

import logging

logging.getLogger().setLevel(logging.INFO)

num_agents = 4
num_envs = 5
episode_length = 5

cuda_data_manager = PyCUDADataManager(num_agents, num_envs, episode_length=episode_length)

random_data = np.random.rand(num_envs, num_agents)

print("准备一份数据：",random_data)

data_feed = DataFeed()
data_feed.add_data(
    name="random_data",
    data=random_data,
    save_copy_and_apply_at_reset=False,
    log_data_across_episode=False,
)

print("上传这个数据（不可以被torch访问）:",data_feed)

cuda_data_manager.push_data_to_device(data_feed)

data_fetched_from_device = cuda_data_manager.pull_data_from_device("random_data")

print("再把数据从显卡拉回来:",data_fetched_from_device)

tensor_feed = DataFeed()
tensor_feed.add_data(name="random_tensor", data=random_data)

cuda_data_manager.push_data_to_device(tensor_feed, torch_accessible=True)

tensor_on_device = cuda_data_manager.data_on_device_via_torch("random_tensor")

print("机器上可以被torch访问的tensor:",tensor_on_device)

large_array = np.random.rand(1000, 1000)

data_feed = DataFeed()
data_feed.add_data(
    name="large_array",
    data=large_array,
)
cuda_data_manager.push_data_to_device(data_feed, torch_accessible=False)

print("比较用时：torch_accessible=False")

print(Timer(lambda: cuda_data_manager.pull_data_from_device("large_array")).timeit(
    number=1000
))

data_feed = DataFeed()
data_feed.add_data(
    name="large_array_torch",
    data=large_array,
)
cuda_data_manager.push_data_to_device(data_feed, torch_accessible=True)

print("比较用时：torch_accessible=True")

print(Timer(lambda: cuda_data_manager.data_on_device_via_torch("random_tensor")).timeit(1000))

print("You can see the time for accessing torch tensors on the GPU is negligible compared to data arrays!")


cuda_function_manager = PyCUDAFunctionManager(
    num_agents=cuda_data_manager.meta_info("n_agents"),
    num_envs=cuda_data_manager.meta_info("n_envs"),
)

source_code = """
// A function to demonstrate how to manipulate data on the GPU.
// This function increments each the random data array we pushed to the GPU before.
// Each index corresponding to (env_id, agent_id) in the array is incremented by "agent_id + env_id".
// Everything inside the if() loop runs in parallel for each agent and environment.
//
extern "C"{
    __global__ void cuda_increment(                               
            float* data,                                  
            int num_agents                                       
    )                                                            
    {                                                            
        int env_id = blockIdx.x;                                 
        int agent_id = threadIdx.x;                             
        if (agent_id < num_agents){                              
            int array_index = env_id * num_agents + agent_id;
            int increment = env_id + agent_id;
            data[array_index] += increment;
        }                                                            
    }   
}
"""

cuda_function_manager.load_cuda_from_source_code(
    source_code, default_functions_included=False
)
cuda_function_manager.initialize_functions(["cuda_increment"])

increment_function = cuda_function_manager.get_function("cuda_increment")

cuda_data_manager.push_data_to_device(
    {
        "num_agents": {
            "data": num_agents,
            "attributes": {
                "save_copy_and_apply_at_reset": False,
                "log_data_across_episode": False,
            },
        }
    }
)
increment_function(
    cuda_data_manager.device_data("random_data"),
    cuda_data_manager.device_data("num_agents"),
    block=cuda_function_manager.block,
    grid=cuda_function_manager.grid,
)

print("Below is the original (random) data that we pushed to the GPU:", random_data)

print("and here's the incremented data:", cuda_data_manager.pull_data_from_device("random_data"))

print("The differences are below:", cuda_data_manager.pull_data_from_device("random_data") - random_data)

increment_function(
    cuda_data_manager.device_data("random_data"),
    cuda_data_manager.device_data("num_agents"),
    block=cuda_function_manager.block,
    grid=cuda_function_manager.grid,
)

print("Again! The differences are below:", cuda_data_manager.pull_data_from_device("random_data") - random_data)


def push_random_data_and_increment_timer(
    num_runs=1,
    num_envs=2,
    num_agents=3,
    source_code=None,
    episode_length=100,
):

    assert source_code is not None

    # Initialize the CUDA data manager
    cuda_data_manager = PyCUDADataManager(
        num_agents=num_agents, num_envs=num_envs, episode_length=episode_length
    )

    # Initialize the CUDA function manager
    cuda_function_manager = PyCUDAFunctionManager(
        num_agents=cuda_data_manager.meta_info("n_agents"),
        num_envs=cuda_data_manager.meta_info("n_envs"),
    )

    # Load source code and initialize function
    cuda_function_manager.load_cuda_from_source_code(
        source_code, default_functions_included=False
    )
    cuda_function_manager.initialize_functions(["cuda_increment"])
    increment_function = cuda_function_manager.get_function("cuda_increment")

    def push_random_data(num_agents, num_envs):
        # Create random data
        random_data = np.random.rand(num_envs, num_agents)

        # Push data from host to device
        data_feed = DataFeed()
        data_feed.add_data(
            name="random_data",
            data=random_data,
        )
        data_feed.add_data(name="num_agents", data=num_agents)
        cuda_data_manager.push_data_to_device(data_feed)

    def increment_data():
        increment_function(
            cuda_data_manager.device_data("random_data"),
            cuda_data_manager.device_data("num_agents"),
            block=cuda_function_manager.block,
            grid=cuda_function_manager.grid,
        )

    # One-time data push
    data_push_time = Timer(lambda: push_random_data(num_agents, num_envs)).timeit(
        number=1
    )
    # Increment the arrays 'num_runs' times
    program_run_time = Timer(lambda: increment_data()).timeit(number=num_runs)

    return {"data push times": data_push_time, "code run time": program_run_time}


num_runs = 10000
times = {}

for scenario in [
    # (1, 1),
    # (1, 10),
    # (1, 100),
    # (10, 10),
    # (1, 1000),
    # (100, 100),
    (1000, 1000),
    (999, 1001)
]:
    num_envs, num_agents = scenario
    times.update(
        {
            f"envs={num_envs}, agents={num_agents}": push_random_data_and_increment_timer(
                num_runs, num_envs, num_agents, source_code
            )
        }
    )

print(f"Times for {num_runs} function calls")
print("*" * 40)
for key, value in times.items():
    print(
        f"{key:30}: data push time: {value['data push times']:10.5}s,\t mean increment times: {value['code run time']:10.5}s"
    )