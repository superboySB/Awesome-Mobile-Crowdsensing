restore_dict = {
    'model_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                  "IPPOTrainer_crowdsim_SanFrancisco_dbd29_00000_0_2023-12-21_11-05-56/"
                  "checkpoint_029000/checkpoint-29000",
    'params_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                   "experiment_state-2023-12-21_11-05-56.json",
    'render': True
}

# new restore dict, small action_space.
restore_dict = {
    'model_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                  "IPPOTrainer_crowdsim_SanFrancisco_138d5_00000_0_2023-12-29_12-45-23/"
                  "checkpoint_019000/checkpoint-19000",
    'params_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                   "experiment_state-2023-12-29_12-45-22.json",
    'render': True
}
# fix points
restore_dict = {
    'model_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                  "IPPOTrainer_crowdsim_SanFrancisco_1eaa4_00000_0_2023-12-30_16-47-53/"
                  "checkpoint_010000/checkpoint-10000",
    'params_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                   "experiment_state-2023-12-30_16-47-53.json",
    'render': True
}
# fully random
restore_dict = {
    'model_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                  "IPPOTrainer_crowdsim_SanFrancisco_e74c0_00000_0_2023-12-31_19-01-09/"
                  "checkpoint_010000/checkpoint-10000",
    'params_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                   "experiment_state-2023-12-31_19-01-09.json",
    'render': True
}
# I forget what is this for...
restore_dict = {
    'model_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                  "IPPOTrainer_crowdsim_SanFrancisco_a4bb8_00000_0_2023-12-31_21-43-56/"
                  "checkpoint_040000/checkpoint-40000",
    'params_path': "/workspace/saved_data/marllib_results/ippo_mlp_SanFrancisco/"
                   "experiment_state-2023-12-31_21-43-56.json",
    'render': True
}

# IPPO updated, all points (536), working
# IPPOTrainer_crowdsim_SanFrancisco_a4d3f_00000_0_2024-01-03_12-14-52
# IPPO updated, less points (200)
# IPPOTrainer_crowdsim_SanFrancisco_e38ae_00000_0_2024-01-03_17-24-26
# IPPO updated, less points (200) with centralized
# IPPOTrainer_crowdsim_SanFrancisco_07298_00000_0_2024-01-03_22-26-05
# 200 points + centralized + dynamic zero shot
# IPPOTrainer_crowdsim_SanFrancisco_7667c_00000_0_2024-01-03_23-26-27
# 200 points + regular reward + dynamic zero shot
# IPPOTrainer_crowdsim_SanFrancisco_52966_00000_0_2024-01-04_10-24-01

# IPPO + CNN, brute force mean:
#         uuid = "882c5"
#         time_str = "2024-01-03_23-26-57"
#         checkpoint_num = 15000
# IPPO + CNN, flatten input to FC:
#         uuid = "a4592"
#         time_str = "2024-01-04_16-24-13"
#         checkpoint_num = 15000
# IPPO fix_target FC
# IPPOTrainer_crowdsim_SanFrancisco_c8dc7_00000_0_2024-01-04_22-51-47
# IPPO fix_target CNN
#         uuid = "3a0b0"
#         time_str = "2024-01-04_22-54-57"

# greedy interval=30
uuid = "67ba0"
time_str = "2024-01-27_22-57-35"
checkpoint_num = 40000
# oracle old version
uuid = "eca47"
time_str = "2024-01-19_19-57-31"
checkpoint_num = 27000
# oracle traffic ppo version
uuid = "9a55f"
time_str = "2024-01-27_16-18-08"
checkpoint_num = 27000
# bootstrap reward in Chengdu
uuid = "87ab4"
time_str = "2024-02-01_16-11-41"
checkpoint_num = 2000
# switch bootstrap in Chengdu
uuid = "8dfe7"
time_str = "2024-02-01_16-11-51"
checkpoint_num = 2000
# NN，San Francisco, gen_interval=10
uuid = "4ab3b"
time_str = "2024-01-30_20-22-54"
checkpoint_num = 30000
backup_str = "2024-01-30_20-22-53"
# Chengdu new dataset, greedy
uuid = "836db"
time_str = "2024-02-02_18-47-52"
checkpoint_num = 4000
backup_str = "2024-02-02_18-47-51"
# Chengdu new dataset, greedy
uuid = "6ffc8"
time_str = "2024-02-02_21-46-16"
checkpoint_num = 8000
# Chengdu new dataset, NN
uuid = "6cc0f"
time_str = "2024-02-02_21-46-11"
checkpoint_num = 8000
# Chengdu, multi-time dataset, greedy
uuid = "aae7e"
time_str = "2024-02-03_14-44-24"
checkpoint_num = 10000
# Chengdu, multi-time dataset, one agent one task, NN
uuid = "8b20a"
time_str = "2024-02-03_20-34-16"
checkpoint_num = 10000
# Chengdu, multi-time dataset, one agent one task, NN new
uuid = "f06d0"
time_str = "2024-02-05_02-26-40"
checkpoint_num = 10000
# Chegndu, multi-time dataset, one agent multi task, greedy
uuid = "bb749"
time_str = "2024-02-05_16-44-11"
checkpoint_num = 7000
backup_str = "2024-02-05_15-44-10"
# San, emergency_queue_length=1, gen_interval=10
uuid = "adbe6"
time_str = "2024-02-13_15-37-03"
checkpoint_num = 15000
backup_str = "2024-02-13_15-37-02"
# San, queue_length=5, policy gradient.
uuid = "9abdc"
time_str = "2024-02-14_20-07-21"
checkpoint_num = 15000
backup_str = "2024-02-14_20-07-20"
# San, queue_length=3, policy gradient new.
uuid = "c58aa"
time_str = "2024-02-16_13-10-59"
checkpoint_num = 2000
backup_str = ""
# Chengdu, queue_length=5, mock_RL greedy, new best.
uuid = "a3cba"
time_str = "2024-02-17_20-25-30"
checkpoint_num = 7000
backup_str = ""
# San, queue_length=1, RL, failing
uuid = "35fef"
time_str = "2024-02-17_20-22-27"
checkpoint_num = 6000
backup_str = "2024-02-17_20-22-26"
# San, queue_length=1, mock_RL greedy
uuid = "a01e1"
time_str = "2024-02-17_20-25-25"
checkpoint_num = 6000
backup_str = "2024-02-17_20-25-24"
# San, queue_length=3, RL greedy, large reward
uuid = "4e133"
checkpoint_num = 4000
time_str = "2024-02-20_01-21-23"
backup_str = ""
"""
San, queue_length=1, RL greedy, large reward
3fff7 5000 2024-02-20_01-21-00 2024-02-20_01-21-00
San, queue_length=3, RL greedy, large reward
4e133 4000 2024-0b2-20_01-21-23 2024-02-20_01-21-23
Chengdu, queue_length=3, RL greedy, large reward
bb7c6 4000 2024-02-20_02-07-25 2024-02-20_02-07-24
San, queue_length=1, RL, large reward
4ed6e 5000 2024-02-20_17-42-06 2024-02-20_17-42-05
San, optim_greedy, large_reward, with end time
69d9c 7000 2024-02-22_16-28-54 2024-02-22_16-28-53
Chengdu, optim_greedy, large_reward, with end time
42045 7000 2024-02-22_14-33-15 2024-02-22_14-33-15
San, with attention
07563 10000 2024-02-27_19-40-38 2024-02-27_19-40-38
"""
