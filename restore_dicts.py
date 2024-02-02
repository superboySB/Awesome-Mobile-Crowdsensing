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
