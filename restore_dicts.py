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

# it seems all experiments are delayed by 3 seconds.
