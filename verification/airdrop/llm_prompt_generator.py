import os
import shutil
from util_misc import file_to_string

task_file = f'parachute_env_test.py'
task_obs_file = f'parachute_env_obs.py'
shutil.copy(task_obs_file, f"env_init_obs.py")
task_code_string = file_to_string(task_file)
task_obs_code_string = file_to_string(task_obs_file)
# Loading all text prompts
prompt_dir = f'prompts'
task_description = file_to_string(f'{prompt_dir}/task_description.txt')
initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')

execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
print(initial_system)
print(initial_user)
