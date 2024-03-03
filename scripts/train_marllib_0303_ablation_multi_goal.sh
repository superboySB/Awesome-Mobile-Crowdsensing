#!/bin/bash
exp_name='66_ablation'
# not completely edited.
session_name=$exp_name
cards=(1 1 2 2 3 4 5 6)
card_num=${#cards[@]}
dry_run=false
# Process command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=true

            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done
# remove NN share_policy all
trains=(
  "--dataset SanFrancisco --selector_type RL --rl_gamma 0 --emergency_queue_length 5 --gen_interval 10 --sibling_rivalry"
  "--dataset SanFrancisco --selector_type RL --rl_gamma 0 --emergency_queue_length 5 --gen_interval 10"
  "--dataset SanFrancisco --selector_type greedy --emergency_queue_length 5 --gen_interval 10 --sibling_rivalry"
  "--dataset SanFrancisco --selector_type greedy --emergency_queue_length 5 --gen_interval 10"
  "--dataset Chengdu --selector_type RL --rl_gamma 0 --emergency_queue_length 4 --gen_interval 10 --sibling_rivalry"
  "--dataset Chengdu --selector_type RL --rl_gamma 0 --emergency_queue_length 4 --gen_interval 10"
  "--dataset Chengdu --selector_type greedy --emergency_queue_length 4 --gen_interval 10 --sibling_rivalry"
  "--dataset Chengdu --selector_type greedy --emergency_queue_length 4 --gen_interval 10"
)


train_num=${#trains[@]}
if [ "$dry_run" = "false" ]
then
    echo "Start running expr $exp_name"
    echo "Will Recreate Session $session_name"
    # Prompt the user for confirmation
    read -rp "Do you want to proceed? (y/n): " choice

    # Check the user's choice
    if [ "$choice" = "y" ] || [ "$choice" = "Y" ] || [ "$choice" = "" ] ; then
        echo "Proceeding with operations."
        tmux kill-session -t $session_name;
        tmux new-session -d -s ${session_name};
        tmux set -g mouse on;
        tmux split-window -h;tmux split-window -h;tmux split-window -h;
        tmux select-pane -t 0;tmux split-window -h;tmux split-window -h;
        tmux split-window -h;tmux select-layout tiled;tmux select-pane -l;
        tmux split-window -h;tmux split-window -h;tmux select-layout tiled;
	      tmux select-pane -l;tmux split-window -h;tmux split-window -h;
	      tmux split-window -h;tmux select-layout tiled;
    fi
fi
for ((i = 0; i < train_num; i++)); do
  if [ "$dry_run" = "false" ] && [ "$choice" != "n" ]
  then
      tmux send-keys -t $session_name:0."$i" 'cd /workspace/Awesome-Mobile-Crowdsensing' Enter;
      tmux send-keys -t $session_name:0."$i" 'conda activate mcs' Enter;
  fi
  card_id=$((i % card_num))
  # shellcheck disable=SC2004
  # if want to add $PATH, remember to add / before $
  command="python warp_drive/marllib_warpdrive_run.py --track --core_arch crowdsim_net --dynamic_zero_shot\
  --num_drones 4 --num_cars 0 --group auto_allocation --algo trafficppo --share_policy all --switch_step 60000000\
  --gpu_id ${cards[card_id]} ${trains[i]} --use_2d_state --tag ablation --look_ahead\
  --intrinsic_mode scaled_dis_aoi --reward_mode greedy --prioritized_buffer --emergency_threshold 20 --buffer_in_obs"
  echo "$command"
  if [ "$dry_run" = "false" ] && [ "$choice" != "n" ]
  then
      tmux send-keys -t $session_name:0."$i" "$command" Enter;
      echo "exp ${i} runs successfully"
      sleep 5
  fi
done
if [ "$dry_run" = "false" ] && [ "$choice" != "n" ]
then
  tmux attach-session -t $session_name
else
  echo "Operations not executed."
  # Add any cleanup or exit code here if needed
fi
