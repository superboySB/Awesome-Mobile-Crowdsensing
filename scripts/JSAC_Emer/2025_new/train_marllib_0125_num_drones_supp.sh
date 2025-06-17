#!/bin/bash
exp_name='77_num_drones_supp'
# not completely edited.
session_name=$exp_name
cards=(0 1 2 3)
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
  "--algo trafficppo --dataset SanFrancisco --num_drones 20 --intrinsic_mode scaled_dis_aoi --core_arch crowdsim_net"
  "--algo trafficppo --dataset Chengdu --num_drones 20 --intrinsic_mode scaled_dis_aoi --core_arch crowdsim_net"
  "--algo trafficppo --dataset SanFrancisco --num_drones 20 --intrinsic_mode aim --core_arch crowdsim_net"
  "--algo trafficppo --dataset Chengdu --num_drones 20 --intrinsic_mode aim --core_arch crowdsim_net"
  "--algo happo --dataset SanFrancisco --num_drones 20 --core_arch mlp"
  "--algo happo --dataset Chengdu --num_drones 20 --core_arch mlp"
  "--algo ippo --dataset SanFrancisco --num_drones 20 --core_arch attention"
  "--algo ippo --dataset Chengdu --num_drones 20 --core_arch attention"
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
  command="python warp_drive/marllib_warpdrive_run.py --track --dynamic_zero_shot\
  --num_cars 0 --group 2025_resubmit --share_policy all --switch_step 60000000\
  --gpu_id ${cards[card_id]} ${trains[i]} --use_2d_state --look_ahead --with_programming_optimization\
  --emergency_threshold 20 --blur_requirement 5 --selector_type RL --use_random --prioritized_buffer\
  --gen_interval 6 --cut_points 300 --surveillance_threshold 35 --tag change_num_drones\
  --display_tags dataset num_drones --reward_mode original --rl_gamma 0\
  --emergency_queue_length 5 --NN_buffer --sibling_rivalry --alpha 0.3"
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
# End of file