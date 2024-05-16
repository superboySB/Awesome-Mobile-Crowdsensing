#!/bin/bash
exp_name='50_blur_requirement_happo'
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
  "--dataset SanFrancisco --blur_requirement 2"
  "--dataset SanFrancisco --blur_requirement 1"
  "--dataset SanFrancisco --blur_requirement 0.75"
  "--dataset SanFrancisco --blur_requirement 0.5"
  "--dataset SanFrancisco --blur_requirement 0.25"
  "--dataset Chengdu --blur_requirement 2"
  "--dataset Chengdu --blur_requirement 1"
  "--dataset Chengdu --blur_requirement 0.75"
  "--dataset Chengdu --blur_requirement 0.5"
  "--dataset Chengdu --blur_requirement 0.25"
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
  command="python warp_drive/marllib_warpdrive_run.py --track --core_arch mlp --dynamic_zero_shot\
  --num_cars 0 --num_drones 4 --group baseline --algo random --share_policy all --switch_step 60000000\
  --gpu_id ${cards[card_id]} ${trains[i]} --use_2d_state --look_ahead --with_programming_optimization\
  --emergency_threshold 20 --selector_type RL --use_random --prioritized_buffer\
  --gen_interval 10 --cut_points 300 --tag change_blur --surveillance_threshold 35\
  --display_tags dataset blur_requirement core_arch --reward_mode original --rl_gamma 0\
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