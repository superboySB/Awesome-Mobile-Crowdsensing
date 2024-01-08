#!/bin/bash

dataset_name='SanFrancisco'
exp_name='WARP'_$dataset_name
session_name=$exp_name
cards=(0 1 3)
card_num=${#cards[@]}
dry_run=false
# Process command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done
trains=(
    "--algo mappo"
    "--algo mappo --centralized"
    "--algo mappo --dynamic_zero_shot"
    "--algo mappo --dynamic_zero_shot --centralized"
    "--algo ippo"
    "--algo ippo --use_2d_state"
    "--algo ippo --centralized"
    "--algo ippo --use_2d_state --centralized"
    "--algo ippo --dynamic_zero_shot"
    "--algo ippo --use_2d_state --dynamic_zero_shot"
    "--algo ippo --centralized --dynamic_zero_shot"
    "--algo ippo --use_2d_state --centralized --dynamic_zero_shot"
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
  command="python warp_drive/marllib_warpdrive_run.py --track\
  --num_drones 4 --num_cars 0 --group baseline --dataset '$dataset_name'\
  --gpu_id ${cards[card_id]} ${trains[i]}"
  echo "$command"
  if [ "$dry_run" = "false" ] && [ "$choice" != "n" ]
  then
      tmux send-keys -t $session_name:0."$i" "$command" Enter;
      echo "exp ${i} runs successfully"
      sleep 3
  fi
done
if [ "$dry_run" = "false" ] && [ "$choice" != "n" ]
then
  tmux attach-session -t $session_name
else
  echo "Operations not executed."
  # Add any cleanup or exit code here if neededd
fi
