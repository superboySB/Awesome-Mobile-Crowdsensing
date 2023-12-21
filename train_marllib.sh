#!/bin/bash

dataset_name='SanFrancisco'
exp_name='WARP'_$dataset_name
session_name=$exp_name'_2'
cards=(3 2 3 3 2 0 0 0)
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
  "--algo ippo"
  "--algo itrpo"
  "--algo matrpo"
  "--algo mappo"
  "--algo mappo --centralized"
  "--algo maa2c --num_workers 1"
  "--algo happo --num_workers 1"
  "--algo hatrpo --num_workers 1"
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
        echo "Proceeding with the operation."
        tmux kill-session -t $session_name;
        tmux new-session -d -s ${session_name};
        tmux set -g mouse on;
        tmux split-window -h;tmux split-window -h;tmux split-window -h;
        tmux select-pane -t 0;tmux split-window -h;tmux split-window -h;
        tmux split-window -h;tmux select-layout tiled;tmux select-pane -l;
        tmux split-window -h;tmux split-window -h;tmux select-layout tiled;
    fi
fi
for ((i = 0; i < train_num; i++)); do
  if [ "$dry_run" = "false" ]
  then
      tmux send-keys -t $session_name:0."$i" 'cd /workspace/Awesome-Mobile-Crowdsensing' Enter;
      tmux send-keys -t $session_name:0."$i" 'conda activate mcs' Enter;
  fi
  card_id=$((i % card_num))
  # shellcheck disable=SC2004
  # if want to add $PATH, remember to add / before $
  command="python warp_drive/marllib_crowdsim_run.py --track --dataset '$dataset_name'\
  --gpu_id ${cards[card_id]} ${trains[i]}"
  echo "$command"
  if [ "$dry_run" = "false" ]
  then
      tmux send-keys -t $session_name:0."$i" "$command" Enter;
      echo "exp ${i} runs successfully"
      sleep 5
  fi
done

if [ "$dry_run" = "false" ]
then
  tmux attach-session -t $session_name
else
  echo "Operation not executed."
  # Add any cleanup or exit code here if needed
fi
