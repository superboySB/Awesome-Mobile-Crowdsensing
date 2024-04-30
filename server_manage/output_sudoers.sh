#!/bin/bash

# List of usernames
users=("liuchi" "hanrui" "lishuang" "gaoguangyu" "liguozheng")

# Define the programs for which you want to generate sudo permissions
programs=("cat" "tail" "apt" "apt-get" "grep" "less" "find" "rsync" "mkdir")

# Check if an IP address is provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <IP_ADDRESS>"
  exit 1
fi

# The IP address is the first command-line argument
ip_address=$1

# Function to find the full paths of programs using SSH and which
get_program_paths() {
  local IFS=" " # Setting internal field separator to space for the command
  ssh admin@"$ip_address" "which ${programs[*]} 2>/dev/null" | while read -r line; do
    if [[ $line == *"not found"* ]]; then
      echo -n ""
    else
      echo -n "$line "
    fi
  done
}

# Function to get the Docker binary path
get_docker_path() {
  ssh admin@"$ip_address" "which docker 2>/dev/null"
}

get_program_path() {
  local program_name="$1"  # This captures the first argument passed to the function.
  ssh admin@"$ip_address" "which $program_name 2>/dev/null"
}


# Retrieve all program paths at once
program_paths=$(get_program_paths)
# Convert the program paths string to an array
read -ra program_paths_arr <<< "$program_paths"

# Retrieve Docker binary path
docker_path=$(get_docker_path)

# Define docker commands allowing all actions except stopping or removing specific containers
docker_commands=(
  "$docker_path"
  "$docker_path stop \*, ! $docker_path stop mcs, ! $docker_path stop mcs_new"
  "$docker_path rm -f \*, ! $docker_path rm -f mcs, ! $docker_path rm -f mcs_new"
)

nvidia_path=$(get_program_path nvidia-smi)

nvidia_commands=(
  "$nvidia_path -pm 0"
  "$nvidia_path -pm 1"
)

# Iterate over each username
for username in "${users[@]}"; do
  # Initialize sudo command permissions string
  sudo_cmd="$username ALL=(ALL) NOPASSWD:"

  # Append found program paths to the sudo command
  for program_path in "${program_paths_arr[@]}"; do
    if [ -n "$program_path" ]; then
      sudo_cmd+=" $program_path,"
    fi
  done

  # Append docker permissions
  for docker_command in "${docker_commands[@]}"; do
    sudo_cmd+=" $docker_command,"
  done
# Append nvidia permissions
  for nvidia_command in "${nvidia_commands[@]}"; do
    sudo_cmd+=" $nvidia_command,"
  done
  # Add permission to read Docker config without a password
  sudo_cmd+=" /bin/cat /home/$username/.docker/config.json,"

  # Trim the last comma
  sudo_cmd=${sudo_cmd%,}

  # Output the generated sudo permissions text
  echo "$sudo_cmd"
done
