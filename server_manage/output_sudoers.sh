#!/bin/bash

# List of usernames
users=("liuchi" "hanrui" "lishuang" "gaoguangyu" "liguozheng")

# Define the programs for which you want to generate sudo permissions
programs=("cat" "docker" "tail" "apt" "apt-get" "grep" "less" "find")

# Check if an IP address is provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <IP_ADDRESS>"
  exit 1
fi

# The IP address is the first command-line argument
ip_address=$1

# Function to find the full path of a program using SSH and which
get_program_path() {
  ssh admin@"$ip_address" "which $1 2>/dev/null"
}

# Additional docker permissions
docker_permissions="!/usr/bin/docker exec -it mcs /bin/*, \
!/usr/bin/docker exec -it mcs_new /bin/*, \
!/usr/bin/docker stop mcs, \
!/usr/bin/docker stop mcs_new, \
!/usr/bin/docker rm -f mcs, \
!/usr/bin/docker rm -f mcs_new"

# Iterate over each username
for username in "${users[@]}"; do
  # Initialize sudo command permissions string
  sudo_cmd="$username ALL=(ALL) NOPASSWD:"

  # Iterate over each program to find its full path and append it to the sudo command
  for program in "${programs[@]}"; do
    program_path=$(get_program_path "$program")
    if [ -n "$program_path" ]; then
      sudo_cmd+=" $program_path,"
    fi
  done

  # Append docker permissions
  sudo_cmd+=" $docker_permissions"

  # Output the generated sudo permissions text
  echo "$sudo_cmd"
done
