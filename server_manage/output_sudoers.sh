#!/bin/bash

# List of usernames
users=("liuchi" "hanrui" "lishuang" "gaoguangyu" "liguozheng")

#!/bin/bash

# Define the programs for which you want to generate sudo permissions
programs=("cat" "docker" "tail" "apt" "apt-get" "grep" "less" "find" "rsync" "mkdir")

# Check if an IP address is provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <IP_ADDRESS>"
  exit 1
fi

# The IP address is the first command-line argument
ip_address=$1

# Function to find the full paths of programs using SSH and which
# Now it handles multiple programs at once and parses output
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

# Retrieve all program paths at once
program_paths=$(get_program_paths "${programs[@]}")
# Convert the program paths string to an array
read -ra program_paths_arr <<< "$program_paths"

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

  # Append found program paths to the sudo command
  for program_path in "${program_paths_arr[@]}"; do
    if [ -n "$program_path" ]; then
      sudo_cmd+=" $program_path,"
    fi
  done

  # Append docker permissions
  sudo_cmd+=" $docker_permissions"

  # Output the generated sudo permissions text
  echo "$sudo_cmd"
done
