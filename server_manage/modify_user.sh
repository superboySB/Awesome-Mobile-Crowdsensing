#!/bin/bash

default_hosts=("10.1.114.50" "10.1.114.56" "10.1.114.66" "10.1.114.75" "10.1.114.76" "10.1.114.77" "10.1.114.103")
# Local script to upload and execute
local_script="change_user_client.sh"

# Check if the script exists
if [ ! -f "$local_script" ]; then
    echo "Error: $local_script not found."
    exit 1
fi

# Check if any IP addresses were provided as command line arguments
if [ $# -gt 0 ]; then
    # Use IP addresses provided as command line arguments
    remote_hosts=("$@")
else
    # Use default IP addresses
    remote_hosts=("${default_hosts[@]}")
fi

# SSH user
ssh_user="admin"

# Iterate over each provided IP address
for ip_address in "${remote_hosts[@]}"; do
    echo "Users on $ip_address:"
    ssh "$ssh_user"@"$ip_address" "getent passwd | awk -F: '\$3 > 999 { print \$1 }'"
    echo "-------------------------"

    # Upload the local script to the remote server
    echo "Uploading $local_script to $ip_address..."
    scp "$local_script" "$ssh_user"@"$ip_address":/tmp/

    # Execute the script using sudo on the remote server
    echo "Executing $local_script on $ip_address..."
    ssh -t "$ssh_user"@"$ip_address" "sudo bash /tmp/$local_script"

    # Remove the script from the remote server after execution
    echo "Removing $local_script from $ip_address..."
    ssh "$ssh_user"@"$ip_address" "rm /tmp/$local_script"
done
