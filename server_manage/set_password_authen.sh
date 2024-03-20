#!/bin/bash

# Default IP addresses
default_hosts=("10.1.114.50" "10.1.114.56" "10.1.114.66" "10.1.114.75" "10.1.114.76" "10.1.114.77" "10.1.114.103")

# Default value for password authentication
password_auth="yes"

# Parse command line options
while getopts ":c" opt; do
    case $opt in
        c)
            password_auth="no"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
    esac
done

remote_hosts=("${default_hosts[@]}")

# Shift the command line arguments so that $1 refers to the first non-option argument
shift "$((OPTIND - 1))"

# Remote user
remote_user="admin"

# Command to modify sshd_config and restart sshd
ssh_command="sudo sed -i 's/^PasswordAuthentication .*/PasswordAuthentication $password_auth/' /etc/ssh/sshd_config && sudo systemctl restart sshd"

# Loop over remote hosts
for remote_host in "${remote_hosts[@]}"; do
    echo "Updating SSH configuration on $remote_host..."
    ssh -t "$remote_user"@"$remote_host" "$ssh_command"
done
