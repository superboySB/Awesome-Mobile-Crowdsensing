#!/bin/bash

# List of IP addresses
ip_addresses=("10.1.114.50" "10.1.114.56" "10.1.114.66" "10.1.114.75" "10.1.114.76" "10.1.114.77" "10.1.114.103")

# SSH user
ssh_user="admin"

# Iterate over each IP address
for ip_address in "${ip_addresses[@]}"; do
    echo "Users on $ip_address:"
    ssh "$ssh_user"@"$ip_address" "getent passwd | awk -F: '\$3 > 999 { print \$1 }'"
    echo "-------------------------"
doneb