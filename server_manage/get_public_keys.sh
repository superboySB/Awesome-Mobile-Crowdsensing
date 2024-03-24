#!/bin/bash

# Check if username is provided as argument
if [ -z "$1" ]; then
    echo "Usage: $0 <username>"
    exit 1
fi

# Extract the username from command line argument
username="$1"

# Define the list of IP addresses
ip_addresses=("10.1.114.50" "10.1.114.56" "10.1.114.66" "10.1.114.75" "10.1.114.76" "10.1.114.77" "10.1.114.103")

# Loop over each IP address
for ip_address in "${ip_addresses[@]}"; do
#    echo "Public key for $username on $ip_address:"
    ssh -q "$username@$ip_address" 'cat ~/.ssh/id_ed25519.pub' || echo "Failed to retrieve public key on $ip_address"
done
