#!/bin/bash

# List of IP addresses
ip_addresses=("10.1.114.50" "10.1.114.56" "10.1.114.66" "10.1.114.75" "10.1.114.76" "10.1.114.77" "10.1.114.103")

# SSH user
ssh_user="admin"

# Iterate over each IP address
for ip_address in "${ip_addresses[@]}"; do
    echo "Users on $ip_address:"
    ssh "$ssh_user"@"$ip_address" "getent passwd | awk -F: '\$3 > 999 { print \$1 }'"

    # Check SSH configuration for PubkeyAuthentication
    pubkey_auth=$(ssh "$ssh_user"@"$ip_address" "grep -i '^PubkeyAuthentication' /etc/ssh/sshd_config | awk '{ print \$2 }'")
    if [ "$pubkey_auth" == "yes" ]; then
        echo "SSH on $ip_address is using PubkeyAuthentication"
    else
        echo "SSH on $ip_address is NOT using PubkeyAuthentication"
    fi

    # Check SSH configuration for PasswordAuthentication
    password_auth=$(ssh "$ssh_user"@"$ip_address" "grep -i '^PasswordAuthentication' /etc/ssh/sshd_config | awk '{ print \$2 }'")
    if [ "$password_auth" == "yes" ]; then
        echo "SSH on $ip_address is using PasswordAuthentication"
    else
        echo "SSH on $ip_address is NOT using PasswordAuthentication"
    fi

    # Display disk usage
    echo "Disk usage on $ip_address:"
    ssh "$ssh_user"@"$ip_address" "df -h | grep -vE 'udev|nvme|docker|snap|boot|tmpfs|cdrom'"

    echo "-------------------------"
done
