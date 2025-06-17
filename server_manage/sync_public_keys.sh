#!/bin/bash

# Define the expected hostname
#expected_hostname="c99244229552"
expected_hostname="dba35451f1c6"
current_hostname=$(hostname)

# Check if the current hostname is equal to the expected hostname
if [ "$current_hostname" = "$expected_hostname" ]; then
    echo "Master Server, syncing public keys to other servers"

    # List of users and their corresponding IP addresses
    users=("liuchi" "lishuang" "liguozheng" "admin")
#    ip_addresses=("10.1.114.50" "10.1.114.56" "10.1.114.66" "10.1.114.75" "10.1.114.76" "10.1.114.77" "10.1.114.103")
    cpu_addresses=("10.1.114.66")
    ip_addresses=("10.1.114.50" "10.1.114.56" "10.1.114.75" "10.1.114.76" "10.1.114.77")
     # File containing additional public keys

    # Loop through the list of IP addresses
    for ip_address in "${ip_addresses[@]}"; do
        echo "Processing IP address: $ip_address"

        # Loop through the list of users
        for user in "${users[@]}"; do
            echo "Processing user: $user"
            local_pub_key_file=~/.ssh/"$user".pub
            cpu_users_file=~/.ssh/"$user"_cpu_users.pub

            # Check if local public key file exists
            if [ -f "$local_pub_key_file" ]; then
                # Use a temporary file to hold the combined keys
                temp_pub_key_file=$(mktemp)

                # Copy the original user's public key to the temporary file
                cat "$local_pub_key_file" > "$temp_pub_key_file"

                # If ip_address is in cpu_addresses, append additional entries from cpu_users_file
                if [[ " ${cpu_addresses[*]} " == *" $ip_address "* ]]; then
                    echo "IP address $ip_address is in cpu_addresses, appending additional entries"
                    if [ -f "$cpu_users_file" ]; then
                        cat "$cpu_users_file" >> "$temp_pub_key_file"
                    else
                        echo "Public key file for cpu users not found"
                    fi
                fi

                # Upload the modified authorized_keys file back to the remote server
                scp "$temp_pub_key_file" "$user@$ip_address:/home/$user/.ssh/authorized_keys"

                # Remove the temporary file
                rm "$temp_pub_key_file"
            else
                echo "Public key file for $user not found"
            fi
        done
    done
else
    echo "The server is not the master server, skipping the sync process."
fi
