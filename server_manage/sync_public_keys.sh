# Define the expected hostname
expected_hostname="c99244229552"
current_hostname=$(hostname)
# Check if the current hostname is equal to the expected hostname
if [ "$current_hostname" = "$expected_hostname" ]; then
    echo "Master Server, syncing public keys to to other servers"
    #!/bin/bash

# Shift the option arguments so they're not parsed as user inputs
shift $((OPTIND - 1))
# List of users and their corresponding IP addresses
users=("liuchi" "hanrui" "lishuang" "gaoguangyu" "liguozheng")

ip_addresses=("10.1.114.50" "10.1.114.56" "10.1.114.66" "10.1.114.75" "10.1.114.76" "10.1.114.77" "10.1.114.103")
# Loop through the list of IP addresses
for ip_address in "${ip_addresses[@]}"; do
    echo "Processing IP address: $ip_address"

    # Loop through the list of users
    for user in "${users[@]}"; do
            echo "Processing user: $user"
            # use ssh-copy-id for local machine if needed
            # Download the original authorized_keys file from the remote server
            local_pub_key_file=~/.ssh/"$user".pub
            # check if local public key file exists
            if [ -f "$local_pub_key_file" ]; then
              # Upload the modified authorized_keys file back to the remote server
              scp $local_pub_key_file "$user@$ip_address:/home/$user/.ssh/authorized_keys"
            else
                echo "Public key file for $user not found"
            fi
    done
done

else
    echo "The server is not the master server, skipping the sync process."
fi
