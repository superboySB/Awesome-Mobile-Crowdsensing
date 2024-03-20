# Define the expected hostname
expected_hostname="c99244229552"
current_hostname=$(hostname)
# Check if the current hostname is equal to the expected hostname
if [ "$current_hostname" = "$expected_hostname" ]; then
    echo "Master Server, syncing public keys to to other servers"
    #!/bin/bash
# Dry run flag
dry_run=false

# Parse command line arguments
while getopts ":n" opt; do
    case $opt in
        n)
            dry_run=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
    esac
done

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
        echo "Processing user: $user on IP address: $ip_address"
        tmp_file_name="/home/$user/local_pub_key.pub"
        # Check if the user is admin
        if [ "$user" == "admin" ]; then
            pub_key_file=~/.ssh/id_rsa.pub
        else
            pub_key_file=~/.ssh/"$user".pub
        fi
        scp "$pub_key_file" "$user@$ip_address:$tmp_file_name"
        # Check if the public key file exists
        if [ -f "$pub_key_file" ]; then
            if $dry_run; then
                echo "Dry run: ssh-copy-id -i $pub_key_file $user@$ip_address"
            else
                # Execute ssh-copy-id
                remote_server="$user@$ip_address"
                ssh-copy-id -f -i "$pub_key_file" "$remote_server"
                ssh "$remote_server" ssh "$remote_server" "comm -23 <(sort ~/.ssh/authorized_keys) <(sort "$tmp_file_name")\
                 >> ~/.ssh/temp_authorized_keys && mv ~/.ssh/temp_authorized_keys ~/.ssh/authorized_keys"

                echo "Public key for $user copied to $ip_address"
                # Remove the temporary public key file
                ssh "$remote_server" "rm $tmp_file_name"
            fi
        else
            echo "Public key file for $user not found"
        fi
    done
done

else
    echo "The server is not the master server, skipping the sync process."
fi
