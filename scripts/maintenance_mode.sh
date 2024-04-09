#!/bin/bash

# Check for correct number of arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <IP_ADDRESS> --enable|--disable"
    exit 1
fi

# Assign command line arguments to variables
IP_ADDRESS=$1
MODE=$2

# Define the list of users
USERS=("liuchi") # Add the usernames you need to operate on

# Path to the admin public key
ADMIN_KEY_PATH="./id_ed25519.pub"

# Check if admin_key.pub exists
if [ ! -f "$ADMIN_KEY_PATH" ]; then
    echo "The file $ADMIN_KEY_PATH does not exist. Exiting."
    exit 1
fi

# Enable maintenance mode
enable_maintenance() {
    for USER in "${USERS[@]}"; do
        echo "Enabling maintenance mode for $USER on $IP_ADDRESS"
        scp "$ADMIN_KEY_PATH" admin@"$IP_ADDRESS":/home/"$USER"/.ssh/authorized_keys
    done
}

# Disable maintenance mode
disable_maintenance() {
    for USER in "${USERS[@]}"; do
      local_path=~/.ssh/"$USER".pub
        if [ ! -f "$local_path" ]; then
            echo "The file $local_path does not exist. Skipping $USER."
            continue
        fi
        echo "Disabling maintenance mode for $USER on $IP_ADDRESS"
        scp "$local_path" admin@"$IP_ADDRESS":/home/"$USER"/.ssh/authorized_keys
    done
}

# Check mode and call the appropriate function
case "$MODE" in
    --enable)
        enable_maintenance
        ;;
    --disable)
        disable_maintenance
        ;;
    *)
        echo "Invalid mode. Use --enable or --disable."
        exit 1
        ;;
esac
