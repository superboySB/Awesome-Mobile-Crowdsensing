#!/bin/bash

admin_name="admin"
# Function to add a new user
add_users() {
    while true; do
        read -p "Enter the username (or 'done' to finish): " username
        if [ "$username" == "done" ]; then
            break
        fi
        sudo adduser "$username"
    done
}

# Function to configure sudoers file for restricted sudo access
configure_sudoers_unrestricted() {
 display the text to be added in visudo
  echo "admin ALL=(ALL) ALL"
#  while true; do
#    read -p "Enter the username (or 'done' to finish): " username
#    if [ "$username" == "done" ]; then
#      break
#    fi
#    if [ -z "$username" ] || [ "$username" == "admin" ]; then
#      echo "Invalid username. Please try again."
#      continue
#    fi
#    sudo sed -i "/^$username/d" /etc/sudoers  # Remove any existing rule for the user
#    echo "$username ALL=(ALL)" | sudo tee -a /etc/sudoers >/dev/null
#  done
}

# Function to configure sudoers file for unrestricted sudo access
configure_sudoers_restricted() {
  # display the text to be append in visudo
  echo "admin ALL=(ALL) /usr/bin/apt, /usr/bin/docker, /usr/bin/apt-get, /bin/cat, /bin/tail, !/usr/bin/docker exec -it mcs /bin/*,\
         !/usr/bin/docker exec -it mcs_new /bin/*, !/usr/bin/docker stop mcs, !/usr/bin/docker stop mcs_new,\
         !/usr/bin/docker rm -f mcs, !/usr/bin/docker rm -f mcs_new"
#    while true; do
#        read -p "Enter the username (or 'done' to finish): " username
#        # if username is admin or done, break
#        if [ "$username" == "done" ]; then
#            break
#        fi
#        if [ -z "$username" ] || [ "$username" == "admin" ]; then
#        echo "Invalid username. Please try again."
#        continue  # Continue to the next iteration if username is empty or "admin"
#        fi
#        sudo sed -i "/^$username/d" /etc/sudoers  # Remove any existing rule for the user
#        echo "$username ALL=(ALL) /usr/bin/apt, /usr/bin/docker, /usr/bin/apt-get, /bin/cat, /bin/tail, !/usr/bin/docker exec -it mcs /bin/*,\
#         !/usr/bin/docker exec -it mcs_new /bin/*, !/usr/bin/docker stop mcs, !/usr/bin/docker stop mcs_new,\
#         !/usr/bin/docker rm -f mcs, !/usr/bin/docker rm -f mcs_new" | sudo tee -a /etc/sudoers >/dev/null
#    done
}

# Function to remove user from sudoers file
remove_user_from_sudoers() {
    while true; do
        read -p "Enter the username (or 'done' to finish): " username
        if [ "$username" == "done" ]; then
            break
        fi
        if [ -z "$username" ] || [ "$username" == "admin" ]; then
        echo "Invalid username. Please try again."
        continue  # Continue to the next iteration if username is empty or "admin"
        fi
        sudo sed -i "/^$username/d" /etc/sudoers  # Remove any existing rule for the user
        sudo deluser "$username" sudo
    done
}

# Function to change username and home directory
change_username() {
    read -p "Enter the old username: " old_username
    read -p "Enter the new username: " new_username
    pkill -KILL -u "$old_username"
    # Change username and move home directory
    sudo usermod -l "$new_username" -d "/home/$new_username" -m "$old_username"

    echo "Username changed from $old_username to $new_username"
}

change_password() {
    read -p "Enter the username: " username
    sudo passwd "$username"
}

# Main menu
while true; do
    echo "1. Add new user"
    echo "2. Configure sudoers file for restricted sudo access (apt and docker)"
    echo "3. Configure sudoers file for unrestricted sudo access"
    echo "4. Remove user from sudoers file"
    echo "5. Change username and copy home directory"
    echo "6. Change password for a user"
    echo "7. Exit"
    read -p "Enter your choice: " choice

    case $choice in
        1) add_users ;;
        2) configure_sudoers_restricted ;;
        3) configure_sudoers_unrestricted ;;
        4) remove_user_from_sudoers ;;
        5) change_username ;;
        6) change_password ;;
        7) exit ;;
        *) echo "Invalid choice. Please try again." ;;
    esac
done
