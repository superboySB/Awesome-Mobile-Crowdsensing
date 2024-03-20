# Define the expected hostname
expected_hostname="c99244229552"
current_hostname=$(hostname)
ip_addresses=("10.1.114.56" "10.1.114.66" "10.1.114.75" "10.1.114.76" "10.1.114.103")
# Check if the current hostname is equal to the expected hostname
if [ "$current_hostname" = "$expected_hostname" ]; then
    echo "Master Server, syncing contents to other servers"
    chmod +x train_marllib*.sh
    # Default values
    upload=false
    download=false
    # Parse command line options
    while getopts "ud" opt; do
        case $opt in
            u)
                upload=true
                ;;
            d)
                download=true
                ;;
            \?)
                echo "Invalid option: -$OPTARG" >&2
                exit 1
                ;;
        esac
    done

    # Check if either upload or download option is set
    if ! $upload && ! $download; then
        echo "Please specify either -u (upload) or -d (download) option."
        exit 1
    fi

    # Change to the /workspace directory
    cd /workspace || return

    # Iterate over each IP address
    for ip_address in "${ip_addresses[@]}"; do
        # Upload directory if upload option is set
        if $upload; then
            rsync -e 'ssh -p 40731' -avuzPr Awesome-Mobile-Crowdsensing "root@$ip_address:/workspace"
        fi

        # Download directory if download option is set
        if $download; then
            rsync -e 'ssh -p 40731' -avuzPr "root@$ip_address:/workspace/saved_data/trajectories" "/workspace/saved_data"
        fi
    done
else
    echo "The server is not the master server, skipping the sync process."
fi
# Sync Server get error “No such file or directory for saved_data": remount 75 data1 and data2 disk.