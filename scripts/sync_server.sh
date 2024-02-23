# Define the expected hostname
expected_hostname="c99244229552"
current_hostname=$(hostname)
# Check if the current hostname is equal to the expected hostname
if [ "$current_hostname" = "$expected_hostname" ]; then
    echo "Master Server, syncing contents to other servers"
    chmod +x train_marllib*.sh
    cd /workspace
    rsync -e 'ssh -p 40731' -avuzPr Awesome-Mobile-Crowdsensing root@10.1.114.75:/workspace
    rsync -e 'ssh -p 40731' -avuzPr Awesome-Mobile-Crowdsensing root@10.1.114.76:/workspace
    rsync -e 'ssh -p 40731' -avuzPr Awesome-Mobile-Crowdsensing root@10.1.114.103:/workspace
    rsync -e 'ssh -p 40731' -avuzPr root@10.1.114.75:/workspace/saved_data/trajectories /workspace/saved_data
    rsync -e 'ssh -p 40731' -avuzPr root@10.1.114.76:/workspace/saved_data/trajectories /workspace/saved_data
    rsync -e 'ssh -p 40731' -avuzPr root@10.1.114.103:/workspace/saved_data/trajectories /workspace/saved_data
else
    echo "The server is not the master server, skipping the sync process."
fi
