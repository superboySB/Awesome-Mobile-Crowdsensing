# Awesome-Mobile-Crowdsensing
This is a list of research resources on **Human-Machine Collaborative Sensing** by AI-Driven Unmanned Vehicles, including related papers, simulation codes and default algorithms. 

Prof. Liu often admonishes us with the phrase "**Talk is cheap, show me your code.**" We should focus on making practical contributions to the Internet of Things and Artificial Intelligence communities. To this end, this repository is continuously updated to help readers get inspired by our researches (including not only our papers but also our codes), and realize their ideas in their own field. To better align our research with industrial-grade applications, we always need to consider much more factors, including: larger scales, more agents, higher model training throughput, and faster simulation speeds. To this end, we explore several industrial tools and improve our somewhat outdated implementations of our previous research. For example, we adopted Salaforce's distributed training framework, called [Warp-Drive](https://catalog.ngc.nvidia.com/orgs/partners/teams/salesforce/containers/warpdrive), an extremely fast end-to-end reinforcement learning architecture on a single or multiple Nvidia GPUs. 

## Related Papers
- [Ensuring Threshold AoI for UAV-assisted Mobile Crowdsensing by Multi-Agent Deep Reinforcement Learning with Transformer](https://ieeexplore.ieee.org/abstract/document/10181012)
- [Exploring both Individuality and Cooperation for Air-Ground Spatial Crowdsourcing by Multi-Agent Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/10184585)
- [Air-Ground Spatial Crowdsourcing with UAV Carriers by Geometric Graph Convolutional Multi-Agent Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/10184614)
- [Delay Sensitive Energy-Efficient UAV Crowdsensing by Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9540290)
- [Human-Drone Collaborative Spatial Crowdsourcing by Memory-Augmented and Distributed Multi-Agent Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9835559)
- [AoI-minimal UAV Crowdsensing by Model-based Graph Convolutional Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9796732)
- [Energy-Efficient 3D Vehicular Crowdsourcing for Disaster Response by Distributed Deep Reinforcement Learning](https://dl.acm.org/doi/abs/10.1145/3447548.3467070)
- [Mobile Crowdsensing for Data Freshness: A Deep Reinforcement Learning Approach](https://ieeexplore.ieee.org/abstract/document/9488791)
- [Social-Aware Incentive Mechanism for Vehicular Crowdsensing by Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9173810)
- [Distributed and Energy-Efficient Mobile Crowdsensing with Charging Stations by Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/8821415)
- [Free Market of Multi-Leader Multi-Follower Mobile Crowdsensing: An Incentive Mechanism Design by Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/8758205)
- [Energy-Efficient Mobile Crowdsensing by Unmanned Vehicles: A Sequential Deep Reinforcement Learning Approach](https://ieeexplore.ieee.org/abstract/document/8944303)
- [Energy-Efficient UAV Crowdsensing with Multiple Charging Stations by Deep Learning](https://ieeexplore.ieee.org/abstract/document/9155535)
- [Multi-Task-Oriented Vehicular Crowdsensing: A Deep Learning Approach](https://ieeexplore.ieee.org/abstract/document/9155393)
- [Distributed Energy-Efficient Multi-UAV Navigation for Long-Term Communication Coverage by Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/8676325)
- [Curiosity Driven Energy-Efficient Worker Scheduling in Vehicular Crowdsourcing: A Deep Reinforcement Learning Approach](https://ieeexplore.ieee.org/abstract/document/9101657)
- [Blockchain-enabled Data Collection and Sharing for Industrial IoT with Deep Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/8594641)
- [Energy-efficient distributed mobile crowd sensing: A deep learning approach](https://ieeexplore.ieee.org/abstract/document/8664596)
- [Online Quality-Aware Incentive Mechanism for Mobile Crowd Sensing with Extra Bonus](https://ieeexplore.ieee.org/abstract/document/8502067)
- [Energy-efficient UAV control for effective and fair communication coverage: A deep reinforcement learning approach](https://ieeexplore.ieee.org/abstract/document/8432464)
- [Learning-based Energy-Efficient Data Collection by Unmanned Vehicles in Smart Cities](https://ieeexplore.ieee.org/abstract/document/8207610/)


## Install
To always get the latest GPU optimized software, we recommend you to use [Nvidia NGC](https://catalog.ngc.nvidia.com/orgs/partners/teams/salesforce/containers/warpdrive)(which fully supports all kinds of Nvidia devices, such as A100, H100), following:
```sh
docker build -t linc_image:v1.0 .
docker run -itd --runtime=nvidia --network=host --user=user --name=mcs linc_image:v1.0 /bin/bash
docker exec -it mcs /bin/bash
```
In the NGC containerFirst, we first install our drl-framework.
```sh
conda create --name mcs python=3.9 --yes && conda activate mcs
cd /workspace/movingpandas && python setup.py develop && cd ..
git clone https://github.com/BIT-MCS/Awesome-Mobile-Crowdsensing.git && cd Awesome-Mobile-Crowdsensing && pip install -e .
```
We recommend you do the following unittests to make sure the whole simulation and training architecture can run successfully.
```sh
python -m warp_drive.utils.unittests.run_unittests_pycuda
python -m warp_drive.utils.unittests.run_trainer_tests
python envs/crowd_sim/run_cpu_gpu_env_consistency_checks.py
```
Now you can start our simulation and baseline algorithms. 

## QuickStart
Run random policy and debug the environment, and you will interactive with an generated html file:
```sh
python run_random_policy.py --plot_loop --moving_line --output_dir="./logs.html"
```
Train independent PPO as the baseline policies:
```sh
python train_rl_policy.py
```
Run the trained RL policy
```sh
python run_rl_policy.py
```
Note that our drl-framework is based on [warp-drive](https://github.com/salesforce/warp-drive), an extremely fast end-to-end MARL architecture on GPUs.

[Optional] Do auto-scaling, and drl-framework will automatically determine the best block size and training batch size to use. It will also determine the number of available GPUs and perform training on all the GPUs.
```sh
python warp_drive/trainer_pytorch.py --env tag_continuous --auto_scale
```

## Roadmap
- [X] Finish Simple Air-Ground Collaborative Simulation for MCS
- [X] Finish CUDA implementation
- [X] Finish PPO baseline
- [X] Add mobile users as PoIs
- [ ] Add macro stragies of mobile users as PoIs
- [ ] Add our proposed key metrics for [emergency response](https://github.com/BIT-MCS/DRL-UCS-AoI-Threshold), inspired by Age of Information (AoI)
- [ ] Add apis for using available actions
- [ ] Add more realistic environment dynamics, e.g., [4G MIMO](https://github.com/BIT-MCS/DRL-freshMCS), [5G NOMA](https://github.com/BIT-MCS/hi-MADRL)