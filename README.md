# Awesome-Mobile-Crowdsensing
This is a list of research resources on **Human-Machine Collaborative Sensing** by AI-Driven Unmanned Vehicles（边缘环境人机协同群智感知与决策关键技术研究）. And the repository will be continuously updated to help readers get inspired by our researches and realize their ideas in their own field.

This repo is based on [warp-drive](https://github.com/salesforce/warp-drive).

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



## Quick Start
We recommend to use Nvidia NGC to start our simulation and baseline algorithms.
```sh
docker build -t linc_image:v1.0 .
docker run -itd --runtime=nvidia --network=host --user=user --name=mcs linc_image:v1.0 /bin/bash
docker exec -it mcs /bin/bash
```
Install our drl-framework.
```sh
git clone https://github.com/BIT-MCS/Awesome-Mobile-Crowdsensing.git && cd Awesome-Mobile-Crowdsensing
conda create --name warp_drive python=3.7 --yes && conda activate warp_drive && pip install -e .
```
Do the following unittests to make sure the whole architecture can run successfully.
```sh
# cd warp_drive/cuda_includes && make compile-test # [Optional]
python -m warp_drive.utils.unittests.run_unittests_pycuda
python -m warp_drive.utils.unittests.run_trainer_tests
```
[Optional] Do auto-scaling, and drl-framework will automatically determine the best block size and training batch size to use. It will also determine the number of available GPUs and perform training on all the GPUs.
```sh
python warp_drive/trainer_pytorch.py --env tag_continuous --auto_scale
```
Note that our drl-framework is based on [warp-drive](https://github.com/salesforce/warp-drive), an extremely fast end-to-end MARL architecture on GPUs.
