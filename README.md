# Awesome-Mobile-Crowdsensing
Research Resources on Human-Machine Collaborative Sensing by AI-Driven Unmanned Vehicles（边缘环境人机协同群智感知与决策关键技术研究）

## Quick Start
```sh
docker build -t linc_image:v1.0 .
docker run -itd --runtime=nvidia --network=host --name=mcs linc_image:v1.0 /bin/bash
git clone https://github.com/BIT-MCS/Awesome-Mobile-Crowdsensing.git && cd Awesome-Mobile-Crowdsensing && pip install -e .

python -m warp_drive.utils.unittests.run_unittests_pycuda  # TODO：或许应该删掉numba的所有功能，太鸡肋了还有bug，保留是否添加wrapper即可，用来debug？

python run.py
```
