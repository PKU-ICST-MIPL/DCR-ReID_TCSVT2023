# Introduction

This is the source code of our TCSVT 2023 paper "DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification". Please cite the following paper if you use our code.

Zhenyu Cui, Jiahuan Zhou, Yuxin Peng, Shiliang Zhang and Yaowei Wang, "DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2023.



# Dependencies

- Python 3.6

- PyTorch 1.6.0

- yacs

- apex



# Data Preparation

- Download the pre-processed datasets that we used from the [link](https://pan.baidu.com/s/1LwAyB1R86P3xMZxIPm1vwQ) (password: dg1a) and unzip them to ./datasets folders.


# Usage

- Replace `_C.DATA.ROOT` and `_C.OUTPUT` in `configs/default_img.py&default_vid.py`with your own `data path` and `output path`, respectively.

- Start training by executing the following commands.

1. For LTCC dataset: `python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --gpu 0,1 --spr 0 --sacr 0.05 --rr 1.0`

2. For PRCC dataset: `python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 2,3 --spr 1.0 --sacr 0.05 --rr 1.0`

3. For CCVID dataset: `python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 main.py --dataset ccvid --cfg configs/c2dres50_ce_cal.yaml --gpu 0,1,2,3`

For any questions, feel free to contact us (cuizhenyu@stu.pku.edu.cn).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl/home/) for more information about our papers, source codes, and datasets.

