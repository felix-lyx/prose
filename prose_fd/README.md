# PROSE-FD: A Multimodal PDE Foundation Model for Learning Multiple Operators for Forecasting Fluid Dynamics

This folder contains code for the paper [PROSE-FD: A Multimodal PDE Foundation Model for Learning Multiple Operators for Forecasting Fluid Dynamics](https://arxiv.org/abs/2409.09811). Accepted by 2024 NeurIPS Foundation Models for Science Workshop. Pretrained PROSE-FD model weights can be found on https://huggingface.co/felix-lyx/prose.

## Run the model

To launch a model training with modified arguments (arg1,val1), (arg2,val2):

```
python main.py arg1=val1 arg2=val2
```

All default arguments can be found in the ```configs``` folder, and are managed using [Hydra](https://hydra.cc/).

Scripts for reproducing the results in the paper are located in `scripts` folder. 

## Data

The dataset we used are collected from [PDEBench](https://github.com/pdebench/PDEBench), [PDEArena](https://github.com/pdearena/pdearena), and [CFDBench](https://github.com/luo-yining/CFDBench). More details about data preprocessing are included in ```data_utils/README.md```.



## Distributed training

Distributed training is available via PyTorch Distributed Data Parallel (DDP)

To launch a run on 1 node with 2 GPU, use 

```
torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py
```

## Citation

 [PROSE-FD: A Multimodal PDE Foundation Model for Learning Multiple Operators for Forecasting Fluid Dynamics](https://arxiv.org/abs/2409.09811)

```
@article{liu2024prose_fd,
  title={{PROSE-FD}: A Multimodal PDE Foundation Model for Learning Multiple Operators for Forecasting Fluid Dynamics},
  author={Liu, Yuxuan and Sun, Jingmin and He, Xinjie and Pinney, Griffin and Zhang, Zecheng and Schaeffer, Hayden},
  journal={arXiv preprint arXiv:2409.09811},
  year={2024}
}
```
