# PROSE: Predicting Multiple Operators and Symbolic Expressions

This folder contains code for the paper [Towards a Foundation Model for Partial Differential Equations: Multi-Operator Learning and Extrapolation](https://arxiv.org/abs/2404.12355). 

The code is based on the repositories [Deep Symbolic Regression](https://github.com/facebookresearch/symbolicregression) and [Deep Learning for Symbolic Mathematics](https://github.com/facebookresearch/SymbolicMathematics).

## Run the model

To launch a model training with additional arguments (arg1,val1), (arg2,val2):

```
python train.py --arg1 val1 --arg2 val2
```

All hyperparameters related to training are specified in ```parsers.py```, and environment hyperparameters are in ```symbolicregression/envs/environment.py```.

To launch evaluation, please use the flag ```eval_from_exp``` to specify in which folder the saved model is located and the flag ```eval_data``` to specify where the testing dataset is located.

```
python train.py --eval_only --eval_from_exp XXX --eval_data XXX
```

## Dataset generation

To pre-generate a dataset for future training/testing, please use

```
python train.py --export_data --max_epoch 1
```

During training, please use the flag ```reload_data``` to specify where the training and validation dataset is located. If datasets are not provided, data will be generated on the fly.

## Distributed training

Distributed training is available via PyTorch Distributed Data Parallel (DDP)

To launch a run on 1 node with 2 GPU, use 

```
torchrun --standalone --nnodes 1 --nproc_per_node 2 train.py
```

## Citation

[Towards a Foundation Model for Partial Differential Equations: Multi-Operator Learning and Extrapolation](https://arxiv.org/abs/2404.12355)

```
@article{sun2024foundation,
  title={Towards a Foundation Model for Partial Differential Equations: Multi-Operator Learning and Extrapolation}, 
  author={Sun, Jingmin and Liu, Yuxuan and Zhang, Zecheng and Schaeffer, Hayden},
  journal={arXiv preprint arXiv:2404.12355},
  year={2024}
}
```
