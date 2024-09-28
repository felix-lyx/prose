# PROSE: Predicting Multiple Operators and Symbolic Expressions

This repository contains code for the following papers:

- [PROSE: Predicting Multiple Operators and Symbolic Expressions using Multimodal Transformers](https://doi.org/10.1016/j.neunet.2024.106707). More details can be found in ``prose_ode/README.md``.

- [Towards a Foundation Model for Partial Differential Equations: Multi-Operator Learning and Extrapolation](https://arxiv.org/abs/2404.12355). More details can be found in ``prose_pde/README.md``.

## Install dependencies

Using conda and the ```env.yml``` file:

```
conda env create --name prose --file=env.yml
```

## Citation

If you find our paper and code useful, please consider citing:

```
@article{liu2024prose,
  title={{PROSE}: Predicting multiple operators and symbolic expressions using multimodal transformers},
  author={Liu, Yuxuan and Zhang, Zecheng and Schaeffer, Hayden},
  journal={Neural Networks},
  pages={106707},
  year={2024},
  publisher={Elsevier}
}
```

```
@article{sun2024foundation,
  title={Towards a Foundation Model for Partial Differential Equations: Multi-Operator Learning and Extrapolation}, 
  author={Sun, Jingmin and Liu, Yuxuan and Zhang, Zecheng and Schaeffer, Hayden},
  journal={arXiv preprint arXiv:2404.12355},
  year={2024}
}
```
