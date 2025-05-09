# PROSE: Predicting Multiple Operators and Symbolic Expressions

This repository contains code for the following papers:

- [PROSE: Predicting Multiple Operators and Symbolic Expressions using Multimodal Transformers](https://doi.org/10.1016/j.neunet.2024.106707). More details can be found in ``prose_ode/README.md``.

- [Towards a Foundation Model for Partial Differential Equations: Multi-Operator Learning and Extrapolation](https://arxiv.org/abs/2404.12355). More details can be found in ``prose_pde/README.md``.

- [PROSE-FD: A Multimodal PDE Foundation Model for Learning Multiple Operators for Forecasting Fluid Dynamics](https://arxiv.org/abs/2409.09811). More details can be found in ``prose_fd/README.md``.

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
  volume={180},
  pages={106707},
  year={2024},
  publisher={Elsevier}
}
```

```
@article{sun2024towards,
  title = {Towards a foundation model for partial differential equations: Multioperator learning and extrapolation},
  author = {Sun, Jingmin and Liu, Yuxuan and Zhang, Zecheng and Schaeffer, Hayden},
  journal = {Phys. Rev. E},
  volume = {111},
  issue = {3},
  pages = {035304},
  numpages = {18},
  year = {2025},
  month = {Mar},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.111.035304},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.111.035304}
}

```

```
@article{liu2024prose_fd,
  title={{PROSE-FD}: A Multimodal PDE Foundation Model for Learning Multiple Operators for Forecasting Fluid Dynamics},
  author={Liu, Yuxuan and Sun, Jingmin and He, Xinjie and Pinney, Griffin and Zhang, Zecheng and Schaeffer, Hayden},
  journal={arXiv preprint arXiv:2409.09811},
  year={2024}
}
```
