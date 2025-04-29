# A Unified Benchmark of Federated Learning with Kolmogorov-Arnold Networks for Medical Imaging

This is an official implementation of the following paper:
> Youngjoon Lee, Jinu Gong, and Joonhyuk Kang.
**[A Unified Benchmark of Federated Learning with Kolmogorov-Arnold Networks for Medical Imaging](https://arxiv.org/abs/2504.19639)**  
_arXiv:2504.19639_.

## Federated Learning Methods
This paper considers the following federated learning techniques:
- **FedAvg**: [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
- **FedDyn**: [Federated Learning Based on Dynamic Regularization](https://openreview.net/pdf?id=B7v4QMR6Z9w)
- **FedSAM**: [Generalized Federated Learning via Sharpness Aware Minimization](https://proceedings.mlr.press/v162/qu22a/qu22a.pdf)
- **FedGamma**: [Fedgamma: Federated learning with global sharpness-aware minimization](https://ieeexplore.ieee.org/abstract/document/10269141)
- **FedSpeed**: [FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy](https://openreview.net/pdf?id=bZjxxYURKT)
- **FedSMOO**: [Dynamic Regularized Sharpness Aware Minimization in Federated Learning: Approaching Global Consistency and Smooth Landscape](https://proceedings.mlr.press/v202/sun23h.html)

## Dataset
- Blood cell classification dataset ([Andrea Acevedo, Anna Merino, et al. Data in Brief 2020](https://www.sciencedirect.com/science/article/pii/S2352340920303681))

## Experiments

To create figure 1:
1. Run `bash execute/fig1.sh`
2. Then, execute `fig1.py`


To create figure 2:
1. Run `bash execute/fig2.sh`
2. Then, execute `fig2.py`

To create figure 3:
1. Run `bash execute/fig3.sh`
2. Then, execute `fig3.py`

To create figure 4:
1. Run `bash execute/ablation.sh`
2. Then, execute `ablation.py`

## Citation
If this codebase can help you, please cite our paper: 
```bibtex
@article{lee2025unifiedbenchmarkfederatedlearning,
  title={A Unified Benchmark of Federated Learning with Kolmogorov-Arnold Networks for Medical Imaging},
  author={Youngjoon Lee and Jinu Gong and Joonhyuk Kang},
  journal={arXiv preprint arXiv:2504.19639},
  year={2025}
}
```

## References
This repository draws inspiration from:
- https://github.com/woodenchild95/FL-Simulator
- https://github.com/ZiyaoLi/fast-kan