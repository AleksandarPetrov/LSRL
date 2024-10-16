![LSRL Logo](lsrl_logo.png)

# LSRL: Linear State Recurrent Language

This repository contains the implementation of LSRL (Linear State Recurrent Language), a programming language for recurrent models.
LSRL is designed as a tool to implement exact algorithms in recurrent neural network architectures such as RNNs, LSTMs, GRUs, Mamba, Griffin/Hawk and more.
The original motivation for developing LSRL was to develop a tool to construct universal in-context approximation programs that can be represented in recurrent architectures.
See the [accompanying paper](https://arxiv.org/abs/2406.01424) for further details.

## Features

- We provide the key building blocks of any reccurent program in `basic_blocks.py`. This includes linear layers, linear state variables, ReLU actications, concatenation blocks and multiplicative gating units.
- To ease the programming experience, we also offer a number of utility functions ("syntactic sugar"). These can be found in `sugar_blocks.py`.
- We support both numeric (with `scipy.sparse`) and symbolic (with `sympy`) backends. The numeric backend is fast but can introduce small numerical precision erros which might be critical for the performance in certain cases, as discussed in the paper. The symbolic backend is much slower but is exact and, as a result, does not exhibit such numerical instabilities.
- We have added extensive unit and integration tests to assure correctness of the implementation.

## Getting started

You can install LSRL by cloning this repository and installing the dependencies in `requirements.txt`.
The `more_ones_or_zeros.ipynb` Jupyter notebook can serve as a basic introduction to the capabilities and use of LSRL.
For more advanced use, see the `continous_universal_approximation.ipynb` and `discrete_universal_approximation.ipynb` notebooks which showcase the full capabilities of LSRL by constructing a universal approximation program for continous functions and an emulator for discrete maps.

## Cite as

```
@inproceedings{petrov2024universal,
  title={Universal In-Context Approximation By Prompting Fully Recurrent Models},
  author={Aleksandar Petrov and Tom A. Lamb and Alasdair Paren and Philip H.S. Torr and Adel Bibi},
  booktitle={Advances in Neural Information Processing Systems},
  url={https://arxiv.org/abs/2406.01424},
  year={2024}
}
```
