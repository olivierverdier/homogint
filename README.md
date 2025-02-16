# Integrators on homogeneous spaces

[![Build Status](https://github.com/olivierverdier/homogint/actions/workflows/python_package.yml/badge.svg?branch=main)](https://github.com/olivierverdier/homogint/actions/workflows/python_package.yml?query=branch%3Amain)
![Python version](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13-blue.svg?logo=python&logoColor=gold)
[![codecov](https://codecov.io/github/olivierverdier/homogint/graph/badge.svg?token=Ea4XsTXw6A)](https://codecov.io/github/olivierverdier/homogint)
[![arXiv](https://img.shields.io/badge/arXiv-1402.6981-b31b1b.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/1402.6981)

## Purpose


This is a proof-of-concept implementation of the general description of Runge–Kutta on homogeneous spaces, from the paper ["Integrators on homogeneous spaces: Isotropy choice and connections"](http://arxiv.org/abs/1402.6981).

## Installation & Examples

1. Install [`uv`](https://docs.astral.sh/uv/) if you haven't already.
2. Clone this repo
3. Inside the repo, run `uv sync`
4. Run `uv run --group example --with jupyter,"." jupyter lab`
5. Open the jupyter URL in a browser
6. Navigate to the `examples` folder and run the `Demo.ipynb`.

## Gallery
The following pictures are extracted from [this Demo Notebook](https://gist.github.com/olivierverdier/ea449d66f856481fd80ab5aa76bb08c0)


Integration on a Stiefel manifold:

<img src="img/oja.png" alt="oja" width="200" />

Quadrature on a sphere:

<img src="img/quad.png" alt="quad" width="200" />

Quadrature on the group SO(3):

<img src="img/so3quad.png" alt="so3quad" width="200" />


Continuous QR flow, converging towards a diagonal matrix:

![matinit](img/matinit.png) ![matfinal](img/matfinal.png)


