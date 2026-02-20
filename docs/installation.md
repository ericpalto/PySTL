---
title: Installation
description: How to install PySTL with optional JAX and PyTorch backends
---

# Installation

PySTL requires Python 3.12 or later. The core package depends only on NumPy. JAX and PyTorch support are optional extras.

## Using uv (recommended)

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is the recommended package manager for PySTL.

```bash
# NumPy only (default)
uv sync

# NumPy + JAX
uv sync --extra jax

# NumPy + PyTorch
uv sync --extra torch

# All backends
uv sync --extra jax --extra torch
```

## Using pip

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# NumPy only
pip install -e .

# NumPy + JAX
pip install -e ".[jax]"

# NumPy + PyTorch
pip install -e ".[torch]"

# All backends
pip install -e ".[jax,torch]"
```

## Verifying the installation

```python
from stl import registry

print(registry.syntaxes())   # ['agm', 'classical', 'cumulative', 'dgmsr', 'smooth']
print(registry.backends())   # ['numpy'] or ['jax', 'numpy'] or ['jax', 'numpy', 'torch']
```

If JAX or PyTorch are not installed, their backends will simply not appear in `registry.backends()`.

## Development install

To contribute to PySTL, install with the `dev` dependency group:

```bash
# Dev tools (pytest, pre-commit, pylint) + NumPy only
uv sync --dev

# Dev tools + JAX
uv sync --dev --extra jax

# Dev tools + PyTorch
uv sync --dev --extra torch
```

Then install the pre-commit hooks:

```bash
pre-commit install
```

See [CONTRIBUTING](../README.md#contributing) for the full contribution workflow.
