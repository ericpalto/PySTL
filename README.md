# PySTL

[![Tests](https://github.com/ericpalto/PySTL/actions/workflows/tests.yml/badge.svg?event=pull_request)](https://github.com/ericpalto/PySTL/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/website?url=https%3A%2F%2Fericpalto.github.io%2FPySTL%2F&label=docs&logo=readthedocs)](https://ericpalto.github.io/PySTL/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![NumPy](https://img.shields.io/badge/NumPy-powered-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![JAX](https://raw.githubusercontent.com/femtomc/jax-badge/main/badge.svg)](https://github.com/google/jax)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)


A Python repository implementing Signal Temporal Logic (STL) quantitative semantics with an explicit syntax/backend split.

## Implemented STL Logics

| STL Semantic | Sound (Sign-Preserving)? | Smooth | What Makes It Special | Citation |
| --- | --- | --- | --- | --- |
| Classical (Space) Robustness | Yes | No | Uses `min`/`max` and `inf`/`sup` style operators to compute signed distance to violation (worst-case semantics). | Donze, A., & Maler, O. (2010). *Robust Satisfaction of Temporal Logic over Real-Valued Signals*. In Formal Modeling and Analysis of Timed Systems (FORMATS). [doi:10.1007/978-3-642-15297-9_9](https://doi.org/10.1007/978-3-642-15297-9_9) |
| Smooth Robustness | Approximate (temperature-dependent) | Yes | Replaces `min`/`max` by softmin/softmax (`logsumexp`) for differentiable approximations of classical robustness. | Leung K, Aréchiga N, Pavone M. *Backpropagation through signal temporal logic specifications: Infusing logical structure into gradient-based methods*. The International Journal of Robotics Research. 2022;42(6):356-370. [doi:10.1177/02783649221082115](https://doi.org/10.1177/02783649221082115) |
| Cumulative Robustness | No (redefines satisfaction) | Usually no (piecewise) | Integrates robustness over time instead of only worst-case aggregation; captures sustained behavior. | Haghighi, I., Mehdipour, N., Bartocci, E., & Belta, C. (2019). *Control from Signal Temporal Logic Specifications with Smooth Cumulative Quantitative Semantics*. IEEE Control Systems Letters. [arXiv:1904.11611](https://arxiv.org/abs/1904.11611) |
| AGM Robustness | Yes | Piecewise smooth (non-smooth on sign boundaries) | Uses arithmetic/geometric means to reward both degree and frequency of satisfaction across subformulae and time. | Mehdipour, N., Vasile, C.-I., & Belta, C. (2019). *Arithmetic-Geometric Mean Robustness for Control from Signal Temporal Logic Specifications*. ACC 2019. [doi:10.23919/ACC.2019.8814487](https://doi.org/10.23919/ACC.2019.8814487) |
| D-GMSR Robustness | Yes | Mostly yes (except boundary points) | Reformulates `min`/`max` with structured generalized means; smooth while preserving sign semantics. | Uzun, S., et al. (2024). *Discrete Generalized Mean Smooth Robustness (D-GMSR) for Signal Temporal Logic*. [arXiv:2405.10996](https://arxiv.org/abs/2405.10996) |

Supported syntax/backend combinations:
- Syntaxes: `classical`, `smooth`, `cumulative`, `agm`, `dgmsr`
- Backends: `numpy` (default), `jax` (with `--extra jax`), `torch` (with `--extra torch`)
- Total combinations: 15 (`5 x 3`) when all extras are installed

## Documentation

- [Docs index](docs/index.md)
- [Unified API guide](docs/unified_api_guide.md)

## Installation

Using [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (recommended):

```bash
# Regular install, NumPy-only
uv sync

# Regular install, NumPy + JAX
uv sync --extra jax

# Regular install, NumPy + PyTorch
uv sync --extra torch
```

Using `pip`:

```bash
# NumPy-only install (default)
python -m venv .venv
source .venv/bin/activate
pip install -e .

# NumPy + JAX install
pip install -e ".[jax]"

# NumPy + PyTorch install
pip install -e ".[torch]"
```

## Contributing

Everyone is welcome to contribute.
Development install using `uv`:

```bash
# Dev install (adds pytest + pre-commit), NumPy-only
uv sync --dev
pre-commit install

# Dev install (adds pytest + pre-commit), NumPy + JAX
uv sync --dev --extra jax
pre-commit install

# Dev install (adds pytest + pre-commit), NumPy + PyTorch
uv sync --dev --extra torch
pre-commit install
```

Instructions:

1. Create a feature branch for your change.
2. Implement the change with tests and docs updates when relevant.
3. Run checks locally (`uv run pytest`).
4. Linter should run automatically on commit.
5. Open a pull request.

## Quick Start

```python
import numpy as np
from stl import Predicate, Interval, create_semantics

# signal shape: (time, state_dim)
signal = np.array([
    [0.2, 0.8],
    [0.3, 0.6],
    [0.5, 0.4],
], dtype=float)

p_speed_ok = Predicate("speed_ok", fn=lambda s, t: 0.6 - s[t, 0])
p_alt_ok = Predicate("alt_ok", fn=lambda s, t: s[t, 1] - 0.2)

phi = (p_speed_ok & p_alt_ok).always(Interval(0, 2))
sem = create_semantics("classical", backend="numpy")
rho0 = phi.evaluate(signal, sem, t=0)

print(float(rho0))
```

JAX semantics + gradients:

```python
import jax
import jax.numpy as jnp
from stl import create_semantics

signal_jax = jnp.asarray(signal)
sem_jax = create_semantics("classical", backend="jax")

rho0_jax = phi.evaluate(signal_jax, sem_jax, t=0)
grad0 = jax.grad(lambda s: phi.evaluate(s, sem_jax, t=0))(signal_jax)
```
