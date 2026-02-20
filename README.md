# PySTL

>A unified API that consolidates different Signal Temporal Logic (STL) semantics under a single interface with native support for NumPy, JAX and PyTorch.


<p align="center">
  <a href="https://github.com/ericpalto/PySTL/actions/workflows/tests.yml">
    <img src="https://github.com/ericpalto/PySTL/actions/workflows/tests.yml/badge.svg?event=pull_request" alt="Tests">
  <a href="https://ericpalto.github.io/PySTL/">
    <img src="https://img.shields.io/website?url=https%3A%2F%2Fericpalto.github.io%2FPySTL%2F&label=docs&logo=readthedocs" alt="License: MIT">
  </a>
  </a>
  <a href="https://doi.org/10.5281/zenodo.18707962">
    <img src="https://zenodo.org/badge/1161400047.svg" alt="DOI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>

</p>

<p align="center">
  <a href="https://numpy.org">
    <img src="https://img.shields.io/badge/NumPy-powered-013243?logo=numpy&logoColor=white" alt="NumPy">
  </a>
  <a href="https://github.com/google/jax">
    <img src="https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq5MTYAhOxXM%2FmvEpsoWmtfP9RxnkAIAr%2F5pVxqPxH93msKodRSXIct2l0OU0%2FL4eY506L%2B3GyJ6UMEZfjjCDbysNcWWmFweJP0Jz%2FA0g2gk80pGkYAAAAAElFTkSuQmCC" alt="JAX">
  </a>
  <a href="https://pytorch.org">
    <img src="https://img.shields.io/badge/PyTorch-supported-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
</p>
 🚀 **TL;DR**

Unified, backend-agnostic STL robustness semantics in one API.  Switch semantics. Switch backend. Keep the same specification:

 - One API
 - Multiple STL semantics (Classic, soft, cumulative, D-GMSR... and more!)
 - NumPy / JAX / PyTorch backends
 - Autograd & GPU ready

---

**Why?**

STL robustness definitions are scattered across separate repositories, programming languages, and incompatible interfaces. Classical, smooth, cumulative, AGM, and D-GMSR semantics often live in isolation, making comparisons and experimentation unnecessarily difficult.

PySTL consolidates them under a single, consistent API. The same STL specification can be evaluated under different quantitative semantics without rewriting logic or changing infrastructure.

---

**How?**

Under PySTL's unified API, switching semantics is as simple as changing a function argument. Switching backend (NumPy, JAX, PyTorch) is equally straightforward, enabling automatic differentiation and GPU acceleration out of the box.

```python
sem_jax = create_semantics("AGM", backend="jax")
```
All semantics share the same internal interface and operator implementation (AND, OR, NOT, Always, Eventually, Until), ensuring consistent behavior across backends.

---

PySTL runs natively on NumPy, JAX, and PyTorch, making every supported semantics immediately usable in gradient-based control synthesis, reinforcement learning, and differentiable optimization pipelines. This includes semantics that were not originally designed for autograd workflows.

By unifying formal STL semantics with modern ML tooling, PySTL turns robustness evaluation from a fragmented research artifact into a backend-agnostic building block for verification, control, and learning over time-series signals.

## Documentation

- [Docs index](docs/index.md)
- [Installation](docs/installation.md)
- [Quick Start](docs/quickstart.md)
- [A Unified API](docs/unified_api.md)
- [JAX backend](docs/backends/jax.md)
- [PyTorch backend](docs/backends/pytorch.md)
- [API Reference](docs/api_reference.md)

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

## Quick Start

```python
import numpy as np
from pystl import Predicate, Interval, create_semantics

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
from pystl import create_semantics

signal_jax = jnp.asarray(signal)
sem_jax = create_semantics("classical", backend="jax")

rho0_jax = phi.evaluate(signal_jax, sem_jax, t=0)
grad0 = jax.grad(lambda s: phi.evaluate(s, sem_jax, t=0))(signal_jax)
```

NumPy D-GMSR semantics + gradients:

```python
sem_dgmsr = create_semantics("dgmsr", backend="numpy", eps=1e-8, p=2)
rho0, grad0 = phi.evaluate_with_grad(signal, sem_dgmsr, t=0)
print(grad0.shape)  # (time, state_dim)
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


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
