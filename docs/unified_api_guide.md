---
title: Unified API Guide
description: How to create STL formulas and choose a semantics backend
---

# Unified API Guide: Formulas and Semantics

This guide explains how to:
- build STL formulas with the unified API
- choose a semantics backend
- evaluate formulas consistently across backends

## Quick Start

```python
import numpy as np
from stl import Predicate, Interval, create_semantics

signal = np.array(
    [
        [0.2, 0.8],
        [0.3, 0.6],
        [0.5, 0.4],
        [0.7, 0.3],
    ],
    dtype=float,
)

p_speed_ok = Predicate("speed_ok", fn=lambda s, t: 0.6 - s[t, 0])
p_alt_ok = Predicate("alt_ok", fn=lambda s, t: s[t, 1] - 0.2)

phi = (p_speed_ok & p_alt_ok).always(Interval(0, 2))
sem = create_semantics("classic")
rho0 = phi.evaluate(signal, sem, t=0)
print(rho0)
```

## Creating Formulas

The API is in `stl/api.py`. Core classes:
- `Predicate(name, fn)`
- Boolean: `~phi`, `phi1 & phi2`, `phi1 | phi2`
- Temporal:
  - `phi.always((start, end))`
  - `phi.eventually((start, end))`
  - `phi1.until(phi2, interval=(start, end))`
- `Interval(start, end)` is equivalent to `(start, end)`

Notes:
- `end=None` means unbounded to the end of the sampled horizon.
- Temporal windows are computed relative to evaluation time `t`.
- `Predicate.fn` must be a callable `fn(signal, t) -> scalar`.

## Choosing a Semantics Backend

Available names:
- `classic`
- `traditional`
- `cumulative`
- `ctstl`
- `dgmsr`
- `jax`
- `jax_classic`
- `jax_traditional`
- `jax_cumulative`
- `jax_ctstl`
- `jax_dgmsr`
- `jax_stljax`
- `stljax`

You can list them at runtime:

```python
from stl import registry
print(registry.names())
```

### Which One Should You Use?

- `classic`: standard hard max-min robustness. Use as baseline/reference.
- `traditional`: same hard robustness as classic, with explicit traditional naming.
- `cumulative`: returns positive/negative cumulative robustness (`rho+`, `rho-`).
- `ctstl`: CT-STL quantitative semantics (`until` as in `ctstl.py`) and cumulative-time helper `C^tau`.
- `dgmsr`: smooth/differentiable semantics with optional weights.
- `stljax`: stljax-compatible backend (hard or approximate min/max depending on stljax settings).

### Semantics Comparison (Optimization View)

| STL Semantic | Sound (Sign-Preserving)? | Smooth | What Makes It Special | Citation |
| --- | --- | --- | --- | --- |
| Classical (Space) Robustness | Yes | No | Uses `min`/`max` and `inf`/`sup` style operators to compute signed distance to violation (worst-case semantics). | Donze, A., & Maler, O. (2010). *Robust Satisfaction of Temporal Logic over Real-Valued Signals*. In Formal Modeling and Analysis of Timed Systems (FORMATS). [doi:10.1007/978-3-642-15297-9_9](https://doi.org/10.1007/978-3-642-15297-9_9) |
| Smooth (Log-Sum-Exp / Soft) Robustness | Not guaranteed (depends on parameter regime) | Yes | Replaces hard `min`/`max` with smooth approximations for gradient-based learning. | Leung K, Arechiga N, Pavone M. *Backpropagation through signal temporal logic specifications: Infusing logical structure into gradient-based methods*. The International Journal of Robotics Research. 2022;42(6):356-370. [doi:10.1177/02783649221082115](https://doi.org/10.1177/02783649221082115) |
| Cumulative Robustness | No (redefines satisfaction) | Usually no (piecewise) | Integrates robustness over time instead of only worst-case aggregation; captures sustained behavior. | Haghighi, I., Mehdipour, N., Bartocci, E., & Belta, C. (2019). *Control from Signal Temporal Logic Specifications with Smooth Cumulative Quantitative Semantics*. IEEE Control Systems Letters. [arXiv:1904.11611](https://arxiv.org/abs/1904.11611) |
| Sound Cumulative (CT-STL) | Yes | No (operator-level kinks) | Adds cumulative-time semantics with sound/complete qualitative correspondence in the CT-STL setting. | Chen, H., Zhang, Z., Roy, S., Bartocci, E., Smolka, S. A., Stoller, S. D., & Lin, S. (2025). *Cumulative-Time Signal Temporal Logic*. [arXiv:2504.10325](https://arxiv.org/abs/2504.10325) |
| D-GMSR Robustness | Yes | Mostly yes (except boundary points) | Reformulates `min`/`max` with structured generalized means; smooth while preserving sign semantics. | Uzun, S., et al. (2024). *Discrete Generalized Mean Smooth Robustness (D-GMSR) for Signal Temporal Logic*. [arXiv:2405.10996](https://arxiv.org/abs/2405.10996) |

Backend mapping in this repo:
- Classical: `classic`, `traditional`, `jax_classic`, `jax_traditional`
- Smooth soft variants: `stljax`, `jax_stljax`, and `jax` with `smooth=True`
- Cumulative: `cumulative`, `jax_cumulative`
- CT-STL: `ctstl`, `jax_ctstl`
- D-GMSR: `dgmsr`, `jax_dgmsr`

### Return Types

- `classic`, `traditional`, `ctstl`, `dgmsr`, `stljax`: return `float`
- `jax`: returns a scalar `jax.Array`
- `cumulative`: returns `CumulativeRobustness(pos: float, neg: float)`

## Semantics Usage Examples

### 1) Classic / Traditional

```python
sem_classic = create_semantics("classic")
sem_traditional = create_semantics("traditional")

rho_classic = phi.evaluate(signal, sem_classic, t=0)
rho_traditional = phi.evaluate(signal, sem_traditional, t=0)
```

### 2) Cumulative

```python
sem_cum = create_semantics("cumulative")
val = phi.evaluate(signal, sem_cum, t=0)
print(val.pos, val.neg)
```

### 3) CT-STL

```python
sem_ctstl = create_semantics("ctstl", delta=1.0)
rho_until = p_speed_ok.until(p_alt_ok, interval=(0, 3)).evaluate(signal, sem_ctstl, t=0)

# CT cumulative-time robustness C^tau over a sampled window:
window_vals = [p_speed_ok.evaluate(signal, sem_ctstl, t=i) for i in [0, 1, 2, 3]]
rho_c_tau = sem_ctstl.temporal_cumulative(window_vals, tau=2.0)
print(rho_until, rho_c_tau)
```

### 4) D-GMSR (weighted smooth robustness)

```python
from stl import And

sem_dgmsr = create_semantics("dgmsr", eps=1e-8, p=2)
phi_weighted = And(p_speed_ok, p_alt_ok, weights=[1.0, 2.0])
rho = phi_weighted.evaluate(signal, sem_dgmsr, t=0)
```

### 5) stljax

```python
sem_stljax = create_semantics("stljax", approx_method="true", temperature=None)
rho = phi.evaluate(signal, sem_stljax, t=0)
```

### 6) jax (autograd)

```python
import jax
import jax.numpy as jnp

sem_jax = create_semantics("jax", smooth=False)
rho = phi.evaluate(jnp.asarray(signal), sem_jax, t=0)
grad = jax.grad(lambda s: phi.evaluate(s, sem_jax, t=0))(jnp.asarray(signal))
```

Other JAX-native semantics:
- `create_semantics("jax_classic")`
- `create_semantics("jax_traditional")`
- `create_semantics("jax_cumulative")`
- `create_semantics("jax_ctstl", delta=1.0)`
- `create_semantics("jax_dgmsr", eps=1e-8, p=1)`
- `create_semantics("jax_stljax", approx_method="true", temperature=None)`

## Weights Support Summary

- `dgmsr`: supports weights in Boolean/temporal aggregations.
- `jax_dgmsr`: supports weights in Boolean/temporal aggregations.
- `jax`, `jax_classic`, `jax_traditional`, `jax_cumulative`, `jax_ctstl`: currently ignore explicit weights.
- `jax_stljax`: rejects explicit weights (`ValueError`).
- `stljax`: rejects explicit weights (`ValueError`).
- `classic`, `traditional`, `cumulative`, `ctstl`: currently do not use weights (treated as unweighted).

## Common Errors

- Empty temporal window:
  - `ALWAYS` / `EVENTUALLY` raise `ValueError`.
- `Until` with `t >= horizon`:
  - raises `IndexError`.
- Missing predicate function:
  - raises `ValueError` in all backends.
- Using weights with stljax:
  - raises `ValueError`.

## Practical Recommendation

Start with `classic` for correctness checks, then switch to:
- `traditional` if you want explicit traditional naming
- `cumulative` for `rho+` / `rho-`
- `ctstl` for `C^tau`-style robustness workflows
- `dgmsr` or `stljax` when you need smooth/approximate behavior
