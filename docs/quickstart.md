---
title: Quick Start
description: Get up and running with PySTL in minutes
---

# Quick Start

This guide walks through the core concepts of PySTL with runnable examples.

## Signals

A signal is a 2-D NumPy array (or JAX/PyTorch equivalent) of shape `(time, state_dim)`. Each row is one time step.

```python
import numpy as np

# 4 time steps, 2 state dimensions: [speed, altitude]
signal = np.array([
    [0.2, 0.8],
    [0.3, 0.6],
    [0.5, 0.4],
    [0.7, 0.3],
], dtype=float)
```

## Predicates

A `Predicate` is the atomic building block of an STL formula. Its `fn` maps `(signal, t)` to a scalar: positive means satisfied, negative means violated.

```python
from pystl import Predicate

# speed < 0.6  →  0.6 - signal[t, 0] > 0 when satisfied
p_speed = Predicate("speed_ok", fn=lambda s, t: 0.6 - s[t, 0])

# altitude > 0.2  →  signal[t, 1] - 0.2 > 0 when satisfied
p_alt = Predicate("alt_ok", fn=lambda s, t: s[t, 1] - 0.2)
```

## Building formulas

Boolean and temporal operators compose predicates into full STL formulas.

```python
from pystl import Interval

# Both conditions must hold at every time step in [0, 3]
phi = (p_speed & p_alt).always(Interval(0, 3))
```

Available operators:

| Syntax | Meaning |
|---|---|
| `~phi` | NOT |
| `phi1 & phi2` | AND |
| `phi1 \| phi2` | OR |
| `phi.always(interval)` | Always (G) |
| `phi.eventually(interval)` | Eventually (F) |
| `phi1.until(phi2, interval)` | Until (U) |

`Interval(start, end)` is a closed integer interval relative to the evaluation time `t`. Set `end=None` for an open-ended horizon.

## Evaluating a formula

Choose a semantics and call `evaluate`:

```python
from pystl import create_semantics

sem = create_semantics("classical", backend="numpy")
rho = phi.evaluate(signal, sem, t=0)
print(float(rho))  # robustness at t=0
```

A positive value means the formula is satisfied; negative means violated. The magnitude indicates how robustly.

## Switching semantics

The same formula works with any semantics — just swap `create_semantics`:

```python
sem_smooth = create_semantics("smooth", backend="numpy", temperature=0.5)
rho_smooth = phi.evaluate(signal, sem_smooth, t=0)

sem_agm = create_semantics("agm", backend="numpy")
rho_agm = phi.evaluate(signal, sem_agm, t=0)
```

## Gradients with NumPy (D-GMSR)

If you want gradients without JAX/PyTorch, use D-GMSR with the NumPy backend:

```python
sem_dgmsr = create_semantics("dgmsr", backend="numpy", eps=1e-8, p=2)
rho0, grad0 = phi.evaluate_with_grad(signal, sem_dgmsr, t=0)
print(grad0.shape)  # (time, state_dim)
```

## Gradients with JAX

Install the JAX extra (`uv sync --extra jax`), then use any JAX-compatible semantics:

```python
import jax
import jax.numpy as jnp
from pystl import create_semantics

signal_jax = jnp.asarray(signal)
sem_jax = create_semantics("smooth", backend="jax", temperature=0.5)

# Robustness at t=0
rho = phi.evaluate(signal_jax, sem_jax, t=0)

# Gradient of robustness w.r.t. the signal
grad = jax.grad(lambda s: phi.evaluate(s, sem_jax, t=0))(signal_jax)
print(grad.shape)  # (4, 2) — same shape as the signal
```

## Next steps

- [A Unified API](./unified_api.md) — full operator and semantics reference
- [JAX backend](./backends/jax.md) — gradients, JIT, and JAX-specific tips
- [PyTorch backend](./backends/pytorch.md) — autograd and GPU usage
- [API Reference](./api_reference.md) — complete class and function docs
