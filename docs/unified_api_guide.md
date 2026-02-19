---
title: Unified API Guide
description: How to create STL formulas and choose syntax/backend
---

# Unified API Guide: Syntax + Backend Split

The STL API is now split into two independent choices:
- `syntax`: `classical`, `cumulative`, `ctstl`, `dgmsr`
- `backend`: `numpy` (always) and `jax` (when installed with the `jax` extra)

Use `create_semantics(syntax, backend=...)`.

## Installation Modes (`uv`)

```bash
# NumPy-only install
uv sync --dev

# NumPy + JAX install
uv sync --dev --extra jax
```

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
sem = create_semantics("classical", backend="numpy")
rho0 = phi.evaluate(signal, sem, t=0)
print(rho0)
```

## Available Options

```python
from stl import registry

print(registry.syntaxes())  # ['classical', 'ctstl', 'cumulative', 'dgmsr']
print(registry.backends())  # ['numpy'] or ['jax', 'numpy']
print(registry.names())
# Includes JAX entries when installed with `--extra jax`
```

## Formula Construction

Core API (from `stl/api.py`):
- `Predicate(name, fn)`
- Boolean: `~phi`, `phi1 & phi2`, `phi1 | phi2`
- Temporal:
  - `phi.always((start, end))`
  - `phi.eventually((start, end))`
  - `phi1.until(phi2, interval=(start, end))`
- `Interval(start, end)` is equivalent to tuple intervals

Notes:
- `end=None` means unbounded to horizon end.
- Windows are relative to evaluation time `t`.
- `Predicate.fn` must be `fn(signal, t) -> scalar`.

## Semantics Matrix

### Classical

```python
sem_np = create_semantics("classical", backend="numpy")
sem_jax = create_semantics("classical", backend="jax")
```

### Cumulative

```python
sem_np = create_semantics("cumulative", backend="numpy")
sem_jax = create_semantics("cumulative", backend="jax")
```

### CT-STL

```python
sem_np = create_semantics("ctstl", backend="numpy", delta=1.0)
sem_jax = create_semantics("ctstl", backend="jax", delta=1.0)
```

### DGMSR

```python
sem_np = create_semantics("dgmsr", backend="numpy", eps=1e-8, p=2)
sem_jax = create_semantics("dgmsr", backend="jax", eps=1e-8, p=2)
```

## Return Types

- `classical`, `ctstl`, `dgmsr`: scalar robustness (`float` for NumPy backend, scalar `jax.Array` for JAX backend)
- `cumulative`: robustness container with `.pos` and `.neg`

## JAX Gradients

```python
import jax
import jax.numpy as jnp

signal_jax = jnp.asarray(signal)
sem_jax = create_semantics("classical", backend="jax")

rho = phi.evaluate(signal_jax, sem_jax, t=0)
grad = jax.grad(lambda s: phi.evaluate(s, sem_jax, t=0))(signal_jax)
```

## Common Errors

- Unknown syntax/backend: `KeyError`
- Requesting `backend="jax"` without JAX extra installed: `ImportError`
- Empty temporal window (`always`/`eventually`): `ValueError`
- `until` with empty trace window: `ValueError`
- Missing `Predicate.fn`: `ValueError`
