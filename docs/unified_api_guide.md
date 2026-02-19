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

### Return Types

- `classic`, `traditional`, `ctstl`, `dgmsr`, `stljax`: return `float`
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

## Weights Support Summary

- `dgmsr`: supports weights in Boolean/temporal aggregations.
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
