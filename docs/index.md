---
title: PySTL Documentation
description: Unified STL robustness semantics for NumPy, JAX, and PyTorch
---

# PySTL

A unified API for **Signal Temporal Logic (STL) robustness evaluation** across multiple semantics and computational backends.

PySTL consolidates five STL quantitative semantics — Classical, Smooth, Cumulative, AGM, and D-GMSR — under one consistent interface. Switch semantics or backends without rewriting your formulas.

```python
from pystl import Predicate, Interval, create_semantics

phi = Predicate("x_positive", fn=lambda s, t: s[t, 0]).always(Interval(0, 4))

rho_classical = phi.evaluate(signal, create_semantics("classical", backend="numpy"))
rho_smooth    = phi.evaluate(signal, create_semantics("smooth",    backend="jax"))
rho_agm       = phi.evaluate(signal, create_semantics("agm",       backend="torch"))
```

## Contents

- [Installation](./installation.md)
- [Quick Start](./quickstart.md)
- [A Unified API](./unified_api.md)
- Backends
  - [JAX](./backends/jax.md)
  - [PyTorch](./backends/pytorch.md)
- [API Reference](./api_reference.md)

## Supported Semantics

| Semantic | Smooth | Sign-preserving | Key idea |
|---|---|---|---|
| `classical` | No | Yes | `min`/`max` worst-case robustness |
| `smooth` | Yes | Approx. | Softmin/softmax via `logsumexp` |
| `cumulative` | Piecewise | No | Integrates robustness over time |
| `agm` | Piecewise | Yes | Arithmetic-Geometric Mean blending |
| `dgmsr` | Mostly yes | Yes | Generalized means, smooth + sign-safe |

Each semantic is available on three backends: `numpy`, `jax`, and `torch`.
