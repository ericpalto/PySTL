---
title: JAX Backend
description: Using PySTL with JAX for automatic differentiation and GPU acceleration
---

# JAX Backend

The JAX backend makes every supported STL semantics natively differentiable. Gradients flow through robustness evaluations via `jax.grad`, and computations can be JIT-compiled or run on GPU/TPU without any changes to the formula.

## Installation

```bash
uv sync --extra jax
# or
pip install -e ".[jax]"
```

## Basic usage

Use any JAX array as the signal. Pass `backend="jax"` to `create_semantics`:

```python
import jax.numpy as jnp
from pystl import Predicate, Interval, create_semantics

signal = jnp.array([
    [0.2, 0.8],
    [0.3, 0.6],
    [0.5, 0.4],
    [0.7, 0.3],
])

p_speed = Predicate("speed_ok", fn=lambda s, t: 0.6 - s[t, 0])
p_alt   = Predicate("alt_ok",   fn=lambda s, t: s[t, 1] - 0.2)
phi     = (p_speed & p_alt).always(Interval(0, 3))

sem = create_semantics("classical", backend="jax")
rho = phi.evaluate(signal, sem, t=0)
print(float(rho))
```

## Gradients

`jax.grad` works directly on the `evaluate` call. The gradient has the same shape as the signal and represents how the robustness changes with each signal value.

```python
import jax

grad_fn = jax.grad(lambda s: phi.evaluate(s, sem, t=0))
grad = grad_fn(signal)
print(grad.shape)  # (4, 2)
```

For semantics with non-differentiable operations (e.g., `classical` uses hard `min`/`max`), JAX will still compute a subgradient. For fully smooth gradients, use the `smooth` or `dgmsr` semantics:

```python
sem_smooth = create_semantics("smooth", backend="jax", temperature=0.5)
grad_smooth = jax.grad(lambda s: phi.evaluate(s, sem_smooth, t=0))(signal)
```

## JIT compilation

Wrap the evaluation in `jax.jit` to compile it for faster repeated evaluation:

```python
import functools

@jax.jit
def robustness(s):
    return phi.evaluate(s, sem_smooth, t=0)

# First call compiles; subsequent calls are fast
rho = robustness(signal)
```

Note: `jax.jit` traces the function with abstract values. The formula and semantics are captured as constants at trace time. If you change the formula or semantics, re-JIT.

## Value and gradient in one call

Use `jax.value_and_grad` to compute robustness and its gradient in a single forward pass:

```python
rho, grad = jax.value_and_grad(lambda s: phi.evaluate(s, sem_smooth, t=0))(signal)
```

## Batching with vmap

To evaluate over a batch of signals, use `jax.vmap`:

```python
signals = jnp.stack([signal, signal * 0.9, signal * 1.1])  # batch of 3

batched_eval = jax.vmap(lambda s: phi.evaluate(s, sem_smooth, t=0))
rhos = batched_eval(signals)
print(rhos.shape)  # (3,)
```

## Semantics-specific notes

### Smooth semantics

The `temperature` parameter controls the softmin/softmax approximation. Lower values track the classical `min`/`max` more closely but can cause numerical issues with very small temperatures.

```python
sem = create_semantics("smooth", backend="jax", temperature=0.25)
```

### Cumulative semantics

Returns a `JaxCumulativeRobustness` object with `.pos` and `.neg` fields. Both are differentiable JAX arrays.

```python
sem_cum = create_semantics("cumulative", backend="jax")
rho = phi.evaluate(signal, sem_cum, t=0)

# Differentiate w.r.t. the positive robustness component
grad_pos = jax.grad(lambda s: phi.evaluate(s, sem_cum, t=0).pos)(signal)
```

### D-GMSR semantics

Mostly smooth (except at exact sign boundaries). Supports `eps` and `p` parameters:

```python
sem_dgmsr = create_semantics("dgmsr", backend="jax", eps=1e-8, p=2)
grad = jax.grad(lambda s: phi.evaluate(s, sem_dgmsr, t=0))(signal)
```

### AGM semantics

Piecewise smooth with a conditional branch at sign boundaries. Gradients are well-defined almost everywhere.

```python
sem_agm = create_semantics("agm", backend="jax")
```

## GPU / TPU

JAX automatically runs on GPU or TPU when available. No changes to PySTL code are needed — just ensure JAX is installed with the appropriate accelerator support (e.g., `jax[cuda12]`).
