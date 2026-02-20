---
title: A Unified API
description: How PySTL separates formula syntax from evaluation semantics
---

# A Unified API

PySTL is built around a clean separation between two independent concerns:

- **Formula syntax** — what the specification says (predicates, operators, intervals)
- **Evaluation semantics** — how robustness is computed (which aggregation functions are used, and with which backend)

The same formula object can be evaluated under any combination of semantics and backend. This makes it straightforward to compare semantics, switch backends, or plug robustness into gradient-based pipelines.

## Formulas

### Predicate

```python
from pystl import Predicate

p = Predicate("name", fn=lambda signal, t: signal[t, 0] - 1.0)
```

The `fn` argument is a callable `(signal, t) -> scalar`. It should return a positive value when the predicate is satisfied and a negative value when violated. The magnitude carries meaning: it represents the margin from the satisfaction boundary.

### Boolean operators

```python
phi1 = Predicate("a", fn=lambda s, t: s[t, 0])
phi2 = Predicate("b", fn=lambda s, t: s[t, 1])

not_phi1      = ~phi1
phi1_and_phi2 = phi1 & phi2
phi1_or_phi2  = phi1 | phi2
```

You can chain as many operands as needed:

```python
from pystl import And, Or

conjunction = And(phi1, phi2, phi3)
disjunction = Or(phi1, phi2, phi3)
```

### Temporal operators

All temporal operators take an `Interval(start, end)`, which is a closed integer interval **relative to the evaluation time `t`**. Setting `end=None` extends to the end of the signal.

```python
from pystl import Interval

# phi holds at every step from t to t+4
phi.always(Interval(0, 4))

# phi holds at some step from t+1 to t+3
phi.eventually(Interval(1, 3))

# phi holds until psi holds, within [t, t+5]
phi.until(psi, Interval(0, 5))
```

Tuple shorthand is also accepted: `phi.always((0, 4))`.

#### Weighted operators

`And`, `Or`, `always`, and `eventually` accept an optional `weights` argument to scale the contribution of each operand or time step. Weights are semantics-dependent — they are used by semantics that support them (e.g., AGM, D-GMSR) and ignored by those that don't (e.g., Classical).

```python
# Give more weight to phi2 in the conjunction
phi1 & phi2  # standard; use And(..., weights=[0.3, 0.7]) for weighted version
And(phi1, phi2, weights=[0.3, 0.7])

# Emphasize later time steps in Always
phi.always(Interval(0, 4), weights=[0.1, 0.2, 0.3, 0.4, 0.5])
```

`until` supports three weight sequences:

```python
phi.until(
    psi,
    Interval(0, 4),
    weights_left=[...],   # weights on phi's prefix trace
    weights_right=[...],  # weights over candidate satisfaction time points
    weights_pair=[1.0, 1.0],  # relative weight of (phi prefix, psi value) pair
)
```

## Semantics

Choose a semantics with `create_semantics(syntax, backend=..., **kwargs)`. The syntax and backend are independent choices.

```python
from pystl import create_semantics

sem = create_semantics("classical", backend="numpy")
```

### Classical robustness

The standard worst-case semantics from Donzé & Maler (2010). Uses `min`/`max` aggregation. Non-smooth, but sound and sign-preserving.

```python
sem = create_semantics("classical", backend="numpy")
sem = create_semantics("classical", backend="jax")
sem = create_semantics("classical", backend="torch")
```

### Smooth robustness

Replaces `min`/`max` with softmin/softmax (`logsumexp`). Differentiable everywhere, which makes it suitable for gradient-based optimization. The `temperature` parameter controls how closely the smooth approximation tracks the classical one: lower temperature → closer to classical, but numerically less stable.

```python
sem = create_semantics("smooth", backend="numpy", temperature=0.5)
sem = create_semantics("smooth", backend="jax",   temperature=0.5)
sem = create_semantics("smooth", backend="torch",  temperature=0.5)
```

### Cumulative robustness

Integrates robustness over time instead of taking worst-case aggregations. Captures sustained behavior more faithfully than classical semantics. Returns a `CumulativeRobustness` container with `.pos` and `.neg` components.

```python
sem = create_semantics("cumulative", backend="numpy")
rho = phi.evaluate(signal, sem, t=0)
print(rho.pos, rho.neg)
```

### AGM robustness

Arithmetic-Geometric Mean robustness (Mehdipour et al., 2019). Rewards both the *degree* and the *frequency* of satisfaction across operands and time. Piecewise smooth; sign-preserving. Supports weighted operators.

```python
sem = create_semantics("agm", backend="numpy")
sem = create_semantics("agm", backend="jax")
sem = create_semantics("agm", backend="torch")
```

### D-GMSR robustness

Discrete Generalized Mean Smooth Robustness (Uzun et al., 2024). Reformulates `min`/`max` with structured generalized means to be smooth while preserving sign semantics. Configurable with `eps` (numerical stability) and `p` (mean order).

```python
sem = create_semantics("dgmsr", backend="numpy", eps=1e-8, p=2)
sem = create_semantics("dgmsr", backend="jax",   eps=1e-8, p=2)
sem = create_semantics("dgmsr", backend="torch",  eps=1e-8, p=2)
```

## Semantics matrix

All 15 syntax/backend combinations are available when the corresponding extras are installed:

| Syntax | numpy | jax | torch |
|---|:---:|:---:|:---:|
| `classical` | ✓ | ✓ | ✓ |
| `smooth` | ✓ | ✓ | ✓ |
| `cumulative` | ✓ | ✓ | ✓ |
| `agm` | ✓ | ✓ | ✓ |
| `dgmsr` | ✓ | ✓ | ✓ |

## Return types

| Semantics | Return type |
|---|---|
| `classical` | `float` (numpy) / scalar `jax.Array` / scalar `torch.Tensor` |
| `smooth` | same as classical |
| `agm` | same as classical |
| `dgmsr` | same as classical |
| `cumulative` | `CumulativeRobustness` with `.pos` and `.neg` |

## Introspecting the registry

```python
from pystl import registry

registry.syntaxes()  # ['agm', 'classical', 'cumulative', 'dgmsr', 'smooth']
registry.backends()  # depends on installed extras
registry.names()     # all available 'syntax/backend' combinations
```

## Common errors

| Error | Cause |
|---|---|
| `KeyError` | Unknown syntax or backend string |
| `ImportError` | Requested `jax` or `torch` backend without installing the extra |
| `ValueError` | Empty temporal window, invalid interval, missing `fn` on a predicate |
