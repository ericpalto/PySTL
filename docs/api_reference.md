---
title: API Reference
description: Complete reference for all PySTL classes and functions
---

# API Reference

All public symbols are importable from the top-level `pystl` package.

```python
from pystl import (
    Predicate, Interval,
    Formula, Not, And, Or, Always, Eventually, Until,
    create_semantics, registry,
)
```

---

## `Interval`

```python
@dataclass(frozen=True)
class Interval:
    start: int = 0
    end: Optional[int] = None
```

A closed integer interval `[start, end]`. Both bounds are inclusive. When `end=None`, the interval extends to the last time step of the signal.

**Raises:** `ValueError` if `start < 0` or `end < start`.

Tuples `(start, end)` are accepted anywhere an `Interval` is expected.

---

## `Predicate`

```python
@dataclass(frozen=True)
class Predicate(Formula):
    name: str
    fn: Optional[Callable[[Signal, int], Any]] = None
    metadata: Optional[dict] = None
    grad: Optional[Callable[[Signal, int], Any]] = None
```

Atomic STL predicate. The function `fn(signal, t)` must return a scalar. A positive value indicates the predicate is satisfied at time `t`; a negative value indicates a violation. The magnitude is the robustness margin.

**Example:**

```python
p = Predicate("x_above_zero", fn=lambda s, t: s[t, 0])
```

**Raises:** `ValueError` at evaluation time if `fn` is `None`.

If provided, `grad(signal, t)` should return the gradient of `fn` w.r.t. the state at time `t` (shape `(state_dim,)`). This is used by semantics/backends that support explicit gradients (e.g., NumPy D-GMSR via `Formula.evaluate_with_grad`).

---

## `Formula`

Abstract base class for all STL formulas.

### `Formula.evaluate(signal, semantics, t=0)`

Evaluate the formula on `signal` at time `t` using the given `semantics`.

- `signal`: array of shape `(time, state_dim)`
- `semantics`: a `Semantics` instance (from `create_semantics`)
- `t`: evaluation time step (default `0`)

Returns a scalar robustness value (type depends on the backend and semantics).

### `Formula.evaluate_with_grad(signal, semantics, t=0, **kwargs)`

Evaluate robustness and the gradient w.r.t. the full signal trace.

This is only supported by some semantics/backends (currently: D-GMSR with the NumPy backend). For JAX/PyTorch backends, prefer their native autodiff (`jax.grad` / `torch.autograd`).

### Operator shorthands

| Expression | Equivalent |
|---|---|
| `~phi` | `Not(phi)` |
| `phi1 & phi2` | `And(phi1, phi2)` |
| `phi1 \| phi2` | `Or(phi1, phi2)` |
| `phi.always(interval)` | `Always(phi, interval)` |
| `phi.eventually(interval)` | `Eventually(phi, interval)` |
| `phi1.until(phi2, interval)` | `Until(phi1, phi2, interval)` |

---

## `Not`

```python
@dataclass(frozen=True)
class Not(Formula):
    child: Formula
```

Logical negation. Robustness is negated: `rho(~phi) = -rho(phi)`.

---

## `And`

```python
class And(Formula):
    def __init__(self, *children: Formula, weights=None): ...
```

Logical conjunction. Aggregates children using the semantics' AND operator (e.g., `min` for classical).

- `weights`: optional sequence of floats controlling per-child importance. Supported by AGM and D-GMSR; ignored by Classical and Smooth.

---

## `Or`

```python
class Or(Formula):
    def __init__(self, *children: Formula, weights=None): ...
```

Logical disjunction. Aggregates children using the semantics' OR operator (e.g., `max` for classical).

- `weights`: same semantics as for `And`.

---

## `Always`

```python
class Always(Formula):
    def __init__(self, child: Formula, interval=Interval(), *, weights=None): ...
```

Temporal ALWAYS (`G`). The formula must hold at every time step in the window.

- `interval`: `Interval` or `(start, end)` tuple, relative to evaluation time `t`
- `weights`: optional per-time-step weights

**Raises:** `ValueError` if the window is empty given the signal horizon.

---

## `Eventually`

```python
class Eventually(Formula):
    def __init__(self, child: Formula, interval=Interval(), *, weights=None): ...
```

Temporal EVENTUALLY (`F`). The formula must hold at some time step in the window.

- `interval`: `Interval` or `(start, end)` tuple
- `weights`: optional per-time-step weights

**Raises:** `ValueError` if the window is empty.

---

## `Until`

```python
class Until(Formula):
    def __init__(
        self,
        left: Formula,
        right: Formula,
        interval=Interval(),
        *,
        weights_left=None,
        weights_right=None,
        weights_pair=(1.0, 1.0),
    ): ...
```

Temporal UNTIL (`U`). `left` must hold until `right` holds, within the interval.

- `weights_left`: weights over `left`'s prefix trace
- `weights_right`: weights over candidate satisfaction time points
- `weights_pair`: relative importance of `(left prefix robustness, right robustness)` pair

**Raises:** `ValueError` if the window is empty or `t` is out of bounds.

---

## `create_semantics`

```python
def create_semantics(syntax: str, *, backend: str = "numpy", **kwargs) -> Semantics:
```

Create a semantics instance by name.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `syntax` | `str` | One of `"classical"`, `"smooth"`, `"cumulative"`, `"agm"`, `"dgmsr"` |
| `backend` | `str` | One of `"numpy"`, `"jax"`, `"torch"` |
| `temperature` | `float` | *(smooth only)* Softmin/softmax temperature. Default `1.0`. Must be `> 0`. |
| `eps` | `float` | *(dgmsr only)* Numerical stability floor. Default `1e-8`. |
| `p` | `int` | *(dgmsr only)* Generalized mean order. Default `1`. |

**Raises:**
- `KeyError` if the `syntax`/`backend` combination is not registered
- `ImportError` if the JAX or PyTorch extra is not installed

**Example:**

```python
sem = create_semantics("smooth", backend="jax", temperature=0.25)
```

---

## `registry`

A global `SemanticsRegistry` instance.

### `registry.syntaxes() -> list[str]`

Returns the list of registered syntax names.

### `registry.backends() -> list[str]`

Returns the list of registered backend names (depends on installed extras).

### `registry.names() -> list[str]`

Returns all registered `"syntax/backend"` strings.

**Example:**

```python
from pystl import registry

registry.syntaxes()  # ['agm', 'classical', 'cumulative', 'dgmsr', 'smooth']
registry.backends()  # ['numpy'] or ['jax', 'numpy', 'torch']
registry.names()     # ['agm/jax', 'agm/numpy', 'agm/torch', ...]
```

---

## Semantics classes

These are returned by `create_semantics` and are not typically instantiated directly.

| Class | Syntax | Backend |
|---|---|---|
| `ClassicRobustSemantics` | `classical` | numpy |
| `SmoothRobustSemantics` | `smooth` | numpy |
| `CumulativeSemantics` | `cumulative` | numpy |
| `AgmRobustSemantics` | `agm` | numpy |
| `DgmsrSemantics` | `dgmsr` | numpy |
| `JaxClassicRobustSemantics` | `classical` | jax |
| `JaxSmoothRobustSemantics` | `smooth` | jax |
| `JaxCumulativeSemantics` | `cumulative` | jax |
| `JaxAgmRobustSemantics` | `agm` | jax |
| `JaxDgmsrSemantics` | `dgmsr` | jax |
| `TorchClassicRobustSemantics` | `classical` | torch |
| `TorchSmoothRobustSemantics` | `smooth` | torch |
| `TorchCumulativeSemantics` | `cumulative` | torch |
| `TorchAgmRobustSemantics` | `agm` | torch |
| `TorchDgmsrSemantics` | `dgmsr` | torch |

---

## `CumulativeRobustness`

```python
@dataclass(frozen=True)
class CumulativeRobustness:
    pos: Any  # positive robustness component
    neg: Any  # negative robustness component
```

Returned by `cumulative` semantics. Both `.pos` and `.neg` carry gradient information when using JAX or PyTorch backends. JAX and PyTorch variants are `JaxCumulativeRobustness` and `TorchCumulativeRobustness` respectively.

---

## Type aliases

```python
Signal = NDArray[np.float64]       # shape: (time, state_dim)
PredicateFn = Callable[[Signal, int], Any]
```
