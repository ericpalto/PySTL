# STL

A collection of different Signal Temporal Logic (STL) semantics.

Documentation:
- Docs index (GitHub-ready): `docs/index.md`
- Unified API guide: `docs/unified_api_guide.md`

## Unified API Skeleton

```python
import numpy as np
from stl import Predicate, Interval, create_semantics

signal = np.array([
    [0.2, 0.8],
    [0.3, 0.6],
    [0.5, 0.4],
], dtype=float)

p_speed_ok = Predicate("speed_ok", fn=lambda s, t: 0.6 - s[t, 0])
p_alt_ok = Predicate("alt_ok", fn=lambda s, t: s[t, 1] - 0.2)

phi = (p_speed_ok & p_alt_ok).always(Interval(0, 2))
sem = create_semantics("classic")  # or: create_semantics("dgmsr", eps=1e-8, p=1)
rho0 = phi.evaluate(signal, sem, t=0)
```

### stljax wrapper

```python
from stl import create_semantics, StlJaxFormulaWrapper

# Unified API + stljax min/max logic
sem = create_semantics("stljax", approx_method="true", temperature=None)
rho0 = phi.evaluate(signal, sem, t=0)

# Compile unified formula to native stljax formula and evaluate trace
wrapped = StlJaxFormulaWrapper(phi, approx_method="true")
trace = wrapped.robustness_trace(signal)
rho0_native = wrapped.robustness(signal, t=0)
```

## D-GMSR API

```python
import numpy as np
from stl.dgmsr import Predicate, And

# signal shape: (time, state_dim)
# if your trajectory is (state_dim, time) like in the notebook, use `signal.T`
signal = np.array([
    [0.2, 0.8],
    [0.3, 0.6],
    [0.5, 0.4],
], dtype=float)

p_speed_ok = Predicate.affine("speed_ok", coeffs=[-1.0, 0.0], bias=0.6)   # 0.6 - x[0]
p_alt_ok = Predicate.affine("alt_ok", coeffs=[0.0, 1.0], bias=-0.2)       # x[1] - 0.2

phi = And(p_speed_ok, p_alt_ok).always(interval=(0, 2))
rho0, grad0 = phi.robustness_with_grad(signal, t=0)
```

Available operators:
- Boolean: `And`, `Or`, `Not` (or `&`, `|`, `~`)
- Temporal: `Always`, `Eventually`, `Until`
