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

### JAX semantics (autograd-friendly)

```python
import jax
import jax.numpy as jnp
from stl import create_semantics

signal = jnp.asarray(signal)
sem_jax = create_semantics("jax", smooth=False)
rho0 = phi.evaluate(signal, sem_jax, t=0)  # returns a JAX scalar
grad = jax.grad(lambda s: phi.evaluate(s, sem_jax, t=0))(signal)
```

Additional JAX backends:
- `jax_classic`
- `jax_traditional`
- `jax_cumulative`
- `jax_ctstl`
- `jax_dgmsr`
- `jax_stljax`

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
