---
title: STL Documentation
description: Unified STL API and semantics backends
---

# STL Documentation

This documentation is structured for GitHub-hosted docs (Markdown pages in `docs/` with front matter).

## Contents

- [Unified API Guide](./unified_api_guide.md)

## Quick Example

```python
import numpy as np
from stl import Predicate, Interval, create_semantics

signal = np.array(
    [
        [0.2, 0.8],
        [0.3, 0.6],
        [0.5, 0.4],
    ],
    dtype=float,
)

p1 = Predicate("speed_ok", fn=lambda s, t: 0.6 - s[t, 0])
p2 = Predicate("alt_ok", fn=lambda s, t: s[t, 1] - 0.2)

phi = (p1 & p2).always(Interval(0, 2))
sem = create_semantics("classical", backend="numpy")
rho = phi.evaluate(signal, sem, t=0)
print(rho)
```
