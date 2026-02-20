---
title: PyTorch Backend
description: Using PySTL with PyTorch for autograd and GPU acceleration
---

# PyTorch Backend

The PyTorch backend integrates PySTL with PyTorch's autograd engine. Robustness values are `torch.Tensor` scalars, and gradients flow through the evaluation graph automatically via standard `.backward()` or `torch.autograd.grad`.

## Installation

```bash
uv sync --extra torch
# or
pip install -e ".[torch]"
```

## Basic usage

Pass PyTorch tensors as the signal and use `backend="torch"`:

```python
import torch
from pystl import Predicate, Interval, create_semantics

signal = torch.tensor([
    [0.2, 0.8],
    [0.3, 0.6],
    [0.5, 0.4],
    [0.7, 0.3],
], dtype=torch.float64)

p_speed = Predicate("speed_ok", fn=lambda s, t: 0.6 - s[t, 0])
p_alt   = Predicate("alt_ok",   fn=lambda s, t: s[t, 1] - 0.2)
phi     = (p_speed & p_alt).always(Interval(0, 3))

sem = create_semantics("classical", backend="torch")
rho = phi.evaluate(signal, sem, t=0)
print(rho.item())
```

## Gradients

Enable gradient tracking on the signal tensor and call `.backward()`:

```python
signal = torch.tensor([
    [0.2, 0.8],
    [0.3, 0.6],
    [0.5, 0.4],
    [0.7, 0.3],
], dtype=torch.float64, requires_grad=True)

sem_smooth = create_semantics("smooth", backend="torch", temperature=0.5)
rho = phi.evaluate(signal, sem_smooth, t=0)
rho.backward()

print(signal.grad)  # shape (4, 2)
```

Alternatively, use `torch.autograd.grad` to avoid in-place gradient accumulation:

```python
(grad,) = torch.autograd.grad(rho, signal)
print(grad.shape)  # (4, 2)
```

## GPU usage

Move the signal to a CUDA device as usual:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
signal = signal.to(device)

sem = create_semantics("smooth", backend="torch", temperature=0.5)
rho = phi.evaluate(signal, sem, t=0)
```

The robustness value and any intermediate tensors will reside on the same device as the input signal.

## Integration with nn.Module

PySTL can be embedded inside a PyTorch training loop. For example, use robustness as a soft constraint loss:

```python
import torch.nn as nn
from pystl import create_semantics

class STLConstrainedModel(nn.Module):
    def __init__(self, formula, semantics):
        super().__init__()
        self.formula = formula
        self.semantics = semantics
        self.linear = nn.Linear(2, 2, dtype=torch.float64)

    def forward(self, x):
        output = self.linear(x)
        rho = self.formula.evaluate(output, self.semantics, t=0)
        return output, rho

sem = create_semantics("smooth", backend="torch", temperature=0.5)
model = STLConstrainedModel(phi, sem)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(100):
    output, rho = model(signal)
    task_loss = output.mean()          # your actual task loss
    stl_loss = -rho                     # maximize robustness
    loss = task_loss + 0.1 * stl_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Semantics-specific notes

### Smooth semantics

Use `temperature` to balance approximation tightness against gradient smoothness:

```python
sem = create_semantics("smooth", backend="torch", temperature=0.25)
```

Lower temperature → closer to `min`/`max`, but gradients may saturate.

### Cumulative semantics

Returns a `TorchCumulativeRobustness` dataclass with `.pos` and `.neg` tensor fields. Both are part of the autograd graph:

```python
sem_cum = create_semantics("cumulative", backend="torch")
rho = phi.evaluate(signal, sem_cum, t=0)

rho.pos.backward()          # gradient w.r.t. positive robustness
# or
rho.neg.backward()          # gradient w.r.t. negative robustness
```

### D-GMSR semantics

Configurable with `eps` (numerical stability floor) and `p` (generalized mean order). Mostly smooth:

```python
sem = create_semantics("dgmsr", backend="torch", eps=1e-8, p=2)
```

### AGM semantics

Piecewise smooth. Supports `weights` on `And`, `Or`, and temporal operators.

```python
sem = create_semantics("agm", backend="torch")
phi_weighted = And(p_speed, p_alt, weights=[0.3, 0.7])
rho = phi_weighted.evaluate(signal, sem, t=0)
```
