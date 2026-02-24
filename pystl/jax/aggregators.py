"""JAX smooth min/max aggregators used by PySTL.

These helpers started life as a small compatibility layer for an older local
project. They are now treated as internal PySTL utilities and intentionally
live alongside the rest of the JAX backend.
"""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _resolve_beta(temperature: Optional[float | tuple[float, int]]) -> Optional[float]:
    if temperature is None:
        return None
    if isinstance(temperature, tuple):
        beta, _steps = temperature
        return float(beta)
    return float(temperature)


def _validate_beta(beta: Optional[float], *, where: str) -> float:
    if beta is None:
        raise ValueError(f"{where} requires `temperature` (beta) to be provided.")
    if beta <= 0.0:
        raise ValueError(f"{where} requires temperature(beta) > 0.")
    return float(beta)


def maxish(
    values: Any,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    approx_method: str = "true",
    temperature: Optional[float | tuple[float, int]] = None,
) -> Any:
    """Approximate max.

    Conventions:
    - `approx_method="true"` returns the hard max.
    - `approx_method="logsumexp"` uses inverse-temperature beta: max(x) ≈
      (1/beta) log ∑ exp(beta x).
    """

    arr = jnp.asarray(values, dtype=float)
    method = str(approx_method).strip().lower()
    if method in ("true", "hard", "max"):
        return jnp.max(arr, axis=axis, keepdims=keepdims)
    if method in ("logsumexp", "lse", "softmax"):
        beta = _validate_beta(_resolve_beta(temperature), where="maxish(logsumexp)")
        return (1.0 / beta) * logsumexp(beta * arr, axis=axis, keepdims=keepdims)
    raise ValueError(f"Unknown approx_method={approx_method!r}.")


def minish(
    values: Any,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    approx_method: str = "true",
    temperature: Optional[float | tuple[float, int]] = None,
) -> Any:
    """Approximate min. See `maxish` for conventions."""

    arr = jnp.asarray(values, dtype=float)
    method = str(approx_method).strip().lower()
    if method in ("true", "hard", "min"):
        return jnp.min(arr, axis=axis, keepdims=keepdims)
    if method in ("logsumexp", "lse", "softmin"):
        beta = _validate_beta(_resolve_beta(temperature), where="minish(logsumexp)")
        return -(1.0 / beta) * logsumexp(-beta * arr, axis=axis, keepdims=keepdims)
    raise ValueError(f"Unknown approx_method={approx_method!r}.")


__all__ = ["minish", "maxish"]
