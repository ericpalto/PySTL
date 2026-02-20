"""CT-STL semantics backend.

This backend implements the quantitative semantics from `ctstl.py` for the
operators available in the unified API, and exposes a helper for the CT-STL
time-cumulative operator C^tau_I.
"""

from __future__ import annotations

from math import ceil
from typing import Optional, Sequence

import numpy as np

from .base import Semantics


def kth_largest(values: Sequence[float], k: int) -> float:
    """Return the k-th largest value (1-indexed)."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("values must be non-empty.")
    if not 1 <= k <= arr.size:
        raise ValueError(f"k={k} must be between 1 and len(values)={arr.size}.")
    idx = arr.size - k
    return float(np.partition(arr, idx)[idx])


def tau_to_k(tau: float, delta: float) -> int:
    """Map CT-STL threshold tau to k = ceil(tau/delta)."""
    if delta <= 0:
        raise ValueError("delta must be > 0.")
    if tau <= 0:
        raise ValueError("tau must be > 0.")
    return int(ceil(tau / delta))


class CtstlSemantics(Semantics[float]):
    """CT-STL quantitative semantics.

    Notes:
    - `predicate`, `not`, `and`, `or` follow standard robustness composition.
    - `until` matches `ctstl.py`: prefix for left operand is over [t, t') (exclusive).
    - `temporal_cumulative` provides robustness for C^tau_I as k-th largest.
    """

    def __init__(self, *, delta: float = 1.0) -> None:
        if delta <= 0:
            raise ValueError("delta must be > 0.")
        self.delta = float(delta)

    def predicate(self, predicate, signal, t: int) -> float:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        return float(predicate.fn(signal, t))

    def boolean_not(self, value: float) -> float:
        return -float(value)

    def boolean_and(
        self, values: Sequence[float], *, weights: Optional[Sequence[float]] = None
    ) -> float:
        if len(values) == 0:
            raise ValueError("boolean_and requires at least one value.")
        return float(np.min(np.asarray(values, dtype=float)))

    def boolean_or(
        self, values: Sequence[float], *, weights: Optional[Sequence[float]] = None
    ) -> float:
        if len(values) == 0:
            raise ValueError("boolean_or requires at least one value.")
        return float(np.max(np.asarray(values, dtype=float)))

    def temporal_until(
        self,
        *,
        left_trace: Sequence[float],
        right_trace: Sequence[float],
        start: int,
        end: Optional[int],
        weights_left: Optional[Sequence[float]] = None,
        weights_right: Optional[Sequence[float]] = None,
        weights_pair: Sequence[float] = (1.0, 1.0),
    ) -> float:
        left = np.asarray(left_trace, dtype=float)
        right = np.asarray(right_trace, dtype=float)
        if left.size == 0 or right.size == 0:
            raise ValueError("UNTIL traces must be non-empty.")
        if left.size != right.size:
            raise ValueError("UNTIL traces must have the same length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = left.size - 1 if end is None else min(end, left.size - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        best = -np.inf
        for idx in range(start, last + 1):
            prefix = np.inf if idx == 0 else float(np.min(left[:idx]))
            candidate = min(float(right[idx]), prefix)
            best = max(best, candidate)
        return float(best)

    def temporal_cumulative(
        self,
        values: Sequence[float],
        *,
        tau: float,
        delta: Optional[float] = None,
    ) -> float:
        """CT-STL robustness for C^tau_I over a sampled trace window.

        Returns the k-th largest value where k=ceil(tau/delta). If the window is
        shorter than k, returns -inf.
        """
        vals = np.asarray(values, dtype=float).reshape(-1)
        if vals.size == 0:
            return float("-inf")

        eff_delta = self.delta if delta is None else float(delta)
        k = tau_to_k(tau=tau, delta=eff_delta)
        if k > vals.size:
            return float("-inf")
        return kth_largest(vals, k)
