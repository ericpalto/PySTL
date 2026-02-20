"""Classic max-min robustness semantics."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import Semantics


class ClassicRobustSemantics(Semantics[float]):
    """Reference semantics using hard min/max operators."""

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
            prefix = np.min(left[: idx + 1])
            candidate = min(prefix, right[idx])
            best = max(best, candidate)
        return float(best)
