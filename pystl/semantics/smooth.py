"""Smooth robustness semantics using soft min/max operators."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import Semantics


def _validate_temperature(temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    return float(temperature)


def _logsumexp(arr: np.ndarray) -> float:
    if arr.size == 0:
        raise ValueError("logsumexp requires at least one value.")
    max_val = float(np.max(arr))
    return max_val + float(np.log(np.sum(np.exp(arr - max_val))))


class SmoothRobustSemantics(Semantics[float]):
    """Soft STL robustness with log-sum-exp min/max approximations."""

    def __init__(self, *, temperature: float = 1.0) -> None:
        self.temperature = _validate_temperature(temperature)

    def _softmin(self, values: Sequence[float] | np.ndarray, where: str) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"{where} requires at least one value.")
        tau = self.temperature
        return -tau * _logsumexp(-arr / tau)

    def _softmax(self, values: Sequence[float] | np.ndarray, where: str) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"{where} requires at least one value.")
        tau = self.temperature
        return tau * _logsumexp(arr / tau)

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
        del weights
        return float(self._softmin(values, where="boolean_and"))

    def boolean_or(
        self, values: Sequence[float], *, weights: Optional[Sequence[float]] = None
    ) -> float:
        del weights
        return float(self._softmax(values, where="boolean_or"))

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
        del weights_left
        del weights_right
        del weights_pair

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

        candidates = []
        for idx in range(start, last + 1):
            prefix = self._softmin(left[: idx + 1], where="temporal_until")
            candidate = self._softmin(
                np.asarray([prefix, right[idx]], dtype=float), where="temporal_until"
            )
            candidates.append(candidate)
        return float(self._softmax(np.asarray(candidates), where="temporal_until"))
