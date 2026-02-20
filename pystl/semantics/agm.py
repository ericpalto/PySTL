"""Arithmetic-Geometric Mean (AGM) robustness semantics.

This module implements the normalized signed robustness from:
Mehdipour, N., Vasile, C.-I., & Belta, C. (ACC 2019)
"Arithmetic-Geometric Mean Robustness for Control from
Signal Temporal Logic Specifications."
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import Semantics


def _resolve_weights(
    weights: Optional[Sequence[float]],
    length: int,
    name: str,
) -> np.ndarray:
    if length <= 0:
        raise ValueError(f"{name} length must be positive.")
    if weights is None:
        return np.ones(length, dtype=float)

    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.size < length:
        raise ValueError(f"{name} requires at least {length} entries.")
    if np.any(arr[:length] < 0.0):
        raise ValueError(f"{name} entries must be >= 0.")
    if np.sum(arr[:length]) <= 0.0:
        raise ValueError(f"{name} entries must contain at least one positive weight.")
    return arr[:length]


def _weighted_arithmetic_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values, dtype=float) / np.sum(weights, dtype=float))


def _weighted_geometric_mean(values: np.ndarray, weights: np.ndarray) -> float:
    norm = weights / np.sum(weights, dtype=float)
    return float(np.exp(np.sum(norm * np.log(values), dtype=float)))


class AgmRobustSemantics(Semantics[float]):
    """AGM robustness semantics for STL formulas."""

    def predicate(self, predicate, signal, t: int) -> float:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        return float(predicate.fn(signal, t))

    def boolean_not(self, value: float) -> float:
        return -float(value)

    def boolean_and(
        self,
        values: Sequence[float],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> float:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("boolean_and requires at least one value.")
        w = _resolve_weights(weights, arr.size, "weights")

        if np.any(arr <= 0.0):
            return _weighted_arithmetic_mean(np.minimum(arr, 0.0), w)
        return _weighted_geometric_mean(1.0 + arr, w) - 1.0

    def boolean_or(
        self,
        values: Sequence[float],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> float:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("boolean_or requires at least one value.")
        w = _resolve_weights(weights, arr.size, "weights")

        if np.any(arr > 0.0):
            return _weighted_arithmetic_mean(np.maximum(arr, 0.0), w)
        return 1.0 - _weighted_geometric_mean(1.0 - arr, w)

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
            raise ValueError("UNTIL traces must have equal length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = left.size - 1 if end is None else min(end, left.size - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        candidate_offsets = np.arange(start, last + 1, dtype=int)
        w_pair = _resolve_weights(weights_pair, 2, "weights_pair")
        w_left = _resolve_weights(weights_left, last + 1, "weights_left")
        w_right = _resolve_weights(
            weights_right, candidate_offsets.size, "weights_right"
        )

        candidates: list[float] = []
        for offset in candidate_offsets:
            prefix = self.boolean_and(left[: offset + 1], weights=w_left[: offset + 1])
            candidate = self.boolean_and(
                np.asarray([prefix, right[offset]], dtype=float),
                weights=w_pair,
            )
            candidates.append(float(candidate))

        return self.boolean_or(candidates, weights=w_right)
