"""Cumulative robustness semantics."""

from __future__ import annotations

from typing import Optional, Sequence
from dataclasses import dataclass

import numpy as np

from stl.semantics.base import Semantics


@dataclass(frozen=True)
class CumulativeRobustness:
    """Container for cumulative positive/negative robustness values."""

    pos: float
    neg: float


class CumulativeSemantics(Semantics[CumulativeRobustness]):
    """Cumulative STL semantics defined in `stls.py` (rho+ and rho-)."""

    def predicate(self, predicate, signal, t: int) -> CumulativeRobustness:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        value = float(predicate.fn(signal, t))
        return CumulativeRobustness(pos=max(0.0, value), neg=min(0.0, value))

    def boolean_not(self, value: CumulativeRobustness) -> CumulativeRobustness:
        return CumulativeRobustness(pos=-value.neg, neg=-value.pos)

    def boolean_and(
        self,
        values: Sequence[CumulativeRobustness],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> CumulativeRobustness:
        if len(values) == 0:
            raise ValueError("boolean_and requires at least one value.")
        return CumulativeRobustness(
            pos=float(min(v.pos for v in values)),
            neg=float(min(v.neg for v in values)),
        )

    def boolean_or(
        self,
        values: Sequence[CumulativeRobustness],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> CumulativeRobustness:
        if len(values) == 0:
            raise ValueError("boolean_or requires at least one value.")
        return CumulativeRobustness(
            pos=float(max(v.pos for v in values)),
            neg=float(max(v.neg for v in values)),
        )

    def temporal_eventually(
        self,
        values: Sequence[CumulativeRobustness],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> CumulativeRobustness:
        if len(values) == 0:
            raise ValueError("temporal_eventually requires at least one value.")
        return CumulativeRobustness(
            pos=float(np.sum([v.pos for v in values], dtype=float)),
            neg=float(np.sum([v.neg for v in values], dtype=float)),
        )

    def temporal_until(
        self,
        *,
        left_trace: Sequence[CumulativeRobustness],
        right_trace: Sequence[CumulativeRobustness],
        start: int,
        end: Optional[int],
        weights_left: Optional[Sequence[float]] = None,
        weights_right: Optional[Sequence[float]] = None,
        weights_pair: Sequence[float] = (1.0, 1.0),
    ) -> CumulativeRobustness:
        if len(left_trace) == 0 or len(right_trace) == 0:
            raise ValueError("UNTIL traces must be non-empty.")
        if len(left_trace) != len(right_trace):
            raise ValueError("UNTIL traces must have the same length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = len(left_trace) - 1 if end is None else min(end, len(left_trace) - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        pos_total = 0.0
        neg_total = 0.0
        for idx in range(start, last + 1):
            prefix_pos = min(v.pos for v in left_trace[: idx + 1])
            prefix_neg = min(v.neg for v in left_trace[: idx + 1])
            pos_total += min(right_trace[idx].pos, prefix_pos)
            neg_total += min(right_trace[idx].neg, prefix_neg)
        return CumulativeRobustness(pos=float(pos_total), neg=float(neg_total))
