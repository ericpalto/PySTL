"""D-GMSR semantics backend for the unified API."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from stl.dgmsr import gmsr_or, gmsr_and
from stl.semantics.base import Semantics


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
    return arr[:length]


class DgmsrSemantics(Semantics[float]):
    """D-GMSR robustness semantics using smooth Boolean/Temporal operators."""

    def __init__(self, *, eps: float = 1e-8, p: int = 1) -> None:
        self.eps = float(eps)
        self.p = int(p)

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
        vals = np.asarray(values, dtype=float)
        if vals.size == 0:
            raise ValueError("boolean_and requires at least one value.")
        w = _resolve_weights(weights, vals.size, "weights")
        out, _ = gmsr_and(eps=self.eps, p=self.p, weights=w, values=vals)
        return float(out)

    def boolean_or(
        self, values: Sequence[float], *, weights: Optional[Sequence[float]] = None
    ) -> float:
        vals = np.asarray(values, dtype=float)
        if vals.size == 0:
            raise ValueError("boolean_or requires at least one value.")
        w = _resolve_weights(weights, vals.size, "weights")
        out, _ = gmsr_or(eps=self.eps, p=self.p, weights=w, values=vals)
        return float(out)

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

        s_values: list[float] = []
        for offset in candidate_offsets:
            y_i, _ = gmsr_and(
                eps=self.eps,
                p=self.p,
                weights=w_left[: offset + 1],
                values=left[: offset + 1],
            )
            s_i, _ = gmsr_and(
                eps=self.eps,
                p=self.p,
                weights=w_pair,
                values=np.asarray([y_i, right[offset]], dtype=float),
            )
            s_values.append(float(s_i))

        z, _ = gmsr_or(
            eps=self.eps,
            p=self.p,
            weights=w_right,
            values=np.asarray(s_values, dtype=float),
        )
        return float(z)
