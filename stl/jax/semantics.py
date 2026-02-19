"""JAX-native STL semantics backends.

This module mirrors the semantics in `stl/semantics` with JAX-based
implementations so values remain differentiable `jax.Array` scalars.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Optional, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from stljax import utils as stljax_utils
from jax.scipy.special import logsumexp

from stl.semantics.base import Semantics


def _validate_temperature(temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    return float(temperature)


def _resolve_weights(
    weights: Optional[Sequence[float]], length: int, name: str
) -> jnp.ndarray:
    if length <= 0:
        raise ValueError(f"{name} length must be positive.")
    if weights is None:
        return jnp.ones((length,), dtype=float)
    arr = jnp.asarray(weights, dtype=float).reshape(-1)
    if arr.size < length:
        raise ValueError(f"{name} requires at least {length} entries.")
    return arr[:length]


def _ensure_no_weights(weights: Optional[Sequence[float]], where: str) -> None:
    if weights is not None:
        raise ValueError(f"`{where}` does not support explicit weights.")


class _JaxHardMinMaxSemantics(Semantics[Any]):
    """Shared min/max robustness helpers for hard JAX semantics."""

    def _reduce_min(self, values: Sequence[Any] | Any, where: str) -> Any:
        arr = jnp.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"{where} requires at least one value.")
        return jnp.min(arr, axis=0)

    def _reduce_max(self, values: Sequence[Any] | Any, where: str) -> Any:
        arr = jnp.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"{where} requires at least one value.")
        return jnp.max(arr, axis=0)

    def predicate(self, predicate: Any, signal, t: int) -> Any:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        return jnp.asarray(predicate.fn(signal, t), dtype=float)

    def boolean_not(self, value: Any) -> Any:
        return -jnp.asarray(value, dtype=float)

    def boolean_and(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        del weights
        return self._reduce_min(values, where="boolean_and")

    def boolean_or(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        del weights
        return self._reduce_max(values, where="boolean_or")

    def temporal_until(
        self,
        *,
        left_trace: Sequence[Any],
        right_trace: Sequence[Any],
        start: int,
        end: Optional[int],
        weights_left: Optional[Sequence[float]] = None,
        weights_right: Optional[Sequence[float]] = None,
        weights_pair: Sequence[float] = (1.0, 1.0),
    ) -> Any:
        del weights_left
        del weights_right
        del weights_pair

        left = jnp.asarray(left_trace, dtype=float)
        right = jnp.asarray(right_trace, dtype=float)
        if left.size == 0 or right.size == 0:
            raise ValueError("UNTIL traces must be non-empty.")
        if left.size != right.size:
            raise ValueError("UNTIL traces must have the same length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = left.shape[0] - 1 if end is None else min(end, left.shape[0] - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        candidates = []
        for idx in range(start, last + 1):
            prefix_min = self._reduce_min(left[: idx + 1], where="temporal_until")
            pair_val = self._reduce_min(
                jnp.stack((prefix_min, right[idx])), where="temporal_until"
            )
            candidates.append(pair_val)

        return self._reduce_max(
            jnp.asarray(candidates, dtype=float), where="temporal_until"
        )


class JaxClassicRobustSemantics(_JaxHardMinMaxSemantics):
    """Classic STL max-min robustness implemented in JAX."""


class JaxTraditionalRobustSemantics(JaxClassicRobustSemantics):
    """Traditional STL robustness implemented in JAX."""


class JaxRobustSemantics(JaxClassicRobustSemantics):
    """Compatibility backend: JAX robustness with optional smooth min/max.

    This keeps the previous `create_semantics("jax", smooth=..., temperature=...)`
    behavior while `jax_classic` is hard-only.
    """

    def __init__(self, *, smooth: bool = False, temperature: float = 1.0) -> None:
        self.smooth = bool(smooth)
        self.temperature = _validate_temperature(temperature)

    def _softmin(self, arr: Any) -> Any:
        tau = self.temperature
        return -tau * logsumexp(-arr / tau, axis=0)

    def _softmax(self, arr: Any) -> Any:
        tau = self.temperature
        return tau * logsumexp(arr / tau, axis=0)

    def _reduce_min(self, values: Sequence[Any] | Any, where: str) -> Any:
        arr = jnp.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"{where} requires at least one value.")
        if self.smooth:
            return self._softmin(arr)
        return jnp.min(arr, axis=0)

    def _reduce_max(self, values: Sequence[Any] | Any, where: str) -> Any:
        arr = jnp.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"{where} requires at least one value.")
        if self.smooth:
            return self._softmax(arr)
        return jnp.max(arr, axis=0)


@dataclass(frozen=True)
class JaxCumulativeRobustness:
    """Container for cumulative positive/negative robustness values."""

    pos: Any
    neg: Any


class JaxCumulativeSemantics(Semantics[JaxCumulativeRobustness]):
    """Cumulative STL semantics implemented in JAX."""

    def predicate(self, predicate: Any, signal, t: int) -> JaxCumulativeRobustness:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        value = jnp.asarray(predicate.fn(signal, t), dtype=float)
        return JaxCumulativeRobustness(
            pos=jnp.maximum(0.0, value), neg=jnp.minimum(0.0, value)
        )

    def boolean_not(self, value: JaxCumulativeRobustness) -> JaxCumulativeRobustness:
        return JaxCumulativeRobustness(pos=-value.neg, neg=-value.pos)

    def boolean_and(
        self,
        values: Sequence[JaxCumulativeRobustness],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> JaxCumulativeRobustness:
        del weights
        if len(values) == 0:
            raise ValueError("boolean_and requires at least one value.")
        pos = jnp.min(jnp.asarray([v.pos for v in values], dtype=float), axis=0)
        neg = jnp.min(jnp.asarray([v.neg for v in values], dtype=float), axis=0)
        return JaxCumulativeRobustness(pos=pos, neg=neg)

    def boolean_or(
        self,
        values: Sequence[JaxCumulativeRobustness],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> JaxCumulativeRobustness:
        del weights
        if len(values) == 0:
            raise ValueError("boolean_or requires at least one value.")
        pos = jnp.max(jnp.asarray([v.pos for v in values], dtype=float), axis=0)
        neg = jnp.max(jnp.asarray([v.neg for v in values], dtype=float), axis=0)
        return JaxCumulativeRobustness(pos=pos, neg=neg)

    def temporal_eventually(
        self,
        values: Sequence[JaxCumulativeRobustness],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> JaxCumulativeRobustness:
        del weights
        if len(values) == 0:
            raise ValueError("temporal_eventually requires at least one value.")
        pos = jnp.sum(jnp.asarray([v.pos for v in values], dtype=float), axis=0)
        neg = jnp.sum(jnp.asarray([v.neg for v in values], dtype=float), axis=0)
        return JaxCumulativeRobustness(pos=pos, neg=neg)

    def temporal_until(
        self,
        *,
        left_trace: Sequence[JaxCumulativeRobustness],
        right_trace: Sequence[JaxCumulativeRobustness],
        start: int,
        end: Optional[int],
        weights_left: Optional[Sequence[float]] = None,
        weights_right: Optional[Sequence[float]] = None,
        weights_pair: Sequence[float] = (1.0, 1.0),
    ) -> JaxCumulativeRobustness:
        del weights_left
        del weights_right
        del weights_pair

        if len(left_trace) == 0 or len(right_trace) == 0:
            raise ValueError("UNTIL traces must be non-empty.")
        if len(left_trace) != len(right_trace):
            raise ValueError("UNTIL traces must have the same length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = len(left_trace) - 1 if end is None else min(end, len(left_trace) - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        pos_total = jnp.asarray(0.0, dtype=float)
        neg_total = jnp.asarray(0.0, dtype=float)
        for idx in range(start, last + 1):
            prefix_pos = jnp.min(
                jnp.asarray([v.pos for v in left_trace[: idx + 1]], dtype=float), axis=0
            )
            prefix_neg = jnp.min(
                jnp.asarray([v.neg for v in left_trace[: idx + 1]], dtype=float), axis=0
            )
            pos_total = pos_total + jnp.minimum(right_trace[idx].pos, prefix_pos)
            neg_total = neg_total + jnp.minimum(right_trace[idx].neg, prefix_neg)

        return JaxCumulativeRobustness(pos=pos_total, neg=neg_total)


def jax_tau_to_k(tau: float, delta: float) -> int:
    """Map CT-STL threshold tau to k = ceil(tau / delta)."""
    if delta <= 0:
        raise ValueError("delta must be > 0.")
    if tau <= 0:
        raise ValueError("tau must be > 0.")
    return int(ceil(tau / delta))


def jax_kth_largest(values: Sequence[Any], k: int) -> Any:
    """Return the k-th largest value (1-indexed) for JAX arrays."""
    arr = jnp.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("values must be non-empty.")
    if not 1 <= k <= arr.size:
        raise ValueError(f"k={k} must be between 1 and len(values)={arr.size}.")
    sorted_vals = jnp.sort(arr)
    return sorted_vals[arr.size - k]


class JaxCtstlSemantics(_JaxHardMinMaxSemantics):
    """CT-STL semantics implemented in JAX."""

    def __init__(self, *, delta: float = 1.0) -> None:
        if delta <= 0:
            raise ValueError("delta must be > 0.")
        self.delta = float(delta)

    def temporal_until(
        self,
        *,
        left_trace: Sequence[Any],
        right_trace: Sequence[Any],
        start: int,
        end: Optional[int],
        weights_left: Optional[Sequence[float]] = None,
        weights_right: Optional[Sequence[float]] = None,
        weights_pair: Sequence[float] = (1.0, 1.0),
    ) -> Any:
        del weights_left
        del weights_right
        del weights_pair

        left = jnp.asarray(left_trace, dtype=float)
        right = jnp.asarray(right_trace, dtype=float)
        if left.size == 0 or right.size == 0:
            raise ValueError("UNTIL traces must be non-empty.")
        if left.size != right.size:
            raise ValueError("UNTIL traces must have the same length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = left.shape[0] - 1 if end is None else min(end, left.shape[0] - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        best = jnp.asarray(-jnp.inf, dtype=float)
        for idx in range(start, last + 1):
            prefix = jnp.asarray(jnp.inf, dtype=float)
            if idx != 0:
                prefix = jnp.min(left[:idx], axis=0)
            candidate = jnp.minimum(right[idx], prefix)
            best = jnp.maximum(best, candidate)
        return best

    def temporal_cumulative(
        self,
        values: Sequence[Any],
        *,
        tau: float,
        delta: Optional[float] = None,
    ) -> Any:
        vals = jnp.asarray(values, dtype=float).reshape(-1)
        if vals.size == 0:
            return jnp.asarray(-jnp.inf, dtype=float)

        eff_delta = self.delta if delta is None else float(delta)
        k = jax_tau_to_k(tau=tau, delta=eff_delta)
        if k > vals.size:
            return jnp.asarray(-jnp.inf, dtype=float)
        return jax_kth_largest(vals, k)


def _gmsr_and_jax(
    *, eps: float, p: int, weights: jnp.ndarray, values: jnp.ndarray
) -> Any:
    vals = jnp.asarray(values, dtype=float).reshape(-1)
    w = jnp.asarray(weights, dtype=float).reshape(-1)
    if vals.size == 0:
        raise ValueError("values must contain at least one value.")
    if w.size < vals.size:
        raise ValueError("weights must have at least len(values) entries.")
    w = w[: vals.size]

    sum_w = jnp.sum(w)
    neg_mask = vals <= 0.0

    neg_terms = jnp.where(neg_mask, w * (vals ** (2 * p)), 0.0)
    sums = jnp.sum(neg_terms)
    mp = (eps**p + (sums / sum_w)) ** (1.0 / p)
    h_neg = eps**0.5 - mp**0.5

    base = jnp.where(neg_mask, 1.0, vals)
    exponent = jnp.where(neg_mask, 0.0, 2.0 * w)
    mult = jnp.prod(base**exponent)
    m0 = (eps**sum_w + mult) ** (1.0 / sum_w)
    h_pos = m0**0.5 - eps**0.5

    any_neg = jnp.any(neg_mask)
    return jnp.where(any_neg, h_neg, h_pos)


def _gmsr_or_jax(
    *, eps: float, p: int, weights: jnp.ndarray, values: jnp.ndarray
) -> Any:
    return -_gmsr_and_jax(eps=eps, p=p, weights=weights, values=-values)


class JaxDgmsrSemantics(Semantics[Any]):
    """D-GMSR semantics implemented in JAX."""

    def __init__(self, *, eps: float = 1e-8, p: int = 1) -> None:
        self.eps = float(eps)
        self.p = int(p)

    def predicate(self, predicate: Any, signal, t: int) -> Any:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        return jnp.asarray(predicate.fn(signal, t), dtype=float)

    def boolean_not(self, value: Any) -> Any:
        return -jnp.asarray(value, dtype=float)

    def boolean_and(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        vals = jnp.asarray(values, dtype=float).reshape(-1)
        if vals.size == 0:
            raise ValueError("boolean_and requires at least one value.")
        w = _resolve_weights(weights, vals.size, "weights")
        return _gmsr_and_jax(eps=self.eps, p=self.p, weights=w, values=vals)

    def boolean_or(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        vals = jnp.asarray(values, dtype=float).reshape(-1)
        if vals.size == 0:
            raise ValueError("boolean_or requires at least one value.")
        w = _resolve_weights(weights, vals.size, "weights")
        return _gmsr_or_jax(eps=self.eps, p=self.p, weights=w, values=vals)

    def temporal_until(
        self,
        *,
        left_trace: Sequence[Any],
        right_trace: Sequence[Any],
        start: int,
        end: Optional[int],
        weights_left: Optional[Sequence[float]] = None,
        weights_right: Optional[Sequence[float]] = None,
        weights_pair: Sequence[float] = (1.0, 1.0),
    ) -> Any:
        left = jnp.asarray(left_trace, dtype=float).reshape(-1)
        right = jnp.asarray(right_trace, dtype=float).reshape(-1)
        if left.size == 0 or right.size == 0:
            raise ValueError("UNTIL traces must be non-empty.")
        if left.size != right.size:
            raise ValueError("UNTIL traces must have equal length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = left.size - 1 if end is None else min(end, left.size - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        candidate_offsets = list(range(start, last + 1))
        w_pair = _resolve_weights(weights_pair, 2, "weights_pair")
        w_left = _resolve_weights(weights_left, last + 1, "weights_left")
        w_right = _resolve_weights(
            weights_right, len(candidate_offsets), "weights_right"
        )

        s_values = []
        for offset in candidate_offsets:
            y_i = _gmsr_and_jax(
                eps=self.eps,
                p=self.p,
                weights=w_left[: offset + 1],
                values=left[: offset + 1],
            )
            s_i = _gmsr_and_jax(
                eps=self.eps,
                p=self.p,
                weights=w_pair,
                values=jnp.stack((y_i, right[offset])),
            )
            s_values.append(s_i)

        s_arr = jnp.asarray(s_values, dtype=float)
        return _gmsr_or_jax(eps=self.eps, p=self.p, weights=w_right, values=s_arr)


class JaxStlJaxSemantics(Semantics[Any]):
    """JAX-friendly semantics using stljax minish/maxish aggregators."""

    def __init__(
        self,
        *,
        approx_method: str = "true",
        temperature: Optional[float | tuple[float, int]] = None,
    ) -> None:
        self.approx_method = approx_method
        self.temperature = temperature

    def _minish(self, values: Sequence[Any] | Any, where: str) -> Any:
        arr = jnp.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"{where} requires at least one value.")
        return stljax_utils.minish(
            arr,
            axis=0,
            keepdims=False,
            approx_method=self.approx_method,
            temperature=self.temperature,
        )

    def _maxish(self, values: Sequence[Any] | Any, where: str) -> Any:
        arr = jnp.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"{where} requires at least one value.")
        return stljax_utils.maxish(
            arr,
            axis=0,
            keepdims=False,
            approx_method=self.approx_method,
            temperature=self.temperature,
        )

    def predicate(self, predicate: Any, signal, t: int) -> Any:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        return jnp.asarray(predicate.fn(signal, t), dtype=float)

    def boolean_not(self, value: Any) -> Any:
        return -jnp.asarray(value, dtype=float)

    def boolean_and(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        _ensure_no_weights(weights, "boolean_and weights")
        return self._minish(values, where="boolean_and")

    def boolean_or(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        _ensure_no_weights(weights, "boolean_or weights")
        return self._maxish(values, where="boolean_or")

    def temporal_until(
        self,
        *,
        left_trace: Sequence[Any],
        right_trace: Sequence[Any],
        start: int,
        end: Optional[int],
        weights_left: Optional[Sequence[float]] = None,
        weights_right: Optional[Sequence[float]] = None,
        weights_pair: Sequence[float] = (1.0, 1.0),
    ) -> Any:
        _ensure_no_weights(weights_left, "Until.weights_left")
        _ensure_no_weights(weights_right, "Until.weights_right")
        if tuple(weights_pair) != (1.0, 1.0):
            raise ValueError("`Until.weights_pair` is not supported by jax_stljax.")

        left = jnp.asarray(left_trace, dtype=float)
        right = jnp.asarray(right_trace, dtype=float)
        if left.size == 0 or right.size == 0:
            raise ValueError("UNTIL traces must be non-empty.")
        if left.size != right.size:
            raise ValueError("UNTIL traces must have the same length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = left.shape[0] - 1 if end is None else min(end, left.shape[0] - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        candidates = []
        for idx in range(start, last + 1):
            prefix_min = self._minish(left[: idx + 1], where="temporal_until")
            pair_val = self._minish(
                jnp.stack((prefix_min, right[idx])), where="temporal_until"
            )
            candidates.append(pair_val)

        return self._maxish(
            jnp.asarray(candidates, dtype=float), where="temporal_until"
        )


__all__ = [
    "JaxClassicRobustSemantics",
    "JaxTraditionalRobustSemantics",
    "JaxRobustSemantics",
    "JaxCumulativeRobustness",
    "JaxCumulativeSemantics",
    "jax_kth_largest",
    "jax_tau_to_k",
    "JaxCtstlSemantics",
    "JaxDgmsrSemantics",
    "JaxStlJaxSemantics",
]
