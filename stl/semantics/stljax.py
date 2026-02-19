"""Wrapper/adapters for using unified STL formulas with the `stljax` package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence
from functools import reduce
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import stljax.formula as stljax_formula
from stljax import utils as stljax_utils

from stl.semantics.base import Semantics

if TYPE_CHECKING:
    from stl.api import Formula


def _ensure_no_weights(weights: Optional[Sequence[float]], where: str) -> None:
    if weights is not None:
        raise ValueError(
            f"`{where}` does not support explicit weights in stljax wrapper."
        )


def _to_stljax_interval(start: int, end: Optional[int]):
    if start == 0 and end is None:
        return None
    if end is None:
        return [start, jnp.inf]
    return [start, end]


def to_stljax_formula(formula: "Formula"):
    """Convert unified API `Formula` into a native `stljax.formula` object."""

    # Delayed import avoids circular dependency with stl.api -> stl.semantics.
    # pylint: disable=import-outside-toplevel
    from stl.api import Or, And, Not, Until, Always, Predicate, Eventually

    if isinstance(formula, Predicate):
        if formula.fn is None:
            raise ValueError(
                f"Predicate '{formula.name}' requires `fn(signal, t)` "
                "for stljax conversion."
            )

        def _predicate_fn(signal):
            sig = np.asarray(signal)
            horizon = sig.shape[0]
            vals = np.asarray([formula.fn(sig, t) for t in range(horizon)], dtype=float)
            return jnp.asarray(vals)

        return stljax_formula.Predicate(
            name=formula.name, predicate_function=_predicate_fn
        )

    if isinstance(formula, Not):
        return stljax_formula.Negation(to_stljax_formula(formula.child))

    if isinstance(formula, And):
        _ensure_no_weights(formula.weights, "And.weights")
        converted = [to_stljax_formula(child) for child in formula.children]
        return reduce(stljax_formula.And, converted)

    if isinstance(formula, Or):
        _ensure_no_weights(formula.weights, "Or.weights")
        converted = [to_stljax_formula(child) for child in formula.children]
        return reduce(stljax_formula.Or, converted)

    if isinstance(formula, Always):
        _ensure_no_weights(formula.weights, "Always.weights")
        interval = _to_stljax_interval(formula.interval.start, formula.interval.end)
        return stljax_formula.Always(
            to_stljax_formula(formula.child), interval=interval
        )

    if isinstance(formula, Eventually):
        _ensure_no_weights(formula.weights, "Eventually.weights")
        interval = _to_stljax_interval(formula.interval.start, formula.interval.end)
        return stljax_formula.Eventually(
            to_stljax_formula(formula.child), interval=interval
        )

    if isinstance(formula, Until):
        _ensure_no_weights(formula.weights_left, "Until.weights_left")
        _ensure_no_weights(formula.weights_right, "Until.weights_right")
        if tuple(formula.weights_pair) != (1.0, 1.0):
            raise ValueError("`Until.weights_pair` is not supported in stljax wrapper.")
        interval = _to_stljax_interval(formula.interval.start, formula.interval.end)
        return stljax_formula.Until(
            to_stljax_formula(formula.left),
            to_stljax_formula(formula.right),
            interval=interval,
        )

    raise TypeError(
        f"Unsupported formula type for stljax conversion: {type(formula)!r}"
    )


@dataclass(frozen=True)
class StlJaxFormulaWrapper:
    """Compile and evaluate a unified API formula with native stljax operators."""

    formula: "Formula"
    approx_method: str = "true"
    temperature: Optional[float | tuple[float, int]] = None
    padding: Optional[str] = None
    large_number: float = 1e9

    def _kwargs(self):
        return {
            "approx_method": self.approx_method,
            "temperature": self.temperature,
            "padding": self.padding,
            "large_number": self.large_number,
        }

    def robustness_trace(self, signal) -> np.ndarray:
        compiled = to_stljax_formula(self.formula)
        trace = compiled.robustness_trace(jnp.asarray(signal), **self._kwargs())
        return np.asarray(trace, dtype=float)

    def robustness(self, signal, t: int = 0) -> float:
        trace = self.robustness_trace(signal)
        if t < 0 or t >= trace.shape[0]:
            raise IndexError(
                f"t={t} is out of bounds for robustness trace "
                f"of length {trace.shape[0]}."
            )
        return float(trace[t])


class StlJaxSemantics(Semantics[float]):
    """Unified API semantics backend using stljax min/max logic.

    This backend supports the unified `Formula.evaluate(signal, semantics, t)` flow.
    """

    def __init__(
        self,
        *,
        approx_method: str = "true",
        temperature: Optional[float | tuple[float, int]] = None,
    ) -> None:
        self.approx_method = approx_method
        self.temperature = temperature

    def _minish(self, values: Sequence[float]) -> float:
        arr = jnp.asarray(values, dtype=float)
        out = stljax_utils.minish(
            arr,
            axis=0,
            keepdims=False,
            approx_method=self.approx_method,
            temperature=self.temperature,
        )
        return float(np.asarray(out))

    def _maxish(self, values: Sequence[float]) -> float:
        arr = jnp.asarray(values, dtype=float)
        out = stljax_utils.maxish(
            arr,
            axis=0,
            keepdims=False,
            approx_method=self.approx_method,
            temperature=self.temperature,
        )
        return float(np.asarray(out))

    def predicate(self, predicate: Any, signal, t: int) -> float:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        sig = np.asarray(signal)
        return float(predicate.fn(sig, t))

    def boolean_not(self, value: float) -> float:
        return -float(value)

    def boolean_and(
        self, values: Sequence[float], *, weights: Optional[Sequence[float]] = None
    ) -> float:
        _ensure_no_weights(weights, "boolean_and weights")
        if len(values) == 0:
            raise ValueError("boolean_and requires at least one value.")
        return self._minish(values)

    def boolean_or(
        self, values: Sequence[float], *, weights: Optional[Sequence[float]] = None
    ) -> float:
        _ensure_no_weights(weights, "boolean_or weights")
        if len(values) == 0:
            raise ValueError("boolean_or requires at least one value.")
        return self._maxish(values)

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
        _ensure_no_weights(weights_left, "Until.weights_left")
        _ensure_no_weights(weights_right, "Until.weights_right")
        if tuple(weights_pair) != (1.0, 1.0):
            raise ValueError(
                "`Until.weights_pair` is not supported by stljax semantics backend."
            )

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

        candidates: list[float] = []
        for idx in range(start, last + 1):
            prefix_min = self._minish(left[: idx + 1])
            pair_val = self._minish([prefix_min, float(right[idx])])
            candidates.append(pair_val)
        return self._maxish(candidates)
