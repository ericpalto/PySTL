"""Reference STL formula evaluation in JAX.

This module provides a tiny formula tree that can compute robustness traces via
`robustness_trace`. It is used as an independent implementation for
cross-checks in tests and for experimentation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol
from dataclasses import dataclass

import jax.numpy as jnp

from .aggregators import maxish, minish


class _HasRobustnessTrace(Protocol):
    """Protocol for objects that can compute robustness traces."""

    def robustness_trace(self, signal, **kwargs: Any) -> Any:
        ...


def _interval_to_bounds(interval: Any) -> tuple[int, Optional[int]]:
    if interval is None:
        return 0, None
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise TypeError(f"interval must be None or [start, end], got {interval!r}.")
    start, end = interval
    start_i = int(start)
    if start_i < 0:
        raise ValueError("interval start must be >= 0.")
    if end is None:
        return start_i, None
    if end == jnp.inf:
        return start_i, None
    end_i = int(end)
    if end_i < start_i:
        raise ValueError("interval end must be >= start.")
    return start_i, end_i


def _clip_window(horizon: int, start: int, end: Optional[int]) -> tuple[int, int]:
    if horizon <= 0:
        raise ValueError("signal horizon must be positive.")
    end_i = horizon - 1 if end is None else min(end, horizon - 1)
    return start, end_i


def _window_slice(
    trace: Any, *, t: int, start: int, end: Optional[int], padding: Optional[str]
) -> Any:
    horizon = int(jnp.asarray(trace).shape[0])
    abs_start, abs_end = _clip_window(
        horizon, t + start, None if end is None else t + end
    )
    if abs_start > abs_end:
        raise ValueError("temporal window is empty.")

    requested_end = horizon - 1 if end is None else (t + end)
    window = jnp.asarray(trace, dtype=float)[abs_start : abs_end + 1]
    if requested_end <= abs_end:
        return window

    if padding is None:
        return window
    pad_len = int(requested_end - abs_end)
    pad_mode = str(padding).strip().lower()
    if pad_mode in ("last", "edge"):
        last = window[-1]
        return jnp.concatenate(
            (window, jnp.full((pad_len,), last, dtype=float)), axis=0
        )
    raise ValueError(f"Unsupported padding={padding!r}.")


def _valid_time_count(*, horizon: int, start: int, end: Optional[int]) -> int:
    """Number of time indices `t` for which the window is well-defined.

    This reference implementation mirrors the common convention that temporal
    operators are only defined when their entire lookahead fits within the
    signal horizon (i.e. no implicit clipping).
    """

    if horizon < 0:
        raise ValueError("horizon must be >= 0.")
    if start < 0:
        raise ValueError("start must be >= 0.")
    if end is None:
        return max(0, horizon - start)
    if end < start:
        raise ValueError("end must be >= start.")
    return max(0, horizon - end)


def _minmax_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "approx_method": kwargs.get("approx_method", "true"),
        "temperature": kwargs.get("temperature", None),
    }


@dataclass(frozen=True)
class Predicate:
    """Atomic predicate providing a full-horizon trace."""

    name: str
    predicate_function: Callable[[Any], Any]

    def robustness_trace(self, signal, **kwargs: Any) -> Any:
        del kwargs
        out = jnp.asarray(self.predicate_function(signal), dtype=float).reshape(-1)
        return out


@dataclass(frozen=True)
class Negation:
    """Logical negation."""

    child: _HasRobustnessTrace

    def robustness_trace(self, signal, **kwargs: Any) -> Any:
        return -jnp.asarray(self.child.robustness_trace(signal, **kwargs), dtype=float)


@dataclass(frozen=True)
class And:
    """Pointwise conjunction."""

    left: _HasRobustnessTrace
    right: _HasRobustnessTrace

    def robustness_trace(self, signal, **kwargs: Any) -> Any:
        lt = jnp.asarray(self.left.robustness_trace(signal, **kwargs), dtype=float)
        rt = jnp.asarray(self.right.robustness_trace(signal, **kwargs), dtype=float)
        return minish(
            jnp.stack((lt, rt), axis=0),
            axis=0,
            keepdims=False,
            **_minmax_kwargs(kwargs),
        )


@dataclass(frozen=True)
class Or:
    """Pointwise disjunction."""

    left: _HasRobustnessTrace
    right: _HasRobustnessTrace

    def robustness_trace(self, signal, **kwargs: Any) -> Any:
        lt = jnp.asarray(self.left.robustness_trace(signal, **kwargs), dtype=float)
        rt = jnp.asarray(self.right.robustness_trace(signal, **kwargs), dtype=float)
        return maxish(
            jnp.stack((lt, rt), axis=0),
            axis=0,
            keepdims=False,
            **_minmax_kwargs(kwargs),
        )


@dataclass(frozen=True)
class Always:
    """Temporal ALWAYS over a window."""

    child: _HasRobustnessTrace
    interval: Any = None

    def robustness_trace(
        self,
        signal,
        *,
        approx_method: str = "true",
        temperature: Optional[float | tuple[float, int]] = None,
        padding: Optional[str] = None,
        large_number: float = 1e9,
    ) -> Any:
        del large_number
        child_trace = jnp.asarray(
            self.child.robustness_trace(
                signal,
                approx_method=approx_method,
                temperature=temperature,
                padding=padding,
            ),
            dtype=float,
        ).reshape(-1)
        start, end = _interval_to_bounds(self.interval)
        horizon = int(child_trace.shape[0])

        out: list[Any] = []
        for t in range(_valid_time_count(horizon=horizon, start=start, end=end)):
            window = _window_slice(
                child_trace, t=t, start=start, end=end, padding=padding
            )
            out.append(
                minish(
                    window,
                    axis=0,
                    keepdims=False,
                    approx_method=approx_method,
                    temperature=temperature,
                )
            )
        return jnp.asarray(out, dtype=float)


@dataclass(frozen=True)
class Eventually:
    """Temporal EVENTUALLY over a window."""

    child: _HasRobustnessTrace
    interval: Any = None

    def robustness_trace(
        self,
        signal,
        *,
        approx_method: str = "true",
        temperature: Optional[float | tuple[float, int]] = None,
        padding: Optional[str] = None,
        large_number: float = 1e9,
    ) -> Any:
        del large_number
        child_trace = jnp.asarray(
            self.child.robustness_trace(
                signal,
                approx_method=approx_method,
                temperature=temperature,
                padding=padding,
            ),
            dtype=float,
        ).reshape(-1)
        start, end = _interval_to_bounds(self.interval)
        horizon = int(child_trace.shape[0])

        out: list[Any] = []
        for t in range(_valid_time_count(horizon=horizon, start=start, end=end)):
            window = _window_slice(
                child_trace, t=t, start=start, end=end, padding=padding
            )
            out.append(
                maxish(
                    window,
                    axis=0,
                    keepdims=False,
                    approx_method=approx_method,
                    temperature=temperature,
                )
            )
        return jnp.asarray(out, dtype=float)


@dataclass(frozen=True)
class Until:
    """Temporal UNTIL over a window."""

    left: _HasRobustnessTrace
    right: _HasRobustnessTrace
    interval: Any = None

    def robustness_trace(
        self,
        signal,
        *,
        approx_method: str = "true",
        temperature: Optional[float | tuple[float, int]] = None,
        padding: Optional[str] = None,
        large_number: float = 1e9,
    ) -> Any:
        del padding
        del large_number
        left_trace = jnp.asarray(
            self.left.robustness_trace(
                signal, approx_method=approx_method, temperature=temperature
            ),
            dtype=float,
        ).reshape(-1)
        right_trace = jnp.asarray(
            self.right.robustness_trace(
                signal, approx_method=approx_method, temperature=temperature
            ),
            dtype=float,
        ).reshape(-1)
        if left_trace.shape[0] != right_trace.shape[0]:
            raise ValueError("UNTIL traces must have the same length.")

        start, end = _interval_to_bounds(self.interval)
        horizon = int(left_trace.shape[0])
        out: list[Any] = []
        for t in range(_valid_time_count(horizon=horizon, start=start, end=end)):
            left_sub = left_trace[t:]
            right_sub = right_trace[t:]
            sub_horizon = int(left_sub.shape[0])
            last = sub_horizon - 1 if end is None else min(end, sub_horizon - 1)
            if start < 0:
                raise ValueError("UNTIL start must be >= 0.")
            if start > last:
                raise ValueError("UNTIL temporal window is empty.")

            candidates: list[Any] = []
            for idx in range(start, last + 1):
                prefix = minish(
                    left_sub[: idx + 1],
                    axis=0,
                    keepdims=False,
                    approx_method=approx_method,
                    temperature=temperature,
                )
                pair = minish(
                    jnp.stack((prefix, right_sub[idx]), axis=0),
                    axis=0,
                    keepdims=False,
                    approx_method=approx_method,
                    temperature=temperature,
                )
                candidates.append(pair)

            out.append(
                maxish(
                    jnp.asarray(candidates, dtype=float),
                    axis=0,
                    keepdims=False,
                    approx_method=approx_method,
                    temperature=temperature,
                )
            )
        return jnp.asarray(out, dtype=float)


__all__ = [
    "Predicate",
    "Negation",
    "And",
    "Or",
    "Always",
    "Eventually",
    "Until",
]
