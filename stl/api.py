"""Unified STL formula API.

This module provides a semantics-agnostic formula syntax tree:
- Predicates
- Boolean operators
- Temporal operators

Each formula is evaluated by a semantics backend implementing
`stl.semantics.base.Semantics`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, TypeAlias
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stl.semantics.base import Semantics

Signal: TypeAlias = NDArray[np.float64]
PredicateFn = Callable[[Signal, int], Any]


@dataclass(frozen=True)
class Interval:
    """Closed integer interval [start, end] with optional open-ended end."""

    start: int = 0
    end: Optional[int] = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("Interval start must be >= 0.")
        if self.end is not None and self.end < self.start:
            raise ValueError("Interval end must be >= start.")


def _as_interval(interval: Interval | tuple[int, Optional[int]]) -> Interval:
    if isinstance(interval, Interval):
        return interval
    return Interval(start=interval[0], end=interval[1])


def _window_indices(t: int, horizon: int, interval: Interval) -> list[int]:
    start = t + interval.start
    end = horizon - 1 if interval.end is None else min(t + interval.end, horizon - 1)
    if start > end:
        return []
    return list(range(start, end + 1))


class Formula(ABC):
    """Base class for all STL formulas."""

    @abstractmethod
    def evaluate(self, signal: Signal, semantics: Semantics[Any], t: int = 0) -> Any:
        pass

    def __and__(self, other: "Formula") -> "And":
        return And(self, other)

    def __or__(self, other: "Formula") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def always(
        self,
        interval: Interval | tuple[int, Optional[int]] = Interval(),
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> "Always":
        return Always(self, interval=interval, weights=weights)

    def eventually(
        self,
        interval: Interval | tuple[int, Optional[int]] = Interval(),
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> "Eventually":
        return Eventually(self, interval=interval, weights=weights)

    def until(
        self,
        other: "Formula",
        interval: Interval | tuple[int, Optional[int]] = Interval(),
        *,
        weights_left: Optional[Sequence[float]] = None,
        weights_right: Optional[Sequence[float]] = None,
        weights_pair: Sequence[float] = (1.0, 1.0),
    ) -> "Until":
        return Until(
            left=self,
            right=other,
            interval=interval,
            weights_left=weights_left,
            weights_right=weights_right,
            weights_pair=weights_pair,
        )


@dataclass(frozen=True)
class Predicate(Formula):
    """Atomic STL predicate.

    `fn` receives `(signal, t)` and should return a scalar quantity used by
    the selected semantics backend.
    """

    name: str
    fn: Optional[PredicateFn] = None
    metadata: dict[str, Any] | None = None

    def evaluate(self, signal: Signal, semantics: Semantics[Any], t: int = 0) -> Any:
        return semantics.predicate(self, signal, t)


@dataclass(frozen=True)
class Not(Formula):
    """Logical negation formula."""

    child: Formula

    def evaluate(self, signal: Signal, semantics: Semantics[Any], t: int = 0) -> Any:
        return semantics.boolean_not(self.child.evaluate(signal, semantics, t))


@dataclass(frozen=True)
class And(Formula):
    """Logical conjunction formula."""

    children: tuple[Formula, ...]
    weights: Optional[Sequence[float]] = None

    def __init__(
        self, *children: Formula, weights: Optional[Sequence[float]] = None
    ) -> None:
        if len(children) == 0:
            raise ValueError("And requires at least one child.")
        object.__setattr__(self, "children", tuple(children))
        object.__setattr__(self, "weights", weights)

    def evaluate(self, signal: Signal, semantics: Semantics[Any], t: int = 0) -> Any:
        values = [child.evaluate(signal, semantics, t) for child in self.children]
        return semantics.boolean_and(values, weights=self.weights)


@dataclass(frozen=True)
class Or(Formula):
    """Logical disjunction formula."""

    children: tuple[Formula, ...]
    weights: Optional[Sequence[float]] = None

    def __init__(
        self, *children: Formula, weights: Optional[Sequence[float]] = None
    ) -> None:
        if len(children) == 0:
            raise ValueError("Or requires at least one child.")
        object.__setattr__(self, "children", tuple(children))
        object.__setattr__(self, "weights", weights)

    def evaluate(self, signal: Signal, semantics: Semantics[Any], t: int = 0) -> Any:
        values = [child.evaluate(signal, semantics, t) for child in self.children]
        return semantics.boolean_or(values, weights=self.weights)


@dataclass(frozen=True)
class Always(Formula):
    """Temporal ALWAYS formula."""

    child: Formula
    interval: Interval = Interval()
    weights: Optional[Sequence[float]] = None

    def __init__(
        self,
        child: Formula,
        interval: Interval | tuple[int, Optional[int]] = Interval(),
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        object.__setattr__(self, "child", child)
        object.__setattr__(self, "interval", _as_interval(interval))
        object.__setattr__(self, "weights", weights)

    def evaluate(self, signal: Signal, semantics: Semantics[Any], t: int = 0) -> Any:
        horizon = np.asarray(signal).shape[0]
        indices = _window_indices(t=t, horizon=horizon, interval=self.interval)
        if not indices:
            raise ValueError(
                f"ALWAYS window is empty at t={t} for interval={self.interval}."
            )
        values = [self.child.evaluate(signal, semantics, tau) for tau in indices]
        return semantics.temporal_always(values, weights=self.weights)


@dataclass(frozen=True)
class Eventually(Formula):
    """Temporal EVENTUALLY formula."""

    child: Formula
    interval: Interval = Interval()
    weights: Optional[Sequence[float]] = None

    def __init__(
        self,
        child: Formula,
        interval: Interval | tuple[int, Optional[int]] = Interval(),
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        object.__setattr__(self, "child", child)
        object.__setattr__(self, "interval", _as_interval(interval))
        object.__setattr__(self, "weights", weights)

    def evaluate(self, signal: Signal, semantics: Semantics[Any], t: int = 0) -> Any:
        horizon = np.asarray(signal).shape[0]
        indices = _window_indices(t=t, horizon=horizon, interval=self.interval)
        if not indices:
            raise ValueError(
                f"EVENTUALLY window is empty at t={t} for interval={self.interval}."
            )
        values = [self.child.evaluate(signal, semantics, tau) for tau in indices]
        return semantics.temporal_eventually(values, weights=self.weights)


@dataclass(frozen=True)
class Until(Formula):
    """Temporal UNTIL formula."""

    left: Formula
    right: Formula
    interval: Interval = Interval()
    weights_left: Optional[Sequence[float]] = None
    weights_right: Optional[Sequence[float]] = None
    weights_pair: Sequence[float] = (1.0, 1.0)

    def __init__(
        self,
        left: Formula,
        right: Formula,
        interval: Interval | tuple[int, Optional[int]] = Interval(),
        *,
        weights_left: Optional[Sequence[float]] = None,
        weights_right: Optional[Sequence[float]] = None,
        weights_pair: Sequence[float] = (1.0, 1.0),
    ) -> None:
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "interval", _as_interval(interval))
        object.__setattr__(self, "weights_left", weights_left)
        object.__setattr__(self, "weights_right", weights_right)
        object.__setattr__(self, "weights_pair", weights_pair)

    def evaluate(self, signal: Signal, semantics: Semantics[Any], t: int = 0) -> Any:
        horizon = np.asarray(signal).shape[0]
        if t >= horizon:
            raise IndexError(f"t={t} is out of bounds for horizon={horizon}.")
        left_trace = [
            self.left.evaluate(signal, semantics, tau) for tau in range(t, horizon)
        ]
        right_trace = [
            self.right.evaluate(signal, semantics, tau) for tau in range(t, horizon)
        ]
        return semantics.temporal_until(
            left_trace=left_trace,
            right_trace=right_trace,
            start=self.interval.start,
            end=self.interval.end,
            weights_left=self.weights_left,
            weights_right=self.weights_right,
            weights_pair=self.weights_pair,
        )


__all__ = [
    "Signal",
    "PredicateFn",
    "Interval",
    "Formula",
    "Predicate",
    "Not",
    "And",
    "Or",
    "Always",
    "Eventually",
    "Until",
]
