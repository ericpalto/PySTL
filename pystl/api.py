"""Unified STL formula API.

This module provides a semantics-agnostic formula syntax tree:
- Predicates
- Boolean operators
- Temporal operators

Each formula is evaluated by a semantics backend implementing
`pystl.semantics.base.Semantics`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Callable, Optional, Sequence, TypeAlias
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .semantics.base import Semantics

Signal: TypeAlias = NDArray[np.float64]
PredicateFn = Callable[[Signal, int], Any]
PredicateGradFn = Callable[[Signal, int], NDArray[np.float64] | Sequence[float]]
FormulaFormat: TypeAlias = Literal["text", "markdown", "latex"]


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

    def evaluate_with_grad(
        self, signal: Signal, semantics: Semantics[Any], t: int = 0, **kwargs: Any
    ) -> tuple[Any, NDArray[np.float64]]:
        """Evaluate robustness and gradient w.r.t. the full signal trace.

        This requires a semantics backend that implements `evaluate_with_grad`.
        """

        evaluator = getattr(semantics, "evaluate_with_grad", None)
        if evaluator is None:
            raise NotImplementedError(
                f"{type(semantics).__name__} does not support gradients."
            )
        value, grad = evaluator(self, signal, t=t, **kwargs)
        return value, np.asarray(grad, dtype=float)

    def export(self, fmt: str = "text", **kwargs: Any) -> str:
        """Export a formula as plain text, Markdown, or LaTeX."""

        return export_formula(self, fmt=fmt, **kwargs)

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
    grad: Optional[PredicateGradFn] = None

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
        horizon = signal.shape[0]
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
        horizon = signal.shape[0]
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
        horizon = signal.shape[0]
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


def _resolve_export_format(fmt: str, kwargs: dict[str, Any]) -> str:
    format_alias = kwargs.pop("format", None)
    if format_alias is not None:
        if fmt != "text":
            raise TypeError("Use either 'fmt' or 'format', not both.")
        if not isinstance(format_alias, str):
            raise TypeError("'format' must be a string.")
        fmt = format_alias
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}.")
    return fmt


def _normalize_format(format_name: str) -> FormulaFormat:
    key = format_name.strip().lower()
    aliases: dict[str, FormulaFormat] = {
        "text": "text",
        "plain": "text",
        "plaintext": "text",
        "txt": "text",
        "markdown": "markdown",
        "md": "markdown",
        "latex": "latex",
        "tex": "latex",
    }
    if key not in aliases:
        msg = "Unsupported export format. Use one of: 'text', 'markdown', 'latex'."
        raise ValueError(msg)
    return aliases[key]


def _format_interval(interval: Interval, fmt: FormulaFormat) -> str:
    if interval.end is None:
        end = r"\infty" if fmt == "latex" else "inf"
    else:
        end = str(interval.end)
    return f"[{interval.start}, {end}]"


def _escape_latex(text: str) -> str:
    chars = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "_": r"\_",
        "^": r"\^{}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "~": r"\~{}",
    }
    return "".join(chars.get(ch, ch) for ch in text)


def _wrap(expr: str, fmt: FormulaFormat) -> str:
    if fmt == "latex":
        return rf"\left({expr}\right)"
    return f"({expr})"


def _format_formula(formula: Formula, fmt: FormulaFormat) -> str:
    if isinstance(formula, Predicate):
        return _escape_latex(formula.name) if fmt == "latex" else formula.name

    if isinstance(formula, Not):
        child = _format_formula(formula.child, fmt)
        if fmt == "latex":
            return rf"\neg{_wrap(child, fmt)}"
        return f"not {_wrap(child, fmt)}"

    if isinstance(formula, And):
        children = [_format_formula(child, fmt) for child in formula.children]
        joiner = r" \wedge " if fmt == "latex" else " and "
        return _wrap(joiner.join(children), fmt)

    if isinstance(formula, Or):
        children = [_format_formula(child, fmt) for child in formula.children]
        joiner = r" \vee " if fmt == "latex" else " or "
        return _wrap(joiner.join(children), fmt)

    if isinstance(formula, Always):
        child = _format_formula(formula.child, fmt)
        interval = _format_interval(formula.interval, fmt)
        if fmt == "text":
            return f"always{interval}{_wrap(child, fmt)}"
        if fmt == "markdown":
            return f"G{interval}{_wrap(child, fmt)}"
        return rf"\square_{{{interval}}}{_wrap(child, fmt)}"

    if isinstance(formula, Eventually):
        child = _format_formula(formula.child, fmt)
        interval = _format_interval(formula.interval, fmt)
        if fmt == "text":
            return f"eventually{interval}{_wrap(child, fmt)}"
        if fmt == "markdown":
            return f"F{interval}{_wrap(child, fmt)}"
        return rf"\lozenge_{{{interval}}}{_wrap(child, fmt)}"

    if isinstance(formula, Until):
        left = _format_formula(formula.left, fmt)
        right = _format_formula(formula.right, fmt)
        interval = _format_interval(formula.interval, fmt)
        if fmt == "text":
            return f"{_wrap(left, fmt)} until{interval} {_wrap(right, fmt)}"
        if fmt == "markdown":
            return f"{_wrap(left, fmt)} U{interval} {_wrap(right, fmt)}"
        return rf"{_wrap(left, fmt)}\ \mathcal{{U}}_{{{interval}}}\ {_wrap(right, fmt)}"

    raise TypeError(f"Unsupported formula node: {type(formula)!r}")


def export_formula(formula: Formula, *, fmt: str = "text", **kwargs: Any) -> str:
    """Export a formula as plain text, Markdown, or LaTeX."""

    selected_fmt = _resolve_export_format(fmt, kwargs)
    normalized_fmt = _normalize_format(selected_fmt)
    return _format_formula(formula, normalized_fmt)


__all__ = [
    "Signal",
    "PredicateFn",
    "FormulaFormat",
    "Interval",
    "Formula",
    "Predicate",
    "Not",
    "And",
    "Or",
    "Always",
    "Eventually",
    "Until",
    "export_formula",
]
