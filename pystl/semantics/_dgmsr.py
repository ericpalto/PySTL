"""Composable D-GMSR STL API.

This module exposes:
- `Predicate` for atomic formulas.
- Boolean operators: `And`, `Or`, `Not`.
- Temporal operators: `Always`, `Eventually`, `Until`.

All operators use D-GMSR smooth robustness primitives from
the underlying D-GMSR derivation and propagate gradients with chain rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, TypeAlias
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array1D: TypeAlias = NDArray[np.float64]
Array2D: TypeAlias = NDArray[np.float64]
Array3D: TypeAlias = NDArray[np.float64]


def _as_1d(values: Sequence[float] | NDArray[np.float64], name: str) -> Array1D:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return arr


def _resolve_weights(
    weights: Optional[Sequence[float] | NDArray[np.float64]],
    length: int,
    name: str = "weights",
) -> Array1D:
    if length <= 0:
        raise ValueError(f"{name} length must be positive.")
    if weights is None:
        return np.ones(length, dtype=float)
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.size < length:
        raise ValueError(f"{name} must have at least {length} entries.")
    return arr[:length]


def _validate_signal(signal: Array2D) -> Array2D:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 2:
        raise ValueError("signal must be a 2D array with shape (time, state_dim).")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("signal must be non-empty in both dimensions.")
    return x


def _parse_interval(interval: tuple[int, Optional[int]]) -> tuple[int, Optional[int]]:
    start, end = interval
    if start < 0:
        raise ValueError("interval start must be >= 0.")
    if end is not None and end < start:
        raise ValueError("interval end must be >= start.")
    return start, end


def gmsr_and(
    eps: float,
    p: int,
    weights: Sequence[float] | NDArray[np.float64],
    values: Sequence[float] | NDArray[np.float64],
) -> tuple[float, Array1D]:
    """D-GMSR smooth conjunction and derivatives wrt input values."""
    fcn_vals = _as_1d(values, "values")
    w = _resolve_weights(weights, fcn_vals.size, "weights")

    pos_idx = np.where(fcn_vals > 0.0)[0]
    neg_idx = np.where(fcn_vals <= 0.0)[0]

    sum_w = float(np.sum(w))
    grad = np.zeros_like(fcn_vals, dtype=float)

    if neg_idx.size > 0:
        neg_vals = fcn_vals[neg_idx]
        neg_w = w[neg_idx]
        sums = float(np.sum(neg_w * (neg_vals ** (2 * p))))
        mp = (eps**p + (sums / sum_w)) ** (1.0 / p)
        h_and = eps**0.5 - mp**0.5

        cp = 0.5 * mp ** (-0.5)
        cpm = (2.0 * p) / (p * sum_w * mp ** (p - 1))
        grad[neg_idx] = cp * cpm * (neg_w * (np.abs(neg_vals) ** (2 * p - 1)))
        return float(h_and), grad

    pos_vals = fcn_vals[pos_idx]
    pos_w = w[pos_idx]

    mult = 1.0
    for idx, pos_val in enumerate(pos_vals):
        mult = mult * (pos_val ** (2.0 * pos_w[idx]))

    m0 = (eps**sum_w + mult) ** (1.0 / sum_w)
    h_and = m0**0.5 - eps**0.5

    c0 = 0.5 * m0 ** (-0.5)
    c0m = (2.0 * mult) / (sum_w * m0 ** (sum_w - 1))
    grad[pos_idx] = c0 * c0m * (pos_w / pos_vals)
    return float(h_and), grad


def gmsr_or(
    eps: float,
    p: int,
    weights: Sequence[float] | NDArray[np.float64],
    values: Sequence[float] | NDArray[np.float64],
) -> tuple[float, Array1D]:
    """D-GMSR smooth disjunction and derivatives wrt input values."""
    vals = -_as_1d(values, "values")
    h_mor, grad = gmsr_and(eps=eps, p=p, weights=weights, values=vals)
    return -h_mor, grad


def gmsr_until(
    eps: float,
    p: int,
    w_f: Sequence[float] | NDArray[np.float64],
    w_g: Sequence[float] | NDArray[np.float64],
    w_fg: Sequence[float] | NDArray[np.float64],
    f: Sequence[float] | NDArray[np.float64],
    g: Sequence[float] | NDArray[np.float64],
) -> tuple[float, Array1D, Array1D]:
    """D-GMSR UNTIL for aligned traces and derivatives wrt f and g traces."""
    f_trace = _as_1d(f, "f")
    g_trace = _as_1d(g, "g")
    if f_trace.size != g_trace.size:
        raise ValueError("f and g must have the same length.")

    k = f_trace.size
    wf = _resolve_weights(w_f, k, "w_f")
    wg = _resolve_weights(w_g, k, "w_g")
    wpair = _resolve_weights(w_fg, 2, "w_fg")

    s_vals: list[float] = []
    ds_df: list[Array1D] = []
    ds_dg: list[float] = []

    for i in range(k):
        y_i, dyi_dfi = gmsr_and(
            eps=eps, p=p, weights=wf[: i + 1], values=f_trace[: i + 1]
        )
        s_i, dsi_pair = gmsr_and(
            eps=eps, p=p, weights=wpair, values=np.array([y_i, g_trace[i]])
        )
        dsi_dyi = dsi_pair[0]
        dsi_dgi = dsi_pair[1]
        s_vals.append(s_i)
        ds_dg.append(float(dsi_dgi))
        ds_df.append(dsi_dyi * dyi_dfi)

    z, dz_ds = gmsr_or(eps=eps, p=p, weights=wg, values=np.array(s_vals))

    dz_df = np.zeros(k, dtype=float)
    for i, dsi_dfi in enumerate(ds_df):
        dz_df[: i + 1] += dz_ds[i] * dsi_dfi
    dz_dg = dz_ds * np.asarray(ds_dg, dtype=float)
    return float(z), dz_df, dz_dg


@dataclass(frozen=True)
class EvalResult:
    """Formula evaluation result on a full signal trace."""

    robustness: Array1D
    grad: Array3D

    def at(self, t: int = 0) -> tuple[float, Array2D]:
        """Return robustness and gradient matrix at one time index."""
        return float(self.robustness[t]), self.grad[t].copy()


class Formula(ABC):
    """Base STL formula class."""

    @abstractmethod
    def evaluate(self, signal: Array2D) -> EvalResult:
        pass

    def robustness(self, signal: Array2D, t: int = 0) -> float:
        return float(self.evaluate(signal).robustness[t])

    def robustness_with_grad(
        self, signal: Array2D, t: int = 0
    ) -> tuple[float, Array2D]:
        return self.evaluate(signal).at(t)

    def __and__(self, other: "Formula") -> "And":
        return And(self, other)

    def __or__(self, other: "Formula") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def always(
        self,
        interval: tuple[int, Optional[int]] = (0, None),
        *,
        eps: float = 1e-8,
        p: int = 1,
        weights: Optional[Sequence[float] | NDArray[np.float64]] = None,
    ) -> "Always":
        return Always(self, interval=interval, eps=eps, p=p, weights=weights)

    def eventually(
        self,
        interval: tuple[int, Optional[int]] = (0, None),
        *,
        eps: float = 1e-8,
        p: int = 1,
        weights: Optional[Sequence[float] | NDArray[np.float64]] = None,
    ) -> "Eventually":
        return Eventually(self, interval=interval, eps=eps, p=p, weights=weights)

    def until(
        self,
        other: "Formula",
        interval: tuple[int, Optional[int]] = (0, None),
        *,
        eps: float = 1e-8,
        p: int = 1,
        weights_left: Optional[Sequence[float] | NDArray[np.float64]] = None,
        weights_event: Optional[Sequence[float] | NDArray[np.float64]] = None,
        weights_pair: Sequence[float] | NDArray[np.float64] = (1.0, 1.0),
    ) -> "Until":
        return Until(
            self,
            other,
            interval=interval,
            eps=eps,
            p=p,
            weights_left=weights_left,
            weights_event=weights_event,
            weights_pair=weights_pair,
        )


class Predicate(Formula):
    """Atomic predicate over a signal trace."""

    def __init__(self, name: str, evaluator: Callable[[Array2D], EvalResult]) -> None:
        self.name = name
        self._evaluator = evaluator

    @classmethod
    def from_state_functions(
        cls,
        name: str,
        value_fn: Callable[[Array1D], float],
        grad_fn: Callable[[Array1D], Sequence[float] | NDArray[np.float64]],
    ) -> "Predicate":
        """Construct a predicate from per-state value and gradient functions."""

        def _evaluate(signal: Array2D) -> EvalResult:
            x = _validate_signal(signal)
            t_horizon, n_state = x.shape
            robustness = np.zeros(t_horizon, dtype=float)
            grad = np.zeros((t_horizon, t_horizon, n_state), dtype=float)
            for t in range(t_horizon):
                robustness[t] = float(value_fn(x[t]))
                gt = np.asarray(grad_fn(x[t]), dtype=float).reshape(-1)
                if gt.size != n_state:
                    raise ValueError(
                        f"Predicate gradient size mismatch at t={t}: "
                        f"expected {n_state}, got {gt.size}."
                    )
                grad[t, t, :] = gt
            return EvalResult(robustness=robustness, grad=grad)

        return cls(name=name, evaluator=_evaluate)

    @classmethod
    def affine(
        cls,
        name: str,
        coeffs: Sequence[float] | NDArray[np.float64],
        bias: float = 0.0,
    ) -> "Predicate":
        """Construct an affine predicate `rho(x_t) = coeffs @ x_t + bias`."""
        w = np.asarray(coeffs, dtype=float).reshape(-1)

        def _value(x_t: Array1D) -> float:
            if x_t.size != w.size:
                raise ValueError(
                    f"Signal state dimension must be {w.size}, got {x_t.size}."
                )
            return float(np.dot(w, x_t) + bias)

        def _grad(x_t: Array1D) -> Array1D:
            if x_t.size != w.size:
                raise ValueError(
                    f"Signal state dimension must be {w.size}, got {x_t.size}."
                )
            return w

        return cls.from_state_functions(name=name, value_fn=_value, grad_fn=_grad)

    def evaluate(self, signal: Array2D) -> EvalResult:
        return self._evaluator(_validate_signal(signal))

    def __repr__(self) -> str:
        return f"Predicate(name={self.name!r})"


class Not(Formula):
    """Logical negation formula."""

    def __init__(self, child: Formula) -> None:
        self.child = child

    def evaluate(self, signal: Array2D) -> EvalResult:
        child_eval = self.child.evaluate(signal)
        return EvalResult(
            robustness=-child_eval.robustness,
            grad=-child_eval.grad,
        )


class And(Formula):
    """Smooth conjunction over one or more child formulas."""

    def __init__(
        self,
        *children: Formula,
        eps: float = 1e-8,
        p: int = 1,
        weights: Optional[Sequence[float] | NDArray[np.float64]] = None,
    ) -> None:
        if len(children) == 0:
            raise ValueError("And requires at least one child.")
        self.children = children
        self.eps = eps
        self.p = p
        self.weights = weights

    def evaluate(self, signal: Array2D) -> EvalResult:
        child_eval = [child.evaluate(signal) for child in self.children]
        t_horizon = child_eval[0].robustness.size
        n_state = child_eval[0].grad.shape[-1]
        for idx in range(1, len(child_eval)):
            if (
                child_eval[idx].robustness.size != t_horizon
                or child_eval[idx].grad.shape[-1] != n_state
            ):
                raise ValueError(
                    "All child formulas must share the same signal dimensions."
                )

        w = _resolve_weights(self.weights, len(child_eval), "weights")
        robustness = np.zeros(t_horizon, dtype=float)
        grad = np.zeros((t_horizon, t_horizon, n_state), dtype=float)

        for t in range(t_horizon):
            vals = np.array([ce.robustness[t] for ce in child_eval], dtype=float)
            rho_t, coeff = gmsr_and(eps=self.eps, p=self.p, weights=w, values=vals)
            robustness[t] = rho_t
            g_t = np.zeros((t_horizon, n_state), dtype=float)
            for i, ce in enumerate(child_eval):
                g_t += coeff[i] * ce.grad[t]
            grad[t] = g_t
        return EvalResult(robustness=robustness, grad=grad)


class Or(Formula):
    """Smooth disjunction over one or more child formulas."""

    def __init__(
        self,
        *children: Formula,
        eps: float = 1e-8,
        p: int = 1,
        weights: Optional[Sequence[float] | NDArray[np.float64]] = None,
    ) -> None:
        if len(children) == 0:
            raise ValueError("Or requires at least one child.")
        self.children = children
        self.eps = eps
        self.p = p
        self.weights = weights

    def evaluate(self, signal: Array2D) -> EvalResult:
        child_eval = [child.evaluate(signal) for child in self.children]
        t_horizon = child_eval[0].robustness.size
        n_state = child_eval[0].grad.shape[-1]
        for idx in range(1, len(child_eval)):
            if (
                child_eval[idx].robustness.size != t_horizon
                or child_eval[idx].grad.shape[-1] != n_state
            ):
                raise ValueError(
                    "All child formulas must share the same signal dimensions."
                )

        w = _resolve_weights(self.weights, len(child_eval), "weights")
        robustness = np.zeros(t_horizon, dtype=float)
        grad = np.zeros((t_horizon, t_horizon, n_state), dtype=float)

        for t in range(t_horizon):
            vals = np.array([ce.robustness[t] for ce in child_eval], dtype=float)
            rho_t, coeff = gmsr_or(eps=self.eps, p=self.p, weights=w, values=vals)
            robustness[t] = rho_t
            g_t = np.zeros((t_horizon, n_state), dtype=float)
            for i, ce in enumerate(child_eval):
                g_t += coeff[i] * ce.grad[t]
            grad[t] = g_t
        return EvalResult(robustness=robustness, grad=grad)


class Always(Formula):
    """Smooth temporal ALWAYS operator over a bounded or unbounded interval."""

    def __init__(
        self,
        child: Formula,
        *,
        interval: tuple[int, Optional[int]] = (0, None),
        eps: float = 1e-8,
        p: int = 1,
        weights: Optional[Sequence[float] | NDArray[np.float64]] = None,
    ) -> None:
        self.child = child
        self.start, self.end = _parse_interval(interval)
        self.eps = eps
        self.p = p
        self.weights = weights

    def evaluate(self, signal: Array2D) -> EvalResult:
        child_eval = self.child.evaluate(signal)
        t_horizon = child_eval.robustness.size
        n_state = child_eval.grad.shape[-1]
        robustness = np.full(t_horizon, np.inf, dtype=float)
        grad = np.zeros((t_horizon, t_horizon, n_state), dtype=float)

        for t in range(t_horizon):
            start = t + self.start
            end = (
                (t_horizon - 1)
                if self.end is None
                else min(t + self.end, t_horizon - 1)
            )
            if start > end:
                continue
            vals = child_eval.robustness[start : end + 1]
            w = _resolve_weights(self.weights, vals.size, "weights")
            rho_t, coeff = gmsr_and(eps=self.eps, p=self.p, weights=w, values=vals)
            robustness[t] = rho_t

            g_t = np.zeros((t_horizon, n_state), dtype=float)
            for i, tau in enumerate(range(start, end + 1)):
                g_t += coeff[i] * child_eval.grad[tau]
            grad[t] = g_t
        return EvalResult(robustness=robustness, grad=grad)


class Eventually(Formula):
    """Smooth temporal EVENTUALLY operator over a bounded or unbounded interval."""

    def __init__(
        self,
        child: Formula,
        *,
        interval: tuple[int, Optional[int]] = (0, None),
        eps: float = 1e-8,
        p: int = 1,
        weights: Optional[Sequence[float] | NDArray[np.float64]] = None,
    ) -> None:
        self.child = child
        self.start, self.end = _parse_interval(interval)
        self.eps = eps
        self.p = p
        self.weights = weights

    def evaluate(self, signal: Array2D) -> EvalResult:
        child_eval = self.child.evaluate(signal)
        t_horizon = child_eval.robustness.size
        n_state = child_eval.grad.shape[-1]
        robustness = np.full(t_horizon, -np.inf, dtype=float)
        grad = np.zeros((t_horizon, t_horizon, n_state), dtype=float)

        for t in range(t_horizon):
            start = t + self.start
            end = (
                (t_horizon - 1)
                if self.end is None
                else min(t + self.end, t_horizon - 1)
            )
            if start > end:
                continue
            vals = child_eval.robustness[start : end + 1]
            w = _resolve_weights(self.weights, vals.size, "weights")
            rho_t, coeff = gmsr_or(eps=self.eps, p=self.p, weights=w, values=vals)
            robustness[t] = rho_t

            g_t = np.zeros((t_horizon, n_state), dtype=float)
            for i, tau in enumerate(range(start, end + 1)):
                g_t += coeff[i] * child_eval.grad[tau]
            grad[t] = g_t
        return EvalResult(robustness=robustness, grad=grad)


class Until(Formula):
    """Smooth temporal UNTIL operator with optional weighting controls."""

    def __init__(
        self,
        left: Formula,
        right: Formula,
        *,
        interval: tuple[int, Optional[int]] = (0, None),
        eps: float = 1e-8,
        p: int = 1,
        weights_left: Optional[Sequence[float] | NDArray[np.float64]] = None,
        weights_event: Optional[Sequence[float] | NDArray[np.float64]] = None,
        weights_pair: Sequence[float] | NDArray[np.float64] = (1.0, 1.0),
    ) -> None:
        self.left = left
        self.right = right
        self.start, self.end = _parse_interval(interval)
        self.eps = eps
        self.p = p
        self.weights_left = weights_left
        self.weights_event = weights_event
        self.weights_pair = weights_pair

    def evaluate(self, signal: Array2D) -> EvalResult:
        left_eval = self.left.evaluate(signal)
        right_eval = self.right.evaluate(signal)
        if left_eval.robustness.size != right_eval.robustness.size:
            raise ValueError("Until operands must share the same horizon.")
        if left_eval.grad.shape[-1] != right_eval.grad.shape[-1]:
            raise ValueError("Until operands must share the same state dimension.")

        t_horizon = left_eval.robustness.size
        n_state = left_eval.grad.shape[-1]
        robustness = np.full(t_horizon, -np.inf, dtype=float)
        grad = np.zeros((t_horizon, t_horizon, n_state), dtype=float)
        w_pair = _resolve_weights(self.weights_pair, 2, "weights_pair")

        for t in range(t_horizon):
            c_start = t + self.start
            c_end = (
                (t_horizon - 1)
                if self.end is None
                else min(t + self.end, t_horizon - 1)
            )
            if c_start > c_end:
                continue

            f_trace = left_eval.robustness[t:]
            g_trace = right_eval.robustness[t:]
            n_future = f_trace.size
            candidate_offsets = np.arange(c_start - t, c_end - t + 1, dtype=int)
            max_offset = int(candidate_offsets[-1])

            w_left = _resolve_weights(self.weights_left, max_offset + 1, "weights_left")
            w_event = _resolve_weights(
                self.weights_event, candidate_offsets.size, "weights_event"
            )

            s_vals: list[float] = []
            ds_df: list[Array1D] = []
            ds_dg: list[float] = []

            for offset in candidate_offsets:
                y_i, dyi_dfi = gmsr_and(
                    eps=self.eps,
                    p=self.p,
                    weights=w_left[: offset + 1],
                    values=f_trace[: offset + 1],
                )
                s_i, dsi_pair = gmsr_and(
                    eps=self.eps,
                    p=self.p,
                    weights=w_pair,
                    values=np.array([y_i, g_trace[offset]], dtype=float),
                )
                dsi_dyi = dsi_pair[0]
                dsi_dgi = dsi_pair[1]
                s_vals.append(s_i)
                ds_dg.append(float(dsi_dgi))
                ds_df.append(dsi_dyi * dyi_dfi)

            z_t, dz_ds = gmsr_or(
                eps=self.eps,
                p=self.p,
                weights=w_event,
                values=np.asarray(s_vals, dtype=float),
            )
            robustness[t] = z_t

            dz_df = np.zeros(n_future, dtype=float)
            dz_dg = np.zeros(n_future, dtype=float)
            for idx, offset in enumerate(candidate_offsets):
                dz_df[: offset + 1] += dz_ds[idx] * ds_df[idx]
                dz_dg[offset] += dz_ds[idx] * ds_dg[idx]

            g_t = np.zeros((t_horizon, n_state), dtype=float)
            for local_idx in range(n_future):
                tau = t + local_idx
                if dz_df[local_idx] != 0.0:
                    g_t += dz_df[local_idx] * left_eval.grad[tau]
                if dz_dg[local_idx] != 0.0:
                    g_t += dz_dg[local_idx] * right_eval.grad[tau]
            grad[t] = g_t

        return EvalResult(robustness=robustness, grad=grad)


__all__ = [
    "EvalResult",
    "Formula",
    "Predicate",
    "Not",
    "And",
    "Or",
    "Always",
    "Eventually",
    "Until",
    "gmsr_and",
    "gmsr_or",
    "gmsr_until",
]
