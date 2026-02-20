"""D-GMSR semantics backend for the unified API."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, cast

import numpy as np

from .base import Semantics
from ._dgmsr import gmsr_or, gmsr_and


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

    def evaluate_with_grad(
        self,
        formula,
        signal,
        *,
        t: int = 0,
        predicate_fd_eps: float = 1e-6,
    ) -> tuple[float, np.ndarray]:
        """Evaluate robustness and gradient for a unified API formula.

        Returns `(rho_t, drho_t_dsignal)` where the gradient has shape
        `(time, state_dim)` and corresponds to the selected time index `t`.

        Notes:
        - Gradients are only supported for the NumPy D-GMSR backend.
        - Predicate gradients are taken from `Predicate.grad` when provided;
          otherwise they are approximated by finite differences w.r.t. `signal[tau]`.
        """

        # Delayed imports avoid cycles: pystl.api -> pystl.semantics.
        # pylint: disable=import-outside-toplevel
        from ..api import Or, And, Not, Until, Always, Formula, Predicate, Eventually
        from ._dgmsr import Or as DgOr
        from ._dgmsr import And as DgAnd
        from ._dgmsr import Not as DgNot
        from ._dgmsr import Until as DgUntil
        from ._dgmsr import Always as DgAlways
        from ._dgmsr import Predicate as DgPredicate
        from ._dgmsr import EvalResult
        from ._dgmsr import Eventually as DgEventually

        if not isinstance(formula, Formula):
            raise TypeError(
                f"formula must be a pystl.api.Formula, got {type(formula)!r}"
            )

        x = np.asarray(signal, dtype=float)
        if x.ndim != 2:
            raise ValueError("signal must be a 2D array with shape (time, state_dim).")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("signal must be non-empty in both dimensions.")
        if t < 0 or t >= x.shape[0]:
            raise IndexError(
                f"t={t} is out of bounds for signal with horizon={x.shape[0]}."
            )

        def _predicate_eval(pred: Predicate) -> DgPredicate:
            if pred.fn is None:
                raise ValueError(
                    f"Predicate '{pred.name}' does not define `fn(signal, t)`."
                )
            pred_fn = cast(Callable[[Any, int], Any], pred.fn)

            grad_fn: Callable[[Any, int], Any] | None = pred.grad
            if grad_fn is None and pred.metadata is not None:
                meta_grad = pred.metadata.get("grad")
                if callable(meta_grad):
                    grad_fn = cast(Callable[[Any, int], Any], meta_grad)

            def _evaluate(signal2d: np.ndarray) -> EvalResult:
                sig = np.asarray(signal2d, dtype=float)
                horizon, state_dim = sig.shape
                robustness = np.zeros(horizon, dtype=float)
                grad = np.zeros((horizon, horizon, state_dim), dtype=float)

                h = float(predicate_fd_eps)
                if h <= 0.0:
                    raise ValueError("predicate_fd_eps must be > 0.")

                for tau in range(horizon):
                    robustness[tau] = float(pred_fn(sig, tau))

                    if grad_fn is not None:
                        g_tau = np.asarray(grad_fn(sig, tau), dtype=float).reshape(-1)
                        if g_tau.size != state_dim:
                            raise ValueError(
                                (
                                    f"Predicate '{pred.name}' grad size mismatch at "
                                    f"t={tau}: expected {state_dim}, got {g_tau.size}."
                                )
                            )
                        grad[tau, tau, :] = g_tau
                        continue

                    base = sig[tau].copy()
                    g_tau = np.zeros(state_dim, dtype=float)
                    for i in range(state_dim):
                        sig_p = sig.copy()
                        sig_m = sig.copy()
                        sig_p[tau, i] = base[i] + h
                        sig_m[tau, i] = base[i] - h
                        fp = float(pred_fn(sig_p, tau))
                        fm = float(pred_fn(sig_m, tau))
                        g_tau[i] = (fp - fm) / (2.0 * h)
                    grad[tau, tau, :] = g_tau

                return EvalResult(robustness=robustness, grad=grad)

            return DgPredicate(name=pred.name, evaluator=_evaluate)

        def _convert(node: Formula):
            if isinstance(node, Predicate):
                return _predicate_eval(node)
            if isinstance(node, Not):
                return DgNot(_convert(node.child))
            if isinstance(node, And):
                return DgAnd(
                    *[_convert(ch) for ch in node.children],
                    eps=self.eps,
                    p=self.p,
                    weights=node.weights,
                )
            if isinstance(node, Or):
                return DgOr(
                    *[_convert(ch) for ch in node.children],
                    eps=self.eps,
                    p=self.p,
                    weights=node.weights,
                )
            if isinstance(node, Always):
                interval = (node.interval.start, node.interval.end)
                return DgAlways(
                    _convert(node.child),
                    interval=interval,
                    eps=self.eps,
                    p=self.p,
                    weights=node.weights,
                )
            if isinstance(node, Eventually):
                interval = (node.interval.start, node.interval.end)
                return DgEventually(
                    _convert(node.child),
                    interval=interval,
                    eps=self.eps,
                    p=self.p,
                    weights=node.weights,
                )
            if isinstance(node, Until):
                interval = (node.interval.start, node.interval.end)
                return DgUntil(
                    _convert(node.left),
                    _convert(node.right),
                    interval=interval,
                    eps=self.eps,
                    p=self.p,
                    weights_left=node.weights_left,
                    weights_event=node.weights_right,
                    weights_pair=node.weights_pair,
                )
            raise TypeError(f"Unsupported formula node type: {type(node)!r}")

        dg_formula = _convert(formula)
        result = dg_formula.evaluate(x)
        rho_t, grad_t = result.at(t)
        return float(rho_t), np.asarray(grad_t, dtype=float)
