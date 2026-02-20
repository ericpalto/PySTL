"""PyTorch-native STL semantics backends."""

from __future__ import annotations

from typing import Any, Optional, Sequence
from dataclasses import dataclass

import torch

from stl.semantics.base import Semantics


def _validate_temperature(temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    return float(temperature)


def _reference_tensor(values: Sequence[Any]) -> torch.Tensor | None:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value
    return None


def _as_float_tensor(value: Any, *, like: torch.Tensor | None = None) -> torch.Tensor:
    if like is None:
        return torch.as_tensor(value, dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64, device=like.device)


def _stack_values(values: Sequence[Any], *, where: str) -> torch.Tensor:
    if len(values) == 0:
        raise ValueError(f"{where} requires at least one value.")
    ref = _reference_tensor(values)
    return torch.stack([_as_float_tensor(v, like=ref) for v in values], dim=0)


def _broadcast_weights(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if values.ndim <= 1:
        return weights
    return weights.reshape((weights.shape[0],) + (1,) * (values.ndim - 1))


def _resolve_weights(
    weights: Optional[Sequence[float]],
    length: int,
    name: str,
    *,
    like: torch.Tensor | None = None,
    non_negative: bool = False,
    positive_sum: bool = False,
) -> torch.Tensor:
    if length <= 0:
        raise ValueError(f"{name} length must be positive.")
    if weights is None:
        return _as_float_tensor(torch.ones(length, dtype=torch.float64), like=like)

    arr = _as_float_tensor(weights, like=like).reshape(-1)
    if arr.numel() < length:
        raise ValueError(f"{name} requires at least {length} entries.")
    out = arr[:length]

    if non_negative and bool(torch.any(out < 0.0).item()):
        raise ValueError(f"{name} entries must be >= 0.")
    if positive_sum and bool((torch.sum(out) <= 0.0).item()):
        raise ValueError(f"{name} entries must contain at least one positive weight.")
    return out


def _weighted_arithmetic_mean_torch(
    values: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    w = _broadcast_weights(weights, values)
    return torch.sum(w * values, dim=0) / torch.sum(w, dim=0)


def _weighted_geometric_mean_torch(
    values: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    w = _broadcast_weights(weights, values)
    norm = w / torch.sum(w, dim=0)
    safe_values = torch.clamp(values, min=1e-12)
    return torch.exp(torch.sum(norm * torch.log(safe_values), dim=0))


class TorchClassicRobustSemantics(Semantics[Any]):
    """Classic STL max-min robustness implemented with Torch operators."""

    def predicate(self, predicate: Any, signal, t: int) -> Any:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        base = signal if isinstance(signal, torch.Tensor) else None
        return _as_float_tensor(predicate.fn(signal, t), like=base)

    def boolean_not(self, value: Any) -> Any:
        return -_as_float_tensor(value)

    def boolean_and(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        del weights
        arr = _stack_values(values, where="boolean_and")
        return torch.min(arr, dim=0).values

    def boolean_or(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        del weights
        arr = _stack_values(values, where="boolean_or")
        return torch.max(arr, dim=0).values

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

        left = _stack_values(left_trace, where="temporal_until")
        right = _stack_values(right_trace, where="temporal_until")
        if left.shape[0] != right.shape[0]:
            raise ValueError("UNTIL traces must have the same length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = left.shape[0] - 1 if end is None else min(end, left.shape[0] - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        best = torch.full_like(right[0], float("-inf"))
        for idx in range(start, last + 1):
            prefix = torch.min(left[: idx + 1], dim=0).values
            candidate = torch.minimum(prefix, right[idx])
            best = torch.maximum(best, candidate)
        return best


class TorchSmoothRobustSemantics(TorchClassicRobustSemantics):
    """Smooth STL robustness using Torch log-sum-exp min/max approximations."""

    def __init__(self, *, temperature: float = 1.0) -> None:
        self.temperature = _validate_temperature(temperature)

    def _softmin(self, values: Sequence[Any], *, where: str) -> torch.Tensor:
        arr = _stack_values(values, where=where)
        tau = self.temperature
        return -tau * torch.logsumexp(-arr / tau, dim=0)

    def _softmax(self, values: Sequence[Any], *, where: str) -> torch.Tensor:
        arr = _stack_values(values, where=where)
        tau = self.temperature
        return tau * torch.logsumexp(arr / tau, dim=0)

    def boolean_and(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        del weights
        return self._softmin(values, where="boolean_and")

    def boolean_or(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        del weights
        return self._softmax(values, where="boolean_or")

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

        left = _stack_values(left_trace, where="temporal_until")
        right = _stack_values(right_trace, where="temporal_until")
        if left.shape[0] != right.shape[0]:
            raise ValueError("UNTIL traces must have the same length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = left.shape[0] - 1 if end is None else min(end, left.shape[0] - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        candidates: list[torch.Tensor] = []
        for idx in range(start, last + 1):
            prefix = self._softmin(left[: idx + 1], where="temporal_until")
            candidates.append(
                self._softmin(
                    (
                        prefix,
                        right[idx],
                    ),
                    where="temporal_until",
                )
            )
        return self._softmax(candidates, where="temporal_until")


@dataclass(frozen=True)
class TorchCumulativeRobustness:
    """Container for cumulative positive/negative robustness values."""

    pos: Any
    neg: Any


class TorchCumulativeSemantics(Semantics[TorchCumulativeRobustness]):
    """Cumulative STL semantics implemented with Torch operators."""

    def predicate(self, predicate: Any, signal, t: int) -> TorchCumulativeRobustness:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        base = signal if isinstance(signal, torch.Tensor) else None
        value = _as_float_tensor(predicate.fn(signal, t), like=base)
        zero = torch.zeros_like(value)
        return TorchCumulativeRobustness(
            pos=torch.maximum(zero, value), neg=torch.minimum(zero, value)
        )

    def boolean_not(
        self, value: TorchCumulativeRobustness
    ) -> TorchCumulativeRobustness:
        return TorchCumulativeRobustness(pos=-value.neg, neg=-value.pos)

    def boolean_and(
        self,
        values: Sequence[TorchCumulativeRobustness],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> TorchCumulativeRobustness:
        del weights
        if len(values) == 0:
            raise ValueError("boolean_and requires at least one value.")
        pos = torch.min(
            torch.stack([_as_float_tensor(v.pos) for v in values], dim=0), dim=0
        ).values
        neg = torch.min(
            torch.stack([_as_float_tensor(v.neg) for v in values], dim=0), dim=0
        ).values
        return TorchCumulativeRobustness(pos=pos, neg=neg)

    def boolean_or(
        self,
        values: Sequence[TorchCumulativeRobustness],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> TorchCumulativeRobustness:
        del weights
        if len(values) == 0:
            raise ValueError("boolean_or requires at least one value.")
        pos = torch.max(
            torch.stack([_as_float_tensor(v.pos) for v in values], dim=0), dim=0
        ).values
        neg = torch.max(
            torch.stack([_as_float_tensor(v.neg) for v in values], dim=0), dim=0
        ).values
        return TorchCumulativeRobustness(pos=pos, neg=neg)

    def temporal_eventually(
        self,
        values: Sequence[TorchCumulativeRobustness],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> TorchCumulativeRobustness:
        del weights
        if len(values) == 0:
            raise ValueError("temporal_eventually requires at least one value.")
        pos = torch.sum(
            torch.stack([_as_float_tensor(v.pos) for v in values], dim=0), dim=0
        )
        neg = torch.sum(
            torch.stack([_as_float_tensor(v.neg) for v in values], dim=0), dim=0
        )
        return TorchCumulativeRobustness(pos=pos, neg=neg)

    def temporal_until(
        self,
        *,
        left_trace: Sequence[TorchCumulativeRobustness],
        right_trace: Sequence[TorchCumulativeRobustness],
        start: int,
        end: Optional[int],
        weights_left: Optional[Sequence[float]] = None,
        weights_right: Optional[Sequence[float]] = None,
        weights_pair: Sequence[float] = (1.0, 1.0),
    ) -> TorchCumulativeRobustness:
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

        pos_total = torch.zeros_like(_as_float_tensor(left_trace[0].pos))
        neg_total = torch.zeros_like(_as_float_tensor(left_trace[0].neg))
        for idx in range(start, last + 1):
            prefix_pos = torch.min(
                torch.stack(
                    [_as_float_tensor(v.pos) for v in left_trace[: idx + 1]], dim=0
                ),
                dim=0,
            ).values
            prefix_neg = torch.min(
                torch.stack(
                    [_as_float_tensor(v.neg) for v in left_trace[: idx + 1]], dim=0
                ),
                dim=0,
            ).values
            pos_total = pos_total + torch.minimum(
                _as_float_tensor(right_trace[idx].pos), prefix_pos
            )
            neg_total = neg_total + torch.minimum(
                _as_float_tensor(right_trace[idx].neg), prefix_neg
            )

        return TorchCumulativeRobustness(pos=pos_total, neg=neg_total)


class TorchAgmRobustSemantics(Semantics[Any]):
    """AGM robustness semantics implemented with Torch operators."""

    def predicate(self, predicate: Any, signal, t: int) -> Any:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        base = signal if isinstance(signal, torch.Tensor) else None
        return _as_float_tensor(predicate.fn(signal, t), like=base)

    def boolean_not(self, value: Any) -> Any:
        return -_as_float_tensor(value)

    def boolean_and(
        self,
        values: Sequence[Any],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> Any:
        arr = _stack_values(values, where="boolean_and").reshape(-1)
        w = _resolve_weights(
            weights,
            arr.numel(),
            "weights",
            like=arr,
            non_negative=True,
            positive_sum=True,
        )

        neg_part = _weighted_arithmetic_mean_torch(
            torch.minimum(arr, arr.new_zeros(())), w
        )
        pos_part = _weighted_geometric_mean_torch(1.0 + arr, w) - 1.0
        return torch.where(torch.all(arr > 0.0), pos_part, neg_part)

    def boolean_or(
        self,
        values: Sequence[Any],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> Any:
        arr = _stack_values(values, where="boolean_or").reshape(-1)
        w = _resolve_weights(
            weights,
            arr.numel(),
            "weights",
            like=arr,
            non_negative=True,
            positive_sum=True,
        )

        pos_part = _weighted_arithmetic_mean_torch(
            torch.maximum(arr, arr.new_zeros(())), w
        )
        neg_part = 1.0 - _weighted_geometric_mean_torch(1.0 - arr, w)
        return torch.where(torch.any(arr > 0.0), pos_part, neg_part)

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
        left = _stack_values(left_trace, where="temporal_until").reshape(-1)
        right = _stack_values(right_trace, where="temporal_until").reshape(-1)
        if left.numel() != right.numel():
            raise ValueError("UNTIL traces must have equal length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = left.numel() - 1 if end is None else min(end, left.numel() - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        candidate_count = last - start + 1
        w_pair = _resolve_weights(
            weights_pair,
            2,
            "weights_pair",
            like=left,
            non_negative=True,
            positive_sum=True,
        )
        w_left = _resolve_weights(
            weights_left,
            last + 1,
            "weights_left",
            like=left,
            non_negative=True,
            positive_sum=True,
        )
        w_right = _resolve_weights(
            weights_right,
            candidate_count,
            "weights_right",
            like=left,
            non_negative=True,
            positive_sum=True,
        )

        candidates: list[torch.Tensor] = []
        for offset in range(start, last + 1):
            prefix = self.boolean_and(left[: offset + 1], weights=w_left[: offset + 1])
            candidates.append(
                self.boolean_and(
                    (_as_float_tensor(prefix, like=left), right[offset]),
                    weights=w_pair,
                )
            )
        return self.boolean_or(candidates, weights=w_right)


def _gmsr_and_torch(
    *,
    eps: float,
    p: int,
    weights: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    vals = values.reshape(-1)
    w = weights.reshape(-1)
    if vals.numel() == 0:
        raise ValueError("values must contain at least one value.")
    if w.numel() < vals.numel():
        raise ValueError("weights must have at least len(values) entries.")
    w = w[: vals.numel()]

    sum_w = torch.sum(w)
    neg_mask = vals <= 0.0

    neg_terms = torch.where(neg_mask, w * (vals ** (2 * p)), torch.zeros_like(vals))
    sums = torch.sum(neg_terms)
    eps_t = _as_float_tensor(eps, like=vals)
    mp = torch.pow(torch.pow(eps_t, p) + (sums / sum_w), 1.0 / p)
    h_neg = torch.pow(eps_t, 0.5) - torch.pow(mp, 0.5)

    base = torch.where(neg_mask, torch.ones_like(vals), vals)
    exponent = torch.where(neg_mask, torch.zeros_like(vals), 2.0 * w)
    mult = torch.prod(torch.pow(base, exponent))
    m0 = torch.pow(torch.pow(eps_t, sum_w) + mult, 1.0 / sum_w)
    h_pos = torch.pow(m0, 0.5) - torch.pow(eps_t, 0.5)

    return torch.where(torch.any(neg_mask), h_neg, h_pos)


def _gmsr_or_torch(
    *,
    eps: float,
    p: int,
    weights: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    return -_gmsr_and_torch(eps=eps, p=p, weights=weights, values=-values)


class TorchDgmsrSemantics(Semantics[Any]):
    """D-GMSR semantics implemented with Torch operators."""

    def __init__(self, *, eps: float = 1e-8, p: int = 1) -> None:
        self.eps = float(eps)
        self.p = int(p)

    def predicate(self, predicate: Any, signal, t: int) -> Any:
        if predicate.fn is None:
            raise ValueError(
                f"Predicate '{predicate.name}' does not define `fn(signal, t)`."
            )
        base = signal if isinstance(signal, torch.Tensor) else None
        return _as_float_tensor(predicate.fn(signal, t), like=base)

    def boolean_not(self, value: Any) -> Any:
        return -_as_float_tensor(value)

    def boolean_and(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        vals = _stack_values(values, where="boolean_and").reshape(-1)
        w = _resolve_weights(weights, vals.numel(), "weights", like=vals)
        return _gmsr_and_torch(eps=self.eps, p=self.p, weights=w, values=vals)

    def boolean_or(
        self, values: Sequence[Any], *, weights: Optional[Sequence[float]] = None
    ) -> Any:
        vals = _stack_values(values, where="boolean_or").reshape(-1)
        w = _resolve_weights(weights, vals.numel(), "weights", like=vals)
        return _gmsr_or_torch(eps=self.eps, p=self.p, weights=w, values=vals)

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
        left = _stack_values(left_trace, where="temporal_until").reshape(-1)
        right = _stack_values(right_trace, where="temporal_until").reshape(-1)
        if left.numel() != right.numel():
            raise ValueError("UNTIL traces must have equal length.")
        if start < 0:
            raise ValueError("UNTIL start must be >= 0.")

        last = left.numel() - 1 if end is None else min(end, left.numel() - 1)
        if start > last:
            raise ValueError("UNTIL temporal window is empty.")

        candidate_offsets = list(range(start, last + 1))
        w_pair = _resolve_weights(weights_pair, 2, "weights_pair", like=left)
        w_left = _resolve_weights(weights_left, last + 1, "weights_left", like=left)
        w_right = _resolve_weights(
            weights_right, len(candidate_offsets), "weights_right", like=left
        )

        s_values: list[torch.Tensor] = []
        for offset in candidate_offsets:
            y_i = _gmsr_and_torch(
                eps=self.eps,
                p=self.p,
                weights=w_left[: offset + 1],
                values=left[: offset + 1],
            )
            s_i = _gmsr_and_torch(
                eps=self.eps,
                p=self.p,
                weights=w_pair,
                values=torch.stack((y_i, right[offset])),
            )
            s_values.append(s_i)

        s_arr = torch.stack(s_values, dim=0)
        return _gmsr_or_torch(eps=self.eps, p=self.p, weights=w_right, values=s_arr)


__all__ = [
    "TorchClassicRobustSemantics",
    "TorchSmoothRobustSemantics",
    "TorchAgmRobustSemantics",
    "TorchCumulativeRobustness",
    "TorchCumulativeSemantics",
    "TorchDgmsrSemantics",
]
