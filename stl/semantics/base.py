"""Semantics backend interface for unified STL formulas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Optional, Sequence, TypeAlias

from numpy.typing import NDArray

if TYPE_CHECKING:
    from stl.api import Predicate


Signal: TypeAlias = NDArray[Any]
ValueT = TypeVar("ValueT")


class Semantics(ABC, Generic[ValueT]):
    """Contract each STL semantics backend should implement."""

    @abstractmethod
    def predicate(self, predicate: "Predicate", signal: Signal, t: int) -> ValueT:
        pass

    @abstractmethod
    def boolean_not(self, value: ValueT) -> ValueT:
        pass

    @abstractmethod
    def boolean_and(
        self,
        values: Sequence[ValueT],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> ValueT:
        pass

    @abstractmethod
    def boolean_or(
        self,
        values: Sequence[ValueT],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> ValueT:
        pass

    def temporal_always(
        self,
        values: Sequence[ValueT],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> ValueT:
        return self.boolean_and(values, weights=weights)

    def temporal_eventually(
        self,
        values: Sequence[ValueT],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> ValueT:
        return self.boolean_or(values, weights=weights)

    @abstractmethod
    def temporal_until(
        self,
        *,
        left_trace: Sequence[ValueT],
        right_trace: Sequence[ValueT],
        start: int,
        end: Optional[int],
        weights_left: Optional[Sequence[float]] = None,
        weights_right: Optional[Sequence[float]] = None,
        weights_pair: Sequence[float] = (1.0, 1.0),
    ) -> ValueT:
        pass
