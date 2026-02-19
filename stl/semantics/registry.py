"""Semantics backend registry."""

from __future__ import annotations

from typing import Any, Callable

from stl.semantics.base import Semantics

SemanticsFactory = Callable[..., Semantics[Any]]


class SemanticsRegistry:
    """Create semantics backends by normalized string name."""

    def __init__(self) -> None:
        self._factories: dict[str, SemanticsFactory] = {}

    def register(self, name: str, factory: SemanticsFactory) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Semantics name cannot be empty.")
        self._factories[key] = factory

    def create(self, name: str, **kwargs: Any) -> Semantics[Any]:
        key = name.strip().lower()
        if key not in self._factories:
            available = ", ".join(sorted(self._factories))
            raise KeyError(f"Unknown semantics '{name}'. Available: [{available}]")
        return self._factories[key](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)
