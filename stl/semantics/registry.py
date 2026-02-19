"""Semantics backend registry keyed by ``(syntax, backend)``."""

from __future__ import annotations

from typing import Any, Callable

from stl.semantics.base import Semantics

SemanticsFactory = Callable[..., Semantics[Any]]


class SemanticsRegistry:
    """Create semantics backends by normalized ``(syntax, backend)`` key."""

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], SemanticsFactory] = {}

    @staticmethod
    def _normalize(value: str, *, field_name: str) -> str:
        key = value.strip().lower().replace("-", "_")
        if not key:
            raise ValueError(f"{field_name} cannot be empty.")
        return key

    def register(
        self,
        *,
        syntax: str,
        backend: str,
        factory: SemanticsFactory,
    ) -> None:
        syntax_key = self._normalize(syntax, field_name="syntax")
        backend_key = self._normalize(backend, field_name="backend")
        self._factories[(syntax_key, backend_key)] = factory

    def create(
        self,
        *,
        syntax: str,
        backend: str = "numpy",
        **kwargs: Any,
    ) -> Semantics[Any]:
        syntax_key = self._normalize(syntax, field_name="syntax")
        backend_key = self._normalize(backend, field_name="backend")
        key = (syntax_key, backend_key)
        if key not in self._factories:
            available = ", ".join(self.names())
            raise KeyError(
                f"Unknown semantics syntax/backend '{syntax}/{backend}'. "
                f"Available: [{available}]"
            )
        return self._factories[key](**kwargs)

    def names(self) -> list[str]:
        return [f"{syntax}/{backend}" for syntax, backend in sorted(self._factories)]

    def syntaxes(self) -> list[str]:
        return sorted({syntax for syntax, _ in self._factories})

    def backends(self) -> list[str]:
        return sorted({backend for _, backend in self._factories})
