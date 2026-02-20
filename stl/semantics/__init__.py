from __future__ import annotations

from stl.semantics.base import Semantics
from stl.semantics.ctstl import CtstlSemantics, tau_to_k, kth_largest
from stl.semantics.dgmsr import DgmsrSemantics
from stl.semantics.classic import ClassicRobustSemantics
from stl.semantics.registry import SemanticsRegistry
from stl.semantics.cumulative import CumulativeSemantics, CumulativeRobustness

_JAX_IMPORT_ERROR: Exception | None = None
_HAS_JAX = False
try:
    from stl.jax import (
        JaxCtstlSemantics,
        JaxDgmsrSemantics,
        JaxCumulativeSemantics,
        JaxCumulativeRobustness,
        JaxClassicRobustSemantics,
        jax_tau_to_k,
        jax_kth_largest,
    )
except ImportError as exc:
    _JAX_IMPORT_ERROR = exc
else:
    _HAS_JAX = True

registry = SemanticsRegistry()
registry.register(syntax="classical", backend="numpy", factory=ClassicRobustSemantics)
registry.register(syntax="cumulative", backend="numpy", factory=CumulativeSemantics)
registry.register(syntax="dgmsr", backend="numpy", factory=DgmsrSemantics)
if _HAS_JAX:
    registry.register(
        syntax="classical", backend="jax", factory=JaxClassicRobustSemantics
    )
    registry.register(
        syntax="cumulative", backend="jax", factory=JaxCumulativeSemantics
    )
    registry.register(syntax="dgmsr", backend="jax", factory=JaxDgmsrSemantics)

_SYNTAX_ALIASES = {
    "classical": "classical",
    "cumulative": "cumulative",
    "dgmsr": "dgmsr",
}
_BACKEND_ALIASES = {
    "numpy": "numpy",
    "np": "numpy",
    "python": "numpy",
    "jax": "jax",
}


def _normalize_syntax(syntax: str) -> str:
    key = syntax.strip().lower().replace("-", "_")
    if key not in _SYNTAX_ALIASES:
        raise KeyError(f"Unknown syntax '{syntax}'. Available: {registry.syntaxes()}")
    return _SYNTAX_ALIASES[key]


def _normalize_backend(backend: str) -> str:
    key = backend.strip().lower().replace("-", "_")
    if key not in _BACKEND_ALIASES:
        raise KeyError(f"Unknown backend '{backend}'. Available: {registry.backends()}")
    return _BACKEND_ALIASES[key]


def create_semantics(syntax: str, *, backend: str = "numpy", **kwargs):
    normalized_syntax = _normalize_syntax(syntax)
    normalized_backend = _normalize_backend(backend)
    if normalized_backend == "jax" and not _HAS_JAX:
        raise ImportError(
            "JAX backend dependencies are not installed. "
            "Install with `uv sync --extra jax` or `pip install -e .[jax]`."
        ) from _JAX_IMPORT_ERROR
    return registry.create(
        syntax=normalized_syntax,
        backend=normalized_backend,
        **kwargs,
    )


__all__ = [
    "Semantics",
    "ClassicRobustSemantics",
    "CumulativeSemantics",
    "CumulativeRobustness",
    "CtstlSemantics",
    "kth_largest",
    "tau_to_k",
    "DgmsrSemantics",
    "SemanticsRegistry",
    "registry",
    "create_semantics",
]
if _HAS_JAX:
    __all__.extend(
        [
            "JaxClassicRobustSemantics",
            "JaxCumulativeRobustness",
            "JaxCumulativeSemantics",
            "jax_kth_largest",
            "jax_tau_to_k",
            "JaxCtstlSemantics",
            "JaxDgmsrSemantics",
        ]
    )
