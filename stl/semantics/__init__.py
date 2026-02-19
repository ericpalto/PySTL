from __future__ import annotations

from stl.jax import (
    JaxCtstlSemantics,
    JaxDgmsrSemantics,
    JaxCumulativeSemantics,
    JaxCumulativeRobustness,
    JaxClassicRobustSemantics,
    jax_tau_to_k,
    jax_kth_largest,
)
from stl.semantics.base import Semantics
from stl.semantics.ctstl import CtstlSemantics, tau_to_k, kth_largest
from stl.semantics.dgmsr import DgmsrSemantics
from stl.semantics.classic import ClassicRobustSemantics
from stl.semantics.registry import SemanticsRegistry
from stl.semantics.cumulative import CumulativeSemantics, CumulativeRobustness

registry = SemanticsRegistry()
registry.register(syntax="classical", backend="numpy", factory=ClassicRobustSemantics)
registry.register(syntax="classical", backend="jax", factory=JaxClassicRobustSemantics)
registry.register(syntax="cumulative", backend="numpy", factory=CumulativeSemantics)
registry.register(syntax="cumulative", backend="jax", factory=JaxCumulativeSemantics)
registry.register(syntax="ctstl", backend="numpy", factory=CtstlSemantics)
registry.register(syntax="ctstl", backend="jax", factory=JaxCtstlSemantics)
registry.register(syntax="dgmsr", backend="numpy", factory=DgmsrSemantics)
registry.register(syntax="dgmsr", backend="jax", factory=JaxDgmsrSemantics)

_SYNTAX_ALIASES = {
    "classical": "classical",
    "cumulative": "cumulative",
    "ctstl": "ctstl",
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
    "JaxClassicRobustSemantics",
    "JaxCumulativeRobustness",
    "JaxCumulativeSemantics",
    "jax_kth_largest",
    "jax_tau_to_k",
    "JaxCtstlSemantics",
    "JaxDgmsrSemantics",
    "SemanticsRegistry",
    "registry",
    "create_semantics",
]
