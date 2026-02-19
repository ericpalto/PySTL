from __future__ import annotations

from stl.semantics.base import Semantics
from stl.semantics.ctstl import CtstlSemantics, tau_to_k, kth_largest
from stl.semantics.dgmsr import DgmsrSemantics
from stl.semantics.stljax import (
    StlJaxSemantics,
    StlJaxFormulaWrapper,
    to_stljax_formula,
)
from stl.semantics.classic import ClassicRobustSemantics
from stl.semantics.registry import SemanticsRegistry
from stl.semantics.cumulative import CumulativeSemantics, CumulativeRobustness
from stl.semantics.traditional import TraditionalRobustSemantics

registry = SemanticsRegistry()
registry.register("classic", ClassicRobustSemantics)
registry.register("traditional", TraditionalRobustSemantics)
registry.register("cumulative", CumulativeSemantics)
registry.register("ctstl", CtstlSemantics)
registry.register("dgmsr", DgmsrSemantics)
registry.register("stljax", StlJaxSemantics)


def create_semantics(name: str, **kwargs):
    return registry.create(name, **kwargs)


__all__ = [
    "Semantics",
    "ClassicRobustSemantics",
    "TraditionalRobustSemantics",
    "CumulativeSemantics",
    "CumulativeRobustness",
    "CtstlSemantics",
    "kth_largest",
    "tau_to_k",
    "DgmsrSemantics",
    "StlJaxSemantics",
    "StlJaxFormulaWrapper",
    "to_stljax_formula",
    "SemanticsRegistry",
    "registry",
    "create_semantics",
]
