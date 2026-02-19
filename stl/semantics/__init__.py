from __future__ import annotations

from stl.jax import (
    JaxCtstlSemantics,
    JaxDgmsrSemantics,
    JaxRobustSemantics,
    JaxStlJaxSemantics,
    JaxCumulativeSemantics,
    JaxCumulativeRobustness,
    JaxClassicRobustSemantics,
    JaxTraditionalRobustSemantics,
    jax_tau_to_k,
    jax_kth_largest,
)
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
registry.register("jax", JaxRobustSemantics)
registry.register("jax_classic", JaxClassicRobustSemantics)
registry.register("jax_traditional", JaxTraditionalRobustSemantics)
registry.register("jax_cumulative", JaxCumulativeSemantics)
registry.register("jax_ctstl", JaxCtstlSemantics)
registry.register("jax_dgmsr", JaxDgmsrSemantics)
registry.register("jax_stljax", JaxStlJaxSemantics)


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
    "JaxClassicRobustSemantics",
    "JaxTraditionalRobustSemantics",
    "JaxRobustSemantics",
    "JaxCumulativeRobustness",
    "JaxCumulativeSemantics",
    "jax_kth_largest",
    "jax_tau_to_k",
    "JaxCtstlSemantics",
    "JaxDgmsrSemantics",
    "JaxStlJaxSemantics",
    "StlJaxSemantics",
    "StlJaxFormulaWrapper",
    "to_stljax_formula",
    "SemanticsRegistry",
    "registry",
    "create_semantics",
]
