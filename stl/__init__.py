from . import dgmsr
from .api import Or, And, Not, Until, Always, Formula, Interval, Predicate, Eventually
from .semantics import (
    CtstlSemantics,
    DgmsrSemantics,
    CumulativeSemantics,
    CumulativeRobustness,
    ClassicRobustSemantics,
    registry,
    tau_to_k,
    kth_largest,
    create_semantics,
)

__all__ = [
    "Formula",
    "Predicate",
    "Not",
    "And",
    "Or",
    "Always",
    "Eventually",
    "Until",
    "Interval",
    "ClassicRobustSemantics",
    "CumulativeSemantics",
    "CumulativeRobustness",
    "CtstlSemantics",
    "kth_largest",
    "tau_to_k",
    "DgmsrSemantics",
    "create_semantics",
    "registry",
    "dgmsr",
]

try:
    from .semantics import (
        JaxCtstlSemantics,
        JaxDgmsrSemantics,
        JaxCumulativeSemantics,
        JaxCumulativeRobustness,
        JaxClassicRobustSemantics,
        jax_tau_to_k,
        jax_kth_largest,
    )
except ImportError:
    pass
else:
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
