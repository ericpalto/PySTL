from . import dgmsr
from .api import Or, And, Not, Until, Always, Formula, Interval, Predicate, Eventually
from .semantics import (
    CtstlSemantics,
    DgmsrSemantics,
    AgmRobustSemantics,
    CumulativeSemantics,
    CumulativeRobustness,
    SmoothRobustSemantics,
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
    "SmoothRobustSemantics",
    "AgmRobustSemantics",
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
        JaxAgmRobustSemantics,
        JaxCumulativeSemantics,
        JaxCumulativeRobustness,
        JaxSmoothRobustSemantics,
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
            "JaxSmoothRobustSemantics",
            "JaxAgmRobustSemantics",
            "JaxCumulativeRobustness",
            "JaxCumulativeSemantics",
            "jax_kth_largest",
            "jax_tau_to_k",
            "JaxCtstlSemantics",
            "JaxDgmsrSemantics",
        ]
    )
