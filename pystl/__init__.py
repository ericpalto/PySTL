from .api import (
    Or,
    And,
    Not,
    Until,
    Always,
    Formula,
    Interval,
    Predicate,
    Eventually,
    FormulaFormat,
    export_formula,
)
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
    "FormulaFormat",
    "Predicate",
    "Not",
    "And",
    "Or",
    "Always",
    "Eventually",
    "Until",
    "Interval",
    "export_formula",
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

try:
    from .semantics import (
        TorchDgmsrSemantics,
        TorchAgmRobustSemantics,
        TorchCumulativeSemantics,
        TorchCumulativeRobustness,
        TorchSmoothRobustSemantics,
        TorchClassicRobustSemantics,
    )
except ImportError:
    pass
else:
    __all__.extend(
        [
            "TorchClassicRobustSemantics",
            "TorchSmoothRobustSemantics",
            "TorchAgmRobustSemantics",
            "TorchCumulativeRobustness",
            "TorchCumulativeSemantics",
            "TorchDgmsrSemantics",
        ]
    )
