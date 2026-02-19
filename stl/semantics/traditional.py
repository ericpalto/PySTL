"""Traditional max-min robustness semantics."""

from __future__ import annotations

from stl.semantics.classic import ClassicRobustSemantics


class TraditionalRobustSemantics(ClassicRobustSemantics):
    """Traditional STL robustness (rho) using hard min/max operators."""
