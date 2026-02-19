from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp
import stljax.formula as stljax_formula

from stl import (
    Interval,
    Predicate,
    StlJaxFormulaWrapper,
    create_semantics,
    to_stljax_formula,
)

# pylint: disable=redefined-outer-name


@pytest.fixture
def signal() -> np.ndarray:
    return np.array(
        [
            [0.2, 0.8],
            [0.3, 0.6],
            [0.5, 0.4],
            [0.7, 0.3],
        ],
        dtype=float,
    )


@pytest.fixture
def predicates():
    p1 = Predicate("speed_ok", fn=lambda s, t: 0.6 - s[t, 0])
    p2 = Predicate("alt_ok", fn=lambda s, t: s[t, 1] - 0.2)
    return p1, p2


def test_to_stljax_formula_returns_native_formula(predicates) -> None:
    p1, p2 = predicates
    phi = (p1 & p2).always(Interval(0, 2))
    native = to_stljax_formula(phi)
    assert isinstance(native, stljax_formula.STL_Formula)


def test_wrapper_matches_native_stljax_robustness_trace(
    signal: np.ndarray, predicates
) -> None:
    p1, p2 = predicates
    phi = (p1 | ~p2).eventually((0, 3))
    wrapped = StlJaxFormulaWrapper(phi, approx_method="true")
    native = to_stljax_formula(phi)

    trace_wrapped = wrapped.robustness_trace(signal)
    trace_native = np.asarray(
        native.robustness_trace(
            jnp.asarray(signal),
            approx_method="true",
            temperature=None,
            padding=None,
            large_number=1e9,
        ),
        dtype=float,
    )

    np.testing.assert_allclose(trace_wrapped, trace_native, atol=1e-12, rtol=1e-12)
    assert wrapped.robustness(signal, t=0) == pytest.approx(
        float(trace_wrapped[0]), abs=1e-12
    )


def test_stljax_semantics_matches_classic_for_true_approx(
    signal: np.ndarray, predicates
) -> None:
    p1, p2 = predicates
    sem_stljax = create_semantics("stljax", approx_method="true", temperature=None)
    sem_classic = create_semantics("classic")

    formulas = [
        (p1 & p2),
        (p1 | ~p2),
        p1.always((0, 2)),
        p2.eventually((1, 3)),
        p1.until(p2, interval=(0, 3)),
    ]

    for phi in formulas:
        for t in [0, 1]:
            v_stljax = phi.evaluate(signal, sem_stljax, t=t)
            v_classic = phi.evaluate(signal, sem_classic, t=t)
            assert v_stljax == pytest.approx(v_classic, abs=1e-6)


def test_stljax_rejects_weighted_operators(signal: np.ndarray, predicates) -> None:
    p1, p2 = predicates
    sem_stljax = create_semantics("stljax")

    phi_and_weighted = type(p1 & p2)(p1, p2, weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        phi_and_weighted.evaluate(signal, sem_stljax, t=0)
    with pytest.raises(ValueError):
        to_stljax_formula(phi_and_weighted)

    phi_until_weighted = p1.until(p2, interval=(0, 3), weights_pair=(2.0, 1.0))
    with pytest.raises(ValueError):
        phi_until_weighted.evaluate(signal, sem_stljax, t=0)
    with pytest.raises(ValueError):
        to_stljax_formula(phi_until_weighted)
