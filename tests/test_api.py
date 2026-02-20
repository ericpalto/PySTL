from __future__ import annotations

import numpy as np
import pytest

from stl import Interval, Predicate, create_semantics
from stl.semantics import registry

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


def test_registry_contains_expected_semantics() -> None:
    names = set(registry.names())
    expected_numpy = {
        "classical/numpy",
        "smooth/numpy",
        "cumulative/numpy",
        "dgmsr/numpy",
    }
    assert expected_numpy.issubset(names)
    if "jax" in registry.backends():
        assert names == expected_numpy | {
            "classical/jax",
            "smooth/jax",
            "cumulative/jax",
            "dgmsr/jax",
        }
    else:
        assert names == expected_numpy
    assert registry.syntaxes() == ["classical", "cumulative", "dgmsr", "smooth"]
    assert registry.backends() in (["jax", "numpy"], ["numpy"])


def test_boolean_and_temporal_always_with_classic_semantics(
    signal: np.ndarray, predicates
) -> None:
    p1, p2 = predicates
    phi = (p1 & p2).always(Interval(0, 2))
    sem = create_semantics("classical")
    rho = phi.evaluate(signal, sem, t=0)

    vals = [min(0.6 - signal[t, 0], signal[t, 1] - 0.2) for t in [0, 1, 2]]
    expected = min(vals)
    assert rho == pytest.approx(expected, abs=1e-12)


def test_eventually_with_classic_semantics(signal: np.ndarray, predicates) -> None:
    p1, _ = predicates
    phi = p1.eventually((1, 3))
    sem = create_semantics("classical")
    rho = phi.evaluate(signal, sem, t=0)
    expected = max(0.6 - signal[t, 0] for t in [1, 2, 3])
    assert rho == pytest.approx(expected, abs=1e-12)


def test_until_with_classic_semantics(signal: np.ndarray, predicates) -> None:
    p1, p2 = predicates
    phi = p1.until(p2, interval=(0, 3))
    sem = create_semantics("classical")
    rho = phi.evaluate(signal, sem, t=0)

    p1_trace = np.array(
        [0.6 - signal[t, 0] for t in range(signal.shape[0])], dtype=float
    )
    p2_trace = np.array(
        [signal[t, 1] - 0.2 for t in range(signal.shape[0])], dtype=float
    )
    best = -np.inf
    for i in range(0, 4):
        candidate = min(float(np.min(p1_trace[: i + 1])), float(p2_trace[i]))
        best = max(best, candidate)
    assert rho == pytest.approx(best, abs=1e-12)


def test_smooth_and_or_are_soft_versions(signal: np.ndarray, predicates) -> None:
    p1, p2 = predicates
    sem = create_semantics("smooth", temperature=0.3)

    and_val = float((p1 & p2).evaluate(signal, sem, t=0))
    or_val = float((p1 | p2).evaluate(signal, sem, t=0))

    vals = np.asarray([0.6 - signal[0, 0], signal[0, 1] - 0.2], dtype=float)
    tau = 0.3
    expected_and = float(-tau * np.log(np.sum(np.exp(-vals / tau))))
    expected_or = float(tau * np.log(np.sum(np.exp(vals / tau))))

    assert and_val == pytest.approx(expected_and, abs=1e-12)
    assert or_val == pytest.approx(expected_or, abs=1e-12)

    classical = create_semantics("classical")
    classical_and = float((p1 & p2).evaluate(signal, classical, t=0))
    classical_or = float((p1 | p2).evaluate(signal, classical, t=0))
    assert and_val <= classical_and
    assert or_val >= classical_or


def test_interval_validation() -> None:
    with pytest.raises(ValueError):
        Interval(3, 2)


def test_unknown_syntax_or_backend_raises() -> None:
    with pytest.raises(KeyError):
        create_semantics("stljax")
    with pytest.raises(KeyError):
        create_semantics("classic")
    with pytest.raises(KeyError):
        create_semantics("traditional")
    with pytest.raises(KeyError):
        create_semantics("classical", backend="torch")


def test_empty_window_raises(signal: np.ndarray, predicates) -> None:
    p1, _ = predicates
    phi = p1.always((2, 3))
    sem = create_semantics("classical")
    with pytest.raises(ValueError):
        phi.evaluate(signal[:2], sem, t=1)


def test_cumulative_semantics_until(signal: np.ndarray, predicates) -> None:
    p1, p2 = predicates
    phi = p1.until(p2, interval=(0, 3))
    sem = create_semantics("cumulative")
    value = phi.evaluate(signal, sem, t=0)

    p1_vals = [0.6 - signal[t, 0] for t in range(signal.shape[0])]
    p2_vals = [signal[t, 1] - 0.2 for t in range(signal.shape[0])]
    p1_pos = [max(0.0, v) for v in p1_vals]
    p1_neg = [min(0.0, v) for v in p1_vals]
    p2_pos = [max(0.0, v) for v in p2_vals]
    p2_neg = [min(0.0, v) for v in p2_vals]

    expected_pos = 0.0
    expected_neg = 0.0
    for i in range(0, 4):
        prefix_min_pos = min(p1_pos[: i + 1])
        prefix_min_neg = min(p1_neg[: i + 1])
        expected_pos += min(p2_pos[i], prefix_min_pos)
        expected_neg += min(p2_neg[i], prefix_min_neg)

    assert value.pos == pytest.approx(expected_pos, abs=1e-12)
    assert value.neg == pytest.approx(expected_neg, abs=1e-12)


def test_cumulative_eventually_is_sum(signal: np.ndarray, predicates) -> None:
    p1, _ = predicates
    phi = p1.eventually((1, 3))
    sem = create_semantics("cumulative")
    value = phi.evaluate(signal, sem, t=0)

    vals = [0.6 - signal[t, 0] for t in [1, 2, 3]]
    expected_pos = sum(max(0.0, v) for v in vals)
    expected_neg = sum(min(0.0, v) for v in vals)

    assert value.pos == pytest.approx(expected_pos, abs=1e-12)
    assert value.neg == pytest.approx(expected_neg, abs=1e-12)


def test_ctstl_syntax_is_disabled() -> None:
    with pytest.raises(KeyError):
        create_semantics("ctstl")
