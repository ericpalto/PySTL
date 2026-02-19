from __future__ import annotations

import numpy as np
import pytest

from stl import Interval, Predicate, tau_to_k, create_semantics
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
    assert "classic" in names
    assert "traditional" in names
    assert "cumulative" in names
    assert "ctstl" in names
    assert "dgmsr" in names
    assert "stljax" in names
    assert "jax" in names
    assert "jax_classic" in names
    assert "jax_traditional" in names
    assert "jax_cumulative" in names
    assert "jax_ctstl" in names
    assert "jax_dgmsr" in names
    assert "jax_stljax" in names


def test_boolean_and_temporal_always_with_classic_semantics(
    signal: np.ndarray, predicates
) -> None:
    p1, p2 = predicates
    phi = (p1 & p2).always(Interval(0, 2))
    sem = create_semantics("classic")
    rho = phi.evaluate(signal, sem, t=0)

    vals = [min(0.6 - signal[t, 0], signal[t, 1] - 0.2) for t in [0, 1, 2]]
    expected = min(vals)
    assert rho == pytest.approx(expected, abs=1e-12)


def test_eventually_with_classic_semantics(signal: np.ndarray, predicates) -> None:
    p1, _ = predicates
    phi = p1.eventually((1, 3))
    sem = create_semantics("classic")
    rho = phi.evaluate(signal, sem, t=0)
    expected = max(0.6 - signal[t, 0] for t in [1, 2, 3])
    assert rho == pytest.approx(expected, abs=1e-12)


def test_until_with_classic_semantics(signal: np.ndarray, predicates) -> None:
    p1, p2 = predicates
    phi = p1.until(p2, interval=(0, 3))
    sem = create_semantics("classic")
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


def test_interval_validation() -> None:
    with pytest.raises(ValueError):
        Interval(3, 2)


def test_empty_window_raises(signal: np.ndarray, predicates) -> None:
    p1, _ = predicates
    phi = p1.always((2, 3))
    sem = create_semantics("classic")
    with pytest.raises(ValueError):
        phi.evaluate(signal[:2], sem, t=1)


def test_traditional_matches_classic(signal: np.ndarray, predicates) -> None:
    p1, p2 = predicates
    formulas = [
        (p1 & p2),
        (p1 | ~p2),
        p1.always((0, 2)),
        p2.eventually((1, 3)),
        p1.until(p2, interval=(0, 3)),
    ]

    sem_classic = create_semantics("classic")
    sem_traditional = create_semantics("traditional")

    for phi in formulas:
        for t in [0, 1]:
            v_classic = phi.evaluate(signal, sem_classic, t=t)
            v_traditional = phi.evaluate(signal, sem_traditional, t=t)
            assert v_traditional == pytest.approx(v_classic, abs=1e-12)


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


def test_ctstl_until_uses_exclusive_left_prefix(signal: np.ndarray, predicates) -> None:
    p1, p2 = predicates
    phi = p1.until(p2, interval=(0, 3))

    sem_ctstl = create_semantics("ctstl")
    sem_classic = create_semantics("classic")
    rho_ctstl = phi.evaluate(signal, sem_ctstl, t=0)
    rho_classic = phi.evaluate(signal, sem_classic, t=0)

    p1_trace = np.array(
        [0.6 - signal[t, 0] for t in range(signal.shape[0])], dtype=float
    )
    p2_trace = np.array(
        [signal[t, 1] - 0.2 for t in range(signal.shape[0])], dtype=float
    )

    expected = -np.inf
    for i in range(0, 4):
        prefix = np.inf if i == 0 else float(np.min(p1_trace[:i]))
        expected = max(expected, min(float(p2_trace[i]), float(prefix)))

    assert rho_ctstl == pytest.approx(expected, abs=1e-12)
    assert rho_ctstl != pytest.approx(rho_classic, abs=1e-12)


def test_ctstl_temporal_cumulative(signal: np.ndarray, predicates) -> None:
    p1, _ = predicates
    sem = create_semantics("ctstl", delta=1.0)
    values = [p1.evaluate(signal, sem, t=t) for t in [0, 1, 2, 3]]

    rho_tau_2 = sem.temporal_cumulative(values, tau=2.0)
    rho_tau_5 = sem.temporal_cumulative(values, tau=5.0)

    # values are [0.4, 0.3, 0.1, -0.1], so 2nd largest is 0.3
    assert rho_tau_2 == pytest.approx(0.3, abs=1e-12)
    assert rho_tau_5 == float("-inf")
    assert tau_to_k(2.0, 1.0) == 2
