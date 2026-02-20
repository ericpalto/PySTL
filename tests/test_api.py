from __future__ import annotations

import numpy as np
import pytest

from stl import Or, And, Interval, Predicate, create_semantics
from stl.semantics import registry

# pylint: disable=redefined-outer-name


def _weights(weights, length: int) -> np.ndarray:
    if weights is None:
        return np.ones(length, dtype=float)
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.size < length:
        raise ValueError("weights too short for test expectation helper.")
    return arr[:length]


def _agm_and(values, weights=None) -> float:
    vals = np.asarray(values, dtype=float).reshape(-1)
    w = _weights(weights, vals.size)
    if np.any(vals <= 0.0):
        return float(np.sum(w * np.minimum(vals, 0.0)) / np.sum(w))
    norm = w / np.sum(w)
    return float(np.exp(np.sum(norm * np.log(1.0 + vals))) - 1.0)


def _agm_or(values, weights=None) -> float:
    vals = np.asarray(values, dtype=float).reshape(-1)
    w = _weights(weights, vals.size)
    if np.any(vals > 0.0):
        return float(np.sum(w * np.maximum(vals, 0.0)) / np.sum(w))
    norm = w / np.sum(w)
    return float(1.0 - np.exp(np.sum(norm * np.log(1.0 - vals))))


def _agm_until_expected(
    *,
    left_trace: np.ndarray,
    right_trace: np.ndarray,
    start: int,
    end: int | None,
    weights_left=None,
    weights_right=None,
    weights_pair=(1.0, 1.0),
) -> float:
    last = left_trace.size - 1 if end is None else min(end, left_trace.size - 1)
    candidates = []
    for idx in range(start, last + 1):
        prefix = _agm_and(left_trace[: idx + 1], weights=weights_left)
        pair = _agm_and([prefix, right_trace[idx]], weights=weights_pair)
        candidates.append(pair)
    return _agm_or(candidates, weights=weights_right)


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
    expected_by_backend = {
        "numpy": {
            "agm/numpy",
            "classical/numpy",
            "smooth/numpy",
            "cumulative/numpy",
            "dgmsr/numpy",
        },
        "jax": {
            "classical/jax",
            "smooth/jax",
            "cumulative/jax",
            "dgmsr/jax",
            "agm/jax",
        },
        "torch": {
            "classical/torch",
            "smooth/torch",
            "cumulative/torch",
            "dgmsr/torch",
            "agm/torch",
        },
    }
    expected_numpy = {
        "agm/numpy",
        "classical/numpy",
        "smooth/numpy",
        "cumulative/numpy",
        "dgmsr/numpy",
    }
    assert expected_numpy.issubset(names)
    backends = registry.backends()
    assert set(backends).issubset({"numpy", "jax", "torch"})
    assert "numpy" in backends

    expected_names: set[str] = set()
    for backend in backends:
        expected_names |= expected_by_backend[backend]
    assert names == expected_names

    assert registry.syntaxes() == ["agm", "classical", "cumulative", "dgmsr", "smooth"]
    assert registry.backends() in (
        ["jax", "numpy", "torch"],
        ["jax", "numpy"],
        ["numpy", "torch"],
        ["numpy"],
    )


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
        create_semantics("classical", backend="torch_backend")


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


def test_agm_syntax_aliases() -> None:
    assert create_semantics("agm").__class__.__name__ == "AgmRobustSemantics"
    assert (
        create_semantics("arithmetic-geometric-mean").__class__.__name__
        == "AgmRobustSemantics"
    )


def test_agm_boolean_branches(signal: np.ndarray, predicates) -> None:
    p1, p2 = predicates
    pneg = Predicate("pneg", fn=lambda _s, _t: -0.2)
    sem = create_semantics("agm")

    and_pos = float((p1 & p2).evaluate(signal, sem, t=0))
    and_neg = float((p1 & p2).evaluate(signal, sem, t=3))
    or_pos = float((p1 | p2).evaluate(signal, sem, t=3))
    or_neg = float((pneg | ~p2).evaluate(signal, sem, t=0))

    assert and_pos == pytest.approx(_agm_and([0.4, 0.6]), abs=1e-12)
    assert and_neg == pytest.approx(_agm_and([-0.1, 0.1]), abs=1e-12)
    assert or_pos == pytest.approx(_agm_or([-0.1, 0.1]), abs=1e-12)
    assert or_neg == pytest.approx(_agm_or([-0.2, -0.6]), abs=1e-12)


def test_agm_weighted_boolean_operators(signal: np.ndarray, predicates) -> None:
    p1, p2 = predicates
    sem = create_semantics("agm")

    phi_and_weighted = And(p1, p2, weights=[1.0, 3.0])
    phi_or_weighted = Or(p1, p2, weights=[1.0, 3.0])

    and_weighted = float(phi_and_weighted.evaluate(signal, sem, t=0))
    or_weighted = float(phi_or_weighted.evaluate(signal, sem, t=3))

    expected_and_weighted = _agm_and([0.4, 0.6], weights=[1.0, 3.0])
    expected_or_weighted = _agm_or([-0.1, 0.1], weights=[1.0, 3.0])
    assert and_weighted == pytest.approx(expected_and_weighted, abs=1e-12)
    assert or_weighted == pytest.approx(expected_or_weighted, abs=1e-12)


def test_agm_temporal_branches(signal: np.ndarray, predicates) -> None:
    p1, p2 = predicates
    sem = create_semantics("agm")

    always_pos = float(p1.always((0, 2)).evaluate(signal, sem, t=0))
    eventually_pos = float(p1.eventually((0, 3)).evaluate(signal, sem, t=0))
    eventually_neg = float((~p2).eventually((0, 3)).evaluate(signal, sem, t=0))

    assert always_pos == pytest.approx(_agm_and([0.4, 0.3, 0.1]), abs=1e-12)
    assert eventually_pos == pytest.approx(_agm_or([0.4, 0.3, 0.1, -0.1]), abs=1e-12)
    assert eventually_neg == pytest.approx(_agm_or([-0.6, -0.4, -0.2, -0.1]), abs=1e-12)


def test_agm_until_with_weights(signal: np.ndarray, predicates) -> None:
    p1, p2 = predicates
    sem = create_semantics("agm")

    phi = p1.until(
        p2,
        interval=(0, 3),
        weights_left=[1.0, 2.0, 1.5, 1.0],
        weights_right=[1.0, 1.2, 0.8, 1.4],
        weights_pair=(1.0, 1.3),
    )

    value = float(phi.evaluate(signal, sem, t=0))

    left_trace = np.asarray([0.6 - signal[t, 0] for t in range(signal.shape[0])])
    right_trace = np.asarray([signal[t, 1] - 0.2 for t in range(signal.shape[0])])
    expected = _agm_until_expected(
        left_trace=left_trace,
        right_trace=right_trace,
        start=0,
        end=3,
        weights_left=[1.0, 2.0, 1.5, 1.0],
        weights_right=[1.0, 1.2, 0.8, 1.4],
        weights_pair=(1.0, 1.3),
    )
    assert value == pytest.approx(expected, abs=1e-12)
