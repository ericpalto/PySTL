from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except ImportError as exc:
    pytest.skip(f"optional dependencies unavailable: {exc}", allow_module_level=True)

from pystl import And, Interval, Predicate, create_semantics

# pylint: disable=redefined-outer-name


@pytest.fixture
def signal_np() -> np.ndarray:
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
def signal_torch(signal_np: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(signal_np, dtype=torch.float64)


@pytest.fixture
def predicates():
    p1 = Predicate("speed_ok", fn=lambda s, t: 0.6 - s[t, 0])
    p2 = Predicate("alt_ok", fn=lambda s, t: s[t, 1] - 0.2)
    return p1, p2


def test_torch_classical_matches_numpy_backend(
    signal_np: np.ndarray, signal_torch: torch.Tensor, predicates
) -> None:
    p1, p2 = predicates
    formulas = [
        (p1 & p2),
        (p1 | ~p2),
        p1.always((0, 2)),
        p2.eventually((1, 3)),
        p1.until(p2, interval=(0, 3)),
    ]

    sem_torch = create_semantics("classical", backend="torch")
    sem_np = create_semantics("classical", backend="numpy")

    for phi in formulas:
        for t in [0, 1]:
            v_torch = float(phi.evaluate(signal_torch, sem_torch, t=t))
            v_np = float(phi.evaluate(signal_np, sem_np, t=t))
            assert v_torch == pytest.approx(v_np, abs=1e-6)


def test_torch_cumulative_matches_numpy_backend(
    signal_np: np.ndarray, signal_torch: torch.Tensor, predicates
) -> None:
    p1, p2 = predicates
    formulas = [
        p1.eventually((1, 3)),
        p1.until(p2, interval=(0, 3)),
    ]

    sem_torch = create_semantics("cumulative", backend="torch")
    sem_np = create_semantics("cumulative", backend="numpy")

    for phi in formulas:
        val_torch = phi.evaluate(signal_torch, sem_torch, t=0)
        val_np = phi.evaluate(signal_np, sem_np, t=0)
        assert float(val_torch.pos) == pytest.approx(val_np.pos, abs=1e-6)
        assert float(val_torch.neg) == pytest.approx(val_np.neg, abs=1e-6)


def test_torch_smooth_matches_numpy_backend(
    signal_np: np.ndarray, signal_torch: torch.Tensor, predicates
) -> None:
    p1, p2 = predicates
    formulas = [
        (p1 & p2),
        (p1 | ~p2),
        p1.always((0, 2)),
        p2.eventually((1, 3)),
        p1.until(p2, interval=(0, 3)),
    ]

    sem_torch = create_semantics("smooth", backend="torch", temperature=0.25)
    sem_np = create_semantics("smooth", backend="numpy", temperature=0.25)

    for phi in formulas:
        for t in [0, 1]:
            v_torch = float(phi.evaluate(signal_torch, sem_torch, t=t))
            v_np = float(phi.evaluate(signal_np, sem_np, t=t))
            assert v_torch == pytest.approx(v_np, abs=1e-6)


def test_torch_dgmsr_matches_numpy_backend_with_weights(
    signal_np: np.ndarray, signal_torch: torch.Tensor, predicates
) -> None:
    p1, p2 = predicates
    phi_and = And(p1, p2, weights=[1.0, 2.0])
    phi_until = p1.until(
        p2,
        interval=Interval(0, 3),
        weights_left=[1.0, 1.5, 1.1, 0.9],
        weights_right=[1.0, 0.8, 1.2, 1.4],
        weights_pair=(1.0, 1.3),
    )

    sem_torch = create_semantics("dgmsr", backend="torch", eps=1e-8, p=1)
    sem_np = create_semantics("dgmsr", backend="numpy", eps=1e-8, p=1)

    v_and_torch = float(phi_and.evaluate(signal_torch, sem_torch, t=0))
    v_and_np = float(phi_and.evaluate(signal_np, sem_np, t=0))
    assert v_and_torch == pytest.approx(v_and_np, abs=1e-6)

    v_until_torch = float(phi_until.evaluate(signal_torch, sem_torch, t=0))
    v_until_np = float(phi_until.evaluate(signal_np, sem_np, t=0))
    assert v_until_torch == pytest.approx(v_until_np, abs=1e-6)


def test_torch_agm_matches_numpy_backend_with_weights(
    signal_np: np.ndarray, signal_torch: torch.Tensor, predicates
) -> None:
    p1, p2 = predicates
    phi_and = And(p1, p2, weights=[1.0, 2.0])
    phi_until = p1.until(
        p2,
        interval=Interval(0, 3),
        weights_left=[1.0, 1.5, 1.1, 0.9],
        weights_right=[1.0, 0.8, 1.2, 1.4],
        weights_pair=(1.0, 1.3),
    )

    sem_torch = create_semantics("agm", backend="torch")
    sem_np = create_semantics("agm", backend="numpy")

    v_and_torch = float(phi_and.evaluate(signal_torch, sem_torch, t=0))
    v_and_np = float(phi_and.evaluate(signal_np, sem_np, t=0))
    assert v_and_torch == pytest.approx(v_and_np, abs=1e-6)

    v_until_torch = float(phi_until.evaluate(signal_torch, sem_torch, t=0))
    v_until_np = float(phi_until.evaluate(signal_np, sem_np, t=0))
    assert v_until_torch == pytest.approx(v_until_np, abs=1e-6)


def test_torch_smooth_backend_supports_autograd(
    signal_torch: torch.Tensor, predicates
) -> None:
    p1, p2 = predicates
    signal = signal_torch.clone().requires_grad_(True)

    sem = create_semantics("smooth", backend="torch", temperature=0.25)
    phi = (p1 & p2).always((0, 2))
    out = phi.evaluate(signal, sem, t=0)
    out.backward()

    assert signal.grad is not None
    assert signal.grad.shape == signal.shape
    assert torch.all(torch.isfinite(signal.grad))
