from __future__ import annotations

import importlib

import numpy as np
import pytest

from pystl import Interval, Predicate, create_semantics


def _finite_difference_grad(phi, signal: np.ndarray, sem, *, t: int, h: float = 1e-6):
    x = np.asarray(signal, dtype=float)
    grad = np.zeros_like(x, dtype=float)
    for tau in range(x.shape[0]):
        for i in range(x.shape[1]):
            xp = x.copy()
            xm = x.copy()
            xp[tau, i] += h
            xm[tau, i] -= h
            fp = float(phi.evaluate(xp, sem, t=t))
            fm = float(phi.evaluate(xm, sem, t=t))
            grad[tau, i] = (fp - fm) / (2.0 * h)
    return grad


def test_dgmsr_numpy_evaluate_with_grad_matches_fd() -> None:
    signal = np.array(
        [
            [0.2, 0.8],
            [0.3, 0.6],
            [0.5, 0.4],
        ],
        dtype=float,
    )

    p_a = Predicate(
        "a",
        fn=lambda s, t: float(s[t, 0] - 0.25),
        grad=lambda _s, _t: np.array([1.0, 0.0], dtype=float),
    )
    p_b = Predicate(
        "b",
        fn=lambda s, t: float(0.55 - s[t, 1]),
        grad=lambda _s, _t: np.array([0.0, -1.0], dtype=float),
    )

    phi = (p_a & p_b).always(Interval(0, 2)).eventually(Interval(0, 2))
    sem = create_semantics("dgmsr", backend="numpy", eps=1e-8, p=2)

    rho, grad = phi.evaluate_with_grad(signal, sem, t=0)
    grad_fd = _finite_difference_grad(phi, signal, sem, t=0, h=1e-6)

    assert np.isfinite(rho)
    assert grad.shape == signal.shape
    np.testing.assert_allclose(grad, grad_fd, atol=5e-4, rtol=5e-4)


def test_evaluate_with_grad_requires_supported_semantics() -> None:
    signal = np.array([[0.0, 0.0]], dtype=float)
    p = Predicate("p", fn=lambda s, t: float(s[t, 0] - 1.0))
    phi = p

    sem = create_semantics("classical", backend="numpy")
    with pytest.raises(NotImplementedError):
        _ = phi.evaluate_with_grad(signal, sem, t=0)


def test_dgmsr_module_is_not_public() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pystl.dgmsr")
