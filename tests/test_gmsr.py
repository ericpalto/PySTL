from __future__ import annotations

import numpy as np
import pytest

from stl.dgmsr import gmsr_or, gmsr_and, gmsr_until

# pylint: disable=redefined-outer-name


def _finite_difference_grad(fn, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        fp = float(fn(xp))
        fm = float(fn(xm))
        grad[i] = (fp - fm) / (2.0 * h)
    return grad


@pytest.fixture
def gmsr_params():
    return 1e-8, 2


def test_or_and_duality(gmsr_params) -> None:
    eps, p = gmsr_params
    x = np.array([-0.7, -0.3, 0.4, 0.9], dtype=float)
    w = np.array([1.0, 2.0, 1.0, 3.0], dtype=float)

    h_or, g_or = gmsr_or(eps, p, w, x)
    h_and_neg, g_and_neg = gmsr_and(eps, p, w, -x)

    assert h_or == pytest.approx(-h_and_neg, abs=1e-12)
    np.testing.assert_allclose(g_or, g_and_neg, atol=1e-9, rtol=1e-9)


def test_gmsr_and_gradient_matches_finite_difference(gmsr_params) -> None:
    eps, p = gmsr_params
    x = np.array([-0.8, -0.35, 0.45, 1.1], dtype=float)
    w = np.array([1.0, 3.0, 2.0, 1.0], dtype=float)

    h, g = gmsr_and(eps, p, w, x)

    def _fn(v: np.ndarray) -> float:
        out, _ = gmsr_and(eps, p, w, v)
        return out

    g_fd = _finite_difference_grad(_fn, x)

    assert np.isfinite(h)
    np.testing.assert_allclose(g, g_fd, atol=5e-5, rtol=5e-5)


def test_gmsr_or_gradient_matches_finite_difference(gmsr_params) -> None:
    eps, p = gmsr_params
    x = np.array([-1.2, -0.5, 0.2, 0.75], dtype=float)
    w = np.array([2.0, 1.0, 1.0, 4.0], dtype=float)

    h, g = gmsr_or(eps, p, w, x)

    def _fn(v: np.ndarray) -> float:
        out, _ = gmsr_or(eps, p, w, v)
        return out

    g_fd = _finite_difference_grad(_fn, x)

    assert np.isfinite(h)
    np.testing.assert_allclose(g, g_fd, atol=5e-5, rtol=5e-5)


def test_gmsr_until_gradients_match_finite_difference(gmsr_params) -> None:
    eps, p = gmsr_params
    f = np.array([0.6, 0.45, 0.3, -0.2], dtype=float)
    g = np.array([-0.25, 0.1, 0.35, 0.5], dtype=float)
    w_f = np.array([1.0, 2.0, 1.0, 1.0], dtype=float)
    w_g = np.array([1.0, 1.0, 2.0, 1.0], dtype=float)
    w_pair = np.array([1.0, 1.0], dtype=float)

    z, dz_df, dz_dg = gmsr_until(eps, p, w_f, w_g, w_pair, f, g)

    x = np.concatenate([f, g], axis=0)

    def _fn(v: np.ndarray) -> float:
        f_v = v[: f.size]
        g_v = v[f.size :]
        out, _, _ = gmsr_until(eps, p, w_f, w_g, w_pair, f_v, g_v)
        return out

    g_fd = _finite_difference_grad(_fn, x)

    assert np.isfinite(z)
    np.testing.assert_allclose(dz_df, g_fd[: f.size], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(dz_dg, g_fd[f.size :], atol=1e-4, rtol=1e-4)
