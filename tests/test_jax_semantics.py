from __future__ import annotations

import jax
import numpy as np
import pytest
import jax.numpy as jnp

from stl import And, Interval, Predicate, create_semantics

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
def signal_jax(signal_np: np.ndarray) -> jax.Array:
    return jnp.asarray(signal_np)


@pytest.fixture
def predicates():
    p1 = Predicate("speed_ok", fn=lambda s, t: 0.6 - s[t, 0])
    p2 = Predicate("alt_ok", fn=lambda s, t: s[t, 1] - 0.2)
    return p1, p2


def test_jax_classic_backends_match_numpy_backend(
    signal_np: np.ndarray, signal_jax: jax.Array, predicates
) -> None:
    p1, p2 = predicates
    formulas = [
        (p1 & p2),
        (p1 | ~p2),
        p1.always((0, 2)),
        p2.eventually((1, 3)),
        p1.until(p2, interval=(0, 3)),
    ]

    for jax_name, np_name in [("jax", "classic"), ("jax_classic", "classic")]:
        sem_jax = (
            create_semantics(jax_name, smooth=False)
            if jax_name == "jax"
            else create_semantics(jax_name)
        )
        sem_np = create_semantics(np_name)

        for phi in formulas:
            for t in [0, 1]:
                v_jax = float(phi.evaluate(signal_jax, sem_jax, t=t))
                v_np = float(phi.evaluate(signal_np, sem_np, t=t))
                assert v_jax == pytest.approx(v_np, abs=1e-6)


def test_jax_cumulative_matches_numpy_backend(
    signal_np: np.ndarray, signal_jax: jax.Array, predicates
) -> None:
    p1, p2 = predicates
    formulas = [
        p1.eventually((1, 3)),
        p1.until(p2, interval=(0, 3)),
    ]

    sem_jax = create_semantics("jax_cumulative")
    sem_np = create_semantics("cumulative")

    for phi in formulas:
        val_jax = phi.evaluate(signal_jax, sem_jax, t=0)
        val_np = phi.evaluate(signal_np, sem_np, t=0)
        assert float(val_jax.pos) == pytest.approx(val_np.pos, abs=1e-6)
        assert float(val_jax.neg) == pytest.approx(val_np.neg, abs=1e-6)


def test_jax_ctstl_matches_numpy_backend(
    signal_np: np.ndarray, signal_jax: jax.Array, predicates
) -> None:
    p1, p2 = predicates
    phi = p1.until(p2, interval=(0, 3))

    sem_jax = create_semantics("jax_ctstl", delta=1.0)
    sem_np = create_semantics("ctstl", delta=1.0)

    rho_jax = float(phi.evaluate(signal_jax, sem_jax, t=0))
    rho_np = float(phi.evaluate(signal_np, sem_np, t=0))
    assert rho_jax == pytest.approx(rho_np, abs=1e-6)

    vals_jax = [p1.evaluate(signal_jax, sem_jax, t=t) for t in [0, 1, 2, 3]]
    vals_np = [p1.evaluate(signal_np, sem_np, t=t) for t in [0, 1, 2, 3]]

    rho_c_jax = float(sem_jax.temporal_cumulative(vals_jax, tau=2.0))
    rho_c_np = float(sem_np.temporal_cumulative(vals_np, tau=2.0))
    assert rho_c_jax == pytest.approx(rho_c_np, abs=1e-6)


def test_jax_dgmsr_matches_numpy_backend_with_weights(
    signal_np: np.ndarray, signal_jax: jax.Array, predicates
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

    sem_jax = create_semantics("jax_dgmsr", eps=1e-8, p=1)
    sem_np = create_semantics("dgmsr", eps=1e-8, p=1)

    v_and_jax = float(phi_and.evaluate(signal_jax, sem_jax, t=0))
    v_and_np = float(phi_and.evaluate(signal_np, sem_np, t=0))
    assert v_and_jax == pytest.approx(v_and_np, abs=1e-6)

    v_until_jax = float(phi_until.evaluate(signal_jax, sem_jax, t=0))
    v_until_np = float(phi_until.evaluate(signal_np, sem_np, t=0))
    assert v_until_jax == pytest.approx(v_until_np, abs=1e-6)


def test_jax_stljax_matches_numpy_stljax_backend(
    signal_np: np.ndarray, signal_jax: jax.Array, predicates
) -> None:
    p1, p2 = predicates
    formulas = [
        (p1 | ~p2).eventually((0, 3)),
        p1.until(p2, interval=(0, 3)),
    ]

    sem_jax = create_semantics("jax_stljax", approx_method="true", temperature=None)
    sem_np = create_semantics("stljax", approx_method="true", temperature=None)

    for phi in formulas:
        v_jax = float(phi.evaluate(signal_jax, sem_jax, t=0))
        v_np = float(phi.evaluate(signal_np, sem_np, t=0))
        assert v_jax == pytest.approx(v_np, abs=1e-6)


def test_jax_backends_support_autograd(signal_jax: jax.Array, predicates) -> None:
    p1, p2 = predicates

    checks = [
        (
            create_semantics("jax", smooth=True, temperature=0.2),
            (p1 & p2).always((0, 2)),
            lambda v: v,
        ),
        (
            create_semantics("jax_cumulative"),
            p1.eventually((0, 2)),
            lambda v: v.pos,
        ),
        (
            create_semantics("jax_ctstl", delta=1.0),
            p1.until(p2, interval=(0, 3)),
            lambda v: v,
        ),
        (
            create_semantics("jax_dgmsr", eps=1e-8, p=1),
            p1.until(p2, interval=(0, 3), weights_pair=(1.0, 1.2)),
            lambda v: v,
        ),
        (
            create_semantics("jax_stljax", approx_method="true", temperature=None),
            p1.eventually((0, 2)),
            lambda v: v,
        ),
    ]

    for sem, phi, projector in checks:

        def objective(
            sig: jax.Array, sem=sem, phi=phi, projector=projector
        ) -> jax.Array:
            out = phi.evaluate(sig, sem, t=0)
            return jnp.asarray(projector(out), dtype=float)

        grad = np.asarray(jax.grad(objective)(signal_jax))
        assert grad.shape == signal_jax.shape
        assert np.all(np.isfinite(grad))


def test_jax_semantics_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError):
        create_semantics("jax", smooth=True, temperature=0.0)
