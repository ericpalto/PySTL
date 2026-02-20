from __future__ import annotations

import numpy as np
import pytest
from stl import And, Interval, Predicate, create_semantics

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

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


def test_jax_classical_matches_numpy_backend(
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

    sem_jax = create_semantics("classical", backend="jax")
    sem_np = create_semantics("classical", backend="numpy")

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

    sem_jax = create_semantics("cumulative", backend="jax")
    sem_np = create_semantics("cumulative", backend="numpy")

    for phi in formulas:
        val_jax = phi.evaluate(signal_jax, sem_jax, t=0)
        val_np = phi.evaluate(signal_np, sem_np, t=0)
        assert float(val_jax.pos) == pytest.approx(val_np.pos, abs=1e-6)
        assert float(val_jax.neg) == pytest.approx(val_np.neg, abs=1e-6)


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

    sem_jax = create_semantics("dgmsr", backend="jax", eps=1e-8, p=1)
    sem_np = create_semantics("dgmsr", backend="numpy", eps=1e-8, p=1)

    v_and_jax = float(phi_and.evaluate(signal_jax, sem_jax, t=0))
    v_and_np = float(phi_and.evaluate(signal_np, sem_np, t=0))
    assert v_and_jax == pytest.approx(v_and_np, abs=1e-6)

    v_until_jax = float(phi_until.evaluate(signal_jax, sem_jax, t=0))
    v_until_np = float(phi_until.evaluate(signal_np, sem_np, t=0))
    assert v_until_jax == pytest.approx(v_until_np, abs=1e-6)


def test_jax_backends_support_autograd(signal_jax: jax.Array, predicates) -> None:
    p1, p2 = predicates

    checks = [
        (
            create_semantics("classical", backend="jax"),
            (p1 & p2).always((0, 2)),
            lambda v: v,
        ),
        (
            create_semantics("cumulative", backend="jax"),
            p1.eventually((0, 2)),
            lambda v: v.pos,
        ),
        (
            create_semantics("dgmsr", backend="jax", eps=1e-8, p=1),
            p1.until(p2, interval=(0, 3), weights_pair=(1.0, 1.2)),
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


def test_jax_backends_are_jittable(signal_jax: jax.Array, predicates) -> None:
    p1, p2 = predicates

    checks = [
        (
            create_semantics("classical", backend="jax"),
            (p1 & p2).always((0, 2)),
            lambda v: v,
        ),
        (
            create_semantics("cumulative", backend="jax"),
            p1.eventually((0, 2)),
            lambda v: v.pos,
        ),
        (
            create_semantics("dgmsr", backend="jax", eps=1e-8, p=1),
            p1.until(p2, interval=(0, 3), weights_pair=(1.0, 1.2)),
            lambda v: v,
        ),
    ]

    for sem, phi, projector in checks:

        def objective(
            sig: jax.Array, sem=sem, phi=phi, projector=projector
        ) -> jax.Array:
            out = phi.evaluate(sig, sem, t=0)
            return jnp.asarray(projector(out), dtype=float)

        expected = objective(signal_jax)
        compiled = jax.jit(objective)(signal_jax)
        assert float(compiled) == pytest.approx(float(expected), abs=1e-6)


def test_jax_backends_are_vmappable(signal_jax: jax.Array, predicates) -> None:
    p1, p2 = predicates

    checks = [
        (
            create_semantics("classical", backend="jax"),
            (p1 & p2).always((0, 2)),
            lambda v: v,
        ),
        (
            create_semantics("cumulative", backend="jax"),
            p1.eventually((0, 2)),
            lambda v: v.pos,
        ),
        (
            create_semantics("dgmsr", backend="jax", eps=1e-8, p=1),
            p1.until(p2, interval=(0, 3), weights_pair=(1.0, 1.2)),
            lambda v: v,
        ),
    ]

    batch = jnp.stack(
        (
            signal_jax,
            signal_jax
            + jnp.array([[0.01, -0.02], [0.02, -0.01], [0.0, 0.01], [-0.01, 0.0]]),
            signal_jax
            + jnp.array([[-0.02, 0.01], [0.01, 0.0], [0.02, -0.02], [0.0, 0.02]]),
        ),
        axis=0,
    )

    for sem, phi, projector in checks:

        def objective(
            sig: jax.Array, sem=sem, phi=phi, projector=projector
        ) -> jax.Array:
            out = phi.evaluate(sig, sem, t=0)
            return jnp.asarray(projector(out), dtype=float)

        vmapped = jax.vmap(objective, in_axes=0)(batch)
        assert vmapped.shape == (batch.shape[0],)
        assert np.all(np.isfinite(np.asarray(vmapped)))


def test_jax_ctstl_syntax_is_disabled() -> None:
    with pytest.raises(KeyError):
        create_semantics("ctstl", backend="jax", delta=1.0)
