from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg

from pcwannier.compute.gradient import Gradient
from pcwannier.compute.matrix import MSet
from pcwannier.compute.mv_optimizer import (
    MVCandidate,
    MVTimeReversalConstraint,
    MVTimeReversalDiagnostics,
    diagnose_diagonal_overlaps,
    precondition_mv_gauge,
    protected_mv_line_search,
)


def _object_grid(shape, factory):
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        result[index] = factory(index)
    return result


def test_protected_line_search_rejects_zero_diagonal_without_consuming_the_step():
    config = SimpleNamespace(composition_of_b=[[1], [-1]])
    state = SimpleNamespace(k_indices=lambda: iter(((0, 0, 0),)))
    holder = {"gauge": _object_grid((1, 1, 1), lambda _idx: np.eye(2, dtype=np.complex128))}

    def update(gauge):
        holder["gauge"] = gauge

    def get(*_args):
        return np.asarray(holder["gauge"][0, 0, 0])

    gradient = SimpleNamespace(
        U=holder["gauge"],
        omega=np.array([1.0, 0.0, 0.0]),
        mset=SimpleNamespace(config=config, state=state, update=update, get=get),
        update=lambda: setattr(gradient, "omega", np.array([0.9, 0.0, 0.0])),
    )

    def candidate(step):
        rotation = np.array(
            [[np.cos(step), -np.sin(step)], [np.sin(step), np.cos(step)]],
            dtype=np.complex128,
        )
        return MVCandidate(_object_grid((1, 1, 1), lambda _idx: rotation.copy()))

    result = protected_mv_line_search(
        gradient,
        candidate,
        initial_step=np.pi / 2.0,
        diagonal_floor=1.0e-8,
        max_steps=4,
    )

    assert result.accepted
    assert result.backtracking_steps == 1
    assert result.trial_step == pytest.approx(np.pi / 4.0)
    assert result.diagnostics.min_abs_diagonal == pytest.approx(2.0**-0.5)


def test_u_n_synchronization_recovers_a_smooth_frame():
    shape = (5, 1, 1)
    config = SimpleNamespace(
        band_calc_num=2,
        kdim=1,
        k_points=[np.arange(5, dtype=float) / 5.0 - 0.5],
        composition_of_b=[[1], [-1]],
        b_vectors=np.array([[2.0 * np.pi / 5.0], [-2.0 * np.pi / 5.0]]),
        wb=np.array([25.0 / (8.0 * np.pi**2), 25.0 / (8.0 * np.pi**2)]),
    )
    state = SimpleNamespace(
        config=config,
        k_shape=shape,
        k_indices=lambda: iter(np.ndindex(shape)),
        get_k_num=lambda: 5,
        gen_matrix_on_kmesh=lambda factory: _object_grid(
            shape, lambda index: factory(*index)
        ),
    )
    rng = np.random.default_rng(41)
    rough = []
    for _ in range(shape[0]):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        antihermitian = 0.5 * (matrix - matrix.conj().T)
        rough.append(scipy.linalg.expm(0.35 * antihermitian))

    centers = np.array([[0.2, -0.3]])
    target_phase = np.diag(np.exp(-1j * config.b_vectors[0, 0] * centers[0]))
    mset = MSet(state, threads=1)
    mset.mMInitial = _object_grid(
        shape,
        lambda index: [
            rough[index[0]].conj().T
            @ target_phase
            @ rough[(index[0] + 1) % shape[0]]
        ],
    )
    mset.mM = _object_grid(shape, lambda _index: [np.eye(2, dtype=np.complex128)])
    gradient = Gradient(state, mset, threads=1)
    mset.update(gradient.U)
    gradient.update()
    before = float(gradient.omega[1] + gradient.omega[2])

    result = precondition_mv_gauge(
        gradient,
        centers,
        diagonal_floor=1.0e-8,
        max_sweeps=50,
    )

    mset.update(gradient.U)
    gradient.update()
    after = float(gradient.omega[1] + gradient.omega[2])
    assert result.accepted
    assert after < before
    assert after < 1.0e-6, (result, np.real(gradient.rn))
    assert diagnose_diagonal_overlaps(mset).min_abs_diagonal > 0.999999
    assert np.allclose(np.real(gradient.rn), centers, atol=1.0e-8)


def test_preconditioner_keeps_original_gauge_when_exact_spread_does_not_improve():
    shape = (1, 1, 1)
    config = SimpleNamespace(
        band_calc_num=1,
        kdim=1,
        k_points=[np.array([0.0])],
        composition_of_b=[[1], [-1]],
        b_vectors=np.array([[1.0], [-1.0]]),
        wb=np.array([1.0, 1.0]),
    )
    state = SimpleNamespace(
        config=config,
        k_shape=shape,
        k_indices=lambda: iter(((0, 0, 0),)),
        get_k_num=lambda: 1,
        gen_matrix_on_kmesh=lambda factory: _object_grid(
            shape, lambda index: factory(*index)
        ),
    )
    mset = MSet(state, threads=1)
    mset.mMInitial = _object_grid(shape, lambda _index: [np.eye(1, dtype=np.complex128)])
    mset.mM = _object_grid(shape, lambda _index: [np.eye(1, dtype=np.complex128)])
    gradient = Gradient(state, mset, threads=1)
    initial = gradient.U[0, 0, 0].copy()

    result = precondition_mv_gauge(
        gradient,
        np.zeros((1, 1)),
        diagonal_floor=1.0e-8,
    )

    assert not result.accepted
    assert np.array_equal(gradient.U[0, 0, 0], initial)


def test_time_reversal_constraint_projects_pairs_and_trim_to_real_structure():
    shape = (4, 1, 1)
    partners = _object_grid(
        shape,
        lambda index: ((0, 0, 0), (3, 0, 0), (2, 0, 0), (1, 0, 0))[index[0]],
    )
    sewing = _object_grid(shape, lambda _index: -np.eye(2, dtype=np.complex128))
    constraint = MVTimeReversalConstraint(
        partners,
        sewing,
        MVTimeReversalDiagnostics(0.0, 0.0),
    )
    rng = np.random.default_rng(812)

    def random_unitary(_index):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        left, _, vh = np.linalg.svd(matrix)
        return left @ vh

    gauge = _object_grid(shape, random_unitary)
    result = constraint.project(gauge)

    assert result.max_residual < 1.0e-10
    assert result.max_unitarity_error < 1.0e-10
    assert np.allclose(result.gauge[3, 0, 0], result.gauge[1, 0, 0].conj())
    assert np.max(np.abs(result.gauge[0, 0, 0].imag)) < 1.0e-10
    assert np.max(np.abs(result.gauge[2, 0, 0].imag)) < 1.0e-10
