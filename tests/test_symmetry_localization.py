from __future__ import annotations

import numpy as np
import scipy.linalg

from pcwannier.compute.gradient import Gradient
from pcwannier.compute.matrix import MSet
from pcwannier.compute.mv_optimizer import precondition_mv_gauge
from pcwannier.symmetry import (
    SpaceGroup,
    SpaceGroupOperation,
    build_symmetry_context,
    build_symmetry_stars,
    project_target_gauge_to_stars,
    propagate_target_gauge,
    symmetrize_gradient,
    WannierTargetSpec,
)

from .symmetry_models import model_from_space_group, p4mm_model, square_2c_model


def test_identity_group_constrained_gradient_is_the_raw_gradient(tmp_path):
    group = SpaceGroup(
        (SpaceGroupOperation(np.eye(2, dtype=int), np.zeros(2), "E"),),
        tolerance=1.0e-10,
    )
    model = model_from_space_group(
        "identity",
        group,
        targets=(WannierTargetSpec("scalar", [0.0, 0.0], "A"),),
    )
    context = build_symmetry_context(model, [np.array([-0.25, 0.25]), np.array([-0.25, 0.25])])
    stars = build_symmetry_stars(context)
    raw = np.empty((2, 2, 1), dtype=object)
    for flat_index, index in enumerate(np.ndindex(raw.shape)):
        raw[index] = np.array([[1j * (flat_index + 1)]])

    constrained = symmetrize_gradient(raw, context, stars)

    assert len(stars.stars) == 4
    for star, matrix in zip(stars.stars, constrained):
        assert np.array_equal(matrix, raw[_state_index(star.representative_index)])


def test_symmetric_target_gauge_is_unchanged_and_random_noise_is_projected_out():
    model = p4mm_model(
        targets=(
            WannierTargetSpec("center_s_A1", [0.0, 0.0], "A1"),
            WannierTargetSpec("center_p_E", [0.0, 0.0], "E"),
        )
    )
    axis = np.arange(-0.5, 0.5, 0.1)
    context = build_symmetry_context(model, [axis, axis])
    stars = build_symmetry_stars(context)
    identity = _matrix_mesh((10, 10, 1), np.eye(3, dtype=np.complex128))

    unchanged = project_target_gauge_to_stars(identity, context, stars)

    assert unchanged.max_path_consistency < 1e-12
    assert unchanged.max_unitarity_error < 1e-12
    for index in np.ndindex(identity.shape):
        assert np.allclose(unchanged.gauge[index], identity[index], atol=1e-12)

    rng = np.random.default_rng(20260713)
    perturbed = np.empty(identity.shape, dtype=object)
    for index in np.ndindex(perturbed.shape):
        raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        antihermitian = 0.5 * (raw - raw.conj().T)
        perturbed[index] = scipy.linalg.expm(1.0e-2 * antihermitian)

    projected = project_target_gauge_to_stars(perturbed, context, stars)

    assert projected.max_path_consistency < 1e-10
    assert projected.max_unitarity_error < 1e-10
    assert _max_target_residual(projected.gauge, context) < 1e-10


def test_square_2c_gradient_pullback_and_star_propagation_follow_right_action_convention():
    model = square_2c_model()
    axis = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    context = build_symmetry_context(model, [axis, axis])
    stars = build_symmetry_stars(context)
    generic_star = next(star for star in stars.stars if len(star.members) == 8)
    raw_gradient = _matrix_mesh((5, 5, 1), np.zeros((2, 2), dtype=np.complex128))
    generator = np.array([[0.0, 0.2 + 0.3j], [-0.2 + 0.3j, 0.0]], dtype=np.complex128)
    representative_k = _fractional_at(context, generic_star.representative_index)
    for member in generic_star.members:
        path = member.canonical_path
        dmat = context.target_matrix(path.operation_index, representative_k)
        raw_gradient[_state_index(member.k_index)] = dmat @ generator @ dmat.conj().T

    constrained = symmetrize_gradient(raw_gradient, context, stars)

    assert np.allclose(
        constrained[generic_star.index],
        len(generic_star.members) * generator,
        atol=1e-12,
    )
    representatives = [np.eye(2, dtype=np.complex128) for _ in stars.stars]
    representatives[generic_star.index] = scipy.linalg.expm(
        0.01 * constrained[generic_star.index]
    )
    propagated = propagate_target_gauge(representatives, context, stars)
    assert propagated.max_path_consistency < 1e-12
    assert _max_target_residual(propagated.gauge, context) < 1e-12


def test_u_n_synchronization_preserves_c4v_target_stars():
    model = p4mm_model(
        targets=(
            WannierTargetSpec("center_s_A1", [0.0, 0.0], "A1"),
            WannierTargetSpec("center_p_E", [0.0, 0.0], "E"),
        )
    )
    axis = np.arange(-0.4, 0.5, 0.2)
    context = build_symmetry_context(model, [axis, axis])
    stars = build_symmetry_stars(context)
    shape = (5, 5, 1)
    config = type(
        "Config",
        (),
        {
            "band_calc_num": 3,
            "kdim": 2,
            "k_points": [axis, axis],
            "composition_of_b": [[1, 0], [0, 1], [-1, 0], [0, -1]],
            "b_vectors": np.array(
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
            ),
            "wb": np.ones(4),
        },
    )()
    state = type(
        "State",
        (),
        {
            "config": config,
            "k_shape": shape,
            "k_indices": lambda self: iter(np.ndindex(shape)),
            "get_k_num": lambda self: 25,
            "gen_matrix_on_kmesh": lambda self, factory: _matrix_mesh_from_factory(
                shape, factory
            ),
        },
    )()
    mset = MSet(state, threads=1)
    mset.mMInitial = _matrix_mesh_from_factory(
        shape, lambda *_index: [np.eye(3), np.eye(3)]
    )
    mset.mM = _matrix_mesh_from_factory(
        shape, lambda *_index: [np.eye(3), np.eye(3)]
    )
    gradient = Gradient(state, mset, threads=1)
    rng = np.random.default_rng(713)
    representatives = []
    for _star in stars.stars:
        raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        left, _, vh = np.linalg.svd(raw)
        representatives.append(left @ vh)
    gradient.U = propagate_target_gauge(representatives, context, stars).gauge
    mset.update(gradient.U)
    gradient.update()
    before = float(gradient.omega[1] + gradient.omega[2])

    def symmetry_projector(candidate):
        projected = project_target_gauge_to_stars(candidate, context, stars)
        return (
            projected.gauge,
            projected.max_path_consistency,
            projected.max_unitarity_error,
        )

    result = precondition_mv_gauge(
        gradient,
        np.zeros((2, 3)),
        diagonal_floor=1.0e-8,
        symmetry_projector=symmetry_projector,
    )

    mset.update(gradient.U)
    gradient.update()
    assert result.accepted
    assert float(gradient.omega[1] + gradient.omega[2]) < before
    assert _max_target_residual(gradient.U, context) < 1.0e-10


def test_unconstrained_two_band_minimum_can_break_target_symmetry():
    dmat = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    overlap = np.array(
        [[0.5 + 0.2j, 0.3], [0.8, 0.5 - 0.2j]],
        dtype=np.complex128,
    )
    assert np.allclose(overlap.conj().T, dmat @ overlap @ dmat)

    left, _, vh = np.linalg.svd(overlap)
    right = vh.conj().T
    unconstrained_overlap = left.conj().T @ overlap @ right
    unconstrained_spread = _offdiagonal_norm(unconstrained_overlap)
    unconstrained_residual = np.linalg.norm(dmat @ left - right @ dmat, ord="fro")

    averaged = 0.5 * (left + dmat.conj().T @ right @ dmat)
    polar_left, _, polar_vh = np.linalg.svd(averaged)
    constrained_left = polar_left @ polar_vh
    constrained_right = dmat @ constrained_left @ dmat.conj().T
    constrained_overlap = constrained_left.conj().T @ overlap @ constrained_right
    constrained_spread = _offdiagonal_norm(constrained_overlap)
    constrained_residual = np.linalg.norm(
        dmat @ constrained_left - constrained_right @ dmat,
        ord="fro",
    )

    assert unconstrained_spread < 1e-24
    assert unconstrained_residual > 1.0
    assert constrained_residual < 1e-12
    assert constrained_spread > unconstrained_spread + 0.5


def _matrix_mesh(shape, matrix):
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        result[index] = np.asarray(matrix, dtype=np.complex128).copy()
    return result


def _matrix_mesh_from_factory(shape, factory):
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        result[index] = factory(*index)
    return result


def _max_target_residual(gauge, context) -> float:
    maximum = 0.0
    for operation_index, mappings in enumerate(context.k_mappings):
        operation = context.model.group.operations[operation_index]
        for mapping in mappings:
            source = _state_index(mapping.source_k_index)
            target = _state_index(mapping.target_k_index)
            source_k = _fractional_at(context, mapping.source_k_index)
            dmat = context.target_matrix(operation_index, source_k)
            source_gauge = gauge[source].conj() if operation.antiunitary else gauge[source]
            maximum = max(
                maximum,
                float(np.linalg.norm(dmat @ source_gauge - gauge[target] @ dmat, ord="fro")),
            )
    return maximum


def _offdiagonal_norm(matrix) -> float:
    array = np.asarray(matrix)
    return float(np.sum(np.abs(array - np.diag(np.diag(array))) ** 2))


def _fractional_at(context, index):
    return np.asarray(
        [context.k_points[axis][index[axis]] for axis in range(context.model.dimension)],
        dtype=float,
    )


def _state_index(index):
    return (tuple(index) + (0, 0, 0))[:3]
