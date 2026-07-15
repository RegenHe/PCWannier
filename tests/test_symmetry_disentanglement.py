from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pcwannier.symmetry.disentanglement as disentanglement_module

from pcwannier.symmetry import (
    OuterWindowClosureReport,
    SpaceGroup,
    SpaceGroupOperation,
    WannierTargetSpec,
    build_symmetry_context,
    combined_target_matrix,
    construct_symmetry_gauge,
    disentangle_symmetry_constrained,
    validate_frozen_window_covariance,
    validate_outer_window_closure,
)

from .symmetry_models import model_from_space_group, square_2c_model


def test_symmetrized_z_uses_little_group_normalization(tmp_path):
    context = _c2_context(tmp_path, target_character=1.0)
    state = _state(context, ((0, 1), (2, 3)))
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    provider = _SyntheticProvider(context, lambda op, *_: np.eye(2) if op == 0 else swap)
    initializer = _initializer(state, target_dimension=1)
    initial = _matrix_grid(
        state.k_shape,
        [np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])],
    )
    gauge = construct_symmetry_gauge(
        state, context, initial, provider=provider, tolerance=1e-12
    )
    bands = gauge.band_indices_by_k
    if bands is None:
        bands = _band_grid(state)
    zmat = disentanglement_module._symmetrized_z(
        initializer,
        context,
        provider,
        bands,
        gauge.gauge,
        gauge.stars.stars[0],
    )

    # There are two group paths and a trivial representative little group.
    # Eq. (35) therefore gives the sum over both distinct star members, not
    # their average.
    assert np.allclose(zmat, np.diag([2.0, 0.0]), atol=1e-12)


def test_flat_omega_does_not_converge_while_projector_changes(tmp_path, monkeypatch):
    context = _identity_context(tmp_path)
    state = _state(context, ((0, 1),))
    provider = _SyntheticProvider(context, lambda *_: np.eye(2, dtype=np.complex128))
    initializer = _initializer(state, target_dimension=1)
    initial = _matrix_grid(
        state.k_shape,
        [np.array([[1.0], [1.0]], dtype=np.complex128) / np.sqrt(2.0)],
    )
    closure = validate_outer_window_closure(state, context, provider, tolerance=1e-12)
    gauge = construct_symmetry_gauge(
        state, context, initial, provider=provider, tolerance=1e-12
    )
    monkeypatch.setattr(disentanglement_module, "_omega_i", lambda *_: 0.0)
    monkeypatch.setattr(disentanglement_module, "_projector_change", lambda *_: 1.0)

    result = disentangle_symmetry_constrained(
        initializer,
        context,
        gauge,
        provider,
        closure,
        err_diff=1e-12,
        projector_tolerance=1e-8,
        max_iter=2,
        mixing=0.5,
        tolerance=1e-12,
        projection_max_iterations=10,
        svd_relative_tolerance=1e-12,
    )

    assert not result.converged
    assert result.iterations[-1].projector_change == pytest.approx(1.0)


def test_covariant_outer_subspace_is_preserved_with_changing_band_ids(tmp_path):
    context = _c2_context(tmp_path, target_character=1.0)
    state = _state(context, ((0, 1), (2, 3)))
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    provider = _SyntheticProvider(context, lambda op, *_: np.eye(2) if op == 0 else swap)
    initializer = _initializer(state, target_dimension=1)
    initial = _matrix_grid(state.k_shape, [np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])])

    closure = validate_outer_window_closure(state, context, provider, tolerance=1e-12)
    gauge = construct_symmetry_gauge(
        state, context, initial, provider=provider, tolerance=1e-12
    )
    result = disentangle_symmetry_constrained(
        initializer,
        context,
        gauge,
        provider,
        closure,
        err_diff=1e-12,
        max_iter=2,
        mixing=0.5,
        tolerance=1e-12,
        projection_max_iterations=10,
        svd_relative_tolerance=1e-12,
    )

    assert closure.max_unitarity_error < 1e-12
    assert result.final_omega_i == pytest.approx(0.0, abs=1e-12)
    assert result.diagnostics.max_projector_residual < 1e-12
    assert result.diagnostics.max_intertwiner_residual < 1e-12
    assert np.allclose(result.optimal_frame[0, 0, 0], [[1.0], [0.0]])
    assert np.allclose(result.optimal_frame[1, 0, 0], [[0.0], [1.0]])


def test_identity_group_update_matches_unconstrained_largest_eigenvector(tmp_path):
    context = _identity_context(tmp_path)
    state = _state(context, ((0, 1),))
    provider = _SyntheticProvider(context, lambda *_: np.eye(2, dtype=np.complex128))
    initializer = _initializer(state, target_dimension=1)
    overlap = np.diag([2.0, 1.0]).astype(np.complex128)
    initializer.mset = SimpleNamespace(get_M0=lambda *_: overlap)
    initial_vector = np.array([[1.0], [1.0]], dtype=np.complex128) / np.sqrt(2.0)
    initial = _matrix_grid(state.k_shape, [initial_vector])
    closure = validate_outer_window_closure(state, context, provider, tolerance=1e-12)
    gauge = construct_symmetry_gauge(
        state, context, initial, provider=provider, tolerance=1e-12
    )
    result = disentangle_symmetry_constrained(
        initializer,
        context,
        gauge,
        provider,
        closure,
        err_diff=1e-12,
        max_iter=1,
        mixing=0.5,
        tolerance=1e-12,
        projection_max_iterations=10,
        svd_relative_tolerance=1e-12,
    )

    expected = overlap @ initial_vector
    expected /= np.linalg.norm(expected)
    actual = result.optimal_frame[0, 0, 0]
    assert np.allclose(actual @ actual.conj().T, expected @ expected.conj().T, atol=1e-12)


def test_outer_window_dimension_mismatch_in_one_star_is_rejected(tmp_path):
    context = _c2_context(tmp_path, target_character=1.0)
    state = _state(context, ((0, 1), (0, 1, 2)))
    provider = _SyntheticProvider(
        context,
        lambda _op, source, _target: np.eye(2 if source[0] == 0 else 3, dtype=np.complex128),
    )

    with pytest.raises(RuntimeError, match="M=2.*M=3"):
        validate_outer_window_closure(state, context, provider, tolerance=1e-12)


def test_incompatible_outer_representation_has_no_target_intertwiner(tmp_path):
    context = _c2_context(tmp_path, target_character=-1.0, axis=np.array([0.0]))
    state = _state(context, ((0, 1),))
    provider = _SyntheticProvider(context, lambda *_: np.eye(2, dtype=np.complex128))
    initial = _matrix_grid(state.k_shape, [np.array([[1.0], [0.0]])])

    with pytest.raises(RuntimeError, match="dim Hom=0"):
        construct_symmetry_gauge(
            state, context, initial, provider=provider, tolerance=1e-12
        )


def test_frozen_window_symmetry_violation_is_rejected(tmp_path):
    context = _c2_context(tmp_path, target_character=1.0)
    state = _state(context, ((0, 1), (0, 1)))
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    provider = _SyntheticProvider(context, lambda op, *_: np.eye(2) if op == 0 else swap)
    initializer = _initializer(state, target_dimension=1, frozen=(0,))

    with pytest.raises(RuntimeError, match="Frozen window violates symmetry covariance"):
        validate_frozen_window_covariance(
            initializer,
            context,
            provider,
            _band_grid(state),
            tolerance=1e-12,
        )


def test_compatible_frozen_subspace_is_retained_at_little_group_point(tmp_path):
    context = _c2_context(tmp_path, target_character=1.0, axis=np.array([0.0]))
    state = _state(context, ((0, 1),))
    physical_c2 = np.diag([1.0, -1.0]).astype(np.complex128)
    provider = _SyntheticProvider(
        context, lambda op, *_: np.eye(2) if op == 0 else physical_c2
    )
    initializer = _initializer(state, target_dimension=1, frozen=(0,))
    initial = _matrix_grid(state.k_shape, [np.array([[1.0], [0.0]])])
    closure = validate_outer_window_closure(state, context, provider, tolerance=1e-12)
    gauge = construct_symmetry_gauge(
        state, context, initial, provider=provider, tolerance=1e-12
    )
    result = disentangle_symmetry_constrained(
        initializer,
        context,
        gauge,
        provider,
        closure,
        err_diff=1e-12,
        max_iter=1,
        mixing=0.5,
        tolerance=1e-12,
        projection_max_iterations=10,
        svd_relative_tolerance=1e-12,
    )

    assert result.diagnostics.max_frozen_residual < 1e-12
    assert np.allclose(result.optimal_frame[0, 0, 0], [[1.0], [0.0]])


def test_square_2c_target_is_selected_from_four_dimensional_outer_space():
    model = square_2c_model()
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    state = _state(context, ((0, 1, 2, 3),))

    def physical(operation_index, source_index, _target_index):
        kpoint = _fractional_at(context, source_index)
        target = combined_target_matrix(model.targets, operation_index, kpoint)
        return np.block([[target, np.zeros_like(target)], [np.zeros_like(target), target]])

    provider = _SyntheticProvider(context, physical)
    initializer = _initializer(state, target_dimension=2)
    initial_frame = np.vstack((np.eye(2), np.zeros((2, 2))))
    initial = _matrix_grid(state.k_shape, [initial_frame])
    closure = validate_outer_window_closure(state, context, provider, tolerance=1e-12)
    gauge = construct_symmetry_gauge(
        state, context, initial, provider=provider, tolerance=1e-12
    )
    result = disentangle_symmetry_constrained(
        initializer,
        context,
        gauge,
        provider,
        closure,
        err_diff=1e-12,
        max_iter=1,
        mixing=0.5,
        tolerance=1e-12,
        projection_max_iterations=10,
        svd_relative_tolerance=1e-12,
    )

    assert result.representative_hom_dimensions[0] > 0
    assert result.diagnostics.max_projector_residual < 1e-12
    assert result.diagnostics.max_intertwiner_residual < 1e-12
    assert np.linalg.norm(
        result.optimal_frame[0, 0, 0].conj().T @ result.optimal_frame[0, 0, 0] - np.eye(2)
    ) < 1e-12


class _SyntheticProvider:
    def __init__(self, context, matrix):
        self.context = context
        self._matrix = matrix

    def mapping(self, operation_index, source_index):
        shape = tuple(len(axis) for axis in self.context.k_points)
        source = tuple(int(value) for value in source_index)
        flat = int(np.ravel_multi_index(source, shape))
        return self.context.k_mappings[operation_index][flat]

    def find_k_index(self, k_fractional):
        point = np.mod(np.asarray(k_fractional, dtype=float) + 0.5, 1.0) - 0.5
        shape = tuple(len(axis) for axis in self.context.k_points)
        points = np.asarray(
            [
                [self.context.k_points[axis][index[axis]] for axis in range(len(shape))]
                for index in np.ndindex(shape)
            ]
        )
        distances = np.max(np.abs(np.mod(points - point + 0.5, 1.0) - 0.5), axis=1)
        return tuple(int(value) for value in np.unravel_index(int(np.argmin(distances)), shape))

    def sewing_matrix_between_mapping(self, mapping, source_band_indices, target_band_indices):
        matrix = self._matrix(
            mapping.operation_index,
            tuple(mapping.source_k_index),
            tuple(mapping.target_k_index),
        )
        return np.asarray(matrix, dtype=np.complex128)

    def request_for_mapping(
        self,
        mapping,
        band_indices,
        *,
        operation=None,
        source_k_fractional=None,
        target_band_indices=None,
    ):
        return SimpleNamespace(
            mapping=mapping,
            operation=operation
            or self.context.model.group.operations[mapping.operation_index],
            band_indices=tuple(band_indices),
            target_band_indices=tuple(target_band_indices or band_indices),
        )

    def sewing_matrix(self, request):
        operation_index = self.context.model.group.operation_index(request.operation)
        matrix = self._matrix(
            operation_index,
            tuple(request.mapping.source_k_index),
            tuple(request.mapping.target_k_index),
        )
        return np.asarray(matrix, dtype=np.complex128)


def _c2_context(tmp_path, *, target_character, axis=None):
    group = SpaceGroup(
        (
            SpaceGroupOperation(np.eye(2, dtype=int), np.zeros(2), "E"),
            SpaceGroupOperation(-np.eye(2, dtype=int), np.zeros(2), "C2"),
        ),
        tolerance=1.0e-10,
    )
    irrep_name = "A" if target_character == 1.0 else "B"
    model = model_from_space_group(
        "c2",
        group,
        targets=(WannierTargetSpec("target", [0.0, 0.0], irrep_name),),
    )
    xaxis = np.array([-0.25, 0.25]) if axis is None else np.asarray(axis)
    return build_symmetry_context(model, [xaxis, np.array([0.0])])


def _identity_context(tmp_path):
    group = SpaceGroup(
        (SpaceGroupOperation(np.eye(2, dtype=int), np.zeros(2), "E"),),
        tolerance=1.0e-10,
    )
    model = model_from_space_group(
        "identity",
        group,
        targets=(WannierTargetSpec("target", [0.0, 0.0], "A"),),
    )
    return build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])


def _state(context, band_sets):
    shape2 = tuple(len(axis) for axis in context.k_points)
    shape = (shape2 + (1, 1, 1))[:3]
    indices = list(np.ndindex(shape))
    assert len(indices) == len(band_sets)
    outer = np.empty(shape, dtype=object)
    inner = np.empty(shape, dtype=object)
    for index, bands in zip(indices, band_sets):
        outer[index] = tuple(bands)
        inner[index] = ()
    return SimpleNamespace(
        E_idx=outer,
        inner_E_idx=inner,
        k_shape=shape,
        k_indices=lambda: iter(indices),
        get_k_num=lambda: len(indices),
    )


def _initializer(state, *, target_dimension, frozen=()):
    for index in state.k_indices():
        state.inner_E_idx[index] = tuple(state.E_idx[index][position] for position in frozen)
    i_idx = np.empty(state.k_shape, dtype=object)
    for index in state.k_indices():
        i_idx[index] = np.asarray(frozen, dtype=np.intp)
    config = SimpleNamespace(
        band_calc_num=target_dimension,
        composition_of_b=[[0, 0]],
        wb=np.array([1.0]),
        kdim=2,
        k_points=[np.arange(state.k_shape[0]), np.arange(state.k_shape[1])],
        proj_iter=True,
    )
    mset = SimpleNamespace(
        get_M0=lambda i, j, k, _b: np.eye(len(state.E_idx[i, j, k]), dtype=np.complex128)
    )
    return SimpleNamespace(state=state, config=config, I_idx=i_idx, mset=mset)


def _matrix_grid(shape, matrices):
    result = np.empty(shape, dtype=object)
    for index, matrix in zip(np.ndindex(shape), matrices):
        result[index] = np.asarray(matrix, dtype=np.complex128)
    return result


def _band_grid(state):
    result = np.empty(state.k_shape, dtype=object)
    for index in state.k_indices():
        result[index] = tuple(state.E_idx[index])
    return result


def _fractional_at(context, index):
    return np.asarray(
        [context.k_points[axis][index[axis]] for axis in range(context.model.dimension)],
        dtype=float,
    )
