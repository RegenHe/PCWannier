from types import SimpleNamespace

import numpy as np
import pytest

from pcwannier import BlochConvention
from pcwannier.conventions import BlochFieldRepresentation
from pcwannier.compute.state import StateCollection
from pcwannier.compute.uniform_grid import UniformGridInnerProduct
from pcwannier.data import InputBundle, PeriodicGrid
from pcwannier.etbc import complete_transverse_bundle, construct_auxiliary_frame
from pcwannier.etbc.completion import prepare_transverse_bundle
from pcwannier.maxwell import MaxwellProblem


def _constant_field(direction, point_count):
    return np.broadcast_to(
        np.asarray(direction, dtype=np.complex128), (point_count, 3)
    ).copy()


def _inner_product(shape=(2, 1, 1), metric=None):
    grid = PeriodicGrid(shape, np.eye(3))
    if metric is None:
        metric = np.ones(grid.point_count)
    return grid, UniformGridInnerProduct(grid, np.asarray(metric, dtype=float))


def test_construct_auxiliary_frame_lowdin_and_complex_nullspace():
    grid, inner = _inner_product(metric=[1.0, 2.0])
    ex = _constant_field([1.0, 0.0, 0.0], grid.point_count)
    ey = _constant_field([0.0, 1.0, 0.0], grid.point_count)
    ez = _constant_field([0.0, 0.0, 1.0], grid.point_count)
    scale = np.sqrt(inner.norm(ex))
    transverse = np.stack((ex / scale, ey / scale))
    trials = np.stack((ex + 0.2j * ey, ey + 0.3 * ez, ez + 0.1 * ex))

    result = construct_auxiliary_frame(
        transverse, trials, inner, rank_tolerance=1.0e-10
    )

    assert result.transverse_rank == 2
    assert result.auxiliary_frame.shape == (1, grid.point_count, 3)
    assert result.trial_gram_eigenvalues.min() > 0.0
    assert result.singular_values[-1] > 0.0
    assert result.transverse_auxiliary_overlap < 1.0e-12
    assert result.augmented_orthonormality_error < 1.0e-12


def test_construct_auxiliary_frame_rejects_missing_transverse_direction():
    grid, inner = _inner_product()
    points = grid.point_count
    ex = _constant_field([1.0, 0.0, 0.0], points)
    ey = _constant_field([0.0, 1.0, 0.0], points)
    ez = _constant_field([0.0, 0.0, 1.0], points)
    alternating = np.asarray([1.0, -1.0])[:, None] * ez
    transverse = np.stack((ex, ey))
    trials = np.stack((ex, ez, alternating))

    with pytest.raises(ValueError, match="overlap has rank 1"):
        construct_auxiliary_frame(
            transverse, trials, inner, rank_tolerance=1.0e-10
        )


def _physical_state_for_gamma(*, include_positive=False):
    grid, _ = _inner_product()
    points = grid.point_count
    ex = _constant_field([1.0, 0.0, 0.0], points)
    ey = _constant_field([0.0, 1.0, 0.0], points)
    fields_list = [ex, ey]
    energies_list = [0.0, 0.0]
    if include_positive:
        alternating = np.asarray([1.0, -1.0])[:, None] * np.asarray(
            [1.0, 0.0, 0.0]
        )[None, :]
        fields_list.append(alternating)
        energies_list.append(2.0)
    fields = np.empty((1, 1, 1), dtype=object)
    fields[0, 0, 0] = np.stack(fields_list)
    energies = np.empty((1, 1, 1), dtype=object)
    energies[0, 0, 0] = np.asarray(energies_list)
    indices = np.empty((1, 1, 1), dtype=object)
    indices[0, 0, 0] = list(range(len(fields_list)))
    inner_indices = np.empty((1, 1, 1), dtype=object)
    inner_indices[0, 0, 0] = []
    n_w = 5 if include_positive else 3
    config = SimpleNamespace(
        kdim=3,
        k_points=[np.asarray([0.0]), np.asarray([0.0]), np.asarray([0.0])],
        gamma_zero_mode_tolerance=1.0e-10,
        band_calc_num=n_w,
        integration_mode="nodal",
        use_cached_data=[],
        real_lattice_vectors=np.eye(3),
        reciprocal_lattice_vectors=2.0 * np.pi * np.eye(3),
        lattice_const=1.0,
        extension=[1, 1, 1],
        maxwell_problem=MaxwellProblem.for_components("full_vector", "magnetic"),
    )
    bundle = InputBundle(
        config=config,
        maxwell=config.maxwell_problem,
        bloch_convention=BlochConvention(1, "synthetic"),
        mesh=grid,
        fields=fields,
        metric_material=np.ones(points),
        energies=energies,
        band_indices=indices,
        inner_band_indices=inner_indices,
        energy_matrix=np.asarray(energies_list).reshape(1, 1, 1, -1),
        field_representation=BlochFieldRepresentation.PERIODIC_PART,
    )
    state = StateCollection(bundle)
    _, need_orth = state.check_orthogonality()
    assert not need_orth
    state.ensure_identity_transform()
    return state


def test_gamma_two_transverse_plus_one_auxiliary_uses_constant_frame():
    state = _physical_state_for_gamma()
    points = state.mesh.point_count
    trials = np.stack(
        [
            _constant_field([1.0, 0.0, 0.0], points),
            _constant_field([0.0, 1.0, 0.0], points),
            _constant_field([0.0, 0.0, 1.0], points),
        ],
        axis=1,
    )

    completion = prepare_transverse_bundle(
        state,
        lambda _: trials,
        auxiliary_eigenvalue=0.0,
        rank_tolerance=1.0e-10,
    )

    result = completion.result
    assert result.gamma_regularized_indices == ((0, 0, 0),)
    assert result.auxiliary_dimension == 1
    block = np.asarray(completion.augmented_bundle.fields[0, 0, 0])
    assert block.shape == (3, points, 3)
    assert np.allclose(
        np.linalg.eigvalsh(completion.augmented_bundle.base_hamiltonians[0, 0, 0]),
        0.0,
    )
    expected_projection = state.inner_product.overlap(
        block,
        np.swapaxes(trials, 0, 1),
        chunk_size=64,
    )
    assert np.allclose(
        completion.projection_seed.matrices[0, 0, 0], expected_projection
    )


def test_public_etbc_completion_does_not_expose_run_scoped_arrays():
    state = _physical_state_for_gamma()
    points = state.mesh.point_count
    trials = np.stack(
        [
            _constant_field([1.0, 0.0, 0.0], points),
            _constant_field([0.0, 1.0, 0.0], points),
            _constant_field([0.0, 0.0, 1.0], points),
        ],
        axis=1,
    )

    result = complete_transverse_bundle(state, lambda _: trials)

    assert result.auxiliary_dimension == 1
    assert not hasattr(result, "augmented_bundle")
    assert not hasattr(result, "projection_seed")


def test_gamma_general_group_preserves_positive_t_and_builds_extra_l():
    state = _physical_state_for_gamma(include_positive=True)
    points = state.mesh.point_count
    alternating = np.asarray([1.0, -1.0])[:, None]
    trial_rows = np.stack(
        (
            _constant_field([1.0, 0.0, 0.0], points),
            _constant_field([0.0, 1.0, 0.0], points),
            _constant_field([0.0, 0.0, 1.0], points),
            alternating * np.asarray([1.0, 0.0, 0.0])[None, :],
            alternating * np.asarray([0.0, 1.0, 0.0])[None, :],
        )
    )
    trials = np.swapaxes(trial_rows, 0, 1)

    completion = prepare_transverse_bundle(
        state,
        lambda _: trials,
        auxiliary_eigenvalue=0.0,
        rank_tolerance=1.0e-10,
    )

    result = completion.result
    assert result.auxiliary_dimension == 2
    h0 = completion.augmented_bundle.base_hamiltonians[0, 0, 0]
    assert np.allclose(np.linalg.eigvalsh(h0), [0.0, 0.0, 0.0, 0.0, 2.0])
    assert result.maximum_augmented_orthonormality_error < 1.0e-12


def test_streaming_etbc_matches_materialized_gamma_completion():
    state = _physical_state_for_gamma()
    points = state.mesh.point_count
    trials = np.stack(
        [
            _constant_field([1.0, 0.0, 0.0], points),
            _constant_field([0.0, 1.0, 0.0], points),
            _constant_field([0.0, 0.0, 1.0], points),
        ],
        axis=1,
    )

    class Source:
        trial_count = 3
        point_count = points

        @staticmethod
        def iter_raw_chunks():
            yield 0, points, trials.reshape((1, 1, 1) + trials.shape)

    materialized = prepare_transverse_bundle(
        state, lambda _: trials, rank_tolerance=1.0e-10
    )
    streaming = prepare_transverse_bundle(
        state, Source(), rank_tolerance=1.0e-10
    )

    assert streaming.result == materialized.result
    assert np.allclose(
        streaming.augmented_bundle.fields[0, 0, 0],
        materialized.augmented_bundle.fields[0, 0, 0],
        atol=1.0e-13,
    )
    assert np.allclose(
        streaming.projection_seed.matrices[0, 0, 0],
        materialized.projection_seed.matrices[0, 0, 0],
        atol=1.0e-13,
    )


def test_streaming_etbc_matches_materialized_regular_k_completion():
    state = _physical_state_for_gamma()
    state.config.k_points = [
        np.asarray([0.25]),
        np.asarray([0.0]),
        np.asarray([0.0]),
    ]
    state.E[0, 0, 0] = np.asarray([1.0, 2.0])
    state.energy_matrix = np.asarray([1.0, 2.0]).reshape(1, 1, 1, 2)
    points = state.mesh.point_count
    trials = np.stack(
        [
            _constant_field([1.0, 0.0, 0.0], points),
            _constant_field([0.0, 1.0, 0.0], points),
            _constant_field([0.0, 0.0, 1.0], points),
        ],
        axis=1,
    )

    class Source:
        trial_count = 3
        point_count = points

        @staticmethod
        def iter_raw_chunks():
            yield 0, points, trials.reshape((1, 1, 1) + trials.shape)

    materialized = prepare_transverse_bundle(
        state, lambda _: trials, rank_tolerance=1.0e-10
    )
    streaming = prepare_transverse_bundle(
        state, Source(), rank_tolerance=1.0e-10
    )

    assert np.allclose(
        streaming.augmented_bundle.fields[0, 0, 0],
        materialized.augmented_bundle.fields[0, 0, 0],
        atol=1.0e-13,
    )
    assert np.allclose(
        streaming.projection_seed.matrices[0, 0, 0],
        materialized.projection_seed.matrices[0, 0, 0],
        atol=1.0e-13,
    )
