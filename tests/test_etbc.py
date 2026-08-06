from types import SimpleNamespace

import numpy as np
import pytest

from pcwannier import BlochConvention
from pcwannier.conventions import BlochFieldRepresentation
from pcwannier.compute.state import StateCollection
from pcwannier.compute.uniform_grid import UniformGridInnerProduct
from pcwannier.compute.projector_interpolation import (
    ProjectorPreservingBandInterpolator,
)
from pcwannier.data import InputBundle, PeriodicGrid
from pcwannier.etbc import complete_transverse_bundle, construct_auxiliary_frame
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


def test_projector_interpolation_flattens_rank_and_pins_auxiliary_energy():
    interpolator = ProjectorPreservingBandInterpolator(
        2, 3, fixed_longitudinal_eigenvalue=-0.25
    )
    approximate = np.asarray(
        [
            [
                [0.93, 0.04 + 0.02j, 0.12],
                [0.04 - 0.02j, 0.82, -0.09j],
                [0.12, 0.09j, 0.18],
            ]
        ],
        dtype=np.complex128,
    )
    h_transverse = np.asarray(
        [[[2.0, 0.3, 0.4j], [0.3, 4.0, -0.2], [-0.4j, -0.2, 0.7]]],
        dtype=np.complex128,
    )

    flattened, _ = interpolator.flatten_projectors(approximate)
    output, diagnostics = interpolator.constrain_hamiltonians(
        h_transverse, np.zeros_like(h_transverse), approximate
    )
    complement = np.eye(3) - flattened[0]

    assert np.allclose(flattened[0] @ flattened[0], flattened[0], atol=1.0e-13)
    assert np.trace(flattened[0]).real == pytest.approx(2.0)
    assert np.linalg.norm(flattened[0] @ output[0] @ complement) < 1.0e-13
    assert np.allclose(
        complement @ output[0] @ complement,
        -0.25 * complement,
        atol=1.0e-13,
    )
    assert diagnostics.minimum_projector_gap > 0.0
    assert diagnostics.maximum_raw_projector_idempotency_error > 0.0
    assert diagnostics.maximum_flattened_projector_idempotency_error < 1.0e-13


def test_projector_uses_metric_output_coefficients_and_source_selector():
    interpolator = ProjectorPreservingBandInterpolator(2, 3)
    overlap = np.asarray(
        [[1.2, 0.1j, 0.05], [-0.1j, 0.9, 0.03j], [0.05, -0.03j, 1.1]],
        dtype=np.complex128,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    coefficients = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.conj().T

    projector = interpolator.projector_in_wannier_basis(
        overlap, coefficients, np.asarray([0, 1])
    )

    assert np.allclose(projector, projector.conj().T, atol=1.0e-13)
    assert np.allclose(projector @ projector, projector, atol=1.0e-13)
    assert np.trace(projector).real == pytest.approx(2.0)


def test_file_longitudinal_hamiltonian_is_restricted_not_pinned():
    interpolator = ProjectorPreservingBandInterpolator(2, 3)
    approximate = np.asarray(
        [[[0.9, 0.1, 0.2], [0.1, 0.85, -0.1j], [0.2, 0.1j, 0.25]]],
        dtype=np.complex128,
    )
    h_transverse = np.asarray(
        [[[2.0, 0.3, 0.4], [0.3, 4.0, 0.2j], [0.4, -0.2j, 0.5]]],
        dtype=np.complex128,
    )
    h_longitudinal = np.asarray(
        [[[0.4, -0.1j, 0.6], [0.1j, 0.2, -0.3], [0.6, -0.3, 7.0]]],
        dtype=np.complex128,
    )

    projector, _ = interpolator.flatten_projectors(approximate)
    output, diagnostics = interpolator.constrain_hamiltonians(
        h_transverse, h_longitudinal, approximate
    )
    complement = np.eye(3) - projector[0]

    assert np.allclose(
        complement @ output[0] @ complement,
        complement @ h_longitudinal[0] @ complement,
        atol=1.0e-13,
    )
    assert np.linalg.norm(projector[0] @ output[0] @ complement) < 1.0e-13
    assert diagnostics.maximum_removed_cross_sector_component > 0.0


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

    result = complete_transverse_bundle(
        state,
        lambda _: trials,
        auxiliary_eigenvalue=0.0,
        rank_tolerance=1.0e-10,
    )

    assert result.gamma_regularized_indices == ((0, 0, 0),)
    assert result.auxiliary_dimension == 1
    block = np.asarray(result.augmented_bundle.fields[0, 0, 0])
    assert block.shape == (3, points, 3)
    assert np.allclose(
        np.linalg.eigvalsh(result.augmented_bundle.base_hamiltonians[0, 0, 0]),
        0.0,
    )
    expected_projection = state.inner_product.overlap(
        block,
        np.swapaxes(trials, 0, 1),
        chunk_size=64,
    )
    assert np.allclose(result.trial_projection_matrices[0, 0, 0], expected_projection)


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

    result = complete_transverse_bundle(
        state,
        lambda _: trials,
        auxiliary_eigenvalue=0.0,
        rank_tolerance=1.0e-10,
    )

    assert result.auxiliary_dimension == 2
    h0 = result.augmented_bundle.base_hamiltonians[0, 0, 0]
    assert np.allclose(np.linalg.eigvalsh(h0), [0.0, 0.0, 0.0, 0.0, 2.0])
    assert result.maximum_augmented_orthonormality_error < 1.0e-12
