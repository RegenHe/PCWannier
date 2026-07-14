from __future__ import annotations

import numpy as np
import pytest

from pcwannier.symmetry import (
    build_symmetry_context,
    build_symmetry_stars,
    combined_target_matrix,
    little_group,
    project_intertwiner,
    solve_intertwiner_space,
)

from .symmetry_models import square_2c_model


def test_one_dimensional_intertwiner_is_identity():
    physical = [np.ones((1, 1)), -np.ones((1, 1))]
    target = [np.ones((1, 1)), -np.ones((1, 1))]
    space = solve_intertwiner_space(physical, target)
    result = project_intertwiner(np.ones((1, 1)), physical, target)

    assert space.dimension == 1
    assert np.allclose(result.matrix, [[1.0]])
    assert result.residual < 1e-12
    assert result.semiunitarity_error < 1e-12


def test_equivalent_randomly_rotated_representation_finds_intertwiner():
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    basis_change, _ = np.linalg.qr(raw)
    target = [
        np.eye(2, dtype=np.complex128),
        np.diag([1j, -1j]),
        -np.eye(2, dtype=np.complex128),
        np.diag([-1j, 1j]),
    ]
    physical = [basis_change @ matrix @ basis_change.conj().T for matrix in target]

    space = solve_intertwiner_space(physical, target)
    result = project_intertwiner(basis_change, physical, target)

    assert space.dimension == 2
    assert result.residual < 1e-12
    assert np.linalg.norm(result.matrix.conj().T @ result.matrix - np.eye(2)) < 1e-12
    assert all(
        np.linalg.norm(dmat @ result.matrix - result.matrix @ target_matrix) < 1e-12
        for dmat, target_matrix in zip(physical, target)
    )


def test_incompatible_irreps_have_zero_hom_dimension():
    physical = [np.ones((1, 1)), -np.ones((1, 1))]
    target = [np.ones((1, 1)), np.ones((1, 1))]
    space = solve_intertwiner_space(physical, target)

    assert space.dimension == 0
    with pytest.raises(RuntimeError, match="rank deficient"):
        project_intertwiner(np.ones((1, 1)), physical, target)


def test_semiunitary_intertwiner_for_larger_physical_space():
    target = [np.eye(2), np.diag([1.0, -1.0])]
    physical = [np.eye(3), np.diag([1.0, -1.0, -1.0])]
    initial = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    result = project_intertwiner(initial, physical, target)

    assert result.matrix.shape == (3, 2)
    assert result.residual < 1e-12
    assert np.linalg.norm(result.matrix.conj().T @ result.matrix - np.eye(2)) < 1e-12


def test_c4v_ten_by_ten_mesh_has_twenty_one_stars():
    model = square_2c_model()
    axis = np.arange(-0.5, 0.5, 0.1)
    context = build_symmetry_context(model, [axis, axis])
    partition = build_symmetry_stars(context)

    assert len(partition.stars) == 21
    assert sum(len(star.members) for star in partition.stars) == 100
    assert sorted({len(star.members) for star in partition.stars}) == [1, 2, 4, 8]
    assert np.all(partition.k_to_star >= 0)


def test_square_2c_target_has_direct_intertwiners_at_high_symmetry_points():
    model = square_2c_model()
    target = model.target("square_2c_A1")
    rng = np.random.default_rng(321)
    raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    basis_change, _ = np.linalg.qr(raw)

    for point in model.representation_analysis.points:
        operation_indices = tuple(
            element.operation_index for element in little_group(model.group, point.k_fractional)
        )
        target_matrices = [
            combined_target_matrix((target,), index, point.k_fractional)
            for index in operation_indices
        ]
        physical_matrices = [
            basis_change @ matrix @ basis_change.conj().T for matrix in target_matrices
        ]
        space = solve_intertwiner_space(physical_matrices, target_matrices)
        projected = project_intertwiner(basis_change, physical_matrices, target_matrices)

        assert space.dimension > 0
        assert projected.residual < 1e-12
