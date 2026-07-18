from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pcwannier import load_config
from pcwannier.maxwell import MaxwellProblem
from pcwannier.symmetry import (
    SymmetryCalculationSpec,
    WannierTargetSpec,
    apply_magnetic_bias_to_model,
    build_symmetry_context,
    build_symmetry_stars,
    build_twisted_representation,
    combined_target_matrix,
    compose_symmetry_model,
    load_symmetry,
    project_intertwiner,
    propagate_target_gauge,
    resolve_symmetry_file,
    solve_intertwiner_space,
)


def _magnetic_p4mm_model():
    path = resolve_symmetry_file("p4mm.yaml", Path.cwd())
    model = apply_magnetic_bias_to_model(
        load_symmetry(path),
        [[1.0, 0.0], [0.0, 1.0]],
        [0.0, 0.0, 1.0],
    )
    return compose_symmetry_model(
        model,
        SymmetryCalculationSpec(
            target_specs=(
                WannierTargetSpec("center_p_E", [0.0, 0.0], "E"),
                WannierTargetSpec("center_s_A1", [0.0, 0.0], "A1"),
            )
        ),
    )


def test_p4mm_z_bias_classifies_rotations_and_mirrors():
    model = _magnetic_p4mm_model()
    flags = {operation.name: operation.antiunitary for operation in model.group.operations}

    assert flags == {
        "E": False,
        "C4": False,
        "C2": False,
        "C4_inv": False,
        "sigma_x": True,
        "sigma_y": True,
        "sigma_d": True,
        "sigma_d2": True,
    }
    assert np.allclose(model.magnetic_bias_direction, [0.0, 0.0, 1.0])


def test_antiunitary_operation_maps_k_to_minus_rotated_k():
    model = _magnetic_p4mm_model()
    mirror = model.group.operation_by_name("sigma_x")
    kpoint = np.array([0.2, 0.3])

    assert np.allclose(mirror.act_reciprocal_spatial(kpoint), [-0.2, 0.3])
    assert np.allclose(mirror.act_reciprocal(kpoint), [0.2, -0.3])


def test_ez_and_hz_time_reversal_have_maxwell_signs():
    values = np.array([1.0 + 2.0j, -3.0 + 4.0j])

    assert np.allclose(MaxwellProblem.for_components("Ez").apply_time_reversal(values), values.conj())
    assert np.allclose(MaxwellProblem.for_components("Hz").apply_time_reversal(values), -values.conj())


def test_magnetic_target_and_equivalent_physical_corepresentation_intertwine():
    model = _magnetic_p4mm_model()
    operation_indices = tuple(range(len(model.group.operations)))
    resolved = model.group_definition.resolve_little_group(operation_indices, [0.0, 0.0])
    target_matrices = tuple(
        combined_target_matrix(model.targets, index, [0.0, 0.0])
        for index in operation_indices
    )
    target = build_twisted_representation(resolved, operation_indices, target_matrices)
    rng = np.random.default_rng(20260717)
    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    basis_change, _ = np.linalg.qr(raw)
    physical_matrices = tuple(
        (
            basis_change @ matrix @ basis_change.T
            if operation.antiunitary
            else basis_change @ matrix @ basis_change.conj().T
        )
        for operation, matrix in zip(model.group.operations, target_matrices)
    )
    physical = build_twisted_representation(resolved, operation_indices, physical_matrices)

    space = solve_intertwiner_space(physical, target)
    projected = project_intertwiner(basis_change, physical, target)

    assert space.scalar_field == "real"
    assert space.dimension > 0
    assert projected.residual < 1.0e-10
    assert projected.semiunitarity_error < 1.0e-10


def test_antiunitary_projection_recovers_a_phase_odd_trial_column():
    model = _magnetic_p4mm_model()
    operation_indices = (0, 5)  # E and antiunitary sigma_y on the k_x=1/2 line.
    kpoint = np.array([0.5, 0.2])
    resolved = model.group_definition.resolve_little_group(operation_indices, kpoint)
    identity_physical = np.eye(4, dtype=np.complex128)
    physical = build_twisted_representation(
        resolved,
        operation_indices,
        (identity_physical, identity_physical),
    )
    target = build_twisted_representation(
        resolved,
        operation_indices,
        tuple(
            combined_target_matrix(model.targets, index, kpoint)
            for index in operation_indices
        ),
    )
    initial = np.eye(4, 3, dtype=np.complex128)

    projected = project_intertwiner(initial, physical, target)

    assert projected.residual < 1.0e-12
    assert projected.semiunitarity_error < 1.0e-12
    assert np.allclose(np.abs(projected.matrix), initial)


def test_antiunitary_star_propagation_satisfies_gauge_relation():
    model = _magnetic_p4mm_model()
    axis = np.array([-0.25, 0.25])
    context = build_symmetry_context(model, [axis, axis])
    stars = build_symmetry_stars(context)
    representatives = [np.eye(3, dtype=np.complex128) for _ in stars.stars]

    propagated = propagate_target_gauge(representatives, context, stars)

    assert propagated.max_path_consistency < 1.0e-12
    for operation_index, operation in enumerate(model.group.operations):
        for mapping in context.k_mappings[operation_index]:
            source = tuple(mapping.source_k_index) + (0,)
            target = tuple(mapping.target_k_index) + (0,)
            source_k = np.asarray(
                [context.k_points[axis_index][mapping.source_k_index[axis_index]] for axis_index in range(2)]
            )
            dmat = combined_target_matrix(model.targets, operation_index, source_k)
            source_gauge = propagated.gauge[source]
            transformed = source_gauge.conj() if operation.antiunitary else source_gauge
            residual = np.linalg.norm(
                dmat @ transformed - propagated.gauge[target] @ dmat,
                ord="fro",
            )
            assert residual < 1.0e-12


def test_magnetic_config_does_not_change_default_when_omitted(tmp_path):
    source = "\n".join(
        [
            "lattice_const = 1",
            "real_lattice_vectors = 1 0, 0 1",
            "k_points = -0.5:0.5:0.5, -0.5:0.5:0.5",
            "composition_of_b = 1 0, 0 1",
            "band_window = 0:3",
            "dataset_file = Ez.txt",
            "metric_file = eps.txt",
            "mesh_file = mesh.mphtxt",
            "E_file = E.txt",
            "extension = 1, 1",
            "symmetry_file = p4mm.yaml",
            "projections",
            "a; [0, 0]; 0; [2, 1, 1]; [2, -1, 1]; [2, 0, 1]",
            "end",
        ]
    )
    ordinary_path = tmp_path / "ordinary.incar"
    magnetic_path = tmp_path / "magnetic.incar"
    ordinary_path.write_text(source, encoding="utf-8")
    magnetic_path.write_text(
        source + "\nmagnetic_bias_direction = 0, 0, 1\n",
        encoding="utf-8",
    )

    ordinary = load_config(ordinary_path)
    magnetic = load_config(magnetic_path)

    assert ordinary.symmetry_context.model.magnetic_bias_direction is None
    assert not any(op.antiunitary for op in ordinary.symmetry_context.model.group.operations)
    assert sum(op.antiunitary for op in magnetic.symmetry_context.model.group.operations) == 4
    assert ordinary.band_calc_num == magnetic.band_calc_num == 3


def test_zero_magnetic_bias_is_rejected(tmp_path):
    path = tmp_path / "incar"
    path.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0, 0 1",
                "k_points = 0:1:1, 0:1:1",
                "composition_of_b = 1 0, 0 1",
                "band_window = 0:1",
                "dataset_file = Ez.txt",
                "metric_file = eps.txt",
                "mesh_file = mesh.mphtxt",
                "E_file = E.txt",
                "extension = 1, 1",
                "symmetry_file = p4mm.yaml",
                "magnetic_bias_direction = 0, 0, 0",
                "projections",
                "a; [0, 0]; 0; [1, 0, 1]",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-zero"):
        load_config(path)
