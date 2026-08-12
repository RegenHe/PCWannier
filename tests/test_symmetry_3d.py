from __future__ import annotations

import numpy as np
import pytest

from pcwannier.symmetry import (
    BlochSymmetryAnalysisResult,
    BlochSymmetryPointAnalysis,
    DegenerateBlock,
    FieldKind,
    SewingDiagnostics,
    SpaceGroupOperation,
    build_symmetry_context,
    cartesian_field_matrix,
    little_group,
    load_point_group_from_spglib,
    load_symmetry_from_spglib,
    regularize_gamma_zero_modes,
)


def test_spglib_point_group_catalog_supports_hm_and_schoenflies_names():
    international = load_point_group_from_spglib("m-3m")
    schoenflies = load_point_group_from_spglib("O_h")

    assert international.name == schoenflies.name == "O_h"
    assert international.point_group_symbol == "m-3m"
    assert international.table.order == 48
    assert {irrep.name for irrep in international.irreps} == {
        "A1g",
        "A2g",
        "Eg",
        "T1g",
        "T2g",
        "A1u",
        "A2u",
        "Eu",
        "T1u",
        "T2u",
    }


def test_spglib_space_group_requires_explicit_hall_for_origin_choices():
    with pytest.raises(ValueError, match=r"hall:521.*hall:522"):
        load_symmetry_from_spglib("Pn-3m")

    choice_one = load_symmetry_from_spglib("hall:521")
    choice_two = load_symmetry_from_spglib("hall:522")
    assert any(
        not np.allclose(left.translation, right.translation)
        for left, right in zip(
            choice_one.group.operations,
            choice_two.group.operations,
            strict=True,
        )
    )


def test_pm3m_database_and_little_groups():
    model = load_symmetry_from_spglib("Pm-3m")
    definition = model.group_definition

    assert definition is not None
    assert definition.dimension == 3
    assert len(definition.group.operations) == 48
    assert definition.point_group.symbol == "m-3m"

    gamma_indices = tuple(
        item.operation_index for item in little_group(definition.group, [0.0, 0.0, 0.0])
    )
    x_indices = tuple(
        item.operation_index for item in little_group(definition.group, [0.5, 0.0, 0.0])
    )
    gamma = definition.resolve_little_group(gamma_indices, [0.0, 0.0, 0.0])
    xpoint = definition.resolve_little_group(x_indices, [0.5, 0.0, 0.0])

    assert gamma.name == "O_h"
    assert xpoint.name == "D4h"
    assert "T1g" in {irrep.name for irrep in gamma.irreps}
    assert "Eg" in {irrep.name for irrep in xpoint.irreps}


def test_fd3m_database_is_rebased_to_fcc_primitive_lattice():
    lattice = np.asarray(
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    )
    model = load_symmetry_from_spglib("hall:525", lattice_vectors=lattice)

    assert len(model.group.operations) == 48
    assert model.group_definition is not None
    assert model.group_definition.point_group.symbol == "m-3m"
    inverse_basis = np.linalg.inv(lattice.T)
    for operation in model.group.operations:
        cartesian = lattice.T @ operation.rotation @ inverse_basis
        assert np.linalg.norm(cartesian.T @ cartesian - np.eye(3)) < 1.0e-12

    axes = [np.asarray([-0.5, -0.25, 0.0, 0.25])] * 3
    context = build_symmetry_context(model, axes)
    assert len(context.k_mappings) == 48
    assert all(len(mappings) == 64 for mappings in context.k_mappings)


def test_fd3m_database_keeps_conventional_centering_translations():
    model = load_symmetry_from_spglib("hall:525", lattice_vectors=np.eye(3))

    assert len(model.group.operations) == 192


def test_fd3m_primitive_wyckoff_uses_conventional_multiplicity():
    lattice = np.asarray(
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    )
    model = load_symmetry_from_spglib("hall:526", lattice_vectors=lattice)

    assert model.group_definition.identify_wyckoff(
        [0.125, 0.125, 0.125], lattice
    ) == (8, "a")
    assert len(
        {
            tuple(np.round(operation.act_real([0.125, 0.125, 0.125]) % 1.0, 12))
            for operation in model.group.operations
        }
    ) == 2


def test_three_dimensional_axial_vector_inversion_and_mirror():
    inversion = SpaceGroupOperation(-np.eye(3, dtype=int), np.zeros(3))
    mirror_z = SpaceGroupOperation(np.diag([1, 1, -1]), np.zeros(3))

    assert np.array_equal(
        cartesian_field_matrix(
            inversion, np.eye(3), FieldKind.MAGNETIC_AXIAL_VECTOR
        ),
        np.eye(3),
    )
    assert np.array_equal(
        cartesian_field_matrix(
            mirror_z, np.eye(3), FieldKind.MAGNETIC_AXIAL_VECTOR
        ),
        np.diag([-1, -1, 1]),
    )
    assert np.array_equal(
        cartesian_field_matrix(mirror_z, np.eye(3), FieldKind.PSEUDOSCALAR),
        [[-1.0]],
    )


def test_gamma_constant_axial_transverse_plus_longitudinal_is_t1g():
    model = load_symmetry_from_spglib("Pm-3m")
    context = build_symmetry_context(
        model,
        [np.array([0.0]), np.array([0.0]), np.array([0.0])],
    )
    operation_indices = tuple(range(len(model.group.operations)))
    resolved = model.group_definition.resolve_little_group(
        operation_indices,
        [0.0, 0.0, 0.0],
    )
    block = DegenerateBlock((0, 1), (0.0j, 0.0j), {}, 0.0)
    point = BlochSymmetryPointAnalysis(
        "Gamma",
        np.zeros(3),
        np.zeros(3),
        (0, 0, 0),
        (0, 1),
        (0, 1),
        operation_indices,
        {},
        {},
        SewingDiagnostics(0.0, 0.0, 0.0),
        (block,),
        None,
        resolved_little_group=resolved,
    )

    updated, regularized = regularize_gamma_zero_modes(
        BlochSymmetryAnalysisResult((point,)),
        context,
        (0,),
        np.eye(3),
        energy_tolerance=1.0e-10,
    )

    assert regularized.decomposition is not None
    assert regularized.decomposition.multiplicities["T1g"] == 1
    assert regularized.unitarity_error < 1.0e-12
    assert regularized.twisted_composition_residual < 1.0e-12
    assert updated.point("Gamma").degenerate_blocks[0].decomposition is None
    assert "direction dependent" in updated.point("Gamma").degenerate_blocks[0].irrep_unavailable_reason
