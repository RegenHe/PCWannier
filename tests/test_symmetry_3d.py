from __future__ import annotations

import numpy as np

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


def test_gamma_constant_axial_t_plus_l_is_t1g():
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
