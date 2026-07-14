from pathlib import Path

import numpy as np
import pytest

from pcwannier import load_config, load_symmetry
from pcwannier.symmetry import (
    FieldKind,
    SpaceGroup,
    SpaceGroupOperation,
    analyze_little_group,
    build_symmetry_context,
    cartesian_field_matrix,
    reduce_fractional,
)
from tests.symmetry_models import P4MM, model_from_space_group, square_2c_model


def test_space_group_algebra_and_reciprocal_duality():
    c4 = SpaceGroupOperation([[0, -1], [1, 0]], [0.25, 0.5], "C4_tau")
    identity = c4 * c4.inverse()

    assert np.array_equal(identity.rotation, np.eye(2, dtype=int))
    assert np.allclose(identity.translation, 0.0)
    point = np.array([0.2, -0.3])
    assert np.allclose(c4.inverse().act_real(c4.act_real(point)), point)

    nonorthogonal = SpaceGroupOperation([[1, 1], [0, -1]], [0.0, 0.0])
    kpoint = np.array([0.17, -0.31])
    assert np.isclose(
        np.dot(nonorthogonal.act_reciprocal(kpoint), nonorthogonal.rotation @ point),
        np.dot(kpoint, point),
    )

    reduced = reduce_fractional(np.array([-0.5, 1.0 + 1e-10]), 1e-8)
    assert np.allclose(reduced.reduced, [0.5, 0.0])
    assert np.array_equal(reduced.lattice_shift, [-1, 1])


def test_square_c4_orbit_actions_and_target_representation():
    model = square_2c_model(analysis=False)
    target = model.target("square_2c_A1")
    c4_index = model.group.operation_index(model.group.operation_by_name("C4"))
    c2_index = model.group.operation_index(model.group.operation_by_name("C2"))

    assert target.multiplicity == 2
    assert target.wannier_dimension == 2
    assert target.site_irrep.finite_group_name == "C2v"
    assert len(target.orbit.site_symmetry.strict_elements) == 2
    assert len(target.orbit.site_symmetry.elements) == 4
    assert np.allclose(target.orbit.points[0].position, [0.5, 0.0])
    assert np.allclose(target.orbit.points[1].position, [0.0, 0.5])

    first_action = target.orbit.action(c4_index, 0)
    second_action = target.orbit.action(c4_index, 1)
    assert first_action.target_index == 1
    assert first_action.lattice_shift == (0, 0)
    assert second_action.target_index == 0
    assert second_action.lattice_shift == (-1, 0)

    kpoint = np.array([0.13, 0.27])
    expected = np.array([[0.0, np.exp(-2j * np.pi * kpoint[1])], [1.0, 0.0]])
    actual = target.matrix(c4_index, kpoint)
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-12)
    transformed_k = model.group.operations[c4_index].act_reciprocal(kpoint)
    composed = target.matrix(c4_index, transformed_k) @ target.matrix(c4_index, kpoint)
    assert np.allclose(composed, target.matrix(c2_index, kpoint), rtol=0.0, atol=1e-12)


def test_square_k_mesh_mapping_and_closure_errors():
    model = load_symmetry(P4MM)
    axes = [np.arange(-0.5, 0.5, 0.25), np.arange(-0.5, 0.5, 0.25)]
    context = build_symmetry_context(model, axes)

    for operation_mappings in context.k_mappings:
        assert len(operation_mappings) == 16
        for mapping in operation_mappings:
            source = np.array([axes[axis][mapping.source_k_index[axis]] for axis in range(2)])
            target = np.array([axes[axis][mapping.target_k_index[axis]] for axis in range(2)])
            transformed = model.group.operations[mapping.operation_index].act_reciprocal(source)
            assert np.allclose(transformed, target + np.asarray(mapping.reciprocal_lattice_shift))

    with pytest.raises(ValueError, match="not closed"):
        build_symmetry_context(model, [np.array([-0.5, 0.0]), np.array([-0.5, -0.25, 0.0, 0.25])])
    with pytest.raises(ValueError, match="periodically duplicate"):
        build_symmetry_context(model, [np.array([-0.5, 0.5]), np.array([-0.5, 0.0])])


def test_incar_loads_relative_space_group_file(tmp_path):
    symmetry_path = tmp_path / "p4mm.yaml"
    symmetry_path.write_text(P4MM.read_text(encoding="utf-8"), encoding="utf-8")
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0, 0 1",
                "k_points = -0.5:0.25:0.5, -0.5:0.25:0.5",
                "composition_of_b = 1 0, 0 1",
                "band_window = 0:2",
                "dataset_file = Ez.txt",
                "dielectric_file = eps.txt",
                "mesh_file = mesh.mphtxt",
                "E_file = E.txt",
                "extension = 1, 1",
                "symmetry_file = p4mm.yaml",
                "wannier_targets",
                "square_2c_A1; 0.5, 0.0; A1",
                "end",
                "projections",
                "a; [0.5, 0]; 0; [1, 0, 1]; [1, 0, 1]",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(incar)

    assert config.symmetry_context.model.group_definition.name == "p4mm"
    assert config.symmetry_context.model.target("square_2c_A1").wannier_dimension == 2


def test_old_boolean_symmetry_input_is_rejected(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text("symmetry = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="use symmetry_file"):
        load_config(incar)


def test_invalid_space_group_and_dimension_are_rejected(tmp_path):
    identity = SpaceGroupOperation.identity(2)
    c4 = SpaceGroupOperation([[0, -1], [1, 0]], [0.0, 0.0])
    with pytest.raises(ValueError, match="not an element"):
        SpaceGroup([identity, c4])
    with pytest.raises(ValueError, match="unimodular"):
        SpaceGroupOperation([[2, 0], [0, 1]], [0.0, 0.0])

    malformed = tmp_path / "wrong-dimension.yaml"
    malformed.write_text(P4MM.read_text(encoding="utf-8").replace("dimension: 2", "dimension: 3", 1), encoding="utf-8")
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        load_symmetry(malformed)


class _FakeSewingProvider:
    def __init__(self):
        self.requests = []

    def sewing_matrix(self, request):
        self.requests.append(request)
        return np.diag([1.0, -1.0]) if request.operation.name == "C2" else np.eye(2)


def test_little_group_sewing_character_and_field_kinds():
    operations = (
        SpaceGroupOperation.identity(2, "E"),
        SpaceGroupOperation([[0, -1], [1, 0]], [0.0, 0.0], "C4"),
        SpaceGroupOperation([[-1, 0], [0, -1]], [0.0, 0.0], "C2"),
        SpaceGroupOperation([[0, 1], [-1, 0]], [0.0, 0.0], "C4_inv"),
    )
    model = model_from_space_group("p4", SpaceGroup(operations))
    provider = _FakeSewingProvider()
    analysis = analyze_little_group(model.group, [0.5, 0.0], [3, 4], provider)

    names = [model.group.operations[entry.element.operation_index].name for entry in analysis.entries]
    assert names == ["E", "C2"]
    assert np.allclose(analysis.characters, [2.0, 0.0])
    c2_request = next(request for request in provider.requests if request.operation.name == "C2")
    assert c2_request.reciprocal_lattice_shift == (-1, 0)

    mirror = SpaceGroupOperation([[-1, 0], [0, 1]], [0.0, 0.0])
    assert np.allclose(cartesian_field_matrix(mirror, np.eye(2), FieldKind.SCALAR), [[1.0]])
    assert np.allclose(
        cartesian_field_matrix(mirror, np.eye(2), FieldKind.ELECTRIC_POLAR_VECTOR),
        [[-1.0, 0.0], [0.0, 1.0]],
    )
    assert np.allclose(
        cartesian_field_matrix(mirror, np.eye(2), FieldKind.MAGNETIC_AXIAL_VECTOR),
        [[1.0, 0.0], [0.0, -1.0]],
    )
