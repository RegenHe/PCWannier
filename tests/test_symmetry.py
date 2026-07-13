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


SQUARE_SYM = Path("tests/data/square_c4.sym.yaml")


def test_space_group_algebra_and_reciprocal_duality():
    c4 = SpaceGroupOperation([[0, -1], [1, 0]], [0.25, 0.5], "C4_tau")
    identity = c4 * c4.inverse()

    assert np.array_equal(identity.rotation, np.eye(2, dtype=int))
    assert np.allclose(identity.translation, 0.0)
    point = np.array([0.2, -0.3])
    assert np.allclose(c4.inverse().act_real(c4.act_real(point)), point)

    nonorthogonal_fractional = SpaceGroupOperation([[1, 1], [0, -1]], [0.0, 0.0])
    kpoint = np.array([0.17, -0.31])
    assert np.isclose(
        np.dot(nonorthogonal_fractional.act_reciprocal(kpoint), nonorthogonal_fractional.rotation @ point),
        np.dot(kpoint, point),
    )

    reduced = reduce_fractional(np.array([-0.5, 1.0 + 1e-10]), 1e-8)
    assert np.allclose(reduced.reduced, [0.5, 0.0])
    assert np.array_equal(reduced.lattice_shift, [-1, 1])


def test_square_c4_orbit_actions_and_target_representation():
    model = load_symmetry(SQUARE_SYM)
    target = model.target("square_2c_A1")
    c4_index = model.group.operation_index(model.group.operation_by_name("C4"))
    c2_index = model.group.operation_index(model.group.operation_by_name("C2"))

    assert target.multiplicity == 2
    assert target.wannier_dimension == 2
    assert len(target.orbit.site_symmetry.strict_elements) == 1
    assert len(target.orbit.site_symmetry.elements) == 2
    assert np.allclose(target.orbit.points[0].position, [0.5, 0.0])
    assert np.allclose(target.orbit.points[1].position, [0.0, 0.5])

    first_action = target.orbit.action(c4_index, 0)
    second_action = target.orbit.action(c4_index, 1)
    assert first_action.target_index == 1
    assert first_action.lattice_shift == (0, 0)
    assert second_action.target_index == 0
    assert second_action.lattice_shift == (-1, 0)
    assert np.array_equal(second_action.site_element.rotation, -np.eye(2, dtype=int))
    assert np.allclose(second_action.site_element.translation, [1.0, 0.0])

    kpoint = np.array([0.13, 0.27])
    expected = np.array([[0.0, np.exp(-2j * np.pi * kpoint[1])], [1.0, 0.0]])
    actual = target.matrix(c4_index, kpoint)
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert np.allclose(actual.conj().T @ actual, np.eye(2), rtol=0.0, atol=1e-12)

    transformed_k = model.group.operations[c4_index].act_reciprocal(kpoint)
    composed = target.matrix(c4_index, transformed_k) @ target.matrix(c4_index, kpoint)
    assert np.allclose(composed, target.matrix(c2_index, kpoint), rtol=0.0, atol=1e-12)


def test_named_site_generator_and_complex_matrix(tmp_path):
    operations = SQUARE_SYM.read_text(encoding="utf-8").split("wannier_targets:", 1)[0]
    symmetry_file = tmp_path / "complex-irrep.sym.yaml"
    symmetry_file.write_text(
        operations
        + """wannier_targets:
  - name: square_1a_complex
    center: [0.0, 0.0]
    site_irrep:
      name: C4_phase
      dimension: 1
      matrices:
        identity:
          - [1.0]
        generators:
          - operation: C4
            matrix:
              - [\"1j\"]
""",
        encoding="utf-8",
    )

    model = load_symmetry(symmetry_file)
    target = model.target("square_1a_complex")

    assert target.multiplicity == 1
    assert np.allclose(target.matrix(model.group.operation_by_name("C4"), [0.17, 0.23]), [[1j]])

    symmetry_file.write_text(symmetry_file.read_text(encoding="utf-8").replace('["1j"]', "[2.0]"), encoding="utf-8")
    with pytest.raises(ValueError, match="not unitary"):
        load_symmetry(symmetry_file)


def test_square_c4_k_mesh_mapping_and_closure_errors():
    model = load_symmetry(SQUARE_SYM)
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


def test_incar_loads_relative_symmetry_file(tmp_path):
    symmetry_path = tmp_path / "square.sym.yaml"
    symmetry_path.write_text(SQUARE_SYM.read_text(encoding="utf-8"), encoding="utf-8")
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0, 0 1",
                "reciprocal_lattice_vectors = 0 0, 0 0",
                "k_points = -0.5:0.25:0.5, -0.5:0.25:0.5",
                "composition_of_b = 1 0, 0 1",
                "band_window = 0:2",
                "dataset_file = Ez.txt",
                "dielectric_file = eps.txt",
                "mesh_file = mesh.mphtxt",
                "E_file = E.txt",
                "extension = 1, 1",
                "symmetry_file = square.sym.yaml",
                "projections",
                "a; [0.5, 0]; 0; [1, 0, 1]; [1, 0, 1]",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(incar)

    assert config.symmetry_file == "square.sym.yaml"
    assert config.symmetry_context is not None
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

    malformed = tmp_path / "wrong-dimension.sym.yaml"
    malformed.write_text(SQUARE_SYM.read_text(encoding="utf-8").replace("dimension: 2", "dimension: 3", 1), encoding="utf-8")
    with pytest.raises(ValueError, match=r"expected \(3, 3\)"):
        load_symmetry(malformed)


class _FakeSewingProvider:
    def __init__(self):
        self.requests = []

    def sewing_matrix(self, request):
        self.requests.append(request)
        if request.operation.name == "C2":
            return np.diag([1.0, -1.0])
        return np.eye(2)


def test_little_group_sewing_character_and_field_kinds():
    model = load_symmetry(SQUARE_SYM)
    provider = _FakeSewingProvider()

    analysis = analyze_little_group(
        model.group,
        [0.5, 0.0],
        [3, 4],
        provider,
        field_kind=FieldKind.SCALAR,
    )

    names = [model.group.operations[entry.element.operation_index].name for entry in analysis.entries]
    assert names == ["E", "C2"]
    assert np.allclose(analysis.characters, [2.0, 0.0])
    c2_request = next(request for request in provider.requests if request.operation.name == "C2")
    assert c2_request.reciprocal_lattice_shift == (-1, 0)
    assert c2_request.band_indices == (3, 4)

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
