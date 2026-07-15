from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

from pcwannier import load_config, load_symmetry
from pcwannier.symmetry import (
    BlochConvention,
    BlochSymmetryAction,
    FieldKind,
    SpaceGroup,
    SpaceGroupOperation,
    SymmetryCalculationSpec,
    WannierTargetSpec,
    analyze_little_group,
    build_symmetry_context,
    build_symmetry_stars,
    cartesian_field_matrix,
    reduce_fractional,
    resolve_little_group,
    compose_symmetry_model,
    symmetrize_gradient,
)
from tests.symmetry_models import P4GM, P4MM, model_from_space_group, square_2c_model


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


def test_p4g_glide_uses_one_bloch_convention_for_field_target_and_factor():
    base = load_symmetry(P4GM)
    kpoint = np.array([0.0, 0.25])
    glide_index = base.group.operation_index(base.group.operation_by_name("glide_x"))
    identity_index = base.group.identity_index
    glide = base.group.operations[glide_index]
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.intp)
    constant = np.ones((1, 4), dtype=np.complex128)

    for sign in (1, -1):
        convention = BlochConvention(sign, f"test-{sign}")
        model = compose_symmetry_model(
            base,
            SymmetryCalculationSpec(
                target_specs=(WannierTargetSpec("origin_A", [0.0, 0.0], "A"),),
                bloch_convention=convention,
            ),
        )
        target = model.target("origin_A")
        resolved = model.group_definition.resolve_little_group(
            (identity_index, glide_index),
            kpoint,
            bloch_convention=convention,
        )
        glide_local = resolved.concrete.local_index(glide_index)
        factor = resolved.factor_system.phases[glide_local, glide_local]
        expected = np.exp(-sign * 2j * np.pi * kpoint[1])
        assert factor == pytest.approx(expected)
        assert resolved.factor_system.bloch_sign == sign

        transformed_k = glide.act_reciprocal(kpoint)
        composed = target.matrix(glide_index, transformed_k) @ target.matrix(
            glide_index, kpoint
        )
        assert np.allclose(
            composed,
            factor * target.matrix(identity_index, kpoint),
            atol=1e-12,
        )

        action = BlochSymmetryAction(
            vertices,
            elements,
            np.eye(2),
            bloch_sign=sign,
        )
        once = action.apply(constant, glide, kpoint, FieldKind.SCALAR)
        twice = action.apply(once, glide, transformed_k, FieldKind.SCALAR)
        assert np.allclose(twice, factor * constant, atol=1e-12)

        projective = resolve_little_group(
            base.group_definition,
            [0.5, 0.0],
            bloch_convention=convention,
        )
        assert not projective.factor_system.cohomologically_trivial
        with pytest.raises(NotImplementedError, match="projective irreps"):
            projective.require_irreps()


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


def test_k_mapping_rejects_ambiguous_targets_and_unphysical_tolerance():
    tolerance = 1.0e-3
    reflection = SpaceGroup(
        (
            SpaceGroupOperation(np.eye(2, dtype=int), np.zeros(2), "E"),
            SpaceGroupOperation([[-1, 0], [0, 1]], np.zeros(2), "mirror_x"),
        ),
        tolerance=tolerance,
    )
    model = model_from_space_group("Cs", reflection)
    xaxis = np.array([0.75, -0.75 - 0.75 * tolerance, -0.75 + 0.75 * tolerance])
    with pytest.raises(ValueError, match="Ambiguous symmetry k mapping"):
        build_symmetry_context(model, [xaxis, np.array([0.0])])

    with pytest.raises(ValueError, match="at least"):
        SpaceGroup(
            (SpaceGroupOperation(np.eye(2, dtype=int), np.zeros(2), "E"),),
            tolerance=np.finfo(float).eps,
        )


def test_constrained_gradient_pullback_matches_directional_finite_difference():
    model = square_2c_model(analysis=False)
    axes = [np.array([-0.25, 0.25]), np.array([-0.25, 0.25])]
    context = build_symmetry_context(model, axes)
    stars = build_symmetry_stars(context)
    assert len(stars.stars) == 1
    star = stars.stars[0]
    shape = (2, 2, 1)
    raw = np.empty(shape, dtype=object)
    rng = np.random.default_rng(2718)
    for index in np.ndindex(shape):
        matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        raw[index] = 0.5 * (matrix - matrix.conj().T)

    representative_gradient = symmetrize_gradient(raw, context, stars)[0]
    representative_k = np.asarray(
        [axes[axis][star.representative_index[axis]] for axis in range(2)]
    )
    target = model.targets[0]

    member_directions = {}
    for member in star.members:
        path = member.paths[0]
        dmat = target.matrix(path.operation_index, representative_k)
        member_directions[member.k_index] = (
            dmat @ representative_gradient @ dmat.conj().T
        )

    def objective(step):
        total = 0.0
        for member in star.members:
            state_index = tuple(member.k_index) + (0,)
            direction = member_directions[member.k_index]
            total += float(
                np.real(
                    np.trace(
                        raw[state_index].conj().T @ scipy.linalg.expm(step * direction)
                    )
                )
            )
        return total

    delta = 1.0e-6
    finite_difference = (objective(delta) - objective(-delta)) / (2.0 * delta)
    analytic = float(
        np.real(np.trace(representative_gradient.conj().T @ representative_gradient))
    )
    assert finite_difference == pytest.approx(analytic, rel=1e-7, abs=1e-9)


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
