from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from pcwannier import load_config
from pcwannier.symmetry import (
    ConcreteFiniteGroup,
    DegeneracyTolerance,
    FieldKind,
    RepresentationAnalysisSpec,
    RepresentationPointSpec,
    SymmetryCalculationSpec,
    WannierTargetSpec,
    compose_symmetry_model,
    identify_finite_group,
    load_builtin_finite_groups,
    load_finite_group,
    load_space_group,
    resolve_little_group,
    resolve_symmetry_file,
)

from .symmetry_models import P4GM, P4MM, p4mm_model


FINITE_GROUPS = Path("pcwannier/symmetry/finite_groups")


def test_c4v_finite_group_generates_classes_and_irreps():
    definition = load_finite_group(FINITE_GROUPS / "C4v.yaml")
    classes = {
        frozenset(definition.table.element_names[index] for index in item.element_indices)
        for item in definition.table.conjugacy_classes
    }

    assert classes == {
        frozenset({"E"}),
        frozenset({"C4", "C4_inv"}),
        frozenset({"C2"}),
        frozenset({"sigma_x", "sigma_y"}),
        frozenset({"sigma_d", "sigma_d2"}),
    }
    assert {irrep.name for irrep in definition.irreps} == {"A1", "A2", "B1", "B2", "E"}
    e_irrep = definition.irrep("E")
    assert len(e_irrep.matrices) == 8
    assert np.trace(e_irrep.matrix(definition.table.element_index("C4"))) == pytest.approx(0.0)


def test_p4mm_little_groups_are_resolved_automatically():
    definition = load_space_group(P4MM)
    expected = {
        "Gamma": ([0.0, 0.0], "C4v", {"A1", "A2", "B1", "B2", "E"}),
        "X": ([0.5, 0.0], "C2v", {"A1", "A2", "B1", "B2"}),
        "M": ([0.5, 0.5], "C4v", {"A1", "A2", "B1", "B2", "E"}),
    }

    for _, (kpoint, group_name, irrep_names) in expected.items():
        resolved = resolve_little_group(definition, kpoint)
        assert resolved.name == group_name
        assert {irrep.name for irrep in resolved.require_irreps()} == irrep_names
        assert resolved.factor_system.is_trivial


def test_c2v_embeddings_keep_distinct_and_stable_element_mappings():
    definition = load_space_group(P4MM)
    group = definition.group
    library = load_builtin_finite_groups()

    def identify(names):
        indices = tuple(group.operation_index(group.operation_by_name(name)) for name in names)
        concrete = ConcreteFiniteGroup.from_space_group(group, indices)
        return identify_finite_group(concrete, library)

    axes = identify(("E", "C2", "sigma_x", "sigma_y"))
    diagonal = identify(("E", "C2", "sigma_d", "sigma_d2"))
    diagonal_reordered = identify(("sigma_d2", "C2", "E", "sigma_d"))

    axes_mapping = {
        group.operations[index].name: axes.canonical_name_for_operation(index)
        for index in axes.concrete.operation_indices
    }
    diagonal_mapping = {
        group.operations[index].name: diagonal.canonical_name_for_operation(index)
        for index in diagonal.concrete.operation_indices
    }
    reordered_mapping = {
        group.operations[index].name: diagonal_reordered.canonical_name_for_operation(index)
        for index in diagonal_reordered.concrete.operation_indices
    }

    assert axes.canonical.name == diagonal.canonical.name == "C2v"
    assert axes.mapping_method == "point_action"
    assert diagonal.mapping_method == "point_action_invariants"
    assert axes_mapping != diagonal_mapping
    assert diagonal_mapping == reordered_mapping


def test_incar_owns_targets_analysis_and_uses_space_group_fallback(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(_minimal_incar(), encoding="utf-8")

    config = load_config(incar)
    model = config.symmetry_context.model

    assert config.symmetry_resolved_path == resolve_symmetry_file("p4mm.yaml", tmp_path)
    assert [(target.name, target.wannier_dimension) for target in model.targets] == [
        ("center_s_A1", 1),
        ("center_p_E", 2),
    ]
    gamma, xpoint = model.representation_analysis.points
    assert gamma.band_indices == (0, 1, 2)
    assert gamma.target_names is None
    assert xpoint.target_names == ("center_s_A1", "center_p_E")
    assert model.symmetry_gauge.enabled
    assert model.symmetry_gauge.tolerance == pytest.approx(1.0e-8)
    assert model.symmetry_gauge.max_iterations == 20


def test_existing_custom_space_group_takes_priority_over_builtin_name(tmp_path):
    raw = yaml.safe_load(P4MM.read_text(encoding="utf-8"))
    raw["name"] = "CustomP4mm"
    custom = tmp_path / "p4mm.yaml"
    custom.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    resolved = resolve_symmetry_file("p4mm.yaml", tmp_path)

    assert resolved == custom.resolve()
    assert load_space_group(resolved).name == "CustomP4mm"


@pytest.mark.parametrize(
    "forbidden",
    ("irreps", "subgroups", "wannier_targets", "representation_analysis"),
)
def test_space_group_schema_rejects_mixed_configuration(tmp_path, forbidden):
    raw = yaml.safe_load(P4MM.read_text(encoding="utf-8"))
    raw[forbidden] = {}
    path = tmp_path / "mixed.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown|forbidden|keys"):
        load_space_group(path)


def test_invalid_finite_group_irrep_and_unknown_target_are_rejected(tmp_path):
    raw = yaml.safe_load((FINITE_GROUPS / "C4v.yaml").read_text(encoding="utf-8"))
    raw["irreps"]["E"]["generators"]["C4"] = [[1, 0], [0, 1]]
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="relation|multiplication|character"):
        load_finite_group(invalid)

    missing = yaml.safe_load((FINITE_GROUPS / "C4v.yaml").read_text(encoding="utf-8"))
    missing["irreps"]["A1"]["characters"].pop("sigma_d2")
    missing_path = tmp_path / "missing-character.yaml"
    missing_path.write_text(yaml.safe_dump(missing, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly|names|elements"):
        load_finite_group(missing_path)

    base = p4mm_model()
    analysis = RepresentationAnalysisSpec(
        FieldKind.SCALAR,
        DegeneracyTolerance(),
        (RepresentationPointSpec("Gamma", np.zeros(2), (0,), ("missing",)),),
    )
    with pytest.raises(ValueError, match="unknown Wannier targets"):
        compose_symmetry_model(base, SymmetryCalculationSpec((), analysis))


def test_finite_group_can_use_explicit_multiplication_without_point_actions(tmp_path):
    path = tmp_path / "abstract-c2.yaml"
    path.write_text(
        """
name: AbstractC2
dimension: 2
elements:
  - name: E
  - name: a
multiplication:
  E: [E, a]
  a: [a, E]
irreps:
  plus:
    dimension: 1
    characters: {E: 1, a: 1}
  minus:
    dimension: 1
    characters: {E: 1, a: -1}
""".strip(),
        encoding="utf-8",
    )

    definition = load_finite_group(path)

    assert definition.point_actions is None
    assert definition.table.element_orders == (1, 2)
    assert {irrep.name for irrep in definition.irreps} == {"plus", "minus"}


def test_site_groups_are_identified_from_center_geometry():
    model = p4mm_model(
        targets=(
            WannierTargetSpec("generic", [0.13, 0.27], "A"),
            WannierTargetSpec("mirror", [0.0, 0.23], "A_prime"),
            WannierTargetSpec("diagonal_mirror", [0.23, 0.23], "A_prime"),
            WannierTargetSpec("edge", [0.5, 0.0], "A1"),
            WannierTargetSpec("origin", [0.0, 0.0], "A1"),
        )
    )

    assert [target.site_irrep.finite_group_name for target in model.targets] == [
        "C1",
        "Cs",
        "Cs",
        "C2v",
        "C4v",
    ]


def test_p4g_factor_system_reports_projective_points_without_fallback():
    definition = load_space_group(P4GM)
    gamma = resolve_little_group(definition, [0.0, 0.0])
    xpoint = resolve_little_group(definition, [0.5, 0.0])
    mpoint = resolve_little_group(definition, [0.5, 0.5])

    assert gamma.factor_system.phase_residual == pytest.approx(0.0)
    assert gamma.factor_system.is_trivial
    assert gamma.factor_system.raw_trivial
    assert gamma.factor_system.cohomologically_trivial
    assert xpoint.factor_system.phase_residual > 1.0
    assert not xpoint.factor_system.is_trivial
    assert not xpoint.factor_system.raw_trivial
    assert not xpoint.factor_system.cohomologically_trivial
    with pytest.raises(NotImplementedError, match="projective irreps"):
        xpoint.require_irreps()

    # The chosen p4g representatives have a non-unit raw factor at M, but it is
    # removed by a one-cochain. Ordinary irreps are therefore valid after rephasing.
    assert mpoint.factor_system.phase_residual > 1.0
    assert mpoint.factor_system.is_trivial
    assert not mpoint.factor_system.raw_trivial
    assert mpoint.factor_system.cohomologically_trivial
    assert mpoint.factor_system.trivializing_cochain is not None


def _minimal_incar() -> str:
    return "\n".join(
        [
            "lattice_const = 1",
            "real_lattice_vectors = 1 0, 0 1",
            "k_points = -0.5:0.5:0.5, -0.5:0.5:0.5",
            "composition_of_b = 1 0, 0 1",
            "band_window = 0:3",
            "dataset_file = Ez.txt",
            "dielectric_file = eps.txt",
            "mesh_file = mesh.mphtxt",
            "E_file = E.txt",
            "extension = 1, 1",
            "symmetry_file = missing/path/p4mm.yaml",
            "symmetry_constrained = true",
            "symmetry_tolerance = 1e-8",
            "symmetry_max_iter = 20",
            "wannier_targets",
            "center_s_A1; 0.0, 0.0; A1",
            "center_p_E; 0.0, 0.0; E",
            "end",
            "representation_analysis",
            "Gamma; 0.0, 0.0; 0:3",
            "X; 0.5, 0.0; 0,1,2; center_s_A1, center_p_E",
            "end",
            "projections",
            "a; [0, 0]; 0; [1, 0, 1]; [2, 1, 1]; [2, -1, 1]",
            "end",
        ]
    )
