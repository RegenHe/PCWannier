from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from pcwannier import load_config
from pcwannier.compute.integration import integrate_overlap_matrix, mesh_integral_view
from pcwannier.data import Mesh
from pcwannier.symmetry import (
    DegeneracyTolerance,
    FieldKind,
    PCWannierDeprecationWarning,
    RepresentationAnalysisSpec,
    RepresentationPointSpec,
    SymmetryCalculationSpec,
    WannierTargetSpec,
    build_symmetry_context,
    compose_symmetry_model,
    decompose_little_group_characters,
    little_group,
    load_symmetry,
    load_symmetry_group,
    resolve_symmetry_file,
    run_symmetry_analysis,
)


LIBRARY = Path("pcwannier/symmetries/c4v.yaml")


def test_c4v_group_definition_generates_classes_and_irreps():
    definition = load_symmetry_group(LIBRARY)

    classes = {
        frozenset(definition.group.operations[index].name for index in item.operation_indices)
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
    e_irrep = next(irrep for irrep in definition.irreps if irrep.name == "E")
    assert e_irrep.matrices is not None and len(e_irrep.matrices) == 8
    assert e_irrep.character_for_global_index(
        definition.group.operation_index(definition.group.operation_by_name("C4"))
    ) == pytest.approx(0.0)


def test_little_groups_are_resolved_without_point_character_tables():
    definition = load_symmetry_group(LIBRARY)
    expected = {
        "Gamma": ([0.0, 0.0], "C4v", {"A1", "A2", "B1", "B2", "E"}),
        "X": ([0.5, 0.0], "C2v_axes", {"A1", "A2", "B1", "B2"}),
        "M": ([0.5, 0.5], "C4v", {"A1", "A2", "B1", "B2", "E"}),
    }
    for _, (kpoint, name, irrep_names) in expected.items():
        indices = tuple(element.operation_index for element in little_group(definition.group, kpoint))
        resolved = definition.resolve_little_group(indices, kpoint)
        assert resolved.name == name
        assert {irrep.name for irrep in resolved.irreps} == irrep_names


def test_incar_owns_targets_analysis_and_uses_builtin_fallback(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(_minimal_incar(), encoding="utf-8")

    config = load_config(incar)
    model = config.symmetry_context.model

    assert config.symmetry_resolved_path == resolve_symmetry_file("c4v.yaml", tmp_path)
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


def test_existing_custom_group_file_takes_priority_over_builtin_name(tmp_path):
    raw = yaml.safe_load(LIBRARY.read_text(encoding="utf-8"))
    raw["name"] = "CustomC4v"
    custom = tmp_path / "c4v.yaml"
    custom.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    resolved = resolve_symmetry_file("c4v.yaml", tmp_path)

    assert resolved == custom.resolve()
    assert load_symmetry_group(resolved).name == "CustomC4v"


def test_analysis_targets_are_optional_and_enable_only_compatibility():
    base = load_symmetry(LIBRARY)
    tolerance = DegeneracyTolerance()
    points = (
        RepresentationPointSpec("physical", np.zeros(2), (0,), None, tolerance),
        RepresentationPointSpec("compare", np.zeros(2), (0,), ("s",), tolerance),
    )
    model = compose_symmetry_model(
        base,
        SymmetryCalculationSpec(
            (WannierTargetSpec("s", np.zeros(2), "A1"),),
            RepresentationAnalysisSpec(FieldKind.SCALAR, tolerance, points),
        ),
    )
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    x = mesh.vertices[:, 0]
    y = mesh.vertices[:, 1]
    state = _synthetic_state(mesh, np.asarray([np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)]))

    physical, compared = run_symmetry_analysis(state, context).points

    assert physical.little_group_name == "C4v"
    assert physical.physical_decomposition.multiplicities["A1"] == 1
    assert physical.target_characters == {}
    assert physical.target_decomposition is None
    assert physical.compatibility is None
    assert compared.target_decomposition.multiplicities["A1"] == 1
    assert compared.compatibility.compatible


def test_square_2c_target_uses_automatic_gamma_x_m_tables():
    model = compose_symmetry_model(
        load_symmetry(LIBRARY),
        SymmetryCalculationSpec((WannierTargetSpec("two_c", [0.5, 0.0], "A1"),)),
    )
    target = model.targets[0]
    expected = {
        "Gamma": ([0.0, 0.0], {"A1": 1, "B1": 1}),
        "X": ([0.5, 0.0], {"A1": 1, "B2": 1}),
        "M": ([0.5, 0.5], {"E": 1}),
    }
    for _, (kpoint, nonzero_expected) in expected.items():
        indices = tuple(element.operation_index for element in little_group(model.group, kpoint))
        resolved = model.group_definition.resolve_little_group(indices, kpoint)
        characters = {
            model.group.operations[index].name: complex(np.trace(target.matrix(index, kpoint)))
            for index in indices
        }
        decomposition = decompose_little_group_characters(resolved, characters)
        nonzero = {name: value for name, value in decomposition.multiplicities.items() if value}
        assert nonzero == nonzero_expected


def test_legacy_mixed_yaml_warns_and_keeps_existing_target_result():
    with pytest.warns(PCWannierDeprecationWarning):
        legacy = load_symmetry("tests/data/square_c4v_analysis.sym.yaml")
    modern = compose_symmetry_model(
        load_symmetry(LIBRARY),
        SymmetryCalculationSpec((WannierTargetSpec("square_2c_A1", [0.5, 0.0], "A1"),)),
    )
    c4_index = modern.group.operation_index(modern.group.operation_by_name("C4"))
    kpoint = np.array([0.13, 0.27])
    assert np.allclose(
        legacy.target("square_2c_A1").matrix(c4_index, kpoint),
        modern.target("square_2c_A1").matrix(c4_index, kpoint),
    )


def test_partly_migrated_group_yaml_warns_but_uses_automatic_analysis(tmp_path):
    raw = yaml.safe_load(LIBRARY.read_text(encoding="utf-8"))
    raw["representation_analysis"] = {
        "points": [
            {
                "name": "Gamma",
                "k": [0.0, 0.0],
                "bands": [0],
                "conjugacy_classes": {
                    "E": ["E"],
                    "C4": ["C4", "C4_inv"],
                    "C2": ["C2"],
                    "sigma_v": ["sigma_x", "sigma_y"],
                    "sigma_d": ["sigma_d", "sigma_d2"],
                },
                "irreps": {
                    "A1": {
                        "class_characters": {
                            "E": 1,
                            "C4": 1,
                            "C2": 1,
                            "sigma_v": 1,
                            "sigma_d": 1,
                        }
                    }
                },
            }
        ]
    }
    path = tmp_path / "mixed.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(PCWannierDeprecationWarning):
        model = load_symmetry(path)

    point = model.representation_analysis.points[0]
    indices = tuple(element.operation_index for element in little_group(model.group, point.k_fractional))
    resolved = model.group_definition.resolve_little_group(indices, point.k_fractional)
    assert resolved.name == "C4v"
    assert {irrep.name for irrep in resolved.irreps} == {"A1", "A2", "B1", "B2", "E"}


def test_invalid_group_irrep_generator_and_unknown_target_are_rejected(tmp_path):
    raw = yaml.safe_load(LIBRARY.read_text(encoding="utf-8"))
    raw["irreps"]["E"]["generators"]["C4"] = [[1, 0], [0, 1]]
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="group relation|multiplication|character"):
        load_symmetry_group(invalid)

    missing_character_raw = yaml.safe_load(LIBRARY.read_text(encoding="utf-8"))
    missing_character_raw["irreps"]["A1"]["characters"].pop("sigma_d2")
    missing_character = tmp_path / "missing-character.yaml"
    missing_character.write_text(
        yaml.safe_dump(missing_character_raw, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cover every operation"):
        load_symmetry_group(missing_character)

    duplicate_operation_raw = yaml.safe_load(LIBRARY.read_text(encoding="utf-8"))
    duplicate_operation_raw["operations"][1]["name"] = "E"
    duplicate_operation = tmp_path / "duplicate-operation.yaml"
    duplicate_operation.write_text(
        yaml.safe_dump(duplicate_operation_raw, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="names must be unique"):
        load_symmetry_group(duplicate_operation)

    base = load_symmetry(LIBRARY)
    analysis = RepresentationAnalysisSpec(
        FieldKind.SCALAR,
        DegeneracyTolerance(),
        (RepresentationPointSpec("Gamma", np.zeros(2), (0,), ("missing",)),),
    )
    with pytest.raises(ValueError, match="unknown Wannier targets"):
        compose_symmetry_model(base, SymmetryCalculationSpec((), analysis))


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
            "symmetry_file = missing/path/c4v.yaml",
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


def _square_mesh(points_per_axis: int = 5) -> Mesh:
    axis = np.linspace(0.0, 1.0, points_per_axis)
    vertices = np.asarray([(x, y) for x in axis for y in axis])
    elements = []
    for i in range(points_per_axis - 1):
        for j in range(points_per_axis - 1):
            a = i * points_per_axis + j
            b = (i + 1) * points_per_axis + j
            c = i * points_per_axis + j + 1
            d = (i + 1) * points_per_axis + j + 1
            elements.extend(((a, b, d), (a, d, c)))
    return Mesh(vertices, np.asarray(elements, dtype=np.intp))


def _synthetic_state(mesh: Mesh, fields: np.ndarray):
    epsilon = np.ones(mesh.vertices.shape[0])
    gram = integrate_overlap_matrix(mesh, fields, fields, epsilon)
    values, vectors = np.linalg.eigh(gram)
    normalized = (fields.T @ (vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.conj().T)).T
    count = normalized.shape[0]
    blocks = np.empty((1, 1, 1), dtype=object)
    blocks[0, 0, 0] = normalized
    band_ids = np.empty((1, 1, 1), dtype=object)
    band_ids[0, 0, 0] = list(range(count))
    transforms = np.empty((1, 1, 1), dtype=object)
    transforms[0, 0, 0] = np.eye(count, dtype=np.complex128)
    config = SimpleNamespace(
        real_lattice_vectors=[[1.0, 0.0], [0.0, 1.0]],
        lattice_const=1.0,
        dataset_type="comsol",
    )
    return SimpleNamespace(
        is_bloch=True,
        config=config,
        mesh=mesh,
        fields=blocks,
        E_idx=band_ids,
        energy_matrix=np.arange(count, dtype=float).reshape(1, 1, 1, count),
        epsilon=epsilon,
        integral_view=mesh_integral_view(mesh),
        compute_backend="python",
        integration_mode="nodal",
        get_block=lambda i, j, k: blocks[i, j, k],
        get_transform=lambda: transforms,
    )
