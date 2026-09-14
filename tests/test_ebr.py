from __future__ import annotations

import json
from dataclasses import replace
from importlib import resources
from types import SimpleNamespace

import numpy as np
import pytest

from pcwannier.maxwell import FieldKind
from pcwannier.ebr.models import (
    BandSymmetryVector,
    EBRAnalysisResult,
    EBRCatalog,
    EBRDecomposition,
    EBRDefinition,
    EBRKPoint,
    EBRMatrix,
    EBRSearchStatistics,
    EBRSubspaceCandidate,
    EBRSubspacePointRepresentation,
    SymmetryVectorKey,
    TETBSolution,
)
from pcwannier.ebr.catalog import infer_builtin_catalog_alias, load_ebr_catalog
from pcwannier.ebr.analysis import (
    _approximate_ebr_diagnostics,
    _build_subspace_inventory,
    _gamma_zero_mode_dimension,
    _has_transverse_gamma_singularity,
    _match_subspace_representations,
    _resolve_ebr_mode,
    _subspace_gamma_sectors,
    _validate_subspace_fixed_bands,
    build_band_symmetry_vector,
    build_ebr_matrix,
)
from pcwannier.ebr.output import write_ebr_outputs
from pcwannier.ebr.solver import (
    EBRSearchLimitError,
    decompose_ebr,
    enumerate_ebr_subspace_solutions,
    enumerate_tetb_decompositions,
)
from pcwannier.symmetry.analysis import (
    BlochSymmetryAnalysisResult,
    BlochSymmetryPointAnalysis,
    SewingDiagnostics,
)
from pcwannier.symmetry.io import load_symmetry_from_spglib
from pcwannier.symmetry import load_symmetry
from pcwannier.symmetry.representation import build_symmetry_context


WALLPAPER_CATALOGS = (
    "p1",
    "p2",
    "pm",
    "pg",
    "c1m1",
    "p2mm",
    "p2mg",
    "p2gg",
    "c2mm",
    "p4",
    "p4mm",
    "p4gm",
    "p3",
    "p3m1",
    "p31m",
    "p6",
    "p6mm",
)

SHORT_WALLPAPER_ALIASES = {
    "cm": "c1m1",
    "pmm": "p2mm",
    "pmg": "p2mg",
    "pgg": "p2gg",
    "cmm": "c2mm",
    "p4m": "p4mm",
    "p4g": "p4gm",
    "p6m": "p6mm",
}


def _definition(name: str) -> EBRDefinition:
    return EBRDefinition(name, "1a", np.zeros(1), "A")


def _matrix(values, dimensions=None) -> EBRMatrix:
    raw = np.asarray(values, dtype=int)
    rows = tuple(
        SymmetryVectorKey("Gamma" if index == 0 else f"K{index}", "A")
        for index in range(raw.shape[0])
    )
    columns = tuple(_definition(chr(ord("a") + index)) for index in range(raw.shape[1]))
    dimensions = np.ones(raw.shape[1], dtype=int) if dimensions is None else dimensions
    return EBRMatrix(rows, columns, raw, np.asarray(dimensions, dtype=int))


def _vector(matrix: EBRMatrix, values, dimension: int) -> BandSymmetryVector:
    return BandSymmetryVector(matrix.row_keys, np.asarray(values, dtype=int), dimension)


@pytest.fixture(scope="module")
def sg221_matrix() -> EBRMatrix:
    model = load_symmetry_from_spglib("hall:517")
    context = build_symmetry_context(model, [np.asarray([-0.5, 0.0])] * 3)
    return build_ebr_matrix(
        context,
        load_ebr_catalog("sg221"),
        lattice_vectors=np.eye(3),
    )


@pytest.fixture(scope="module")
def sg213_matrix() -> EBRMatrix:
    model = load_symmetry_from_spglib("hall:509")
    context = build_symmetry_context(model, [np.asarray([-0.5, 0.0])] * 3)
    return build_ebr_matrix(
        context,
        load_ebr_catalog("sg213"),
        lattice_vectors=np.eye(3),
    )


@pytest.fixture(scope="module")
def sg224_matrix() -> EBRMatrix:
    model = load_symmetry_from_spglib("hall:522")
    context = build_symmetry_context(model, [np.asarray([-0.5, 0.0])] * 3)
    return build_ebr_matrix(
        context,
        load_ebr_catalog("224", hall_number=522),
        lattice_vectors=np.eye(3),
    )


@pytest.fixture(scope="module")
def sg227_matrix() -> EBRMatrix:
    lattice = np.asarray(
        (
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        )
    )
    model = load_symmetry_from_spglib("hall:526", lattice_vectors=lattice)
    context = build_symmetry_context(model, [np.asarray([-0.5, 0.0])] * 3)
    return build_ebr_matrix(
        context,
        load_ebr_catalog("sg227", hall_number=526),
        lattice_vectors=lattice,
    )


@pytest.fixture(scope="module")
def p4mm_matrix() -> EBRMatrix:
    path = resources.files("pcwannier.symmetry").joinpath(
        "space_groups", "p4mm.yaml"
    )
    model = load_symmetry(path)
    context = build_symmetry_context(
        model, [np.asarray([-0.5, 0.0]), np.asarray([-0.5, 0.0])]
    )
    return build_ebr_matrix(
        context,
        load_ebr_catalog("p4mm"),
        lattice_vectors=np.eye(2),
    )


def test_catalog_alias_and_strict_custom_schema(tmp_path):
    builtin = load_ebr_catalog("221")
    assert builtin.space_group_number == 221
    assert builtin.hall_number == 517
    assert len(builtin.ebrs) == 40

    custom = tmp_path / "catalog.yaml"
    custom.write_text(
        "\n".join(
            (
                "name: custom",
                "space_group_number: 1",
                "hall_number: 1",
                "k_points:",
                "  - {name: Gamma, k: [0, 0, 0]}",
                "ebrs:",
                "  - {name: A@1a, wyckoff: 1a, center: [0, 0, 0], site_irrep: A}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = load_ebr_catalog(custom)
    assert parsed.name == "custom"
    assert parsed.k_points[0].name == "Gamma"

    custom.write_text(custom.read_text(encoding="utf-8") + "unknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        load_ebr_catalog(custom)


def test_multi_setting_space_group_requires_explicit_hall():
    with pytest.raises(ValueError, match="multiple built-in Hall catalogs"):
        load_ebr_catalog("sg224")
    with pytest.raises(ValueError, match="multiple built-in Hall catalogs"):
        infer_builtin_catalog_alias(227)

    choice_one = load_ebr_catalog("sg224", hall_number=521)
    choice_two = load_ebr_catalog("hall:522")
    assert choice_one.hall_number == 521
    assert choice_two.hall_number == 522
    assert choice_one.source == "builtin:sg224/hall521.yaml"
    assert choice_two.source == "builtin:sg224/hall522.yaml"
    assert infer_builtin_catalog_alias(224, hall_number=521) == "sg224/hall521"
    assert infer_builtin_catalog_alias(224, hall_number=522) == "sg224/hall522"


def test_hall_catalog_reference_rejects_conflicting_context():
    with pytest.raises(ValueError, match="selects Hall 522, not requested Hall 521"):
        load_ebr_catalog("sg224/hall522", hall_number=521)


@pytest.mark.parametrize(
    ("space_group", "hall_number", "expected_columns"),
    (
        (198, 492, 3),
        (199, 493, 3),
        (205, 501, 12),
        (206, 502, 12),
        (212, 508, 6),
        (213, 509, 6),
        (214, 510, 14),
        (216, 512, 20),
        (221, 517, 40),
        (224, 521, 25),
        (224, 522, 25),
        (225, 523, 33),
        (227, 525, 22),
        (227, 526, 22),
        (229, 529, 31),
        (230, 530, 17),
    ),
)
def test_all_builtin_3d_hall_catalog_resources_load(
    space_group, hall_number, expected_columns
):
    catalog = load_ebr_catalog(f"sg{space_group}", hall_number=hall_number)
    assert catalog.space_group_number == space_group
    assert catalog.hall_number == hall_number
    assert len(catalog.ebrs) == expected_columns


@pytest.mark.parametrize(
    ("space_group", "hall_number", "lattice", "expected_shape"),
    (
        (224, 521, np.eye(3), (28, 25)),
        (
            227,
            525,
            np.asarray(((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))),
            (22, 22),
        ),
    ),
)
def test_origin_choice_one_hall_catalogs_build_dynamic_matrices(
    space_group, hall_number, lattice, expected_shape
):
    model = load_symmetry_from_spglib(
        f"hall:{hall_number}", lattice_vectors=lattice
    )
    context = build_symmetry_context(model, [np.asarray([-0.5, 0.0])] * 3)
    matrix = build_ebr_matrix(
        context,
        load_ebr_catalog(f"sg{space_group}", hall_number=hall_number),
        lattice_vectors=lattice,
    )
    assert matrix.values.shape == expected_shape
    assert np.all(matrix.values >= 0)


@pytest.mark.parametrize(
    ("space_group", "hall_number", "lattice", "expected_shape"),
    (
        (198, 492, np.eye(3), (12, 3)),
        (
            199,
            493,
            np.asarray(((-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5))),
            (13, 3),
        ),
        (205, 501, np.eye(3), (18, 12)),
        (
            206,
            502,
            np.asarray(((-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5))),
            (23, 12),
        ),
        (212, 508, np.eye(3), (15, 6)),
        (
            214,
            510,
            np.asarray(((-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5))),
            (17, 14),
        ),
        (
            216,
            512,
            np.asarray(((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))),
            (17, 20),
        ),
        (
            225,
            523,
            np.asarray(((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))),
            (31, 33),
        ),
        (
            229,
            529,
            np.asarray(((-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5))),
            (30, 31),
        ),
        (
            230,
            530,
            np.asarray(((-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5))),
            (22, 17),
        ),
    ),
)
def test_common_photonic_hall_catalogs_build_dynamic_matrices(
    space_group, hall_number, lattice, expected_shape
):
    model = load_symmetry_from_spglib(
        f"hall:{hall_number}", lattice_vectors=lattice
    )
    context = build_symmetry_context(model, [np.asarray([-0.5, 0.0])] * 3)
    matrix = build_ebr_matrix(
        context,
        load_ebr_catalog(f"sg{space_group}", hall_number=hall_number),
        lattice_vectors=lattice,
    )
    assert matrix.values.shape == expected_shape
    assert np.all(matrix.values >= 0)


def test_sg227_catalog_builds_in_fcc_primitive_setting(sg227_matrix):
    catalog = load_ebr_catalog("227", hall_number=526)

    assert catalog.space_group_number == 227
    assert catalog.hall_number == 526
    assert sg227_matrix.values.shape == (22, 22)
    assert tuple(definition.name for definition in sg227_matrix.columns)[-6:] == (
        "A1g@16d",
        "A2g@16d",
        "Eg@16d",
        "A1u@16d",
        "A2u@16d",
        "Eu@16d",
    )
    assert sg227_matrix.dimensions.tolist() == [
        2,
        2,
        4,
        6,
        6,
        2,
        2,
        4,
        6,
        6,
        4,
        4,
        8,
        4,
        4,
        8,
        4,
        4,
        8,
        4,
        4,
        8,
    ]

def test_sg213_catalog_uses_maximal_wyckoff_positions(sg213_matrix):
    catalog = load_ebr_catalog("213")
    assert catalog.space_group_number == 213
    assert catalog.hall_number == 509
    assert tuple(column.name for column in catalog.ebrs) == (
        "A1@4a",
        "A2@4a",
        "E@4a",
        "A1@4b",
        "A2@4b",
        "E@4b",
    )
    assert np.array_equal(sg213_matrix.dimensions, [4, 4, 8, 4, 4, 8])
    assert np.all(sg213_matrix.values >= 0)


def test_p4mm_catalog_and_dynamic_2d_ebr_matrix(p4mm_matrix):
    catalog = load_ebr_catalog("p4mm")
    assert catalog.dimension == 2
    assert catalog.space_group_name == "p4mm"
    assert catalog.space_group_number is None
    assert catalog.hall_number is None
    assert len(catalog.ebrs) == 14

    matrix = p4mm_matrix
    assert matrix.values.shape == (14, 14)
    names = tuple(column.name for column in matrix.columns)
    expected = (
        matrix.values[:, names.index("A1@1a")]
        + matrix.values[:, names.index("E@1a")]
    )
    vector = BandSymmetryVector(matrix.row_keys, expected, 3)
    solutions = decompose_ebr(vector, matrix)
    assert len(solutions) == 1
    assert np.array_equal(
        solutions[0].multiplicities,
        np.asarray([int(name in {"A1@1a", "E@1a"}) for name in names]),
    )

    assert _resolve_ebr_mode(
        SimpleNamespace(ebr_subspace_dimension=None, wannier_subspace="T"),
        SimpleNamespace(model=SimpleNamespace(dimension=2)),
    ) == "regular"


@pytest.mark.parametrize(
    ("dimension", "wannier_subspace", "subspace_dimension", "expected"),
    (
        (2, "T", None, "regular"),
        (3, "T", None, "transverse"),
        (3, "T+L", None, "regular"),
        (2, "T", 6, "subspace"),
        (3, "T+L", 8, "subspace"),
    ),
)
def test_ebr_analysis_kind_is_inferred_from_physical_configuration(
    dimension, wannier_subspace, subspace_dimension, expected
):
    config = SimpleNamespace(
        ebr_subspace_dimension=subspace_dimension,
        wannier_subspace=wannier_subspace,
    )
    context = SimpleNamespace(model=SimpleNamespace(dimension=dimension))

    assert _resolve_ebr_mode(config, context) == expected


@pytest.mark.parametrize("name", WALLPAPER_CATALOGS)
def test_all_wallpaper_catalogs_load_by_name_and_generate_ebr_matrices(name):
    catalog = load_ebr_catalog(name)
    model = load_symmetry(
        resources.files("pcwannier.symmetry").joinpath(
            "space_groups", f"{name}.yaml"
        )
    )
    context = build_symmetry_context(
        model, [np.asarray([0.0])] * model.dimension
    )

    matrix = build_ebr_matrix(context, catalog, lattice_vectors=np.eye(2))

    assert catalog.space_group_name == name
    assert matrix.values.shape[1] == len(catalog.ebrs)
    assert matrix.values.shape[0] == len(matrix.row_keys)
    assert np.all(matrix.values >= 0)
    assert np.all(matrix.dimensions > 0)
    assert infer_builtin_catalog_alias(name) == name


@pytest.mark.parametrize(("alias", "canonical"), SHORT_WALLPAPER_ALIASES.items())
def test_short_wallpaper_catalog_aliases_resolve(alias, canonical):
    catalog = load_ebr_catalog(alias)
    assert catalog.space_group_name == canonical
    assert infer_builtin_catalog_alias(alias) == canonical


def test_wallpaper_number_aliases_follow_international_order():
    for number, canonical in enumerate(WALLPAPER_CATALOGS, start=1):
        assert load_ebr_catalog(f"wp{number}").space_group_name == canonical
        assert load_ebr_catalog(f"wp{number:02d}").space_group_name == canonical


def test_hexagonal_catalogs_parse_exact_fractional_coordinates():
    p3 = load_ebr_catalog("p3")
    points = {point.name: point.k_fractional for point in p3.k_points}
    assert np.allclose(points["K"], [1.0 / 3.0, 1.0 / 3.0])
    assert np.allclose(points["K_prime"], [-1.0 / 3.0, -1.0 / 3.0])
    assert np.allclose(p3.ebrs[3].center, [1.0 / 3.0, 2.0 / 3.0])

    p31m = load_ebr_catalog("p31m")
    assert {point.name for point in p31m.k_points} == {
        "Gamma",
        "K",
        "K_prime",
        "M",
    }


def test_named_2d_catalog_rejects_a_different_space_group():
    path = resources.files("pcwannier.symmetry").joinpath(
        "space_groups", "p4mm.yaml"
    )
    model = load_symmetry(path)
    context = build_symmetry_context(
        model, [np.asarray([-0.5, 0.0]), np.asarray([-0.5, 0.0])]
    )
    wrong = replace(load_ebr_catalog("p4mm"), space_group_name="p4gm")
    with pytest.raises(ValueError, match="does not match"):
        build_ebr_matrix(context, wrong, lattice_vectors=np.eye(2))


def test_regular_solver_unique_multiple_and_no_solution():
    identity = _matrix([[1, 0], [0, 1]])
    solutions = decompose_ebr(_vector(identity, [1, 1], 2), identity)
    assert len(solutions) == 1
    assert np.array_equal(solutions[0].multiplicities, [1, 1])

    duplicate = _matrix([[1, 1]])
    solutions = decompose_ebr(_vector(duplicate, [1], 1), duplicate)
    assert {tuple(item.multiplicities) for item in solutions} == {(1, 0), (0, 1)}

    impossible = _matrix([[1], [0]])
    assert decompose_ebr(_vector(impossible, [1, 1], 1), impossible) == ()


def test_tetb_solver_keeps_composite_solutions_and_honors_search_limit():
    matrix = _matrix([[1, 0], [0, 1]])
    vector = _vector(matrix, [0, 1], 1)
    solutions = enumerate_tetb_decompositions(
        vector,
        matrix,
        gamma_point_name="Gamma",
        max_auxiliary_bands=1,
        stop_at_first_physical=False,
    )
    assert any(
        item.composite
        and np.array_equal(item.n_t_plus_l, [1, 1])
        and np.array_equal(item.n_l, [1, 0])
        for item in solutions
    )

    crowded = _matrix([[1, 1, 1, 1]])
    with pytest.raises(EBRSearchLimitError, match="ebr_max_states"):
        decompose_ebr(_vector(crowded, [3], 3), crowded, max_states=1)


def test_sg221_dynamic_matrix_reproduces_two_transverse_solutions(sg221_matrix):
    matrix = sg221_matrix
    names = [column.name for column in matrix.columns]
    magnetic = (
        matrix.values[:, names.index("A2g@3c")]
        - matrix.values[:, names.index("A1u@1b")]
    )
    electric = (
        matrix.values[:, names.index("A2u@3d")]
        - matrix.values[:, names.index("A1g@1a")]
    )
    gamma = np.asarray([key.point_name == "Gamma" for key in matrix.row_keys])
    assert np.array_equal(magnetic[~gamma], electric[~gamma])
    assert [(key.irrep_name, int(magnetic[index])) for index, key in enumerate(matrix.row_keys) if gamma[index] and magnetic[index]] == [
        ("A1u", -1),
        ("T1g", 1),
    ]
    assert [(key.irrep_name, int(electric[index])) for index, key in enumerate(matrix.row_keys) if gamma[index] and electric[index]] == [
        ("A1g", -1),
        ("T1u", 1),
    ]

    transverse = magnetic.copy()
    transverse[gamma] = 0
    vector = BandSymmetryVector(matrix.row_keys, transverse, 2)
    for required_irrep, positive_name, negative_name in (
        ("A1u", "A2g@3c", "A1u@1b"),
        ("A1g", "A2u@3d", "A1g@1a"),
    ):
        required_row = next(
            index
            for index, key in enumerate(matrix.row_keys)
            if key.point_name == "Gamma" and key.irrep_name == required_irrep
        )
        solutions = enumerate_tetb_decompositions(
            vector,
            matrix,
            gamma_point_name="Gamma",
            max_auxiliary_bands=1,
            required_auxiliary_gamma_rows=(required_row,),
        )
        expected_positive = names.index(positive_name)
        expected_negative = names.index(negative_name)
        assert any(
            item.physical
            and item.n_t_plus_l[expected_positive] == 1
            and item.n_l[expected_negative] == 1
            for item in solutions
        )


def test_sg224_dynamic_matrix_reproduces_published_solution(sg224_matrix):
    matrix = sg224_matrix
    names = [column.name for column in matrix.columns]
    signed = (
        matrix.values[:, names.index("A2u@4b")]
        + matrix.values[:, names.index("A2u@4c")]
        - matrix.values[:, names.index("A1@2a")]
    )
    physical = signed.copy()
    required_row = None
    for index, key in enumerate(matrix.row_keys):
        if key.point_name != "Gamma":
            continue
        if key.irrep_name == "A1g":
            physical[index] += 1
            required_row = index
        elif key.irrep_name == "T1u":
            physical[index] -= 1
    assert required_row is not None
    vector = BandSymmetryVector(matrix.row_keys, physical, 6)
    solutions = enumerate_tetb_decompositions(
        vector,
        matrix,
        gamma_point_name="Gamma",
        max_auxiliary_bands=2,
        required_auxiliary_gamma_rows=(required_row,),
    )
    expected_positive = {names.index("A2u@4b"), names.index("A2u@4c")}
    expected_negative = names.index("A1@2a")
    assert any(
        item.physical
        and set(np.flatnonzero(item.n_t_plus_l)) == expected_positive
        and item.n_l[expected_negative] == 1
        and np.array_equal(item.gamma_surrogate, signed - physical)
        for item in solutions
    )


def test_dynamic_matrix_rejects_wrong_sg224_origin_choice():
    model = load_symmetry_from_spglib("hall:521")
    context = build_symmetry_context(model, [np.asarray([-0.5, 0.0])] * 3)
    with pytest.raises(ValueError, match="catalog Hall 522.*calculation Hall 521"):
        build_ebr_matrix(
            context,
            load_ebr_catalog("sg224", hall_number=522),
            lattice_vectors=np.eye(3),
        )


def test_dynamic_matrix_rejects_different_space_group(sg221_matrix):
    model = load_symmetry_from_spglib("hall:517")
    context = build_symmetry_context(model, [np.asarray([-0.5, 0.0])] * 3)
    with pytest.raises(ValueError, match="space group 224.*space group 221"):
        build_ebr_matrix(
            context,
            load_ebr_catalog("sg224", hall_number=522),
            lattice_vectors=np.eye(3),
        )


def test_subspace_solver_keeps_positive_gamma_constraints_and_zero_mode_surrogate():
    rows = (
        SymmetryVectorKey("Gamma", "T"),
        SymmetryVectorKey("Gamma", "L"),
        SymmetryVectorKey("X", "A"),
    )
    columns = (EBRDefinition("target", "1a", np.zeros(1), "A"),)
    matrix = EBRMatrix(rows, columns, np.asarray([[1], [1], [1]]), np.asarray([4]))
    inventory = BandSymmetryVector(rows, np.asarray([1, 0, 1]), 8)
    enumeration = enumerate_ebr_subspace_solutions(
        inventory,
        matrix,
        gamma_point_name="Gamma",
        target_dimension=4,
        gamma_surrogate=np.asarray([0, 1, 0]),
        max_auxiliary_bands=0,
    )
    solutions = enumeration.solutions
    assert len(solutions) == 1
    assert enumeration.statistics.complete
    assert enumeration.statistics.weighted_vectors_generated > 0
    assert enumeration.statistics.signed_combinations_tested > 0
    solution, selected = solutions[0]
    assert np.array_equal(solution.n_t, [1])
    assert np.array_equal(selected.multiplicities, [1, 0, 1])

    unavailable = BandSymmetryVector(rows, np.asarray([0, 0, 1]), 8)
    assert enumerate_ebr_subspace_solutions(
        unavailable,
        matrix,
        gamma_point_name="Gamma",
        target_dimension=4,
        gamma_surrogate=np.asarray([0, 1, 0]),
        max_auxiliary_bands=0,
    ).solutions == ()

    two_band_matrix = EBRMatrix(
        (SymmetryVectorKey("Gamma", "A"),),
        (_definition("two_band"),),
        np.asarray([[1]]),
        np.asarray([2]),
    )
    two_band_inventory = BandSymmetryVector(
        two_band_matrix.row_keys, np.asarray([1]), 2
    )
    assert len(
        enumerate_ebr_subspace_solutions(
            two_band_inventory,
            two_band_matrix,
            gamma_point_name="Gamma",
            target_dimension=2,
            gamma_surrogate=np.zeros(1, dtype=int),
            max_auxiliary_bands=0,
        ).solutions
    ) == 1


def test_subspace_search_covers_both_gamma_zero_mode_sectors_and_sg224_target(
    sg224_matrix,
):
    assert _subspace_gamma_sectors(False, 2) == (False, True)
    assert _subspace_gamma_sectors(True, 2) == (True,)
    assert _subspace_gamma_sectors(False, 0) == (False,)

    zero_blocks = (
        SimpleNamespace(band_indices=(0, 1), energies=(0.0, 0.0)),
        SimpleNamespace(band_indices=(2,), energies=(1.0,)),
    )
    gamma_point = SimpleNamespace(
        name="Gamma",
        requested_k_fractional=np.zeros(1),
        degenerate_blocks=zero_blocks,
    )
    analysis = SimpleNamespace(points=(gamma_point,))
    context = SimpleNamespace(model=SimpleNamespace(tolerance=1.0e-8))
    catalog_point = SimpleNamespace(name="Gamma", k_fractional=np.zeros(1))
    catalog = SimpleNamespace(gamma_point=catalog_point)
    assert _gamma_zero_mode_dimension(
        analysis, context, catalog, 1.0e-10
    ) == 2

    matrix = sg224_matrix
    names = [column.name for column in matrix.columns]
    signed = np.zeros(len(names), dtype=np.int64)
    signed[names.index("A2g@4b")] = 1
    signed[names.index("A2g@4c")] = 1
    signed[names.index("A2@2a")] = -1
    formal = matrix.values @ signed
    surrogate = np.zeros(len(matrix.row_keys), dtype=np.int64)
    required_row = None
    for index, key in enumerate(matrix.row_keys):
        if key.point_name != "Gamma":
            continue
        if key.irrep_name == "T1g":
            surrogate[index] = 1
        elif key.irrep_name == "A1u":
            surrogate[index] = -1
            required_row = index
    assert required_row is not None
    selected = formal - surrogate
    assert np.all(selected >= 0)
    inventory = BandSymmetryVector(matrix.row_keys, selected, 30)
    enumeration = enumerate_ebr_subspace_solutions(
        inventory,
        matrix,
        gamma_point_name="Gamma",
        target_dimension=6,
        gamma_surrogate=surrogate,
        max_auxiliary_bands=2,
        required_auxiliary_gamma_rows=(required_row,),
        gamma_sector="include_gamma_zero_modes",
    )
    assert enumeration.statistics.complete
    assert enumeration.statistics.gamma_sectors == ("include_gamma_zero_modes",)
    assert any(np.array_equal(solution.n_t, signed) for solution, _ in enumeration.solutions)


def test_subspace_inventory_ignores_incomplete_outer_window_boundary_block():
    exact = SimpleNamespace(multiplicities={"A": 1})
    blocks = (
        SimpleNamespace(
            band_indices=(0, 1),
            energies=(0.0, 0.0),
            decomposition=None,
            irrep_unavailable_reason="Gamma transverse zero modes",
        ),
        SimpleNamespace(
            band_indices=(2,),
            energies=(1.0,),
            decomposition=exact,
            irrep_unavailable_reason=None,
        ),
        SimpleNamespace(
            band_indices=(3,),
            energies=(2.0,),
            decomposition=None,
            irrep_unavailable_reason="outer-window boundary truncates the representation",
        ),
    )
    resolved = SimpleNamespace(
        require_irreps=lambda: (SimpleNamespace(name="A", dimension=1),)
    )
    point = SimpleNamespace(
        name="Gamma",
        k_fractional=np.zeros(1),
        requested_k_fractional=np.zeros(1),
        band_indices=(0, 1, 2, 3),
        degenerate_blocks=blocks,
        resolved_little_group=resolved,
        antiunitary_operation_names=(),
    )
    analysis = SimpleNamespace(points=(point,))
    context = SimpleNamespace(model=SimpleNamespace(tolerance=1.0e-8))
    catalog_point = SimpleNamespace(name="Gamma", k_fractional=np.zeros(1))
    catalog = SimpleNamespace(
        k_points=(catalog_point,),
        gamma_point=catalog_point,
    )

    skipped: list[str] = []
    inventory = _build_subspace_inventory(
        analysis,
        context,
        catalog,
        zero_tolerance=1.0e-10,
        skipped_blocks=skipped,
    )

    assert np.array_equal(inventory.multiplicities, [1])
    assert len(skipped) == 1
    assert "bands(0-based)=(3,)" in skipped[0]

    selected = BandSymmetryVector(
        inventory.row_keys, inventory.multiplicities, total_dimension=1
    )
    representations = _match_subspace_representations(
        analysis,
        context,
        catalog,
        selected,
        zero_tolerance=1.0e-10,
    )
    assert representations is not None
    assert len(representations) == 1
    assert representations[0].irrep_multiplicities == (("A", 1),)
    assert representations[0].unresolved_dimension == 0
    assert representations[0].alternative_count == 1

    assert not _validate_subspace_fixed_bands(
        analysis,
        context,
        catalog,
        fixed_bands=(),
        target_dimension=1,
        zero_tolerance=1.0e-10,
    )
    assert _validate_subspace_fixed_bands(
        analysis,
        context,
        catalog,
        fixed_bands=(0, 1),
        target_dimension=3,
        zero_tolerance=1.0e-10,
    )
    fixed_representation = _match_subspace_representations(
        analysis,
        context,
        catalog,
        BandSymmetryVector(
            inventory.row_keys, inventory.multiplicities, total_dimension=3
        ),
        zero_tolerance=1.0e-10,
        fixed_band_indices=(0, 1),
        include_gamma_zero_modes=True,
    )
    assert fixed_representation is not None
    assert fixed_representation[0].irrep_multiplicities == (("A", 1),)
    assert fixed_representation[0].unresolved_dimension == 2


def test_subspace_reports_numerical_channel_dimension_realizations():
    block_t1 = SimpleNamespace(
        band_indices=(0, 1, 12),
        energies=(0.0, 0.0, 0.0),
        decomposition=SimpleNamespace(multiplicities={"T1": 1}),
        irrep_unavailable_reason=None,
    )
    block_e = SimpleNamespace(
        band_indices=(2, 3),
        energies=(1.0, 1.0),
        decomposition=SimpleNamespace(multiplicities={"E": 1}),
        irrep_unavailable_reason=None,
    )
    block_t2_h = SimpleNamespace(
        band_indices=(8, 9, 10),
        energies=(2.0, 2.0, 2.0),
        decomposition=SimpleNamespace(multiplicities={"T2": 1}),
        irrep_unavailable_reason=None,
    )
    block_t2_l = SimpleNamespace(
        band_indices=(13, 14, 15),
        energies=(3.0, 3.0, 3.0),
        decomposition=SimpleNamespace(multiplicities={"T2": 1}),
        irrep_unavailable_reason=None,
    )
    resolved = SimpleNamespace(
        require_irreps=lambda: (
            SimpleNamespace(name="E", dimension=2),
            SimpleNamespace(name="T1", dimension=3),
            SimpleNamespace(name="T2", dimension=3),
        )
    )
    point = SimpleNamespace(
        name="Gamma",
        requested_k_fractional=np.zeros(3),
        band_indices=tuple(range(16)),
        degenerate_blocks=(block_t1, block_e, block_t2_h, block_t2_l),
        resolved_little_group=resolved,
    )
    analysis = SimpleNamespace(points=(point,))
    context = SimpleNamespace(model=SimpleNamespace(tolerance=1.0e-8))
    catalog_point = SimpleNamespace(name="Gamma", k_fractional=np.zeros(3))
    catalog = SimpleNamespace(k_points=(catalog_point,), gamma_point=catalog_point)
    selected = BandSymmetryVector(
        (
            SymmetryVectorKey("Gamma", "E"),
            SymmetryVectorKey("Gamma", "T1"),
            SymmetryVectorKey("Gamma", "T2"),
        ),
        np.ones(3, dtype=np.int64),
        total_dimension=8,
    )
    channels = {
        **{index: SimpleNamespace(channel="H") for index in range(12)},
        **{index: SimpleNamespace(channel="L") for index in range(12, 16)},
    }

    representations = _match_subspace_representations(
        analysis,
        context,
        catalog,
        selected,
        zero_tolerance=1.0e-10,
        fixed_band_indices=(0, 1, 12),
        singular_gamma_zero_modes=False,
        band_channels=channels,
    )

    assert representations is not None
    assert representations[0].alternative_count == 2
    assert representations[0].channel_dimension_realizations == (
        (("H", 4), ("L", 4)),
        (("H", 7), ("L", 1)),
    )
    assert representations[0].channel_assignment_complete


def test_subspace_can_select_an_irrep_from_a_reducible_numerical_block():
    block = SimpleNamespace(
        band_indices=(0, 1, 2, 3, 4, 5),
        energies=(1.0,) * 6,
        decomposition=SimpleNamespace(multiplicities={"T1": 1, "T2": 1}),
        irrep_unavailable_reason=None,
    )
    resolved = SimpleNamespace(
        require_irreps=lambda: (
            SimpleNamespace(name="T1", dimension=3),
            SimpleNamespace(name="T2", dimension=3),
        )
    )
    point = SimpleNamespace(
        name="Gamma",
        requested_k_fractional=np.zeros(3),
        band_indices=block.band_indices,
        degenerate_blocks=(block,),
        resolved_little_group=resolved,
    )
    analysis = SimpleNamespace(points=(point,))
    context = SimpleNamespace(model=SimpleNamespace(tolerance=1.0e-8))
    catalog_point = SimpleNamespace(name="Gamma", k_fractional=np.zeros(3))
    catalog = SimpleNamespace(k_points=(catalog_point,), gamma_point=catalog_point)
    selected = BandSymmetryVector(
        (
            SymmetryVectorKey("Gamma", "T1"),
            SymmetryVectorKey("Gamma", "T2"),
        ),
        np.asarray([1, 0]),
        total_dimension=3,
    )

    representations = _match_subspace_representations(
        analysis,
        context,
        catalog,
        selected,
        zero_tolerance=1.0e-10,
        singular_gamma_zero_modes=False,
    )
    assert representations is not None
    assert representations[0].irrep_multiplicities == (("T1", 1),)
    assert representations[0].alternative_count == 1

    # Fixing any raw numerical eigenvector keeps the entire reducible block mandatory;
    # an irrep projector generally does not preserve that individual eigenvector.
    assert (
        _match_subspace_representations(
            analysis,
            context,
            catalog,
            selected,
            zero_tolerance=1.0e-10,
            fixed_band_indices=(0,),
            singular_gamma_zero_modes=False,
        )
        is None
    )


def test_2d_subspace_treats_a_zero_frequency_scalar_irrep_as_regular():
    exact = SimpleNamespace(multiplicities={"A": 1})
    zero_block = SimpleNamespace(
        band_indices=(0,),
        energies=(0.0,),
        decomposition=exact,
        irrep_unavailable_reason=None,
    )
    resolved = SimpleNamespace(
        require_irreps=lambda: (SimpleNamespace(name="A", dimension=1),)
    )
    point = SimpleNamespace(
        name="Gamma",
        k_fractional=np.zeros(2),
        requested_k_fractional=np.zeros(2),
        band_indices=(0,),
        degenerate_blocks=(zero_block,),
        resolved_little_group=resolved,
        antiunitary_operation_names=(),
    )
    analysis = SimpleNamespace(points=(point,))
    context = SimpleNamespace(model=SimpleNamespace(tolerance=1.0e-8))
    catalog_point = SimpleNamespace(name="Gamma", k_fractional=np.zeros(2))
    catalog = SimpleNamespace(k_points=(catalog_point,), gamma_point=catalog_point)

    inventory = _build_subspace_inventory(
        analysis,
        context,
        catalog,
        zero_tolerance=1.0e-10,
        singular_gamma_zero_modes=False,
    )
    assert np.array_equal(inventory.multiplicities, [1])
    assert not _validate_subspace_fixed_bands(
        analysis,
        context,
        catalog,
        fixed_bands=(0,),
        target_dimension=1,
        zero_tolerance=1.0e-10,
        singular_gamma_zero_modes=False,
    )
    representations = _match_subspace_representations(
        analysis,
        context,
        catalog,
        inventory,
        zero_tolerance=1.0e-10,
        fixed_band_indices=(0,),
        singular_gamma_zero_modes=False,
    )
    assert representations is not None
    assert representations[0].irrep_multiplicities == (("A", 1),)
    assert representations[0].unresolved_dimension == 0


def test_t_plus_l_treats_the_three_dimensional_gamma_zero_block_as_regular():
    context = SimpleNamespace(model=SimpleNamespace(dimension=3))

    assert _has_transverse_gamma_singularity(
        context,
        FieldKind.MAGNETIC_AXIAL_VECTOR,
        wannier_subspace="T",
    )
    assert not _has_transverse_gamma_singularity(
        context,
        FieldKind.MAGNETIC_AXIAL_VECTOR,
        wannier_subspace="T + L",
    )


def test_band_vector_rejects_point_without_exact_or_approximate_decomposition():
    catalog = EBRCatalog(
        "fixture",
        1,
        1,
        (EBRKPoint("Gamma", np.zeros(3)),),
        (EBRDefinition("A@1a", "1a", np.zeros(3), "A"),),
    )
    point = BlochSymmetryPointAnalysis(
        "Gamma",
        np.zeros(3),
        np.zeros(3),
        (0, 0, 0),
        (0,),
        (0,),
        (0,),
        {},
        {},
        SewingDiagnostics(0.0, 0.1, 0.0),
        (),
        None,
    )
    with pytest.raises(ValueError, match="no usable physical decomposition"):
        build_band_symmetry_vector(BlochSymmetryAnalysisResult((point,)), catalog)


def test_band_vector_uses_accepted_approximate_block_decomposition():
    catalog = EBRCatalog(
        "fixture",
        1,
        1,
        (EBRKPoint("Gamma", np.zeros(1)),),
        (EBRDefinition("A@1a", "1a", np.zeros(1), "A"),),
    )
    approximate = SimpleNamespace(multiplicities={"A": 1}, max_residual=2.0e-12)
    block = SimpleNamespace(
        band_indices=(0,),
        energies=(1.0,),
        decomposition=None,
        approximate_decomposition=approximate,
        character_fit_error=3.0e-12,
        leakage=0.02,
    )
    resolved = SimpleNamespace(
        require_irreps=lambda: (SimpleNamespace(name="A", dimension=1),)
    )
    point = SimpleNamespace(
        name="Gamma",
        k_fractional=np.zeros(1),
        requested_k_fractional=np.zeros(1),
        band_indices=(0,),
        degenerate_blocks=(block,),
        physical_decomposition=None,
        resolved_little_group=resolved,
        antiunitary_operation_names=(),
    )

    vector = build_band_symmetry_vector(SimpleNamespace(points=(point,)), catalog)

    assert np.array_equal(vector.multiplicities, [1])
    assert vector.total_dimension == 1
    diagnostics = _approximate_ebr_diagnostics(
        SimpleNamespace(points=(point,)), catalog, 1.0e-8
    )
    assert "bands(0-based)=(0,)" in diagnostics[0]
    assert "character_error=3e-12" in diagnostics[0]


def test_ebr_json_output_round_trip(tmp_path):
    matrix = _matrix([[1]])
    vector = _vector(matrix, [1], 1)
    result = EBRAnalysisResult(
        "regular",
        EBRCatalog(
            "fixture",
            1,
            1,
            (EBRKPoint("Gamma", np.zeros(1)),),
            matrix.columns,
        ),
        vector,
        matrix,
        regular_decompositions=(EBRDecomposition(np.asarray([1])),),
    )
    config = SimpleNamespace(
        ebr_report_file="ebr.txt",
        ebr_data_file="ebr.json",
        base_dir=tmp_path,
    )
    write_ebr_outputs(result, config, tmp_path)
    payload = json.loads((tmp_path / "ebr.json").read_text(encoding="utf-8"))
    assert payload["regular_decompositions"][0]["expression"] == "a"
    assert payload["ebr_matrix"]["values"] == [[1]]
    assert "Regular decompositions: 1" in (tmp_path / "ebr.txt").read_text(
        encoding="utf-8"
    )


def test_subspace_output_reports_representations_without_arbitrary_band_assignment(
    tmp_path,
):
    rows = (
        SymmetryVectorKey("Gamma", "A1u"),
        SymmetryVectorKey("Gamma", "T1g"),
    )
    columns = (
        EBRDefinition("negative", "1a", np.zeros(1), "A1u"),
        EBRDefinition("positive", "1b", np.zeros(1), "T1g"),
    )
    matrix = EBRMatrix(
        rows,
        columns,
        np.asarray([[1, 0], [0, 1]]),
        np.asarray([1, 3]),
    )
    selected = BandSymmetryVector(rows, np.asarray([0, 1]), 5)
    solution = TETBSolution(
        np.asarray([0, 1]),
        np.asarray([1, 0]),
        np.asarray([-1, 1]),
        np.asarray([-1, 1]),
        1,
        True,
        "fixture",
    )
    candidate = EBRSubspaceCandidate(
        solution,
        selected,
        (
            EBRSubspacePointRepresentation(
                "Gamma", (("T1g", 1),), unresolved_dimension=2, alternative_count=3
            ),
        ),
        includes_gamma_zero_modes=True,
        gamma_sector="include_gamma_zero_modes",
    )
    result = EBRAnalysisResult(
        "subspace",
        EBRCatalog(
            "fixture",
            1,
            1,
            (EBRKPoint("Gamma", np.zeros(1)),),
            columns,
        ),
        selected,
        matrix,
        subspace_candidates=(candidate,),
        search_statistics=EBRSearchStatistics(
            complete=True,
            realizable_candidates=1,
            block_realization_count=3,
        ),
    )
    config = SimpleNamespace(
        ebr_report_file="ebr.txt",
        ebr_data_file="ebr.json",
        base_dir=tmp_path,
    )

    write_ebr_outputs(result, config, tmp_path)

    report = (tmp_path / "ebr.txt").read_text(encoding="utf-8")
    assert "physical=2 unresolved transverse zero modes + T1g" in report
    assert "formal_signed=-A1u + T1g" in report
    assert "bands=" not in report
    assert "blocks=" not in report
    assert "n_L convention: the subtracted auxiliary EBR" in report
    payload = json.loads((tmp_path / "ebr.json").read_text(encoding="utf-8"))
    point = payload["subspace_candidates"][0]["point_representations"][0]
    assert payload["subspace_candidates"][0]["gamma_sector"] == "include_gamma_zero_modes"
    assert point["physical_irreps"] == {"T1g": 1}
    assert point["formal_signed_irreps"] == {"A1u": -1, "T1g": 1}
    assert point["unresolved_dimension"] == 2
    assert "bands" not in point
    assert "blocks" not in point
