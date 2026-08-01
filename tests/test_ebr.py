from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pcwannier.ebr import (
    BandSymmetryVector,
    EBRAnalysisResult,
    EBRCatalog,
    EBRDecomposition,
    EBRDefinition,
    EBRKPoint,
    EBRMatrix,
    EBRSearchLimitError,
    SymmetryVectorKey,
    build_band_symmetry_vector,
    build_ebr_matrix,
    decompose_ebr,
    enumerate_tetb_decompositions,
    load_ebr_catalog,
    write_ebr_outputs,
)
from pcwannier.symmetry.analysis import (
    BlochSymmetryAnalysisResult,
    BlochSymmetryPointAnalysis,
    SewingDiagnostics,
)
from pcwannier.symmetry.io import load_symmetry_from_spglib
from pcwannier.symmetry.representation import build_symmetry_context


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
def sg224_matrix() -> EBRMatrix:
    model = load_symmetry_from_spglib("hall:522")
    context = build_symmetry_context(model, [np.asarray([-0.5, 0.0])] * 3)
    return build_ebr_matrix(
        context,
        load_ebr_catalog("224"),
        lattice_vectors=np.eye(3),
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


def test_dynamic_matrix_rejects_wrong_hall_setting(sg221_matrix):
    model = load_symmetry_from_spglib("hall:517")
    context = build_symmetry_context(model, [np.asarray([-0.5, 0.0])] * 3)
    with pytest.raises(ValueError, match="Hall 522.*Hall 517"):
        build_ebr_matrix(context, load_ebr_catalog("sg224"), lattice_vectors=np.eye(3))


def test_band_vector_rejects_nonexact_physical_decomposition():
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
    with pytest.raises(ValueError, match="no exact physical decomposition"):
        build_band_symmetry_vector(BlochSymmetryAnalysisResult((point,)), catalog)


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
