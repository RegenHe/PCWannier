from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from pcwannier import load_input, run_calculation
from pcwannier.config import IncarParser, load_config
from pcwannier.projections import (
    LocalFrame3D,
    ProjectionRecord3D,
    TrialLinearCombination,
    VectorHydrogenicOrbital,
    real_spherical_harmonic,
)
from pcwannier.symmetry import cartesian_field_matrix
import pcwannier.compute.vector_trials as vector_trials_module
from pcwannier.compute.wannier import generate_wannier


def test_local_frame_and_fixed_vector_orbitals():
    frame = LocalFrame3D.from_xz([1.0, 1.0, 0.0], [0.0, 0.0, 2.0])
    assert np.allclose(frame.matrix.T @ frame.matrix, np.eye(3), atol=1.0e-14)
    assert np.linalg.det(frame.matrix) == pytest.approx(1.0)

    orbital = VectorHydrogenicOrbital(1, 0, 0, 2.0, [0.0, 0.0, 4.0])
    values = orbital.evaluate(
        np.asarray([[0.1, 0.0, 0.0], [0.2, 0.1, 0.0]]),
        frame,
        1.0,
        lambda *_: lambda radius, zeta: np.exp(-zeta * radius),
    )
    direction = frame.ez
    assert np.allclose(np.cross(values.real, direction), 0.0, atol=1.0e-14)


def test_real_spherical_harmonic_px_py_convention():
    theta = np.asarray([np.pi / 2, np.pi / 2])
    phi = np.asarray([0.0, np.pi / 2])
    px = real_spherical_harmonic(1, 1, theta, phi)
    py = real_spherical_harmonic(1, -1, theta, phi)
    assert px[0] > 0.0
    assert py[1] > 0.0
    assert abs(px[1]) < 1.0e-14
    assert abs(py[0]) < 1.0e-14


def test_parse_3d_vector_projection_and_complex_combination(tmp_path):
    parser = IncarParser(tmp_path / "incar")
    records = parser._parse_projections(
        "3c; [0.5,0.5,0.0]; (z=[0,0,1], x=[1,0,0]); "
        "{[2,1,1,10]@[1,0,0],[2,1,-1,10]@[0,1,0]}{1/sqrt(2),i/sqrt(2)}"
    )
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, ProjectionRecord3D)
    assert record.wyckoff == "3c"
    assert len(record.states) == 1
    assert isinstance(record.states[0], TrialLinearCombination)
    assert record.states[0].coefficients == pytest.approx(
        (1.0 / np.sqrt(2), 1.0j / np.sqrt(2))
    )
    assert [orbital.m for orbital in record.states[0].orbitals] == [1, -1]


@pytest.mark.parametrize(
    "text,match",
    [
        ("1a; [0,0,0]; (z=[0,0,1], x=[0,0,2]); [1,0,0,2]@[1,0,0]", "parallel"),
        ("1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [2,1,2,2]@[1,0,0]", "-l <= m <= l"),
        ("1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,2]@[0,0,0]", "non-zero"),
    ],
)
def test_invalid_3d_projection_inputs(tmp_path, text, match):
    parser = IncarParser(tmp_path / "incar")
    with pytest.raises(ValueError, match=match):
        parser._parse_projections(text)


def test_pm3m_wyckoff_projection_target_binding(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
                "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
                "k_points = 0:1:1, 0:1:1, 0:1:1",
                "composition_of_b = 1 0 0, 0 1 0, 0 0 1",
                "dataset_type = mpb",
                "field_components = full_vector",
                "primary_field = magnetic",
                "dataset_file = H.h5",
                "mesh_file = grid.h5",
                "E_file = E.h5",
                "metric_file = false",
                "band_window = 0:3",
                "extension = 1,1,1",
                "wannier_figures = false",
                "symmetry_file = Pm-3m",
                "projections",
                "3c; [0.5,0.5,0.0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]",
                "end",
                "wannier_targets",
                "center_A2g_3c; 3c; A2g",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(incar)

    assert len(config.projection_target_bindings) == 1
    assert config.band_calc_num == 3
    target = config.symmetry_context.model.target("center_A2g_3c")
    assert target.multiplicity == 3
    assert target.site_irrep.dimension == 1
    positions = {
        tuple(np.round(point.position, 12)) for point in target.orbit.points
    }
    assert positions == {
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    }
    record = config.projection_target_bindings[0].projection
    directions = []
    for point in target.orbit.points:
        matrix = cartesian_field_matrix(
            point.representative_operation,
            config.real_lattice_vectors,
            config.maxwell_problem.symmetry_field_kind,
        )
        directions.append(matrix @ record.frame.ez)
    assert np.allclose(
        np.sort(np.abs(np.asarray(directions)), axis=0),
        np.sort(np.eye(3), axis=0),
        atol=1.0e-14,
    )


def _write_pm3m_vector_binding_incar(
    path: Path,
    projection_lines: list[str],
    target_lines: list[str] | None,
) -> None:
    lines = [
        "lattice_const = 1",
        "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
        "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
        "k_points = 0:1:1, 0:1:1, 0:1:1",
        "composition_of_b = 1 0 0, 0 1 0, 0 0 1",
        "dataset_type = mpb",
        "field_components = full_vector",
        "primary_field = magnetic",
        "dataset_file = H.h5",
        "mesh_file = grid.h5",
        "E_file = E.h5",
        "metric_file = false",
        "band_window = 0:3",
        "extension = 1,1,1",
        "wannier_figures = false",
        "symmetry_file = Pm-3m",
        "projections",
        *projection_lines,
        "end",
    ]
    if target_lines is not None:
        lines.extend(("wannier_targets", *target_lines, "end"))
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.parametrize(
    ("projection_lines", "target_lines", "match"),
    [
        (
            ["1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]"],
            None,
            "matching wannier_targets",
        ),
        (
            [
                "1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]",
                "1b; [0.5,0.5,0.5]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]",
            ],
            ["center_A2g_1a; 1a; A2g"],
            "same number of records",
        ),
        (
            ["1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]"],
            ["center_A2g_1b; 1b; A2g"],
            "does not match",
        ),
        (
            ["1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]"],
            ["center_T1g_1a; 1a; T1g"],
            "defines 1 functions",
        ),
    ],
)
def test_invalid_3d_projection_target_bindings(
    tmp_path,
    projection_lines,
    target_lines,
    match,
):
    incar = tmp_path / "incar"
    _write_pm3m_vector_binding_incar(incar, projection_lines, target_lines)
    with pytest.raises(ValueError, match=match):
        load_config(incar)


def test_synthetic_3d_vector_projection_to_wannier_and_hopping(tmp_path):
    shape = (4, 4, 4)
    kpoints = np.zeros((1, 3), dtype=float)
    with h5py.File(tmp_path / "grid.h5", "w") as handle:
        handle.attrs["dimension"] = 3
        handle.create_dataset("shape", data=np.asarray(shape, dtype=np.int64))
        handle.create_dataset("basis", data=np.eye(3))
        for axis in range(3):
            handle.create_dataset(
                f"u{axis + 1}", data=np.arange(shape[axis]) / shape[axis] - 0.5
            )
    with h5py.File(tmp_path / "E.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("E", data=np.ones((1, 3), dtype=float))
    fields = np.zeros((1, 3) + shape + (3,), dtype=np.complex128)
    for component in range(3):
        fields[0, component, ..., component] = 1.0
    with h5py.File(tmp_path / "H.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("H_periodic", data=fields)

    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "dataset_type = mpb",
                "field_components = full_vector",
                "primary_field = magnetic",
                "lattice_const = 1",
                "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
                "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
                "k_points = 0:1:1, 0:1:1, 0:1:1",
                "composition_of_b = 1 0 0, 0 1 0, 0 0 1",
                "band_window = 0:3",
                "dataset_file = ./H.h5",
                "mesh_file = ./grid.h5",
                "metric_file = false",
                "E_file = ./E.h5",
                "extension = 1,1,1",
                "max_iter = 0",
                "wannier_figures = false",
                "symmetry_file = Pm-3m",
                "symmetry_constrained = true",
                "symmetry_validate_wannier = true",
                "symmetry_minimum_retained_norm = 0.4",
                "projections",
                "1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); "
                "[1,0,0,4]@[1,0,0]; [1,0,0,4]@[0,1,0]; [1,0,0,4]@[0,0,1]",
                "end",
                "wannier_targets",
                "center_T1g_1a; 1a; T1g",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    result = run_calculation(load_input(load_config(incar)), threads=1)

    wannier = result.wanniers[(0, 0, 0)]
    assert wannier.shape == (np.prod(shape), 3, 3)
    assert np.all(np.isfinite(wannier))
    assert np.all(result.wannier_norms > 0.0)
    assert result.A[0, 0, 0].shape == (3, 3)
    assert result.V[0, 0, 0].shape == (3, 3)
    assert np.allclose(
        result.hoppings[(0, 0, 0)],
        result.hoppings[(0, 0, 0)].conj().T,
        atol=1.0e-12,
    )


def test_orbit_expanded_vector_trials_normalize_all_six_columns(monkeypatch):
    point_count = 5
    operation = SimpleNamespace(antiunitary=False)
    orbit = SimpleNamespace(
        points=(
            SimpleNamespace(position=np.zeros(3), representative_operation=operation),
            SimpleNamespace(position=np.full(3, 0.5), representative_operation=operation),
        )
    )
    target = SimpleNamespace(orbit=orbit)
    context = SimpleNamespace(model=SimpleNamespace(target=lambda _name: target))
    record = SimpleNamespace(states=(1.0, 2.0, 3.0))
    binding = SimpleNamespace(target_name="T1_2a", projection=record)

    class InnerProduct:
        def norms(self, values, *, name):
            assert name == "3D projection Bloch-sum norms"
            assert values.shape == (point_count, 3, 6)
            return np.sum(np.abs(values) ** 2, axis=(0, 1), dtype=np.float64)

    state = SimpleNamespace(
        config=SimpleNamespace(
            symmetry_context=context,
            projection_target_bindings=(binding,),
            real_lattice_vectors=np.eye(3),
            lattice_const=1.0,
            k_points=(np.asarray([0.0]),) * 3,
            extension=(1, 1, 1),
        ),
        mesh=SimpleNamespace(vertices=np.zeros((point_count, 3))),
        bloch_sign=1,
        inner_product=InnerProduct(),
    )

    monkeypatch.setattr(
        vector_trials_module,
        "_evaluate_transformed_trial",
        lambda _state, _record, trial, points, _center, _operation: np.full(
            (points.shape[0], 3), trial, dtype=np.complex128
        ),
    )

    values = vector_trials_module.build_vector_bloch_trials(state, (0, 0, 0))

    assert values.shape == (point_count, 6, 3)
    assert np.allclose(
        np.sum(np.abs(values) ** 2, axis=(0, 2), dtype=np.float64),
        np.ones(6),
    )


def test_vector_wannier_norms_follow_six_wannier_columns():
    point_count = 6
    base = np.zeros((6, point_count, 3), dtype=np.complex128)
    for band in range(6):
        base[band, band, band % 3] = 1.0

    class InnerProduct:
        def norms(self, values, *, chunk_size, name):
            assert name == "Wannier norms"
            assert chunk_size == 2048
            assert values.shape == (point_count, 3, 6)
            return np.sum(np.abs(values) ** 2, axis=(0, 1), dtype=np.float64)

    config = SimpleNamespace(
        real_lattice_vectors=np.eye(3),
        reciprocal_lattice_vectors=np.eye(3),
        lattice_const=1.0,
        kdim=3,
        k_points=(np.asarray([0.0]),) * 3,
        extension=(1, 1, 1),
        band_calc_num=6,
    )
    state = SimpleNamespace(
        extended_mesh=SimpleNamespace(vertices=np.zeros((point_count, 3))),
        space_to_original_mapping=np.arange(point_count),
        extended_inner_product=InnerProduct(),
        bloch_sign=1,
        k_indices=lambda: iter(((0, 0, 0),)),
        get_k_num=lambda: 1,
        get_block=lambda *_: base,
    )
    context = SimpleNamespace(
        config=config,
        state=state,
        output_state_coefficients_at=lambda *_: np.eye(6, dtype=np.complex128),
    )

    _, values, norms = generate_wannier(context)

    assert values.shape == (point_count, 6, 3)
    assert norms.shape == (6,)
    assert np.allclose(norms, np.ones(6))
