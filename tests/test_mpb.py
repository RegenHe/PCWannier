from __future__ import annotations

import h5py
import numpy as np
import pytest

from pcwannier.compute import (
    UniformGridInnerProduct,
    integrate_components,
    integrate_scalar,
    periodic_grid_coordinates,
)
from pcwannier.compute.vector_diagnostics import periodic_vector_field_residuals
from pcwannier.config import load_config
from pcwannier.conventions import BlochConvention, BlochFieldRepresentation
from pcwannier.data import PeriodicGrid
from pcwannier.sources import load_input
from pcwannier.sources.mpb import load_mpb_grid
from pcwannier.symmetry.bloch import PeriodicGridInterpolator


def test_uniform_grid_integrates_real_and_complex_constants():
    real_values = np.ones((4, 6), dtype=np.float32)
    complex_values = np.full((4, 6), 1.0 + 2.0j, dtype=np.complex64)

    assert integrate_scalar(real_values, cell_volume=3.5) == pytest.approx(3.5)
    assert integrate_scalar(complex_values, cell_volume=3.5) == pytest.approx(
        3.5 + 7.0j
    )


def test_uniform_grid_integrates_periodic_cosine_to_zero():
    nx, ny = 32, 9
    x = np.arange(nx, dtype=float) / nx - 0.5
    values = np.cos(2.0 * np.pi * x)[:, None] * np.ones((1, ny))

    assert integrate_scalar(values, cell_volume=2.0) == pytest.approx(
        0.0,
        abs=1.0e-14,
    )


def test_uniform_grid_integrates_components_pointwise():
    values = np.empty((3, 5, 3), dtype=np.complex128)
    values[..., 0] = 1.0
    values[..., 1] = 2.0j
    values[..., 2] = 3.0

    assert integrate_components(values, cell_volume=2.5) == pytest.approx(
        (4.0 + 2.0j) * 2.5
    )


def test_uniform_grid_coordinates_use_oblique_half_open_cell():
    lattice = np.array([[2.0, 0.0], [0.5, 3.0]])
    coordinates = periodic_grid_coordinates((4, 2), lattice)
    fractional = coordinates @ np.linalg.inv(lattice)

    assert abs(np.linalg.det(lattice)) == pytest.approx(6.0)
    assert coordinates.shape == (4, 2, 2)
    assert np.min(fractional[..., 0]) == pytest.approx(-0.5)
    assert np.max(fractional[..., 0]) == pytest.approx(0.25)
    assert np.min(fractional[..., 1]) == pytest.approx(-0.5)
    assert np.max(fractional[..., 1]) == pytest.approx(0.0)
    assert not np.any(np.isclose(fractional, 0.5))


def test_uniform_grid_coordinates_center_degenerate_axis():
    coordinates = periodic_grid_coordinates((4, 1, 2), np.eye(3))

    assert np.allclose(coordinates[:, 0, :, 1], 0.0)
    assert np.allclose(coordinates[:, 0, 0, 0], [-0.5, -0.25, 0.0, 0.25])
    assert np.allclose(coordinates[0, 0, :, 2], [-0.5, 0.0])


def test_periodic_vector_differential_diagnostics():
    shape = (4, 4, 4)
    axes = [np.arange(size, dtype=float) / size for size in shape]
    x, _y, _z = np.meshgrid(*axes, indexing="ij")
    phase = np.exp(2j * np.pi * x)
    transverse = np.stack(
        (np.zeros_like(phase), phase, np.zeros_like(phase)), axis=-1
    )
    longitudinal = np.stack(
        (phase, np.zeros_like(phase), np.zeros_like(phase)), axis=-1
    )
    fields = np.stack((transverse, longitudinal), axis=0).reshape(2, -1, 3)

    divergence, curl = periodic_vector_field_residuals(
        fields,
        shape,
        np.eye(3),
        np.zeros(3),
        bloch_sign=1,
    )

    assert divergence[0] < 1.0e-12
    assert curl[1] < 1.0e-12
    assert divergence[1] == pytest.approx(1.0)
    assert curl[0] == pytest.approx(1.0)


def test_periodic_grid_centers_single_sample_direction():
    grid = PeriodicGrid((3, 1), np.eye(2))

    assert np.allclose(grid.fractional_vertices[:, 1], 0.0)
    stencil = PeriodicGridInterpolator(grid).stencil(grid.fractional_vertices)
    values = np.arange(grid.point_count, dtype=float)[None, :]
    assert np.allclose(stencil.apply(values), values)


def test_uniform_grid_rejects_nonfinite_input():
    values = np.ones((2, 2))
    values[0, 1] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        integrate_scalar(values, cell_volume=1.0)


def test_uniform_metric_inner_product_and_extension_mapping():
    grid = PeriodicGrid((2, 3), np.array([[2.0, 0.0], [0.0, 3.0]]))
    metric = np.arange(1.0, 7.0)
    left = np.ones(grid.point_count, dtype=np.complex128)
    right = np.arange(grid.point_count, dtype=float) + 1.0j
    inner = UniformGridInnerProduct(grid, metric)

    expected = (
        grid.cell_volume
        / grid.point_count
        * np.sum(metric * right, dtype=np.complex128)
    )
    assert inner.overlap(left, right)[0, 0] == pytest.approx(expected)
    assert inner.norm(left) == pytest.approx(
        grid.cell_volume / grid.point_count * np.sum(metric)
    )

    mapping = grid.extension([2, 2], [[2.0, 0.0], [0.0, 3.0]], 1.0)
    assert grid.shape == (4, 6)
    assert grid.cell_volume == pytest.approx(24.0)
    assert np.array_equal(
        metric[mapping].reshape(grid.shape),
        np.tile(metric.reshape(2, 3), (2, 2)),
    )


def test_uniform_metric_inner_product_supports_vector_fields():
    grid = PeriodicGrid((2, 2, 2), np.eye(3))
    metric = np.linspace(1.0, 2.0, grid.point_count)
    left = np.zeros((2, grid.point_count, 3), dtype=np.complex128)
    right = np.zeros_like(left)
    left[0, :, 0] = 1.0
    left[1, :, 1] = 1.0j
    right[0, :, 0] = 2.0
    right[1, :, 1] = 3.0j
    inner = UniformGridInnerProduct(grid, metric)

    expected_weight = np.sum(metric) / grid.point_count
    overlap = inner.overlap(left, right)
    assert np.allclose(overlap, np.diag([2.0, 3.0]) * expected_weight)
    columns = np.moveaxis(left, 0, -1)
    assert np.allclose(inner.norms(columns), [expected_weight, expected_weight])


def test_mpb_grid_rejects_shifted_coordinate_metadata(tmp_path):
    with h5py.File(tmp_path / "grid.h5", "w") as handle:
        handle.attrs["dimension"] = 2
        handle.create_dataset("shape", data=np.array([2, 1]))
        handle.create_dataset("basis", data=np.eye(3))
        handle.create_dataset("u1", data=np.array([-0.25, 0.25]))
        handle.create_dataset("u2", data=np.array([0.0]))

    with pytest.raises(ValueError, match="i/N-1/2"):
        load_mpb_grid(tmp_path / "grid.h5")


def test_mpb_source_loads_three_dimensional_vector_fields_without_reordering(tmp_path):
    shape = (2, 2, 2)
    kpoints = np.array([[0.0, 0.0, 0.0]])
    with h5py.File(tmp_path / "grid.h5", "w") as handle:
        handle.attrs["dimension"] = 3
        handle.create_dataset("shape", data=np.asarray(shape, dtype=np.int64))
        handle.create_dataset("basis", data=np.eye(3))
        for axis in range(3):
            handle.create_dataset(f"u{axis + 1}", data=np.array([-0.5, 0.0]))
    with h5py.File(tmp_path / "E.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("E", data=np.array([[1.0, 2.0]]))
    raw = np.arange(1 * 2 * 2 * 2 * 2 * 3, dtype=float).reshape(
        1, 2, 2, 2, 2, 3
    )
    with h5py.File(tmp_path / "fields.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("H_periodic", data=raw)
    with h5py.File(tmp_path / "EL.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("E", data=np.array([[0.0, 2.0]]))
    with h5py.File(tmp_path / "longitudinal.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("H_periodic", data=raw + 100.0)
    scalar = np.arange(1 * 2 * 2 * 2 * 2, dtype=float).reshape(1, 2, 2, 2, 2)
    with h5py.File(tmp_path / "phi.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("phi_periodic", data=scalar)
        handle.create_dataset("zero_mode", data=np.array([[True, False]]))
    with h5py.File(tmp_path / "epsilon.h5", "w") as handle:
        handle.create_dataset("epsilon", data=np.full(shape, 2.0))
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
                "origin = 0, 0, 0",
                "k_points = 0:1:1, 0:1:1, 0:1:1",
                "band_window = 0:2",
                "dataset_file = ./fields.h5",
                "mesh_file = ./grid.h5",
                "metric_file = false",
                "E_file = ./E.h5",
                "longitudinal_field_file = ./longitudinal.h5",
                "longitudinal_energy_file = ./EL.h5",
                "pseudoscalar_file = ./phi.h5",
                "pseudoscalar_metric_file = ./epsilon.h5",
                "symmetry_file = Pm-3m",
                "representation_analysis",
                "Gamma; 0, 0, 0; 0:2",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    bundle = load_input(load_config(incar, mode="bloch_symmetry"))

    assert bundle.fields[0, 0, 0].shape == (2, 8, 3)
    assert np.array_equal(bundle.fields[0, 0, 0], raw[0].reshape(2, 8, 3))
    assert np.array_equal(bundle.metric_material, np.ones(8))
    assert tuple(bundle.auxiliary_bundle_loaders) == ("longitudinal", "pseudoscalar")

    longitudinal = bundle.auxiliary_bundle_loaders["longitudinal"]()
    pseudoscalar = bundle.auxiliary_bundle_loaders["pseudoscalar"]()
    assert longitudinal.fields[0, 0, 0].shape == (1, 8, 3)
    assert longitudinal.band_indices[0, 0, 0] == [1]
    assert np.array_equal(
        longitudinal.fields[0, 0, 0][0],
        (raw[0, 1] + 100.0).reshape(8, 3),
    )
    assert pseudoscalar.fields[0, 0, 0].shape == (2, 8)
    assert np.array_equal(pseudoscalar.metric_material, np.full(8, 2.0))
    assert np.array_equal(pseudoscalar.zero_modes[0, 0, 0], [True, False])


def test_mpb_transverse_plus_longitudinal_combines_independent_windows_and_regularizes_gamma(tmp_path):
    shape = (2, 2, 2)
    kpoints = np.zeros((1, 3), dtype=float)
    with h5py.File(tmp_path / "grid.h5", "w") as handle:
        handle.attrs["dimension"] = 3
        handle.create_dataset("shape", data=np.asarray(shape, dtype=np.int64))
        handle.create_dataset("basis", data=np.eye(3))
        for axis in range(3):
            handle.create_dataset(f"u{axis + 1}", data=np.array([-0.5, 0.0]))
    with h5py.File(tmp_path / "E.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("E", data=np.array([[0.0, 0.0, 2.0]]))
    physical = np.ones((1, 3) + shape + (3,), dtype=np.complex128)
    with h5py.File(tmp_path / "H.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("H_periodic", data=physical)
    with h5py.File(tmp_path / "EL.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("E", data=np.array([[0.0, 3.0]]))
    longitudinal = np.full((1, 2) + shape + (3,), 2.0, dtype=np.complex128)
    with h5py.File(tmp_path / "HL.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("H_periodic", data=longitudinal)
    scalar = np.ones((1, 2) + shape, dtype=np.complex128)
    with h5py.File(tmp_path / "phi.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("phi_periodic", data=scalar)
        handle.create_dataset("zero_mode", data=np.array([[True, False]]))
    with h5py.File(tmp_path / "epsilon.h5", "w") as handle:
        handle.create_dataset("epsilon", data=np.ones(shape))
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "dataset_type = mpb",
                "field_components = full_vector",
                "primary_field = magnetic",
                "wannier_subspace = T + L",
                "invert_longitudinal_energies = true",
                "lattice_const = 1",
                "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
                "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
                "k_points = 0:1:1, 0:1:1, 0:1:1",
                "band_window = 0:3",
                "inner_window = false",
                "longitudinal_band_window = 0:2",
                "longitudinal_inner_window = false",
                "dataset_file = ./H.h5",
                "mesh_file = ./grid.h5",
                "metric_file = false",
                "E_file = ./E.h5",
                "longitudinal_field_file = ./HL.h5",
                "longitudinal_energy_file = ./EL.h5",
                "pseudoscalar_file = ./phi.h5",
                "pseudoscalar_metric_file = ./epsilon.h5",
                "symmetry_file = Pm-3m",
                "representation_analysis",
                "Gamma; 0,0,0; H[0:3],L[0:2]",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(incar, mode="bloch_symmetry")
    assert config.wannier_subspace == "T+L"
    assert config.invert_longitudinal_energies is True
    bundle = load_input(config)

    assert bundle.fields[0, 0, 0].shape == (5, 8, 3)
    assert bundle.band_indices[0, 0, 0] == [0, 1, 2, 3, 4]
    assert bundle.band_channels[0].label == "H:0"
    assert bundle.band_channels[3].label == "L:0"
    selected = bundle.symmetry.model.representation_analysis.points[0].band_indices
    assert selected == (0, 1, 2, 3, 4)
    zero_fields = bundle.fields[0, 0, 0][[0, 1, 3]]
    gram = UniformGridInnerProduct(bundle.mesh, bundle.metric_material).overlap(
        zero_fields, zero_fields
    )
    assert np.allclose(gram, np.eye(3), atol=1.0e-14)


def test_periodic_grid_quasiperiodic_stencil_tracks_each_corner_shift():
    grid = PeriodicGrid((2, 2), np.eye(2))
    stencil = PeriodicGridInterpolator(grid).stencil(
        np.array([[0.5, -0.5], [0.5, 0.5]])
    )
    values = np.ones((1, grid.point_count), dtype=np.complex128)
    kpoint = np.array([0.25, 0.125])

    actual = stencil.apply_quasiperiodic(values, kpoint, bloch_sign=1)
    expected = np.exp(
        2j * np.pi * np.array(
            [
                kpoint[0],
                kpoint[0] + kpoint[1],
            ]
        )
    )

    assert np.allclose(actual[0], expected, rtol=0.0, atol=1.0e-14)


def test_mpb_source_loads_reordered_hdf5_kpoints(tmp_path):
    shape = (2, 2)
    configured_kpoints = np.array(
        [
            [-0.5, -0.5, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    eigenvalue_order = np.array([2, 0, 3, 1])
    field_order = np.array([1, 3, 0, 2])
    eigenvalues = np.column_stack(
        (
            1.1 + 0.1 * np.arange(4),
            2.2 + 0.1 * np.arange(4),
        )
    )

    with h5py.File(tmp_path / "grid.h5", "w") as handle:
        handle.attrs["dimension"] = 2
        handle.create_dataset("shape", data=np.asarray(shape, dtype=np.int64))
        handle.create_dataset("basis", data=np.eye(3))
        handle.create_dataset("u1", data=np.array([-0.5, 0.0]))
        handle.create_dataset("u2", data=np.array([-0.5, 0.0]))

    with h5py.File(tmp_path / "E.h5", "w") as handle:
        handle.create_dataset("kpoints", data=configured_kpoints[eigenvalue_order])
        handle.create_dataset("E", data=eigenvalues[eigenvalue_order])

    field_values = np.zeros((4, 2, 2, 2, 1, 3), dtype=np.complex128)
    for configured_index in range(4):
        for band in range(2):
            field_values[configured_index, band, ..., 0, 2] = (
                10.0 * configured_index + band + 1.0
            )
    with h5py.File(tmp_path / "fields.h5", "w") as handle:
        handle.create_dataset("kpoints", data=configured_kpoints[field_order])
        handle.create_dataset("H_periodic", data=field_values[field_order])

    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "name = synthetic_mpb",
                "dataset_type = mpb",
                "field_components = Hz",
                "lattice_const = 1",
                "real_lattice_vectors = 1 0, 0 1",
                "reciprocal_lattice_vectors = 0 0, 0 0",
                "k_points = -0.5:0.5:0.5, -0.5:0.5:0.5",
                "composition_of_b = 1 0, 0 1",
                "band_window = 0:2",
                "dataset_file = ./fields.h5",
                "mesh_file = ./grid.h5",
                "metric_file = false",
                "E_file = ./E.h5",
                "extension = 1, 1",
                "projections",
                "a; [0, 0]; 0; [1, 0, 1]; [2, 1, 1]",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    bundle = load_input(load_config(incar))

    assert isinstance(bundle.mesh, PeriodicGrid)
    assert bundle.mesh.shape == shape
    assert bundle.bloch_convention == BlochConvention(1, "mpb")
    assert bundle.field_representation is BlochFieldRepresentation.PERIODIC_PART
    assert np.array_equal(bundle.metric_material, np.ones(4))
    for flat, index in enumerate(np.ndindex((2, 2, 1))):
        assert np.all(bundle.fields[index][0] == 10.0 * flat + 1.0)
        assert np.all(bundle.fields[index][1] == 10.0 * flat + 2.0)
        assert np.allclose(bundle.energies[index], eigenvalues[flat])
