from __future__ import annotations

from pathlib import Path
import logging

import h5py
import numpy as np

from ..config import EnergyWindow, IncarConfig
from ..conventions import (
    BlochConvention,
    BlochFieldRepresentation,
)
from ..data import InputBundle, PeriodicGrid
from ..maxwell import FieldComponents
from ..timing import timed_step
from .base import SourceAdapter

LOGGER = logging.getLogger(__name__)
MPB_BLOCH_CONVENTION = BlochConvention(1, "mpb")
_K_TOLERANCE = 1.0e-10


def _read_required_dataset(handle: h5py.File, name: str) -> np.ndarray:
    if name not in handle:
        raise ValueError(
            f"MPB HDF5 file {handle.filename!r} is missing dataset /{name}."
        )
    result = np.asarray(handle[name])
    if not np.issubdtype(result.dtype, np.number):
        raise ValueError(
            f"MPB dataset /{name} in {handle.filename!r} must be numeric."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(
            f"MPB dataset /{name} in {handle.filename!r} contains NaN or Inf."
        )
    return result


def load_mpb_grid(filename: str | Path) -> PeriodicGrid:
    path = Path(filename)
    with timed_step("read MPB grid", LOGGER, file=path):
        with h5py.File(path, "r") as handle:
            shape_array = _read_required_dataset(handle, "shape")
            shape = tuple(int(value) for value in shape_array.reshape(-1))
            dimension = int(
                handle.attrs.get(
                    "dimension",
                    handle.attrs.get("dimensions", len(shape)),
                )
            )
            if dimension != len(shape):
                raise ValueError(
                    f"MPB grid dimension {dimension} does not match shape {shape}."
                )
            if dimension not in {2, 3}:
                raise NotImplementedError(
                    f"MPB grid dimension {dimension} is not supported; expected 2 or 3."
                )
            basis = _read_required_dataset(handle, "basis")
            if basis.shape != (3, 3):
                raise ValueError(
                    f"MPB /basis must have shape (3, 3); got {basis.shape}."
                )
            # MPB stores a1, a2, a3 as columns. Internal lattice vectors are rows.
            lattice = np.asarray(basis[:dimension, :dimension].T, dtype=float)
            grid = PeriodicGrid(shape, lattice, sample_offset=0.0)
            _check_exported_coordinate_axes(handle, grid)

    LOGGER.info(
        "MPB grid loaded: dimension=%s shape=%s points=%s volume=%.12g",
        grid.dimension,
        grid.shape,
        grid.point_count,
        grid.cell_volume,
    )
    return grid


def _check_exported_coordinate_axes(
    handle: h5py.File,
    grid: PeriodicGrid,
) -> None:
    """Diagnose exporter metadata without overriding MPB sample indexing.

    MPB arrays use the half-open sample indexing i/N-1/2 requested by the
    adapter contract. Some exporters write cell-center helper axes while the
    actual field arrays remain node indexed; those helper arrays are therefore
    informative only.
    """

    for axis, size in enumerate(grid.shape):
        name = f"u{axis + 1}"
        if name not in handle:
            continue
        values = np.asarray(handle[name], dtype=float).reshape(-1)
        expected = np.arange(size, dtype=float) / size - 0.5
        if values.shape != expected.shape:
            raise ValueError(
                f"MPB coordinate helper /{name} has shape {values.shape}; "
                f"expected {expected.shape}."
            )
        spacing = np.diff(values)
        if spacing.size and not np.allclose(
            spacing,
            1.0 / size,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"MPB coordinate helper /{name} is not uniform.")
        if not np.allclose(values, expected, rtol=0.0, atol=1.0e-12):
            LOGGER.warning(
                "MPB coordinate helper /%s uses an offset different from the "
                "field-array indexing; using the half-open i/N-1/2 convention",
                name,
            )


def load_mpb_input(config: IncarConfig) -> InputBundle:
    if config.maxwell_problem is None:
        raise ValueError("Maxwell field configuration has not been initialized.")
    if config.maxwell_problem.field_components is not FieldComponents.HZ:
        raise NotImplementedError(
            "The MPB adapter currently supports the scalar Hz component only."
        )
    mesh_path = config.input_path(config.mesh_file)
    field_path = config.input_path(config.dataset_file)
    eigenvalue_path = config.input_path(config.E_file)
    if mesh_path is None or field_path is None or eigenvalue_path is None:
        raise ValueError("MPB mesh_file, dataset_file, and E_file are required.")

    grid = load_mpb_grid(mesh_path)
    if grid.dimension != int(config.kdim or 0):
        raise ValueError(
            f"MPB grid dimension {grid.dimension} does not match incar kdim={config.kdim}."
        )
    if grid.dimension != 2:
        raise NotImplementedError(
            "The MPB data model is 3D-ready, but the current Wannier pipeline "
            "implements only 2D scalar fields."
        )
    _validate_lattice(config, grid)

    with timed_step("read MPB Maxwell eigenvalues", LOGGER, file=eigenvalue_path):
        with h5py.File(eigenvalue_path, "r") as handle:
            eigenvalues = np.asarray(
                _read_required_dataset(handle, "E"),
                dtype=np.float64,
            )
            energy_kpoints = np.asarray(
                _read_required_dataset(handle, "kpoints"),
                dtype=np.float64,
            )
    if eigenvalues.ndim != 2:
        raise ValueError(
            "MPB Maxwell eigenvalues /E must have shape (Nk, bands); "
            f"got {eigenvalues.shape}."
        )
    if energy_kpoints.shape[0] != eigenvalues.shape[0]:
        raise ValueError(
            "MPB eigenvalue and kpoint row counts differ: "
            f"{eigenvalues.shape[0]} != {energy_kpoints.shape[0]}."
        )
    row_for_k = _map_kpoints(config, energy_kpoints)
    # /E stores the dimensionless Maxwell eigenvalue (omega*a/c)^2.
    energy_rows = eigenvalues[row_for_k] / float(config.lattice_const) ** 2
    k_shape = _configured_k_shape(config)
    energy_matrix = energy_rows.reshape(k_shape + (eigenvalues.shape[1],))
    energies, band_indices, inner_band_indices = _select_bands(
        config, energy_matrix
    )

    with timed_step("read MPB periodic Hz fields", LOGGER, file=field_path):
        fields = _read_periodic_hz_fields(
            field_path,
            config,
            grid,
            band_indices,
            eigenvalues.shape[1],
        )
    metric_material = _load_metric_material(config, grid)

    band_lengths = [
        len(band_indices[index]) for index in np.ndindex(band_indices.shape)
    ]
    LOGGER.info(
        "MPB input prepared: field=Hz representation=periodic_part k_shape=%s "
        "grid_shape=%s bands_per_k=min:%s max:%s metric=%s",
        fields.shape,
        grid.shape,
        min(band_lengths) if band_lengths else 0,
        max(band_lengths) if band_lengths else 0,
        config.maxwell_problem.metric_material.value,
    )
    return InputBundle(
        config=config,
        maxwell=config.maxwell_problem,
        bloch_convention=MPB_BLOCH_CONVENTION,
        mesh=grid,
        fields=fields,
        metric_material=metric_material,
        energies=energies,
        band_indices=band_indices,
        inner_band_indices=inner_band_indices,
        energy_matrix=energy_matrix,
        field_representation=BlochFieldRepresentation.PERIODIC_PART,
        symmetry=config.symmetry_context,
    )


def _validate_lattice(config: IncarConfig, grid: PeriodicGrid) -> None:
    configured = (
        np.asarray(config.real_lattice_vectors, dtype=float)
        * float(config.lattice_const)
    )
    if configured.shape != grid.lattice_vectors.shape or not np.allclose(
        configured,
        grid.lattice_vectors,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            "MPB /basis does not match lattice_const * real_lattice_vectors: "
            f"file={grid.lattice_vectors.tolist()}, configured={configured.tolist()}."
        )


def _configured_k_shape(config: IncarConfig) -> tuple[int, int, int]:
    dimension = int(config.kdim or 0)
    return tuple(
        len(config.k_points[axis]) if axis < dimension else 1
        for axis in range(3)
    )


def _configured_k_rows(config: IncarConfig) -> np.ndarray:
    axes = [
        np.asarray(config.k_points[axis], dtype=float)
        for axis in range(int(config.kdim or 0))
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(axes)
    )


def _map_kpoints(config: IncarConfig, file_kpoints: np.ndarray) -> np.ndarray:
    dimension = int(config.kdim or 0)
    points = np.asarray(file_kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] < dimension:
        raise ValueError(
            f"MPB kpoints must have at least {dimension} columns; got {points.shape}."
        )
    if points.shape[1] > dimension and not np.allclose(
        points[:, dimension:],
        0.0,
        rtol=0.0,
        atol=_K_TOLERANCE,
    ):
        raise ValueError("MPB kpoints contain non-zero unused dimensions.")
    points = points[:, :dimension]
    expected = _configured_k_rows(config)
    if points.shape[0] != expected.shape[0]:
        raise ValueError(
            f"MPB kpoint count {points.shape[0]} does not match incar "
            f"k-grid count {expected.shape[0]}."
        )

    result = np.empty(expected.shape[0], dtype=np.intp)
    used: set[int] = set()
    for index, target in enumerate(expected):
        difference = points - target
        difference -= np.rint(difference)
        distances = np.max(np.abs(difference), axis=1)
        matches = np.flatnonzero(distances <= _K_TOLERANCE)
        if matches.size != 1:
            raise ValueError(
                "MPB kpoint mapping is missing or ambiguous: "
                f"k={target.tolist()}, matches={matches.tolist()}."
            )
        row = int(matches[0])
        if row in used:
            raise ValueError(f"MPB kpoint row {row} is mapped more than once.")
        used.add(row)
        result[index] = row
    return result


def _select_bands(
    config: IncarConfig,
    energy_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_shape = energy_matrix.shape[:3]
    energies = np.empty(k_shape, dtype=object)
    band_indices = np.empty(k_shape, dtype=object)
    inner_indices = np.empty(k_shape, dtype=object)

    for index in np.ndindex(k_shape):
        line = np.asarray(energy_matrix[index], dtype=float)
        if isinstance(config.band_window, EnergyWindow):
            outer = np.flatnonzero(
                (line >= config.band_window.emin)
                & (line <= config.band_window.emax)
            )
        else:
            outer = np.asarray(config.band_window, dtype=int)
        if outer.size == 0 or np.any(outer < 0) or np.any(outer >= line.size):
            raise ValueError(
                f"MPB outer band window is empty or out of range at k={index}."
            )
        band_indices[index] = outer.tolist()
        energies[index] = line[outer]

        if config.inner_window is False:
            inner = np.empty(0, dtype=int)
        elif isinstance(config.inner_window, EnergyWindow):
            inner = np.flatnonzero(
                (line >= config.inner_window.emin)
                & (line <= config.inner_window.emax)
            )
        else:
            inner = np.asarray(config.inner_window, dtype=int)
        missing = sorted(set(int(value) for value in inner) - set(outer.tolist()))
        if missing:
            raise ValueError(
                f"MPB frozen bands {missing} are outside the outer window at k={index}."
            )
        inner_indices[index] = inner.tolist()
    return energies, band_indices, inner_indices


def _read_periodic_hz_fields(
    path: Path,
    config: IncarConfig,
    grid: PeriodicGrid,
    band_indices: np.ndarray,
    band_count: int,
) -> np.ndarray:
    k_shape = _configured_k_shape(config)
    fields = np.empty(k_shape, dtype=object)
    with h5py.File(path, "r") as handle:
        if "H_periodic" not in handle:
            raise ValueError(
                f"MPB field file {path} is missing dataset /H_periodic."
            )
        dataset = handle["H_periodic"]
        expected_tail = (
            grid.shape + ((1,) if grid.dimension == 2 else ()) + (3,)
        )
        expected_ndim = 2 + len(expected_tail)
        if dataset.ndim != expected_ndim or dataset.shape[1] != band_count:
            raise ValueError(
                "MPB /H_periodic must have shape "
                f"(Nk, {band_count}, {', '.join(str(v) for v in expected_tail)}); "
                f"got {dataset.shape}."
            )
        if tuple(dataset.shape[2:]) != expected_tail:
            raise ValueError(
                f"MPB field grid shape {dataset.shape[2:]} does not match "
                f"grid metadata {expected_tail}."
            )
        field_kpoints = np.asarray(
            _read_required_dataset(handle, "kpoints"), dtype=float
        )
        try:
            field_row_for_k = _map_kpoints(config, field_kpoints)
        except ValueError as exc:
            raise ValueError(
                "MPB field kpoints do not match the configured k grid."
            ) from exc

        for flat, index in enumerate(np.ndindex(k_shape)):
            selected = np.asarray(band_indices[index], dtype=np.intp)
            row_block = np.asarray(
                dataset[int(field_row_for_k[flat])],
                dtype=np.complex128,
            )
            block = np.asarray(
                row_block[selected],
                dtype=np.complex128,
            )
            if grid.dimension == 2:
                block = block[..., 0, :]
            if not np.all(np.isfinite(block)):
                raise ValueError(
                    f"MPB Hz fields contain NaN or Inf at k={index}."
                )
            transverse = float(
                np.max(np.abs(block[..., :2]), initial=0.0)
            )
            longitudinal = max(
                float(np.max(np.abs(block[..., 2]), initial=0.0)),
                np.finfo(float).tiny,
            )
            if transverse > 1.0e-10 * longitudinal:
                raise ValueError(
                    "MPB scalar Hz mode contains non-zero Hx/Hy components: "
                    f"k={index}, relative={transverse / longitudinal:.6g}."
                )
            fields[index] = np.ascontiguousarray(
                block[..., 2].reshape(selected.size, grid.point_count)
            )
    return fields


def _load_metric_material(
    config: IncarConfig,
    grid: PeriodicGrid,
) -> np.ndarray:
    path = config.input_path(config.metric_file)
    if path is None:
        LOGGER.info(
            "MPB metric_file is disabled; using unit %s for scalar Hz",
            config.maxwell_problem.metric_material.value,
        )
        return np.ones(grid.point_count, dtype=float)

    expected_name = config.maxwell_problem.metric_material.value
    with timed_step("read MPB metric material", LOGGER, file=path):
        with h5py.File(path, "r") as handle:
            candidates = (expected_name, "metric")
            name = next((value for value in candidates if value in handle), None)
            if name is None:
                available = ", ".join(sorted(handle.keys()))
                raise ValueError(
                    f"MPB metric file {path} must contain /{expected_name} or "
                    f"/metric; available datasets: {available or '<none>'}."
                )
            values = np.asarray(_read_required_dataset(handle, name)).squeeze()
    if values.shape != grid.shape:
        raise ValueError(
            f"MPB metric dataset has shape {values.shape}; expected {grid.shape}."
        )
    if np.iscomplexobj(values) and np.max(np.abs(values.imag), initial=0.0) > 1.0e-12:
        raise ValueError("MPB metric material must be real.")
    return np.asarray(values.real, dtype=float).reshape(-1)


MPB_SOURCE = SourceAdapter(
    name="mpb",
    bloch_convention=MPB_BLOCH_CONVENTION,
    supported_field_components=frozenset({FieldComponents.HZ}),
    input_loader=load_mpb_input,
    mesh_loader=load_mpb_grid,
    required_config_fields=("mesh_file", "dataset_file", "E_file"),
    supported_dimensions=frozenset({2}),
)
