from __future__ import annotations

from pathlib import Path
from copy import copy
import logging

import h5py
import numpy as np

from ..config import EnergyWindow, IncarConfig
from ..conventions import (
    BlochConvention,
    BlochFieldRepresentation,
)
from ..data import InputBundle, PeriodicGrid, periodic_axis_coordinates
from ..maxwell import FieldComponents, FieldKind, PrimaryField
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
    """Validate exported axes without changing the MPB array indexing."""

    for axis, size in enumerate(grid.shape):
        name = f"u{axis + 1}"
        if name not in handle:
            raise ValueError(
                f"MPB grid file {handle.filename!r} is missing coordinate helper /{name}."
            )
        values = np.asarray(handle[name], dtype=float).reshape(-1)
        expected = periodic_axis_coordinates(size)
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
            raise ValueError(
                f"MPB coordinate helper /{name} does not use the required "
                "u_i=i/N-1/2 convention (with u=0 for N=1)."
            )


def load_mpb_input(config: IncarConfig) -> InputBundle:
    if config.maxwell_problem is None:
        raise ValueError("Maxwell field configuration has not been initialized.")
    components = config.maxwell_problem.field_components
    if components is FieldComponents.FULL_VECTOR and (
        config.maxwell_problem.primary_field is not PrimaryField.MAGNETIC
    ):
        raise NotImplementedError(
            "The MPB 3D adapter currently supports magnetic full-vector fields only."
        )
    if components not in {FieldComponents.HZ, FieldComponents.FULL_VECTOR}:
        raise NotImplementedError(
            "The MPB adapter supports scalar Hz and 3D magnetic full-vector fields."
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
    if components is FieldComponents.HZ and grid.dimension != 2:
        raise ValueError("field_components=Hz requires a two-dimensional MPB grid.")
    if components is FieldComponents.FULL_VECTOR and grid.dimension != 3:
        raise ValueError("field_components=full_vector requires a three-dimensional MPB grid.")
    _validate_lattice(config, grid)

    eigenvalues, row_for_k = _read_eigenvalues(
        eigenvalue_path,
        config,
        description="Maxwell",
    )
    # /E stores the dimensionless Maxwell eigenvalue (omega*a/c)^2.
    energy_rows = eigenvalues[row_for_k] / float(config.lattice_const) ** 2
    k_shape = _configured_k_shape(config)
    energy_matrix = energy_rows.reshape(k_shape + (eigenvalues.shape[1],))
    energies, band_indices, inner_band_indices = _select_bands(
        config, energy_matrix
    )

    with timed_step("read MPB periodic magnetic fields", LOGGER, file=field_path):
        fields = _read_periodic_fields(
            field_path,
            config,
            grid,
            band_indices,
            eigenvalues.shape[1],
            dataset_name="H_periodic",
            vector=components is FieldComponents.FULL_VECTOR,
        )
    metric_material = _load_metric_material(config, grid)
    auxiliary_loaders = _build_auxiliary_channel_loaders(config, grid)

    band_lengths = [
        len(band_indices[index]) for index in np.ndindex(band_indices.shape)
    ]
    LOGGER.info(
        "MPB input prepared: field=%s representation=periodic_part k_shape=%s "
        "grid_shape=%s bands_per_k=min:%s max:%s metric=%s",
        config.maxwell_problem.field_components.value,
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
        analysis_field_kind=config.maxwell_problem.symmetry_field_kind,
        auxiliary_bundle_loaders=auxiliary_loaders,
    )


def _read_eigenvalues(
    path: Path,
    config: IncarConfig,
    *,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    with timed_step(f"read MPB {description} eigenvalues", LOGGER, file=path):
        with h5py.File(path, "r") as handle:
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
            f"MPB {description} eigenvalues /E must have shape (Nk, bands); "
            f"got {eigenvalues.shape}."
        )
    if energy_kpoints.shape[0] != eigenvalues.shape[0]:
        raise ValueError(
            f"MPB {description} eigenvalue and kpoint row counts differ: "
            f"{eigenvalues.shape[0]} != {energy_kpoints.shape[0]}."
        )
    return eigenvalues, _map_kpoints(config, energy_kpoints)


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


def _read_periodic_fields(
    path: Path,
    config: IncarConfig,
    grid: PeriodicGrid,
    band_indices: np.ndarray,
    band_count: int,
    *,
    dataset_name: str,
    vector: bool,
) -> np.ndarray:
    k_shape = _configured_k_shape(config)
    fields = np.empty(k_shape, dtype=object)
    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            raise ValueError(
                f"MPB field file {path} is missing dataset /{dataset_name}."
            )
        dataset = handle[dataset_name]
        expected_tail = grid.shape
        if vector:
            expected_tail = expected_tail + (3,)
        elif dataset_name == "H_periodic":
            expected_tail = expected_tail + ((1,) if grid.dimension == 2 else ()) + (3,)
        expected_ndim = 2 + len(expected_tail)
        if dataset.ndim != expected_ndim or dataset.shape[1] != band_count:
            raise ValueError(
                f"MPB /{dataset_name} must have shape "
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
            if dataset_name == "H_periodic" and not vector:
                block = block[..., 0, :]
            if not np.all(np.isfinite(block)):
                raise ValueError(
                    f"MPB Hz fields contain NaN or Inf at k={index}."
                )
            if vector:
                fields[index] = np.ascontiguousarray(
                    block.reshape(selected.size, grid.point_count, 3)
                )
            elif dataset_name == "H_periodic":
                transverse = float(np.max(np.abs(block[..., :2]), initial=0.0))
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
            else:
                fields[index] = np.ascontiguousarray(
                    block.reshape(selected.size, grid.point_count)
                )
    return fields


def _build_auxiliary_channel_loaders(
    config: IncarConfig,
    grid: PeriodicGrid,
):
    values = {
        "longitudinal_field_file": config.input_path(config.longitudinal_field_file),
        "longitudinal_energy_file": config.input_path(config.longitudinal_energy_file),
        "pseudoscalar_file": config.input_path(config.pseudoscalar_file),
        "pseudoscalar_metric_file": config.input_path(config.pseudoscalar_metric_file),
    }
    if not any(path is not None for path in values.values()):
        if config.gamma_zero_regularization:
            raise ValueError(
                "gamma_zero_regularization requires longitudinal and pseudoscalar MPB files."
            )
        return {}
    missing = [name for name, path in values.items() if path is None]
    if missing:
        raise ValueError(
            "MPB longitudinal analysis requires all auxiliary files; missing "
            + ", ".join(missing)
        )
    if grid.dimension != 3:
        raise NotImplementedError("MPB longitudinal auxiliary channels require a 3D grid.")

    energy_path = values["longitudinal_energy_file"]
    assert energy_path is not None
    raw_energy, rows = _read_eigenvalues(
        energy_path,
        config,
        description="longitudinal",
    )
    k_shape = _configured_k_shape(config)
    energy_matrix = (
        raw_energy[rows] / float(config.lattice_const) ** 2
    ).reshape(k_shape + (raw_energy.shape[1],))
    energies, bands, inner = _select_bands(config, energy_matrix)

    field_path = values["longitudinal_field_file"]
    scalar_path = values["pseudoscalar_file"]
    metric_path = values["pseudoscalar_metric_file"]
    assert field_path is not None and scalar_path is not None and metric_path is not None
    zero_modes = _read_selected_zero_modes(scalar_path, config, bands, raw_energy.shape[1])
    longitudinal_config = copy(config)
    longitudinal_config.S_file = config.longitudinal_S_file
    longitudinal_config.D_file = config.longitudinal_D_file
    pseudoscalar_config = copy(config)
    pseudoscalar_config.S_file = config.pseudoscalar_S_file
    pseudoscalar_config.D_file = config.pseudoscalar_D_file

    def load_longitudinal() -> InputBundle:
        longitudinal_fields = _read_periodic_fields(
            field_path,
            config,
            grid,
            bands,
            raw_energy.shape[1],
            dataset_name="H_periodic",
            vector=True,
        )
        longitudinal_energies = np.empty(energies.shape, dtype=object)
        longitudinal_bands = np.empty(bands.shape, dtype=object)
        longitudinal_inner = np.empty(inner.shape, dtype=object)
        for index in np.ndindex(bands.shape):
            mask = ~np.asarray(zero_modes[index], dtype=bool)
            selected_bands = np.asarray(bands[index], dtype=int)
            longitudinal_fields[index] = np.asarray(longitudinal_fields[index])[mask]
            longitudinal_energies[index] = np.asarray(energies[index])[mask]
            longitudinal_bands[index] = selected_bands[mask].tolist()
            retained = set(longitudinal_bands[index])
            longitudinal_inner[index] = [
                int(value) for value in inner[index] if int(value) in retained
            ]
        return InputBundle(
            config=longitudinal_config,
            maxwell=config.maxwell_problem,
            bloch_convention=MPB_BLOCH_CONVENTION,
            mesh=grid,
            fields=longitudinal_fields,
            metric_material=np.ones(grid.point_count, dtype=float),
            energies=longitudinal_energies,
            band_indices=longitudinal_bands,
            inner_band_indices=longitudinal_inner,
            energy_matrix=energy_matrix,
            field_representation=BlochFieldRepresentation.PERIODIC_PART,
            symmetry=config.symmetry_context,
            analysis_field_kind=FieldKind.MAGNETIC_AXIAL_VECTOR,
            zero_modes=zero_modes,
        )

    def load_pseudoscalar() -> InputBundle:
        scalar_fields = _read_periodic_fields(
            scalar_path,
            config,
            grid,
            bands,
            raw_energy.shape[1],
            dataset_name="phi_periodic",
            vector=False,
        )
        scalar_metric = _load_grid_material(
            metric_path,
            grid,
            candidates=("epsilon", "eta", "metric"),
            description="pseudoscalar metric",
        )
        return InputBundle(
            config=pseudoscalar_config,
            maxwell=config.maxwell_problem,
            bloch_convention=MPB_BLOCH_CONVENTION,
            mesh=grid,
            fields=scalar_fields,
            metric_material=scalar_metric,
            energies=energies,
            band_indices=bands,
            inner_band_indices=inner,
            energy_matrix=energy_matrix,
            field_representation=BlochFieldRepresentation.PERIODIC_PART,
            symmetry=config.symmetry_context,
            analysis_field_kind=FieldKind.PSEUDOSCALAR,
            zero_modes=zero_modes,
        )

    return {
        "longitudinal": load_longitudinal,
        "pseudoscalar": load_pseudoscalar,
    }


def _read_selected_zero_modes(
    path: Path,
    config: IncarConfig,
    band_indices: np.ndarray,
    band_count: int,
) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "zero_mode" not in handle:
            raise ValueError(f"MPB HDF5 file {str(path)!r} is missing dataset /zero_mode.")
        stored = np.asarray(handle["zero_mode"])
        if stored.dtype != np.bool_:
            if not np.issubdtype(stored.dtype, np.number) or not np.all(
                np.isin(stored, (0, 1))
            ):
                raise ValueError("MPB /zero_mode must contain boolean or 0/1 values.")
        raw = np.asarray(stored, dtype=bool)
        kpoints = np.asarray(_read_required_dataset(handle, "kpoints"), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != band_count:
        raise ValueError(
            f"MPB /zero_mode must have shape (Nk, {band_count}); got {raw.shape}."
        )
    rows = _map_kpoints(config, kpoints)
    output = np.empty(band_indices.shape, dtype=object)
    for flat, index in enumerate(np.ndindex(band_indices.shape)):
        selected = np.asarray(band_indices[index], dtype=np.intp)
        output[index] = np.asarray(raw[int(rows[flat]), selected], dtype=bool)
    return output


def _load_metric_material(
    config: IncarConfig,
    grid: PeriodicGrid,
) -> np.ndarray:
    path = config.input_path(config.metric_file)
    if path is None:
        LOGGER.info(
            "MPB metric_file is disabled; using unit %s",
            config.maxwell_problem.metric_material.value,
        )
        return np.ones(grid.point_count, dtype=float)

    expected_name = config.maxwell_problem.metric_material.value
    return _load_grid_material(
        path,
        grid,
        candidates=(expected_name, "metric"),
        description="metric material",
    )


def _load_grid_material(
    path: Path,
    grid: PeriodicGrid,
    *,
    candidates: tuple[str, ...],
    description: str,
) -> np.ndarray:
    with timed_step(f"read MPB {description}", LOGGER, file=path):
        with h5py.File(path, "r") as handle:
            name = next((value for value in candidates if value in handle), None)
            if name is None:
                available = ", ".join(sorted(handle.keys()))
                expected = ", ".join(f"/{value}" for value in candidates)
                raise ValueError(
                    f"MPB {description} file {path} must contain one of {expected}; "
                    f"available datasets: {available or '<none>'}."
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
    supported_field_components=frozenset(
        {FieldComponents.HZ, FieldComponents.FULL_VECTOR}
    ),
    input_loader=load_mpb_input,
    mesh_loader=load_mpb_grid,
    required_config_fields=("mesh_file", "dataset_file", "E_file"),
    supported_dimensions=frozenset({2, 3}),
)
