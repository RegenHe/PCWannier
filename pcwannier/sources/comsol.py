from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging
import re

import numpy as np
from scipy.spatial import cKDTree

from ..config import EnergyWindow, IncarConfig
from ..data import InputBundle, Mesh, RawData
from ..timing import timed_step

LOGGER = logging.getLogger(__name__)


def load_comsol_mesh(filename: str | Path) -> Mesh:
    path = Path(filename)
    with timed_step("read COMSOL mesh", LOGGER, file=path):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        vertices = _read_vertex_block(lines)
        elements = _read_element_block(lines, "tri", 3)
        edges = _read_element_block(lines, "edg", 2, required=False)
        mesh = Mesh(vertices, elements, edges)
    LOGGER.info(
        "COMSOL mesh loaded: vertices=%s triangles=%s edges=%s mindist=%.6g",
        mesh.vertices.shape[0],
        mesh.elements.shape[0],
        0 if mesh.edge is None else mesh.edge.shape[0],
        mesh.mindist,
    )
    return mesh


def _read_vertex_block(lines: list[str]) -> np.ndarray:
    count = None
    start = None
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if "# number of mesh vertices" in line:
            count = int(line.split()[0])
        elif line.startswith("# Mesh vertex coordinates"):
            start = idx + 1
            break
    if count is None or start is None:
        raise ValueError("COMSOL mesh is missing the vertex block.")
    data = np.fromstring("\n".join(lines[start : start + count]), sep=" ", dtype=float)
    if data.size < count * 2 or data.size % count != 0:
        raise ValueError("COMSOL mesh vertex block has an invalid shape.")
    return data.reshape(count, -1)[:, :2]


def _read_element_block(lines: list[str], name: str, width: int, *, required: bool = True) -> np.ndarray | None:
    marker_idx = None
    marker = f"3 {name}"
    for idx, raw in enumerate(lines):
        if raw.strip().startswith(marker):
            marker_idx = idx
            break
    if marker_idx is None:
        if required:
            raise ValueError(f"COMSOL mesh is missing the {name!r} element block.")
        return None

    count = None
    element_header = None
    for idx in range(marker_idx + 1, len(lines)):
        line = lines[idx].strip()
        if "number of elements" in line:
            count = int(line.split()[0])
        elif line.startswith("# Elements"):
            element_header = idx + 1
            break
    if count is None or element_header is None:
        raise ValueError(f"COMSOL {name!r} element block is incomplete.")

    data = np.fromstring("\n".join(lines[element_header : element_header + count]), sep=" ", dtype=np.intp)
    if data.size != count * width:
        raise ValueError(f"COMSOL {name!r} element block has an invalid shape.")
    return data.reshape(count, width)


def load_comsol_data(filename: str | Path, *, real_only: bool = False) -> RawData:
    path = Path(filename)
    column_parameters = _read_column_parameters(path)
    with timed_step("read COMSOL data", LOGGER, file=path):
        if real_only:
            try:
                points, values = _load_real_table(path)
                parser = "float"
            except ValueError:
                points, values = _load_token_table(path, real_values=True)
                parser = "complex-real"
            LOGGER.info(
                "COMSOL data loaded: file=%s points=%s value_cols=%s real_only=True parser=%s",
                path,
                points.shape[0],
                values.shape[1],
                parser,
            )
            return RawData(points, values, column_parameters)

        points, values = _load_token_table(path, real_values=False)
    LOGGER.info("COMSOL data loaded: file=%s points=%s value_cols=%s real_only=False", path, points.shape[0], values.shape[1])
    return RawData(points, values, column_parameters)


_PARAMETER_RE = re.compile(
    r"([A-Za-z_]\w*)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)


def _read_column_parameters(path: Path) -> dict[str, np.ndarray] | None:
    header = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if not raw.startswith("%"):
                break
            if "@" in raw and (header is None or raw.count("@") > header.count("@")):
                header = raw
    if header is None:
        return None

    rows = []
    for segment in header.split("@")[1:]:
        matches = _PARAMETER_RE.findall(segment)
        if matches:
            rows.append({name: float(value) for name, value in matches})
    if not rows:
        return None

    names = set(rows[0])
    if any(set(row) != names for row in rows):
        LOGGER.warning("COMSOL column parameter header is inconsistent in %s; metadata validation skipped", path)
        return None
    return {name: np.asarray([row[name] for row in rows], dtype=float) for name in sorted(names)}


def _load_real_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        raw = np.loadtxt(handle, comments="%", dtype=float, ndmin=2)
    if raw.shape[1] < 3:
        raise ValueError(f"Not enough COMSOL data columns in {path}.")
    return np.asarray(raw[:, :2], dtype=float), np.asarray(raw[:, 2:], dtype=float)


def _load_token_table(path: Path, *, real_values: bool) -> tuple[np.ndarray, np.ndarray]:
    rows, value_cols = _scan_data_shape(path)
    points = np.empty((rows, 2), dtype=float)
    value_dtype = float if real_values else np.complex128
    values = np.empty((rows, value_cols), dtype=value_dtype)

    row = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            tokens = line.split()
            points[row, 0] = float(tokens[0])
            points[row, 1] = float(tokens[1])
            parsed = np.fromiter(
                (_parse_comsol_complex(token) for token in tokens[2:]),
                dtype=np.complex128,
                count=value_cols,
            )
            values[row] = np.real(parsed) if real_values else parsed
            row += 1
    return points, values


def _parse_comsol_complex(token: str) -> complex:
    text = token.strip().strip("'\"").replace("−", "-").replace("i", "j")
    if text == "j":
        text = "1j"
    elif text == "-j":
        text = "-1j"
    return complex(text)


def _scan_data_shape(path: Path) -> tuple[int, int]:
    rows = 0
    value_cols = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            cols = len(line.split())
            if cols < 3:
                raise ValueError(f"Not enough COMSOL data columns in {path}: {line!r}")
            current_values = cols - 2
            if value_cols is None:
                value_cols = current_values
            elif current_values != value_cols:
                raise ValueError(
                    f"Ragged COMSOL data in {path}: expected {value_cols} value columns, got {current_values}."
                )
            rows += 1
    if rows == 0 or value_cols is None:
        raise ValueError(f"No data rows found in {path}.")
    return rows, value_cols


def match_data_to_mesh(
    mesh: Mesh,
    data: RawData,
    *,
    value_col: Optional[int] = None,
    max_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(mesh.vertices)
    data_to_mesh_dist, data_to_mesh_idx = tree.query(data.point_matrix, k=1)
    threshold = max_distance
    if threshold is None:
        threshold = max(float(mesh.mindist) * 0.5, 1e-8)

    order = np.lexsort((np.arange(data_to_mesh_idx.size), data_to_mesh_dist))
    mesh_to_data_idx = np.full(mesh.vertices.shape[0], -1, dtype=np.intp)
    mesh_dists = np.full(mesh.vertices.shape[0], np.inf, dtype=float)

    for data_idx in order:
        mesh_idx = int(data_to_mesh_idx[data_idx])
        if data_to_mesh_dist[data_idx] > threshold:
            continue
        if mesh_to_data_idx[mesh_idx] >= 0:
            continue
        mesh_to_data_idx[mesh_idx] = int(data_idx)
        mesh_dists[mesh_idx] = float(data_to_mesh_dist[data_idx])

    return mesh_to_data_idx, mesh_dists


def load_input(config: IncarConfig) -> InputBundle:
    mesh_path, dataset_path, metric_path, energy_path = _required_input_paths(config)
    if config.maxwell_problem is None:
        raise ValueError("Maxwell field configuration has not been initialized.")
    LOGGER.info(
        "COMSOL input paths: mesh=%s dataset=%s metric=%s(%s) energy=%s dataset_order=%s",
        mesh_path,
        dataset_path,
        metric_path,
        config.maxwell_problem.metric_material.value,
        energy_path,
        config.dataset_order,
    )

    mesh = load_comsol_mesh(mesh_path)
    raw_data = load_comsol_data(dataset_path)
    metric_raw = load_comsol_data(metric_path)
    energy_raw = load_comsol_data(energy_path, real_only=config.E_is_real or config.hermitian)
    _validate_header_k_grid(config, raw_data, dataset_path)
    _validate_header_k_grid(config, energy_raw, energy_path)
    if energy_raw.point_matrix.shape[0] != 1:
        raise ValueError(f"COMSOL energy file must contain exactly one spatial row; got {energy_raw.point_matrix.shape[0]}.")
    if raw_data.value_matrix.shape[1] != energy_raw.value_matrix.shape[1]:
        raise ValueError(
            "COMSOL field and energy column counts differ: "
            f"field={raw_data.value_matrix.shape[1]}, energy={energy_raw.value_matrix.shape[1]}."
        )

    with timed_step("prepare COMSOL tensors", LOGGER):
        energy_matrix, energies, band_indices, inner_band_indices = _handle_energy_data(config, energy_raw)
        fields = _distribute_fields(config, _values_on_mesh(mesh, raw_data), band_indices)
        metric_material = _metric_on_mesh(
            mesh,
            metric_raw,
            material=config.maxwell_problem.metric_material.value,
            path=metric_path,
        )

    periodic_residuals = _periodic_boundary_residuals(config, mesh, fields)
    for axis, residuals in enumerate(periodic_residuals):
        if residuals.size == 0:
            continue
        level = logging.WARNING if float(np.max(residuals)) > 5e-2 else logging.INFO
        LOGGER.log(
            level,
            "COMSOL periodic boundary residual: axis=%s median=%.6g max=%.6g samples=%s",
            axis + 1,
            float(np.median(residuals)),
            float(np.max(residuals)),
            residuals.size,
        )

    band_lengths = [len(band_indices[idx]) for idx in np.ndindex(band_indices.shape)]
    LOGGER.info(
        "Input bundle prepared: field=%s primary=%s metric=%s curl=%s k_shape=%s "
        "energy_shape=%s field_blocks=%s bands_per_k=min:%s max:%s metric_shape=%s",
        config.maxwell_problem.field_components.value,
        config.maxwell_problem.primary_field.value,
        config.maxwell_problem.metric_material.value,
        config.maxwell_problem.curl_material.value,
        fields.shape,
        energy_matrix.shape,
        fields.size,
        min(band_lengths) if band_lengths else 0,
        max(band_lengths) if band_lengths else 0,
        metric_material.shape,
    )

    return InputBundle(
        config=config,
        maxwell=config.maxwell_problem,
        mesh=mesh,
        fields=fields,
        metric_material=metric_material,
        energies=energies,
        band_indices=band_indices,
        inner_band_indices=inner_band_indices,
        energy_matrix=energy_matrix,
        symmetry=config.symmetry_context,
    )


def _validate_header_k_grid(config: IncarConfig, data: RawData, path: Path) -> None:
    parameters = data.column_parameters
    if not parameters:
        return
    k_sizes = {}
    for axis in range(int(config.kdim)):
        name = f"k{axis + 1}"
        if name not in parameters:
            continue
        header_values = np.unique(parameters[name])
        expected_count = len(config.k_points[axis])
        k_sizes[name] = expected_count
        if header_values.size != expected_count:
            raise ValueError(
                f"COMSOL k-grid mismatch in {path}: header {name} has {header_values.size} values "
                f"({header_values[0]:.10g} to {header_values[-1]:.10g}), but incar k_points[{axis}] "
                f"has {expected_count}. Check k_points and dataset_order before calculating Wannier functions."
            )

    if len(k_sizes) != int(config.kdim):
        return
    column_count = data.value_matrix.shape[1]
    k_count = int(np.prod(list(k_sizes.values())))
    if column_count % k_count != 0:
        raise ValueError(f"COMSOL column count {column_count} is incompatible with the declared k-grid in {path}.")
    dimensions = list(config.dataset_order)
    if "E" not in dimensions:
        dimensions.append("E")
    sizes = {**k_sizes, "E": column_count // k_count}
    shape = tuple(sizes[name] for name in dimensions)
    energy_count = sizes["E"]
    k_names = set(k_sizes)
    energy_parameters = [
        name
        for name, values in parameters.items()
        if name not in k_names and np.unique(values).size == energy_count
    ]
    if energy_count > 1 and len(energy_parameters) != 1:
        raise ValueError(
            f"COMSOL header in {path} must contain exactly one complete energy/band parameter "
            f"with {energy_count} unique values; found {energy_parameters}."
        )
    parameter_for_dimension = {name: name for name in k_names}
    if energy_count > 1:
        parameter_for_dimension["E"] = energy_parameters[0]

    rank_columns = []
    for dimension in dimensions:
        name = parameter_for_dimension.get(dimension)
        if name is None:
            continue
        values = parameters[name]
        if values.size != column_count:
            raise ValueError(
                f"COMSOL header metadata in {path} describes {values.size} columns, but the data has {column_count}."
            )
        unique = np.unique(values)
        actual = np.searchsorted(unique, values)
        expected = np.indices(shape, sparse=False)[dimensions.index(dimension)].reshape(-1)
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"COMSOL column order in {path} does not match dataset_order={config.dataset_order}; "
                f"the {name} parameter varies in a different position."
            )
        rank_columns.append(actual)
    if rank_columns:
        combinations = np.stack(rank_columns, axis=1)
        if np.unique(combinations, axis=0).shape[0] != column_count:
            raise ValueError(f"COMSOL header in {path} contains duplicate or missing k/band parameter combinations.")


def _required_input_paths(config: IncarConfig) -> tuple[Path, Path, Path, Path]:
    paths = (
        config.input_path(config.mesh_file),
        config.input_path(config.dataset_file),
        config.input_path(config.metric_file),
        config.input_path(config.E_file),
    )
    if any(path is None for path in paths):
        raise ValueError("mesh_file, dataset_file, metric_file, and E_file are required.")
    return paths


def _metric_on_mesh(
    mesh: Mesh,
    data: RawData,
    *,
    material: str,
    path: str | Path,
) -> np.ndarray:
    if data.value_matrix.shape[1] != 1:
        raise ValueError(
            "COMSOL metric material file must contain exactly one value column; "
            f"got {data.value_matrix.shape[1]} in {path}."
        )
    mapped = np.asarray(_values_on_mesh(mesh, data)[:, 0])
    imag_scale = max(float(np.max(np.abs(mapped.real), initial=0.0)), 1.0)
    imag_residual = float(np.max(np.abs(mapped.imag), initial=0.0))
    if imag_residual > 1.0e-12 * imag_scale:
        raise ValueError(
            f"COMSOL {material} metric must be real; "
            f"imaginary residual={imag_residual:.6g}."
        )
    metric = np.asarray(mapped.real, dtype=float)
    expected = (mesh.vertices.shape[0],)
    if metric.shape != expected or not np.all(np.isfinite(metric)):
        raise ValueError(
            f"COMSOL {material} metric must contain one finite real value per mesh vertex."
        )
    return metric


def _values_on_mesh(mesh: Mesh, data: RawData) -> np.ndarray:
    tree = cKDTree(mesh.vertices)
    dists, mesh_idxs = tree.query(data.point_matrix, k=1)
    threshold = max(float(mesh.mindist) * 0.5, 1e-8)
    far = np.where(dists > threshold)[0]
    if far.size:
        raise ValueError(
            f"{far.size} COMSOL data points are farther than the mesh matching tolerance "
            f"(max distance={float(np.max(dists[far])):.6g}, tolerance={threshold:.6g})."
        )

    counts = np.bincount(mesh_idxs, minlength=mesh.vertices.shape[0])
    missing = np.where(counts == 0)[0]
    if missing.size:
        raise ValueError(f"{missing.size} mesh vertices could not be matched to COMSOL data.")

    values = np.asarray(data.value_matrix)
    aggregated = np.zeros((mesh.vertices.shape[0], values.shape[1]), dtype=values.dtype)
    np.add.at(aggregated, mesh_idxs, values)
    aggregated /= counts[:, None]

    duplicate_vertices = int(np.count_nonzero(counts > 1))
    LOGGER.info(
        "COMSOL mesh mapping: data_points=%s mesh_vertices=%s duplicates=%s max_distance=%.6g",
        data.point_matrix.shape[0],
        mesh.vertices.shape[0],
        duplicate_vertices,
        float(np.max(dists)) if dists.size else 0.0,
    )
    return aggregated


def _k_shape(config: IncarConfig) -> tuple[int, int, int]:
    kdim = int(config.kdim or len(config.k_points or []))
    return (
        len(config.k_points[0]) if kdim >= 1 else 1,
        len(config.k_points[1]) if kdim >= 2 else 1,
        len(config.k_points[2]) if kdim >= 3 else 1,
    )


def _periodic_boundary_residuals(config: IncarConfig, mesh: Mesh, fields: np.ndarray) -> list[np.ndarray]:
    """Estimate Bloch-boundary continuity on non-conforming 2D edge meshes."""
    if int(config.kdim or 0) != 2:
        return []

    avec = np.asarray(config.real_lattice_vectors, dtype=float) * float(config.lattice_const)
    if avec.shape != (2, 2):
        return []
    fractional = mesh.vertices @ np.linalg.inv(avec)
    sign = -1 if config.dataset_type.lower() == "comsol" else 1
    reciprocal = np.asarray(config.reciprocal_lattice_vectors, dtype=float)
    output: list[np.ndarray] = []

    for axis in range(2):
        coordinate = fractional[:, axis]
        low_value = float(np.min(coordinate))
        high_value = float(np.max(coordinate))
        tolerance = max(np.finfo(float).eps * max(abs(low_value), abs(high_value), 1.0) * 128.0, 1e-10)
        low_idx = np.flatnonzero(np.abs(coordinate - low_value) <= tolerance)
        high_idx = np.flatnonzero(np.abs(coordinate - high_value) <= tolerance)
        if low_idx.size < 2 or high_idx.size < 2:
            output.append(np.empty(0, dtype=float))
            continue

        tangent_axis = 1 - axis
        low_param, low_idx = _sorted_unique_boundary(fractional[low_idx, tangent_axis], low_idx)
        high_param, high_idx = _sorted_unique_boundary(fractional[high_idx, tangent_axis], high_idx)
        start = max(float(low_param[0]), float(high_param[0]))
        stop = min(float(low_param[-1]), float(high_param[-1]))
        sample = np.unique(
            np.concatenate((low_param[(low_param >= start) & (low_param <= stop)], high_param[(high_param >= start) & (high_param <= stop)]))
        )
        if sample.size < 2:
            output.append(np.empty(0, dtype=float))
            continue

        residuals = []
        for i, j, k in np.ndindex(fields.shape):
            block = np.asarray(fields[i, j, k], dtype=np.complex128)
            low = _interpolate_boundary(block[:, low_idx], low_param, sample)
            high = _interpolate_boundary(block[:, high_idx], high_param, sample)
            k_values = [config.k_points[0][i], config.k_points[1][j]]
            k_cart = (2.0 * np.pi / float(config.lattice_const)) * (
                k_values[0] * reciprocal[0] + k_values[1] * reciprocal[1]
            )
            phase = np.exp(1j * sign * np.dot(k_cart, avec[axis]))
            scale = max(float(np.linalg.norm(low)), float(np.linalg.norm(high)), np.finfo(float).tiny)
            residuals.append(float(np.linalg.norm(high - low * phase) / scale))
        output.append(np.asarray(residuals, dtype=float))
    return output


def _sorted_unique_boundary(parameters: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(parameters, kind="stable")
    sorted_parameters = np.asarray(parameters[order], dtype=float)
    sorted_indices = np.asarray(indices[order], dtype=np.intp)
    keep = np.ones(sorted_parameters.size, dtype=bool)
    keep[1:] = np.diff(sorted_parameters) > 1e-12
    return sorted_parameters[keep], sorted_indices[keep]


def _interpolate_boundary(values: np.ndarray, parameters: np.ndarray, sample: np.ndarray) -> np.ndarray:
    out = np.empty((values.shape[0], sample.size), dtype=np.complex128)
    for row in range(values.shape[0]):
        out[row] = np.interp(sample, parameters, values[row].real) + 1j * np.interp(
            sample, parameters, values[row].imag
        )
    return out


def _handle_energy_data(config: IncarConfig, energy_raw: RawData):
    nk1, nk2, nk3 = _k_shape(config)
    raw = energy_raw.value_matrix[0]
    n_k = nk1 * nk2 * nk3
    if raw.size % n_k != 0:
        raise ValueError(f"Energy size mismatch: total={raw.size}, Nk product={n_k}.")
    nbands = raw.size // n_k

    energy_matrix = _reshape_parameter_tensor(raw, config, nbands)
    if config.hermitian or config.E_is_real:
        energy_matrix = np.real(energy_matrix)

    band_indices = np.empty((nk1, nk2, nk3), dtype=object)
    energies = np.empty((nk1, nk2, nk3), dtype=object)
    if isinstance(config.band_window, EnergyWindow):
        for idx in np.ndindex(nk1, nk2, nk3):
            line = energy_matrix[idx]
            selected = np.where((line >= config.band_window.emin) & (line <= config.band_window.emax))[0]
            band_indices[idx] = selected.tolist()
            energies[idx] = line[selected].tolist()
    else:
        selected = np.asarray(config.band_window, dtype=int)
        for idx in np.ndindex(nk1, nk2, nk3):
            band_indices[idx] = selected.tolist()
            energies[idx] = energy_matrix[idx][selected]

    inner = _select_inner_window(config, energy_matrix)
    return energy_matrix, energies, band_indices, inner


def _reshape_parameter_tensor(raw: np.ndarray, config: IncarConfig, nbands: int) -> np.ndarray:
    nk1, nk2, nk3 = _k_shape(config)
    order = list(config.dataset_order)
    sizes = {"k1": nk1, "k2": nk2, "k3": nk3, "E": nbands}
    _validate_dataset_order(order, sizes)
    shaped = raw.reshape(tuple(sizes[dim] for dim in order), order="C")
    axes = [order.index(dim) for dim in ("k1", "k2", "k3", "E") if dim in order]
    return np.transpose(shaped, axes=axes).reshape(nk1, nk2, nk3, nbands)


def _select_inner_window(config: IncarConfig, energy_matrix: np.ndarray) -> np.ndarray:
    nk1, nk2, nk3, _ = energy_matrix.shape
    inner = np.empty((nk1, nk2, nk3), dtype=object)
    if config.inner_window is False:
        for idx in np.ndindex(nk1, nk2, nk3):
            inner[idx] = []
    elif isinstance(config.inner_window, EnergyWindow):
        for idx in np.ndindex(nk1, nk2, nk3):
            line = energy_matrix[idx]
            inner[idx] = np.where((line >= config.inner_window.emin) & (line <= config.inner_window.emax))[0].tolist()
    else:
        selected_inner = np.asarray(config.inner_window, dtype=int).tolist()
        for idx in np.ndindex(nk1, nk2, nk3):
            inner[idx] = selected_inner.copy()
    return inner


def _distribute_fields(config: IncarConfig, values: np.ndarray, band_indices: np.ndarray) -> np.ndarray:
    tensor = _reshape_field_tensor(values, config)
    nk1, nk2, nk3 = tensor.shape[1:4]
    fields = np.empty((nk1, nk2, nk3), dtype=object)

    if isinstance(config.band_window, EnergyWindow):
        for i, j, k in np.ndindex(nk1, nk2, nk3):
            selected = np.asarray(band_indices[i, j, k], dtype=int)
            fields[i, j, k] = np.asarray(tensor[:, i, j, k, selected].T, dtype=np.complex128).copy()
        return fields

    selected = np.asarray(config.band_window, dtype=int)
    selected_tensor = tensor[:, :, :, :, selected]
    for i, j, k in np.ndindex(nk1, nk2, nk3):
        fields[i, j, k] = np.asarray(selected_tensor[:, i, j, k, :].T, dtype=np.complex128).copy()
    return fields


def _reshape_field_tensor(values: np.ndarray, config: IncarConfig) -> np.ndarray:
    nk1, nk2, nk3 = _k_shape(config)
    nv = values.shape[0]
    order = list(config.dataset_order)
    k_sizes = {"k1": nk1, "k2": nk2, "k3": nk3}
    _validate_dataset_order(order, {**k_sizes, "E": None})

    known = 1
    for dim in order:
        if dim != "E":
            known *= k_sizes[dim]
    if values.shape[1] % known != 0:
        raise ValueError(f"Field data size mismatch: value columns={values.shape[1]}, k-grid product={known}.")
    nbands = values.shape[1] // known

    sizes = {**k_sizes, "E": nbands}
    if "E" in order:
        shape_in = (nv,) + tuple(sizes[dim] for dim in order)
        positions = {dim: order.index(dim) + 1 for dim in order}
    else:
        shape_in = (nv,) + tuple(sizes[dim] for dim in order) + (nbands,)
        positions = {dim: order.index(dim) + 1 for dim in order}
        positions["E"] = len(shape_in) - 1

    shaped = values.reshape(shape_in, order="C")
    present_k = [dim for dim in ("k1", "k2", "k3") if dim in positions]
    axes = (0,) + tuple(positions[dim] for dim in present_k) + (positions["E"],)
    return np.transpose(shaped, axes=axes).reshape(nv, nk1, nk2, nk3, nbands, order="C")


def _validate_dataset_order(order: list[str], sizes: dict[str, int | None]) -> None:
    allowed = set(sizes)
    unknown = [dim for dim in order if dim not in allowed]
    if unknown:
        raise ValueError(f"dataset_order contains unknown dimension(s): {unknown}")
    if len(set(order)) != len(order):
        raise ValueError(f"dataset_order contains duplicate dimensions: {order}")
    for dim, size in sizes.items():
        if dim.startswith("k") and size not in (None, 1) and dim not in order:
            raise ValueError(f"dataset_order is missing required dimension {dim!r}.")
