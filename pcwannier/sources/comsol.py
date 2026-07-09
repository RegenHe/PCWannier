from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging

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
            return RawData(points, values)

        points, values = _load_token_table(path, real_values=False)
    LOGGER.info("COMSOL data loaded: file=%s points=%s value_cols=%s real_only=False", path, points.shape[0], values.shape[1])
    return RawData(points, values)


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
    mesh_path, dataset_path, dielectric_path, energy_path = _required_input_paths(config)
    LOGGER.info(
        "COMSOL input paths: mesh=%s dataset=%s dielectric=%s energy=%s dataset_order=%s",
        mesh_path,
        dataset_path,
        dielectric_path,
        energy_path,
        config.dataset_order,
    )

    mesh = load_comsol_mesh(mesh_path)
    raw_data = load_comsol_data(dataset_path)
    epsilon_raw = load_comsol_data(dielectric_path, real_only=True)
    energy_raw = load_comsol_data(energy_path, real_only=config.E_is_real or config.hermitian)

    with timed_step("prepare COMSOL tensors", LOGGER):
        energy_matrix, energies, band_indices, inner_band_indices = _handle_energy_data(config, energy_raw)
        fields = _distribute_fields(config, _values_on_mesh(mesh, raw_data), band_indices)
        epsilon = _values_on_mesh(mesh, epsilon_raw).reshape(-1)

    band_lengths = [len(band_indices[idx]) for idx in np.ndindex(band_indices.shape)]
    LOGGER.info(
        "Input bundle prepared: k_shape=%s energy_shape=%s field_blocks=%s bands_per_k=min:%s max:%s epsilon_shape=%s",
        fields.shape,
        energy_matrix.shape,
        fields.size,
        min(band_lengths) if band_lengths else 0,
        max(band_lengths) if band_lengths else 0,
        epsilon.shape,
    )

    return InputBundle(
        config=config,
        mesh=mesh,
        fields=fields,
        epsilon=epsilon,
        energies=energies,
        band_indices=band_indices,
        inner_band_indices=inner_band_indices,
        energy_matrix=energy_matrix,
    )


def _required_input_paths(config: IncarConfig) -> tuple[Path, Path, Path, Path]:
    paths = (
        config.input_path(config.mesh_file),
        config.input_path(config.dataset_file),
        config.input_path(config.dielectric_file),
        config.input_path(config.E_file),
    )
    if any(path is None for path in paths):
        raise ValueError("mesh_file, dataset_file, dielectric_file, and E_file are required.")
    return paths


def _values_on_mesh(mesh: Mesh, data: RawData) -> np.ndarray:
    idxs, dists = match_data_to_mesh(mesh, data)
    missing = np.where(idxs < 0)[0]
    if missing.size:
        finite = dists[np.isfinite(dists)]
        max_dist = float(np.max(finite)) if finite.size else float("nan")
        raise ValueError(f"{missing.size} mesh vertices could not be matched to COMSOL data (max matched distance={max_dist:.6g}).")
    return data.value_matrix[idxs]


def _k_shape(config: IncarConfig) -> tuple[int, int, int]:
    kdim = int(config.kdim or len(config.k_points or []))
    return (
        len(config.k_points[0]) if kdim >= 1 else 1,
        len(config.k_points[1]) if kdim >= 2 else 1,
        len(config.k_points[2]) if kdim >= 3 else 1,
    )


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
