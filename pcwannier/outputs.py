from __future__ import annotations

from pathlib import Path
import logging
import math

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation

from .config import EnergyWindow, IncarConfig
from .conventions import SpatialDiscretization
from .data import (
    BandResult,
    BlochSymmetryRunResult,
    Mesh,
    PeriodicGrid,
    RunResult,
    TopologyResult,
)
from .matrix_io import save_cell_matrix
from .symmetry.cache import save_sewing_matrix_cache
from .symmetry.reporting import (
    format_bloch_symmetry_report,
    format_gamma_zero_regularization_report,
    format_symmetry_analysis_report,
)
from .timing import timed_step

LOGGER = logging.getLogger(__name__)


def _is_false_path(value) -> bool:
    return value is None or value is False or str(value).lower() == "false"


def _resolve_output(path_value, config: IncarConfig, out_dir: str | Path | None = None) -> Path | None:
    if _is_false_path(path_value):
        return None
    path = Path(str(path_value))
    if out_dir is not None:
        out = Path(out_dir)
        return out / path if not path.is_absolute() else out / path.name
    return path if path.is_absolute() else config.base_dir / path


def _ensure_parent(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def _save_text(path: str | Path, text: str) -> None:
    output = Path(path)
    _ensure_parent(output)
    output.write_text(text, encoding="utf-8")


def _fmt_c(value, tol=1e-12, prec=8, force_complex=False, spaced=True, zero_small_imag=True):
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return ", ".join(
            _fmt_c(x, tol=tol, prec=prec, force_complex=force_complex, spaced=spaced, zero_small_imag=zero_small_imag)
            for x in arr
        )
    real = float(np.real(value))
    imag = float(np.imag(value))
    if zero_small_imag and abs(imag) < tol:
        imag = 0.0
    if not force_complex and abs(imag) < tol:
        return f"{real:.{prec}f}"
    if spaced:
        sign = " - " if math.copysign(1.0, imag) < 0 else " + "
        return f"{real:.{prec}f}{sign}{abs(imag):.{prec}f}j"
    sign = "+" if imag >= 0 else ""
    return f"{real:.{prec}f}{sign}{imag:.{prec}f}j"


def save_dict(filename: str | Path, data: dict) -> None:
    path = Path(filename)
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# dict_size={len(data)}\n")
        for key, value in data.items():
            key_text = ", ".join(str(x) for x in key) if isinstance(key, tuple) else str(key)
            arr = np.asarray(value)
            handle.write(f"CELL({key_text}) shape={arr.shape}:\n")
            if arr.ndim == 0:
                handle.write(_fmt_c(arr.item()) + "\n")
            elif arr.ndim == 1:
                handle.write(_fmt_c(arr) + "\n")
            else:
                for row in arr:
                    handle.write(_fmt_c(row) + "\n")
            handle.write("\n")


def save_vector_wanniers(filename: str | Path, result: RunResult) -> None:
    """Write 3D vector Wannier fields as coordinates and complex components."""

    path = Path(filename)
    _ensure_parent(path)
    coordinates = np.asarray(result.extended_mesh.vertices, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Vector Wannier text output requires 3D mesh coordinates.")
    maxwell = result.config.maxwell_problem
    field_symbol = (
        "E"
        if maxwell is not None and maxwell.primary_field.value == "electric"
        else "H"
    )
    component_columns = " ".join(
        f"Re({field_symbol}{axis}) Im({field_symbol}{axis})"
        for axis in "xyz"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 3D vector Wannier fields\n")
        handle.write(
            f"# columns: x y z, then {component_columns} per Wannier\n"
        )
        for cell, values in result.wanniers.items():
            array = np.asarray(values, dtype=np.complex128)
            if array.ndim != 3 or array.shape[0] != coordinates.shape[0] or array.shape[2] != 3:
                raise ValueError(
                    f"Vector Wannier cell {cell} has invalid shape {array.shape}."
                )
            handle.write(
                f"CELL({', '.join(str(value) for value in cell)}) "
                f"shape={array.shape}:\n"
            )
            column_count = 3 + 6 * array.shape[1]
            chunk_size = 1 << 14
            for start in range(0, coordinates.shape[0], chunk_size):
                stop = min(start + chunk_size, coordinates.shape[0])
                matrix = np.empty((stop - start, column_count), dtype=np.float64)
                matrix[:, :3] = coordinates[start:stop]
                column = 3
                for index in range(array.shape[1]):
                    for component in range(3):
                        values = array[start:stop, index, component]
                        matrix[:, column] = values.real
                        matrix[:, column + 1] = values.imag
                        column += 2
                np.savetxt(handle, matrix, fmt="%.12e")
            handle.write("\n")


def save_band(filename: str | Path, energies: np.ndarray, k_path: np.ndarray | None, other_info: dict | None = None) -> None:
    path = Path(filename)
    _ensure_parent(path)
    e = np.asarray(energies)
    if e.ndim == 1:
        e = e.reshape(-1, 1)
    elif e.ndim > 2:
        e = e.reshape(e.shape[0], -1)

    with path.open("w", encoding="utf-8") as handle:
        if k_path is None:
            handle.write(f"# k-points: 1, Bands: {e.shape[1]}\n")
            if other_info:
                for key, value in other_info.items():
                    handle.write(f"# {key}: {value}\n")
            handle.write(",".join(_fmt_c(x, prec=8, spaced=False) for x in e[0]) + "\n")
            return

        k = np.asarray(k_path)
        if k.ndim == 1:
            k = k.reshape(-1, 1)
        elif k.ndim > 2:
            k = k.reshape(k.shape[0], -1)
        handle.write(f"# k-points: {k.shape[0]}, Bands: {e.shape[1]}\n")
        if other_info:
            for key, value in other_info.items():
                handle.write(f"# {key}: {value}\n")
        for idx in range(k.shape[0]):
            k_text = ", ".join(f"{x:.8f}" for x in k[idx])
            e_text = ", ".join(_fmt_c(x, prec=8, spaced=False) for x in e[idx])
            handle.write(f"{k_text},{e_text}\n")


def load_interpolation_points(
    filename: str | Path,
    dimension: int | None = None,
) -> np.ndarray:
    path = Path(filename)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        points = np.loadtxt(handle, delimiter=",")
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    allowed_dimensions = (2, 3) if dimension is None else (int(dimension),)
    if points.ndim != 2 or points.shape[1] not in allowed_dimensions:
        columns = " or ".join(
            ",".join(("x", "y", "z")[:item]) for item in allowed_dimensions
        )
        raise ValueError(
            f"Invalid interpolation mesh {path}: each row must contain {columns}."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"Invalid interpolation mesh {path}: coordinates must be finite.")
    return points


def save_points_with_values(
    filename: str | Path,
    points: np.ndarray,
    values: np.ndarray,
    *,
    value_labels: list[str] | None = None,
) -> None:
    path = Path(filename)
    _ensure_parent(path)
    points = np.asarray(points, dtype=float)
    vals = np.asarray(values)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points must have shape (n, 2) or (n, 3).")
    if vals.shape[1] != points.shape[0]:
        raise ValueError("Number of interpolation points and value rows differ.")
    if value_labels is not None and len(value_labels) != vals.shape[0]:
        raise ValueError("Number of interpolation labels and value rows differ.")

    with path.open("w", encoding="utf-8") as handle:
        if value_labels is not None:
            coordinate_labels = list(("x", "y", "z")[: points.shape[1]])
            handle.write(
                "# columns: "
                + ", ".join(coordinate_labels + value_labels)
                + "\n"
            )
        for idx, point in enumerate(points):
            row = [f"{coordinate:.10f}" for coordinate in point]
            for value in vals[:, idx]:
                if np.iscomplexobj(value):
                    row.append(f"{np.real(value):.10f}{np.imag(value):+.10f}j")
                else:
                    row.append(f"{float(value):.10f}")
            handle.write(",".join(row) + "\n")


def write_interpolation_outputs(
    result: RunResult,
    interp_path: str | Path,
    interp_wannier: str | Path | None = None,
    interp_metric: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> None:
    mesh = result.extended_mesh
    dimension = int(np.asarray(mesh.vertices).shape[1])
    points = load_interpolation_points(interp_path, dimension=dimension)
    interp_path = Path(interp_path)
    tile_count = int(np.prod(result.config.extension[: result.config.kdim]))

    wannier_path = _resolve_interpolation_output(interp_path, interp_wannier, "interp-wannier", out_dir)
    if wannier_path is not None:
        values = []
        labels = []
        for cell, wmat in result.wanniers.items():
            wmat = np.asarray(wmat)
            cell_label = "_".join(str(index) for index in cell)
            if wmat.ndim == 2:
                for band in range(wmat.shape[1]):
                    values.append(
                        _interpolate_complex_mesh(
                            mesh, wmat[:, band], points, tile_count
                        )
                    )
                    labels.append(f"W[{cell_label},{band}]")
            elif wmat.ndim == 3 and wmat.shape[2] == dimension:
                for band in range(wmat.shape[1]):
                    interpolated = _interpolate_complex_mesh(
                        mesh, wmat[:, band], points, tile_count
                    )
                    values.extend(
                        interpolated[:, component]
                        for component in range(dimension)
                    )
                    labels.extend(
                        f"W[{cell_label},{band}].{axis}"
                        for axis in ("x", "y", "z")[:dimension]
                    )
            else:
                raise ValueError(
                    f"Wannier field has unsupported interpolation shape {wmat.shape}."
                )
        with timed_step("write interpolated Wannier data", LOGGER, file=wannier_path):
            save_points_with_values(
                wannier_path,
                points,
                np.asarray(values),
                value_labels=labels if dimension == 3 else None,
            )

    metric_path = _resolve_interpolation_output(
        interp_path, interp_metric, "interp-metric", out_dir
    )
    if metric_path is not None:
        metric = _interpolate_real_mesh(
            mesh, result.extended_metric_material, points, tile_count
        )
        material = result.config.maxwell_problem.metric_material.value
        with timed_step(
            "write interpolated metric material",
            LOGGER,
            file=metric_path,
            material=material,
        ):
            save_points_with_values(
                metric_path,
                points,
                metric.reshape(1, -1),
                value_labels=[material] if dimension == 3 else None,
            )


def _resolve_interpolation_output(
    interp_path: Path,
    output_path: str | Path | None,
    suffix: str,
    out_dir: str | Path | None = None,
) -> Path | None:
    if output_path is None:
        return interp_path.with_name(f"{interp_path.stem}-{suffix}.txt")
    if _is_false_path(output_path):
        return None
    path = Path(output_path)
    if out_dir is not None and not path.is_absolute():
        return Path(out_dir) / path
    return path


def _interpolate_real(triang: Triangulation, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    interpolator = LinearTriInterpolator(triang, np.asarray(np.real(values), dtype=float))
    out = interpolator(points[:, 0], points[:, 1])
    if isinstance(out, np.ma.MaskedArray):
        return np.asarray(out.filled(np.nan), dtype=float)
    return np.asarray(out, dtype=float)


def _interpolate_complex(triang: Triangulation, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    real = _interpolate_real(triang, np.real(values), points)
    imag = _interpolate_real(triang, np.imag(values), points)
    return real + 1j * imag


def _interpolate_real_mesh(mesh: Mesh, values: np.ndarray, points: np.ndarray, tile_count: int) -> np.ndarray:
    """Interpolate a tiled non-conforming mesh without building a global triangle finder."""
    discretization = getattr(mesh, "discretization", None)
    if discretization is SpatialDiscretization.PERIODIC_FOURIER_COLLOCATION:
        if not isinstance(mesh, PeriodicGrid):
            raise TypeError("Fourier-grid interpolation requires PeriodicGrid metadata.")
        if mesh.dimension == 3:
            interpolated = _interpolate_uniform_grid(mesh, values, points)
            if interpolated.ndim != 1:
                raise ValueError("Real scalar interpolation received component-valued data.")
            return np.asarray(np.real(interpolated), dtype=float)
        triang = Triangulation(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.elements,
        )
        return _interpolate_real(triang, values, points)
    if discretization is not SpatialDiscretization.TRIANGLE_FEM_P1:
        raise ValueError(f"Unsupported spatial discretization {discretization!r}.")

    tile_count = max(1, int(tile_count))
    if mesh.elements.shape[0] % tile_count != 0:
        raise ValueError("Extended mesh element count is incompatible with its tile count.")
    elements_per_tile = mesh.elements.shape[0] // tile_count
    totals = np.zeros(points.shape[0], dtype=float)
    counts = np.zeros(points.shape[0], dtype=np.intp)
    real_values = np.asarray(np.real(values), dtype=float)

    for tile in range(tile_count):
        start = tile * elements_per_tile
        stop = start + elements_per_tile
        triang = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements[start:stop])
        interpolated = _interpolate_real(triang, real_values, points)
        valid = np.isfinite(interpolated)
        totals[valid] += interpolated[valid]
        counts[valid] += 1

    out = np.full(points.shape[0], np.nan, dtype=float)
    valid = counts > 0
    out[valid] = totals[valid] / counts[valid]
    return out


def _interpolate_complex_mesh(mesh: Mesh, values: np.ndarray, points: np.ndarray, tile_count: int) -> np.ndarray:
    values = np.asarray(values)
    if (
        getattr(mesh, "discretization", None)
        is SpatialDiscretization.PERIODIC_FOURIER_COLLOCATION
        and isinstance(mesh, PeriodicGrid)
        and mesh.dimension == 3
    ):
        return np.asarray(
            _interpolate_uniform_grid(mesh, values, points),
            dtype=np.complex128,
        )
    real = _interpolate_real_mesh(mesh, np.real(values), points, tile_count)
    imag = _interpolate_real_mesh(mesh, np.imag(values), points, tile_count)
    return real + 1j * imag


def _interpolate_uniform_grid(
    grid: PeriodicGrid,
    values: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Multilinearly interpolate an extended regular grid without wrapping."""

    coordinates = np.asarray(points, dtype=float)
    data = np.asarray(values)
    if coordinates.ndim != 2 or coordinates.shape[1] != grid.dimension:
        raise ValueError(
            f"Interpolation points must have shape (n, {grid.dimension})."
        )
    if data.ndim < 1 or data.shape[0] != grid.point_count:
        raise ValueError(
            f"Uniform-grid values have shape {data.shape}; "
            f"expected first dimension {grid.point_count}."
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("Uniform-grid interpolation values must be finite.")

    fractional = coordinates @ np.linalg.inv(grid.lattice_vectors)
    lower = np.zeros((coordinates.shape[0], grid.dimension), dtype=np.intp)
    upper = np.zeros_like(lower)
    fractions = np.zeros_like(fractional)
    valid = np.ones(coordinates.shape[0], dtype=bool)
    coordinate_tolerance = max(
        np.finfo(float).eps
        * max(float(np.max(np.abs(grid.fractional_vertices))), 1.0)
        * 256.0,
        1.0e-12,
    )

    fractional_grid = grid.fractional_vertices.reshape(
        grid.shape + (grid.dimension,)
    )
    for axis, size in enumerate(grid.shape):
        selector: list[int | slice] = [0] * grid.dimension
        selector[axis] = slice(None)
        axis_values = fractional_grid[tuple(selector) + (axis,)]
        if axis_values.size != size:
            raise ValueError("PeriodicGrid fractional coordinates do not match its shape.")
        if size == 1:
            valid &= np.abs(fractional[:, axis] - axis_values[0]) <= coordinate_tolerance
            continue

        step = float(axis_values[1] - axis_values[0])
        if step <= 0.0 or not np.allclose(
            np.diff(axis_values), step, rtol=1.0e-10, atol=coordinate_tolerance
        ):
            raise ValueError("Uniform-grid interpolation requires evenly spaced axes.")
        position = (fractional[:, axis] - axis_values[0]) / step
        valid &= (position >= -coordinate_tolerance / step) & (
            position <= (size - 1) + coordinate_tolerance / step
        )
        position = np.clip(position, 0.0, float(size - 1))
        axis_lower = np.floor(position).astype(np.intp)
        at_upper_edge = axis_lower == size - 1
        axis_lower[at_upper_edge] = size - 2
        lower[:, axis] = axis_lower
        upper[:, axis] = axis_lower + 1
        fractions[:, axis] = position - axis_lower

    dtype = np.result_type(data.dtype, np.float64)
    output = np.full(
        (coordinates.shape[0],) + data.shape[1:],
        np.nan,
        dtype=dtype,
    )
    if not np.any(valid):
        return output

    valid_lower = lower[valid]
    valid_upper = upper[valid]
    valid_fractions = fractions[valid]
    interpolated = np.zeros(
        (int(np.count_nonzero(valid)),) + data.shape[1:],
        dtype=dtype,
    )
    for corner in np.ndindex(*(2,) * grid.dimension):
        indices = tuple(
            valid_upper[:, axis] if corner[axis] else valid_lower[:, axis]
            for axis in range(grid.dimension)
        )
        flat = np.ravel_multi_index(indices, grid.shape)
        weight = np.ones(valid_lower.shape[0], dtype=float)
        for axis in range(grid.dimension):
            fraction = valid_fractions[:, axis]
            weight *= fraction if corner[axis] else 1.0 - fraction
        weight_shape = (weight.shape[0],) + (1,) * (data.ndim - 1)
        interpolated += data[flat] * weight.reshape(weight_shape)
    output[valid] = interpolated
    return output


def write_base_figures(
    config: IncarConfig,
    mesh: Mesh,
    out_dir: str | Path | None = None,
    directory: str | Path = "base",
) -> None:
    from .compute.initializer import StateBases

    target = _resolve_output(directory, config, out_dir)
    if target is None:
        return
    target.mkdir(parents=True, exist_ok=True)

    ext_mesh = mesh.__deepcopy__()
    ext_mesh.extension(config.extension, config.real_lattice_vectors, float(config.lattice_const))

    idx = 0
    for projection in config.projections:
        frac = projection["frac_position"]
        cart_position = (
            frac[0] * np.asarray(config.real_lattice_vectors[0])
            + frac[1] * np.asarray(config.real_lattice_vectors[1])
            + np.asarray(config.origin)
        ) * float(config.lattice_const)
        for state_spec in projection["states"]:
            fn = _projection_function(config, StateBases, state_spec)
            values = ext_mesh.rfunc(fn, cart_position, projection["xaxis_angluar"])
            _save_tri_field(target / f"base-{idx}-real.png", ext_mesh, np.real(values), "Real Part")
            if np.max(np.abs(np.imag(values))) > 1e-12:
                _save_tri_field(target / f"base-{idx}-imag.png", ext_mesh, np.imag(values), "Imaginary Part")
            idx += 1


def _projection_function(config: IncarConfig, bases, state_spec):
    if isinstance(state_spec, dict) and "lc_states" in state_spec:
        lc_states = state_spec["lc_states"]
        lc_coeffs = state_spec["lc_coeffs"]

        def fn(r, phi, _states=lc_states, _coeffs=lc_coeffs):
            total = 0.0 + 0.0j
            for (n, l, z), coeff in zip(_states, _coeffs):
                rr = r / float(config.lattice_const)
                total += coeff * bases.Radial(n, l)(rr, z) * bases.Angular(l)(phi)
            return total

        return fn

    n, l, z = state_spec

    def fn(r, phi, _n=n, _l=l, _z=z):
        rr = r / float(config.lattice_const)
        return bases.Radial(_n, _l)(rr, _z) * bases.Angular(_l)(phi)

    return fn


def _save_tri_field(filename: str | Path, mesh: Mesh, values: np.ndarray, label: str) -> None:
    path = Path(filename)
    _ensure_parent(path)
    triang = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements)
    fig, ax = plt.subplots()
    contour = ax.tricontourf(triang, values, levels=255, cmap="bwr")
    vmax = float(np.max(np.abs(values))) if values.size else 1.0
    if vmax == 0.0:
        vmax = 1.0
    contour.set_clim(-vmax, vmax)
    fig.colorbar(contour, ax=ax, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    min_x, max_x = np.min(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 0])
    min_y, max_y = np.min(mesh.vertices[:, 1]), np.max(mesh.vertices[:, 1])
    margin = 0.1
    ax.set_xlim(min_x - margin * (max_x - min_x), max_x + margin * (max_x - min_x))
    ax.set_ylim(min_y - margin * (max_y - min_y), max_y + margin * (max_y - min_y))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_bloch_symmetry_outputs(
    result: BlochSymmetryRunResult,
    config: IncarConfig | None = None,
    out_dir: str | Path | None = None,
) -> None:
    """Write the reusable raw-overlap and full outer-window sewing caches."""

    config = config or result.config
    s_path = _resolve_output(
        "S.txt" if _is_false_path(config.S_file) else config.S_file,
        config,
        out_dir,
    )
    d_path = _resolve_output(
        "D.txt" if _is_false_path(config.D_file) else config.D_file,
        config,
        out_dir,
    )
    if s_path is None or d_path is None:
        raise RuntimeError("Bloch symmetry cache output paths could not be resolved.")
    with timed_step("write raw S matrix", LOGGER, file=s_path):
        save_cell_matrix(s_path, result.primary.S, result.primary.S.shape)
    with timed_step(
        "write D symmetry matrices",
        LOGGER,
        file=d_path,
        count=len(result.primary.sewing_matrices),
    ):
        save_sewing_matrix_cache(
            d_path,
            result.primary.sewing_matrices,
            dimension=result.symmetry.model.dimension,
            bloch_sign=result.symmetry.model.bloch_convention.sign,
            k_shape=tuple(len(axis) for axis in result.symmetry.k_points),
        )
    channel_paths = {
        "longitudinal": (
            getattr(config, "longitudinal_S_file", "S_L.txt"),
            getattr(config, "longitudinal_D_file", "D_L.txt"),
        ),
    }
    for channel_name, channel in result.auxiliary_channels.items():
        if channel_name not in channel_paths:
            raise ValueError(f"Unknown Bloch symmetry output channel {channel_name!r}.")
        s_value, d_value = channel_paths[channel_name]
        channel_s_path = _resolve_output(s_value, config, out_dir)
        channel_d_path = _resolve_output(d_value, config, out_dir)
        if channel_s_path is None or channel_d_path is None:
            raise RuntimeError(
                f"Bloch symmetry cache paths are disabled for channel {channel_name!r}."
            )
        with timed_step(
            "write auxiliary raw S matrix",
            LOGGER,
            channel=channel_name,
            file=channel_s_path,
        ):
            save_cell_matrix(channel_s_path, channel.S, channel.S.shape)
        with timed_step(
            "write auxiliary D symmetry matrices",
            LOGGER,
            channel=channel_name,
            file=channel_d_path,
            count=len(channel.sewing_matrices),
        ):
            save_sewing_matrix_cache(
                channel_d_path,
                channel.sewing_matrices,
                dimension=result.symmetry.model.dimension,
                bloch_sign=result.symmetry.model.bloch_convention.sign,
                k_shape=tuple(len(axis) for axis in result.symmetry.k_points),
            )

    report_path = _resolve_output(
        getattr(config, "symmetry_report_file", "./sym.txt"), config, out_dir
    )
    if report_path is not None:
        sections = [
            "# PCWannier symmetry analysis",
            format_bloch_symmetry_report(result.primary.analysis, title="physical"),
        ]
        if result.gamma_zero_regularization is not None:
            sections.append(
                format_gamma_zero_regularization_report(
                    result.gamma_zero_regularization
                )
            )
        sections.extend(
            format_bloch_symmetry_report(channel.analysis, title=channel_name)
            for channel_name, channel in result.auxiliary_channels.items()
        )
        with timed_step("write symmetry report", LOGGER, file=report_path):
            _save_text(report_path, "\n\n".join(sections) + "\n")


def write_outputs(result: RunResult, config: IncarConfig | None = None, out_dir: str | Path | None = None) -> None:
    config = config or result.config
    s_path = _resolve_output(config.S_file, config, out_dir)
    if s_path is not None and result.S is not None:
        with timed_step("write raw S matrix", LOGGER, file=s_path):
            save_cell_matrix(s_path, result.S, result.S.shape)
    p_path = _resolve_output(getattr(config, "P_file", "./P.txt"), config, out_dir)
    transverse_projectors = getattr(result, "transverse_projectors", None)
    if p_path is not None and transverse_projectors is not None:
        with timed_step("write transverse P matrix", LOGGER, file=p_path):
            save_cell_matrix(
                p_path,
                transverse_projectors,
                transverse_projectors.shape,
                header_comments=(
                    "P_T(k) in the final Wannier basis.",
                    "CELL indices follow the configured sampled k grid.",
                ),
            )
    m_path = _resolve_output(config.M_file, config, out_dir)
    if m_path is not None:
        with timed_step("write M0 matrix", LOGGER, file=m_path):
            save_cell_matrix(
                m_path,
                result.M0,
                result.M0.shape + (len(config.composition_of_b) // 2,),
            )
    v_path = _resolve_output(config.V_file, config, out_dir)
    if v_path is not None:
        with timed_step("write V matrix", LOGGER, file=v_path):
            save_cell_matrix(v_path, result.V, result.V.shape)
    a_path = _resolve_output(config.A_file, config, out_dir)
    if a_path is not None:
        with timed_step("write A matrix", LOGGER, file=a_path):
            save_cell_matrix(a_path, result.A, result.A.shape)
    u_path = _resolve_output(config.U_file, config, out_dir)
    if u_path is not None:
        with timed_step("write U matrix", LOGGER, file=u_path):
            save_cell_matrix(u_path, result.U, result.U.shape)

    d_path = _resolve_output(config.D_file, config, out_dir)
    if d_path is not None and result.sewing_matrices:
        if result.symmetry is None:
            raise ValueError("Sewing matrices are present without a symmetry context.")
        with timed_step(
            "write D symmetry matrices",
            LOGGER,
            file=d_path,
            count=len(result.sewing_matrices),
        ):
            save_sewing_matrix_cache(
                d_path,
                result.sewing_matrices,
                dimension=result.symmetry.model.dimension,
                bloch_sign=result.symmetry.model.bloch_convention.sign,
                k_shape=tuple(len(axis) for axis in result.symmetry.k_points),
            )

    report_path = _resolve_output(
        getattr(config, "symmetry_report_file", "./sym.txt"), config, out_dir
    )
    symmetry_analysis = getattr(result, "symmetry_analysis", None)
    if report_path is not None and symmetry_analysis is not None:
        with timed_step("write symmetry report", LOGGER, file=report_path):
            _save_text(
                report_path,
                format_symmetry_analysis_report(symmetry_analysis),
            )

    hopping_path = _resolve_output(config.hopping_file, config, out_dir)
    if hopping_path is not None:
        with timed_step("write hopping", LOGGER, file=hopping_path, count=len(result.hoppings)):
            save_dict(hopping_path, result.hoppings)

    wannier_path = _resolve_output(config.wannier_file, config, out_dir)
    if wannier_path is not None:
        with timed_step("write Wannier data", LOGGER, file=wannier_path, count=len(result.wanniers)):
            first_wannier = np.asarray(next(iter(result.wanniers.values())))
            if first_wannier.ndim == 3:
                save_vector_wanniers(wannier_path, result)
            else:
                save_dict(wannier_path, result.wanniers)

    if result.band is not None:
        band_path = _resolve_output(config.band_file, config, out_dir)
        if band_path is not None:
            with timed_step("write band data", LOGGER, file=band_path):
                save_band(band_path, result.band.energies, result.band.k_path)
        figure_path = _resolve_output(config.band_figure, config, out_dir)
        if figure_path is not None:
            with timed_step("write band figure", LOGGER, file=figure_path):
                plot_band(figure_path, result.band, config)

    wannier_dir = _resolve_output(config.wannier_figures, config, out_dir)
    if wannier_dir is not None:
        with timed_step("write Wannier figures", LOGGER, directory=wannier_dir):
            write_wannier_figures(wannier_dir, result)

    topo_dir = _resolve_output(config.topo_output, config, out_dir)
    if topo_dir is not None and result.topology is not None:
        with timed_step("write topology figures", LOGGER, directory=topo_dir):
            write_topology_figures(topo_dir, result.topology)


def write_wannier_figures(directory: str | Path, result: RunResult) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mesh = result.extended_mesh
    if np.asarray(mesh.vertices).shape[1] != 2:
        raise NotImplementedError("Three-dimensional Wannier volume figures are not implemented.")
    triang = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements)
    for r_key, wmat in result.wanniers.items():
        suffix = "-".join(str(x) for x in r_key)
        for band in range(wmat.shape[1]):
            for real_part, label in ((True, "real"), (False, "imag")):
                values = np.real(wmat[:, band]) if real_part else np.imag(wmat[:, band])
                fig, ax = plt.subplots()
                contour = ax.tricontourf(triang, values, levels=255, cmap="bwr")
                vmax = float(np.max(np.abs(values))) if values.size else 1.0
                contour.set_clim(-vmax, vmax)
                fig.colorbar(contour, ax=ax)
                ax.set_aspect("equal")
                fig.savefig(directory / f"wannier-{suffix}-{band}-{label}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)


def plot_band(filename: str | Path, band: BandResult, config: IncarConfig) -> None:
    path = Path(filename)
    _ensure_parent(path)
    if band.dos_components is None or band.dos_energy is None or config.DOS == 0:
        fig, ax = plt.subplots()
        for idx in range(band.energies.shape[1]):
            ax.plot(band.k_axis, np.real(band.energies[:, idx]), color="blue")
        _decorate_band_axis(ax, band, config)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
    ax_band = fig.add_subplot(grid[0])
    for idx in range(band.energies.shape[1]):
        ax_band.plot(band.k_axis, np.real(band.energies[:, idx]), color="blue")
    _decorate_band_axis(ax_band, band, config)
    ax_dos = fig.add_subplot(grid[1], sharey=ax_band)
    cmap = plt.get_cmap("tab10")
    for idx, dos in enumerate(band.dos_components):
        color = cmap(idx)
        ax_dos.plot(np.real(dos), np.real(band.dos_energy), color=color, label=f"DOS {idx + 1}")
        ax_dos.fill_betweenx(np.real(band.dos_energy), 0, np.real(dos), color=color, alpha=0.3)
    ax_dos.set_xlabel("PDOS" if config.DOS == 2 else "DOS")
    ax_dos.tick_params(labelleft=False)
    ax_dos.grid(True)
    ax_dos.legend(loc="upper right", fontsize="small")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _decorate_band_axis(ax, band: BandResult, config: IncarConfig) -> None:
    for pos in [p[1] for p in band.high_sym_points]:
        ax.axvline(x=pos, color="black", linestyle="--", linewidth=0.5)
    ax.set_xticks([p[1] for p in band.high_sym_points])
    ax.set_xticklabels([p[0] for p in band.high_sym_points])
    ax.set_xlim(0, band.k_axis[-1])
    if isinstance(config.band_window, EnergyWindow):
        ax.axhline(y=config.band_window.emin, color="black", linestyle="--", linewidth=1)
        ax.axhline(y=config.band_window.emax, color="black", linestyle="--", linewidth=1)
    if isinstance(config.inner_window, EnergyWindow):
        ax.axhline(y=config.inner_window.emin, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(y=config.inner_window.emax, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("Band Structure", fontsize=14)
    ax.set_ylabel("E", fontsize=12)


def write_topology_figures(directory: str | Path, topology: TopologyResult) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for (gid, direction), (centers, k_param, z2) in topology.wilson.items():
        fig, ax = plt.subplots()
        for band in range(centers.shape[1]):
            ax.plot(k_param, centers[:, band] % 1)
        ax.axvline(x=0.5, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(y=0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$k (2\pi / a)$")
        ax.set_ylabel(r"$x$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        z2_label = "unavailable" if z2 is None else str(z2)
        ax.set_title(f"Wilson loop (direction = {direction}, Z2 = {z2_label})")
        fig.savefig(directory / f"Hybrid_Wilson_Loop-{gid}-d-{direction}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
    for key, (flux, chern) in topology.chern.items():
        fig, ax = plt.subplots()
        nk1, nk2 = flux.shape
        img = ax.imshow(flux.T / (2 * np.pi) * (nk1 * nk2), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5])
        fig.colorbar(img, ax=ax)
        ax.set_xlabel(r"$k_1 (2\pi / a)$")
        ax.set_ylabel(r"$k_2 (2\pi / a)$")
        bands = topology.chern_bands.get(key, ())
        if bands and bands == tuple(range(bands[0], bands[-1] + 1)):
            band_label = str(bands[0]) if len(bands) == 1 else f"{bands[0]}-{bands[-1]}"
        else:
            band_label = ",".join(str(band) for band in bands) or "unknown"
        ax.set_title(f"Chern number = {chern:.4f} (bands {band_label})")
        fig.savefig(directory / f"Chern_Number-{key}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
