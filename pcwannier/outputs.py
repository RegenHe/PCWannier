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
from .data import BandResult, Mesh, RunResult, TopologyResult
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


def save_cell_matrix(filename: str | Path, data, shape: tuple | None = None) -> None:
    path = Path(filename)
    _ensure_parent(path)

    def as_matrix(obj):
        arr = np.asarray(obj)
        if arr.dtype == object:
            return None
        if arr.ndim == 2:
            return arr
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 0 and np.issubdtype(arr.dtype, np.number):
            return arr.reshape(1, 1)
        return None

    def iter_cells(obj, prefix=()):
        mat = as_matrix(obj)
        if mat is not None:
            yield prefix, mat
            return
        if isinstance(obj, (list, tuple, np.ndarray)):
            for idx, sub in enumerate(obj):
                yield from iter_cells(sub, prefix + (idx,))
            return
        raise TypeError(f"Unsupported cell data at {prefix}: {type(obj)}")

    with path.open("w", encoding="utf-8") as handle:
        if shape is not None:
            handle.write(f"# Declared grid shape (top-level): {shape}\n")
        handle.write("# Each CELL may have its own matrix shape (ragged supported).\n")
        for idx, matrix in iter_cells(data):
            matrix = np.asarray(matrix)
            handle.write(f"CELL{idx if idx else '(root)'} shape={tuple(matrix.shape)}:\n")
            if matrix.size:
                for row in matrix:
                    handle.write(_fmt_c(row, force_complex=True, spaced=True, zero_small_imag=True) + "\n")
            handle.write("\n")


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


def load_interpolation_points(filename: str | Path) -> np.ndarray:
    path = Path(filename)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        points = np.loadtxt(handle, delimiter=",")
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Invalid interpolation mesh {path}: each row must contain x,y.")
    return points


def save_points_with_values(filename: str | Path, points: np.ndarray, values: np.ndarray) -> None:
    path = Path(filename)
    _ensure_parent(path)
    points = np.asarray(points, dtype=float)
    vals = np.asarray(values)
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2).")
    if vals.shape[1] != points.shape[0]:
        raise ValueError("Number of interpolation points and value rows differ.")

    with path.open("w", encoding="utf-8") as handle:
        for idx, (x, y) in enumerate(points):
            row = [f"{x:.10f}", f"{y:.10f}"]
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
    interp_epsilon: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> None:
    points = load_interpolation_points(interp_path)
    interp_path = Path(interp_path)
    mesh = result.extended_mesh
    triang = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements)

    wannier_path = _resolve_interpolation_output(interp_path, interp_wannier, "interp-wannier", out_dir)
    if wannier_path is not None:
        values = []
        for _, wmat in result.wanniers.items():
            wmat = np.asarray(wmat)
            for band in range(wmat.shape[1]):
                values.append(_interpolate_complex(triang, wmat[:, band], points))
        with timed_step("write interpolated Wannier data", LOGGER, file=wannier_path):
            save_points_with_values(wannier_path, points, np.asarray(values))

    epsilon_path = _resolve_interpolation_output(interp_path, interp_epsilon, "interp-epsilon", out_dir)
    if epsilon_path is not None:
        epsilon = _interpolate_real(triang, result.extended_epsilon, points)
        with timed_step("write interpolated epsilon data", LOGGER, file=epsilon_path):
            save_points_with_values(epsilon_path, points, epsilon.reshape(1, -1))


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


def write_outputs(result: RunResult, config: IncarConfig | None = None, out_dir: str | Path | None = None) -> None:
    config = config or result.config
    m_path = _resolve_output(config.M_file, config, out_dir)
    if m_path is not None:
        with timed_step("write M0 matrix", LOGGER, file=m_path):
            save_cell_matrix(m_path, result.M0, result.M0.shape + (len(config.composition_of_b) // 2,))
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

    hopping_path = _resolve_output(config.hopping_file, config, out_dir)
    if hopping_path is not None:
        with timed_step("write hopping", LOGGER, file=hopping_path, count=len(result.hoppings)):
            save_dict(hopping_path, result.hoppings)

    wannier_path = _resolve_output(config.wannier_file, config, out_dir)
    if wannier_path is not None:
        with timed_step("write Wannier data", LOGGER, file=wannier_path, count=len(result.wanniers)):
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
        ax.set_title(f"Wilson loop (direction = {direction}, Z2 = {z2})")
        fig.savefig(directory / f"Hybrid_Wilson_Loop-{gid}-d-{direction}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
    for key, (flux, chern) in topology.chern.items():
        fig, ax = plt.subplots()
        nk1, nk2 = flux.shape
        img = ax.imshow(flux.T / (2 * np.pi) * (nk1 * nk2), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5])
        fig.colorbar(img, ax=ax)
        ax.set_xlabel(r"$k_1 (2\pi / a)$")
        ax.set_ylabel(r"$k_2 (2\pi / a)$")
        ax.set_title(f"Chern number = {chern:.4f}")
        fig.savefig(directory / f"Chern_Number-{key}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
