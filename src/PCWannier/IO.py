import math
import numpy as np
import re
from typing import Dict, Tuple

from .Log import Logger

class IO:
    @staticmethod
    def load_cell_matrix(filename: str, shape=None) -> np.ndarray:
        header_re = re.compile(
            r'^CELL\s*([\(\[])\s*([^\)\]]*)\s*[\)\]]\s*'
            r'(?:shape\s*=\s*\(([^)]*)\))?\s*:\s*$',
            re.IGNORECASE
        )

        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()

        blocks = []
        current = None
        buf = []

        def flush_current():
            if current is not None:
                blocks.append((current, ''.join(buf)))
                buf.clear()

        for raw in lines:
            line = raw.rstrip("\n")
            if line.lstrip().startswith('#'):
                continue
            m = header_re.match(line.strip())
            if m:
                flush_current()
                _bracket, idx_str, _shape_str = m.groups()
                idx_str = (idx_str or "").strip()
                if idx_str == "" or idx_str.lower() == "root":
                    idx_tuple = ()
                else:
                    try:
                        idx_tuple = tuple(int(s.strip()) for s in idx_str.split(',') if s.strip() != "")
                    except Exception as e:
                        Logger.error(f"Invalid CELL index '{idx_str}'")
                        raise
                current = (idx_tuple, _shape_str)
            else:
                buf.append(raw)

        flush_current()

        if not blocks:
            Logger.error("No CELL blocks found in file.")
            raise

        cells = []
        ndims = 0

        def parse_token(tok: str) -> complex:
            s = tok.strip()
            if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
                s = s[1:-1]

            s = re.sub(r'\s+', '', s, flags=re.UNICODE).replace('−', '-')

            if len(s) >= 2 and s[0] == '(' and s[-1] == ')':
                s = s[1:-1]

            try:
                return complex(s)
            except Exception:
                Logger.error(f"Cannot parse token {tok!r} after normalization -> {s!r}")
                raise

        for ((idx_tuple, _shape_str), text) in blocks:
            ndims = max(ndims, len(idx_tuple))
            rows = []
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                tokens = [tok for tok in s.split(',') if tok.strip()]
                row = [parse_token(tok) for tok in tokens]
                rows.append(row)

            if rows:
                L = {len(r) for r in rows}
                if len(L) != 1:
                    Logger.error(f"Ragged rows in CELL{idx_tuple}: row lengths = {sorted(L)}")
                    raise
                mat = np.array(rows, dtype=complex)
            else:
                mat = np.empty((0, 0), dtype=complex)

            cells.append((idx_tuple, mat))

        if all(len(idx) == 0 for idx, _ in cells):
            if len(cells) != 1:
                Logger.error("Multiple 'CELL(root)' blocks found.")
                raise
            return cells[0][1]

        if shape is None:
            max_per_dim = [0] * ndims
            for idx, _ in cells:
                for d, val in enumerate(idx):
                    if val > max_per_dim[d]:
                        max_per_dim[d] = val
            shape = tuple(v + 1 for v in max_per_dim)
        else:
            if len(shape) < ndims:
                Logger.error(f"Provided shape {shape} has fewer dims than index tuples ({ndims}).")
                raise

        data = np.empty(shape, dtype=object)
        data.flat[:] = None

        for idx, mat in cells:
            if len(idx) < len(shape):
                idx = idx + (0,) * (len(shape) - len(idx))
            for d, val in enumerate(idx):
                if not (0 <= val < shape[d]):
                    Logger.error(f"Index {idx} out of bounds for shape {shape}")
                    raise
            data[idx] = mat

        return data



    @staticmethod
    def save_to_txt(filename: str, data, shape: tuple | None = None) -> None:
        def _as_matrix(x):
            try:
                arr = np.asarray(x)
            except Exception:
                return None
            if arr.dtype == object:
                return None
            if arr.ndim == 2:
                return arr
            if arr.ndim == 1:
                return arr.reshape(1, -1)
            if arr.ndim == 0 and np.issubdtype(arr.dtype, np.number):
                return arr.reshape(1, 1)
            return None

        def _iter_cells(obj, prefix=()):
            mat = _as_matrix(obj)
            if mat is not None:
                yield prefix, mat
                return

            if isinstance(obj, (list, tuple, np.ndarray)):
                for i, sub in enumerate(obj):
                    yield from _iter_cells(sub, prefix + (i,))
                return
            Logger.error(f"Unsupported data type at {prefix}: {type(obj)}")
            raise

        def _grid_overview(obj):
            dims = []
            cur = obj
            try:
                while isinstance(cur, (list, tuple, np.ndarray)) and (_as_matrix(cur) is None):
                    n = len(cur)
                    dims.append(n)
                    cur = cur[0] if n > 0 else []
            except Exception:
                pass
            return tuple(dims)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                inferred = _grid_overview(data)
                if shape is not None:
                    f.write(f"# Declared grid shape (top-level): {shape}\n")
                if inferred:
                    f.write(f"# Inferred grid dims (top-level): {inferred}\n")
                f.write("# Each CELL may have its own matrix shape (ragged supported).\n")

                for idx, matrix in _iter_cells(data):
                    matrix = np.asarray(matrix)
                    f.write(f"CELL{idx if idx else '(root)'} shape={tuple(matrix.shape)}:\n")
                    if matrix.size == 0:
                        f.write("\n")
                        continue
                    for row in matrix:
                        f.write(IO._fmt_c(row, tol=1e-12, prec=8, force_complex=True, spaced=True, zero_small_imag=True) + "\n")
                    f.write("\n")

            Logger.info(f"Data successfully saved to {filename}")
        except Exception as e:
            Logger.error(f"Error saving data to {filename}: {str(e)}")
            raise


    @staticmethod
    def save_band(filename: str, data: np.ndarray, k_path: np.ndarray, other_info: Dict = None) -> None:
        try:
            E = np.asarray(data)
            if k_path is None:
                if E.ndim == 1:
                    E = E.reshape(1, -1)
                elif E.ndim > 2:
                    E = E.reshape(E.shape[0], -1)
                if E.shape[0] != 1:
                    Logger.error(f"When k_path is None, data must be 1D or have first dim == 1, got shape {data.shape}.")
                    raise

                num_k_points, num_bands = 1, E.shape[1]
                with open(filename, 'w') as f:
                    f.write(f"# k-points: {num_k_points}, Bands: {num_bands}\n")
                    f.write("# band_1, band_2, ..., band_n\n")
                    if other_info is not None:
                        for key, value in other_info.items():
                            f.write(f"# {key}: {value}\n")

                    band_energies = ", ".join(
                        (f"{E[0, j].real:.8f}{E[0, j].imag:+.8f}j"
                        if E[0, j].imag != 0 else f"{E[0, j].real:.8f}")
                        for j in range(num_bands)
                    )
                    f.write(f"{band_energies}\n")

                Logger.info(f"Band structure data has been saved to {filename}")
                return

            k = np.asarray(k_path)
            if k.ndim == 1:
                k = k.reshape(-1, 1)
            elif k.ndim > 2:
                k = k.reshape(k.shape[0], -1)
            num_k_points, k_dim = k.shape

            if E.ndim == 1:
                E = E.reshape(-1, 1)
            elif E.ndim > 2:
                E = E.reshape(E.shape[0], -1)

            if E.shape[0] != num_k_points:
                Logger.error(f"The first dimension of k_path and data do not match: {num_k_points} vs {E.shape[0]}")
                raise

            num_bands = E.shape[1]

            if k_dim == 1:
                k_header = "k"
            else:
                k_header = ", ".join([f"k{i+1}" for i in range(k_dim)])

            with open(filename, 'w') as f:
                f.write(f"# k-points: {num_k_points}, Bands: {num_bands}\n")
                f.write(f"# {k_header}, band_1, band_2, ..., band_n\n")
                
                if other_info is not None:
                    for key, value in other_info.items():
                        f.write(f"# {key}: {value}\n")

                for i in range(num_k_points):
                    k_point = ", ".join(f"{k[i, j]:.8f}" for j in range(k_dim))
                    band_energies = ", ".join(
                        (f"{E[i, j].real:.8f}{E[i, j].imag:+.8f}j"
                        if E[i, j].imag != 0 else f"{E[i, j].real:.8f}")
                        for j in range(num_bands)
                    )
                    f.write(f"{k_point},{band_energies}\n")

            Logger.info(f"Band structure data has been saved to {filename}")

        except Exception as e:
            Logger.error(f"An error occurred while saving the band structure: {e}")
            raise


    
    @staticmethod
    def load_mesh_points(filename: str) -> np.ndarray:
        try:
            points = np.loadtxt(filename, delimiter=',')
            if points.ndim != 2 or points.shape[1] != 2:
                Logger.error("Invalid mesh file format: each line must contain exactly two comma-separated values (x, y).")
                raise
            return points
        except Exception as e:
            Logger.error(f"Failed to load mesh points from '{filename}': {e}")
            raise

    def save_points_with_values(filename: str, points: np.ndarray, values: np.ndarray):
        values = np.array(values)
        if points.shape[0] != values[0].shape[0]:
            Logger.error("Number of points and number of value rows must match.")
            raise

        with open(filename, 'w') as f:
            for i in range(points.shape[0]):
                x, y = points[i]
                val_strs = []

                for v in values[:, i]:
                    if isinstance(v, complex):
                        val_strs.append(f"{v.real:.10f}{v.imag:+.10f}j")
                    else:
                        val_strs.append(f"{v:.10f}")

                row = f"{x:.10f},{y:.10f}," + ",".join(val_strs)
                f.write(row + "\n")
        Logger.info(f"Data successfully saved to {filename}")

    @staticmethod
    def save_dict(filename: str, d: dict):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                for k, v in d.items():
                    f.write(f"{k}\n")
                    if isinstance(v, np.ndarray):
                        arr = v
                        if arr.ndim == 0:
                            f.write(IO._fmt_c(arr.item()) + "\n")
                        elif arr.ndim == 1:
                            f.write(IO._fmt_c(arr) + "\n")
                        else:
                            for i in range(arr.shape[0]):
                                f.write(IO._fmt_c(arr[i]) + "\n")

                    elif isinstance(v, (list, tuple)):
                        if len(v) > 0 and isinstance(v[0], (list, tuple, np.ndarray)):
                            for row in v:
                                row = np.asarray(row)
                                f.write(IO._fmt_c(row) + "\n")
                        else:
                            f.write(IO._fmt_c(v) + "\n")

                    else:
                        f.write(f"{repr(v)}\n")

                    f.write("\n")
        except Exception as e:
            Logger.error(f"Error saving dict to {filename}: {e}")
            raise
        else:
            Logger.info(f"Dict saved to {filename}")


    @staticmethod
    def _fmt_c(x, tol=1e-12, prec=10, force_complex=False, spaced=False, zero_small_imag=True):
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            return ", ".join(IO._fmt_c(t, tol=tol, prec=prec,
                                    force_complex=force_complex,
                                    spaced=spaced,
                                    zero_small_imag=zero_small_imag) for t in arr)
        try:
            xr = float(np.real(x))
            xi = float(np.imag(x))
            sign_neg = math.copysign(1.0, xi) < 0.0
            if zero_small_imag and abs(xi) < tol:
                xi_disp = 0.0
            else:
                xi_disp = xi

            if not force_complex and abs(xi) < tol:
                return f"{xr:.{prec}f}"

            if spaced:
                sign_str = " - " if sign_neg else " + "
                return f"{xr:.{prec}f}{sign_str}{abs(xi_disp):.{prec}f}j"
            else:
                sign_str = "+" if xi_disp >= 0 else ""
                return f"{xr:.{prec}f}{sign_str}{xi_disp:.{prec}f}j"
        except Exception:
            return repr(x)

