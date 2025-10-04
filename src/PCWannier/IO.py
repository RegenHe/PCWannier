import numpy as np
import re
from typing import Dict, Tuple

from .Log import Logger

class IO:
    @staticmethod
    def load_cell_matrix(filename: str, shape=None) -> np.ndarray:
        with open(filename, "r") as f:
            content = f.read()
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)

        pattern = re.compile(
            r"CELL\s*[\(\[]\s*([0-9,\s]*)\s*[\)\]]\s*"
            r"(?:shape\s*=\s*\(([^)]*)\)\s*)?"
            r":\s*\n"
            r"((?:.*?(?:\n|$))*?)(?=CELL\s*[\(\[]|\Z)",
            re.IGNORECASE
        )

        cells = []
        ndims = 0
        for m in pattern.finditer(content):
            idx_str = (m.group(1) or "").strip()
            if not idx_str:
                continue
            try:
                idx_tuple = tuple(int(x.strip()) for x in idx_str.split(",") if x.strip() != "")
            except ValueError:
                raise ValueError(f"Invalid index format in CELL: {idx_str}")

            ndims = max(ndims, len(idx_tuple))

            block_text = (m.group(3) or "").strip("\n")
            if not block_text:
                mat = np.empty((0, 0), dtype=complex)
            else:
                rows = []
                for line in block_text.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    tokens = [tok.strip().replace(' ', '') for tok in s.split(',') if tok.strip()]
                    try:
                        row = [complex(tok) for tok in tokens]
                    except Exception:
                        row = [complex(eval(tok)) for tok in tokens]
                    rows.append(row)
                mat = np.array(rows, dtype=complex)

            cells.append((idx_tuple, mat))

        if not cells:
            raise ValueError("No CELL blocks found in file.")

        if shape is None:
            max_per_dim = [0] * ndims
            for idx, _ in cells:
                for d, val in enumerate(idx):
                    if val > max_per_dim[d]:
                        max_per_dim[d] = val
            shape = tuple(v + 1 for v in max_per_dim)
        else:
            if len(shape) < ndims:
                raise ValueError(f"Provided shape {shape} has fewer dims than index tuples ({ndims}).")

        data = np.empty(shape, dtype=object)
        data.flat[:] = None

        for idx, mat in cells:
            if len(idx) < len(shape):
                idx = idx + (0,) * (len(shape) - len(idx))
            for d, val in enumerate(idx):
                if val < 0 or val >= shape[d]:
                    raise IndexError(f"Index {idx} out of bounds for shape {shape}")
            data[idx] = mat

        return data

    @staticmethod
    def save_to_txt(filename: str, data, shape: tuple | None = None) -> None:

        def _is_matrix_like(x) -> bool:
            try:
                arr = np.asarray(x)
            except Exception:
                return False
            return (arr.ndim == 2) and (arr.dtype != object)

        def _iter_cells(obj, prefix=()):
            if _is_matrix_like(obj):
                yield prefix, np.asarray(obj)
                return
            if isinstance(obj, (list, tuple, np.ndarray)):
                for i, sub in enumerate(obj):
                    yield from _iter_cells(sub, prefix + (i,))
                return
            raise TypeError(f"Unsupported data type at {prefix}: {type(obj)}")

        def _grid_overview(obj):
            dims = []
            cur = obj
            try:
                while isinstance(cur, (list, tuple, np.ndarray)) and not _is_matrix_like(cur):
                    dims.append(len(cur))
                    cur = cur[0] if len(cur) > 0 else []
            except Exception:
                pass
            return tuple(dims)

        try:
            with open(filename, 'w') as f:
                inferred = _grid_overview(data)
                if shape is not None:
                    f.write(f"# Declared grid shape (top-level): {shape}\n")
                if inferred:
                    f.write(f"# Inferred grid dims (top-level): {inferred}\n")
                f.write("# Each CELL may have its own matrix shape (ragged supported).\n")

                for idx, matrix in _iter_cells(data):
                    is_complex = np.iscomplexobj(matrix)
                    f.write(f"CELL{idx} shape={tuple(matrix.shape)}:\n")
                    if matrix.size == 0:
                        f.write("\n")
                        continue
                    for row in matrix:
                        if is_complex:
                            row_str = ', '.join(
                                f"{val.real:.8f}" + (" + " if val.imag >= 0 else " - ") + f"{abs(val.imag):.8f}j"
                                for val in row
                            )
                        else:
                            row_str = ', '.join(f"{float(val):.8f}" for val in row)
                        f.write(row_str + '\n')

            Logger.info(f"Data successfully saved to {filename}")
        except Exception as e:
            Logger.error(f"Error saving data to {filename}: {str(e)}")
            raise

    @staticmethod
    def save_band(filename: str, data: np.ndarray, k_path: np.ndarray):
        try:
            with open(filename, 'w') as f:
                num_k_points = k_path.shape[0]
                num_bands = data.shape[1]
                f.write(f"# k-points: {num_k_points}, Bands: {num_bands}\n")
                f.write("# kx, ky, band_1, band_2, ..., band_n\n")

                for i in range(num_k_points):
                    k_point = ", ".join([f"{k_path[i, j]:.8f}" for j in range(k_path.shape[1])])
                    
                    band_energies = ", ".join([f"{data[i, j].real:.8f}+{data[i, j].imag:.8f}j" if data[i, j].imag != 0 else f"{data[i, j].real:.8f}" for j in range(num_bands)])
                    
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
                raise ValueError("Invalid file format.")
            return points
        except Exception as e:
            Logger.error(f"Failed to load mesh points from '{filename}': {e}")
            raise

    def save_points_with_values(filename: str, points: np.ndarray, values: np.ndarray):
        values = np.array(values)
        if points.shape[0] != values[0].shape[0]:
            Logger.error("Number of points and number of value rows must match.")
            raise ValueError("Number of points and number of value rows must match.")

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
    def _fmt_c(x, tol=1e-12):
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            return ", ".join(IO._fmt_c(t, tol) for t in arr)
        try:
            xr = float(np.real(x))
            xi = float(np.imag(x))
            if abs(xi) < tol:
                return f"{xr:.10f}"
            sign = '+' if xi >= 0 else ''
            return f"{xr:.10f}{sign}{xi:.10f}j"
        except Exception:
            return repr(x)
