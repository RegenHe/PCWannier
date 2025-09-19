import numpy as np
import re
from typing import Dict, Tuple

from .Log import Logger

class IO:
    @staticmethod
    def load_cell_matrix(filename: str, shape) -> np.ndarray:
        data = np.empty(shape, dtype=object)

        with open(filename, "r") as f:
            content = f.read()

        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)

        pattern = re.compile(
            r"CELL\s*[\(\[]\s*([0-9,\s]*)\s*[\)\]]\s*:\s*\n((?:.*?\n)*?)(?=CELL\s*[\(\[]|\Z)", re.MULTILINE)

        for match in pattern.finditer(content):
            index_str = match.group(1).strip()
            lines = match.group(2).strip().splitlines()

            if not index_str:
                continue

            try:
                indices = [int(i.strip()) for i in index_str.split(',') if i.strip() != '']
            except ValueError:
                raise ValueError(f"Invalid index format in CELL: {index_str}")
            
            if len(indices) > len(shape):
                raise ValueError(f"Index {indices} exceeds shape dimensions {shape}")
            
            while len(indices) < len(shape):
                indices.append(0)

            for dim, val in zip(shape, indices):
                if val < 0 or val >= dim:
                    raise IndexError(f"Index {indices} out of bounds for shape {shape}")

            matrix = []
            for line in lines:
                if not line.strip():
                    continue
                entries = [complex(eval(e.strip().replace(' ', '')))
                        for e in line.split(',') if e.strip()]
                matrix.append(entries)

            data[tuple(indices)] = np.array(matrix, dtype=complex)

        return data

    @staticmethod
    def save_to_txt(filename: str, data: np.ndarray, shape: tuple) -> None:
        data = np.array(data)
        try:
            with open(filename, 'w') as f:
                shape_info = f"Shape of the data array: {shape}"
                f.write(f"# {shape_info}\n")

                data = np.array(data)
                for idx in np.ndindex(shape):
                    matrix = data[idx]
                    f.write(f"CELL{idx}:\n")
                    for row in matrix:
                        row_str = ', '.join([f"{entry.real:.8f}" + (' + ' if entry.imag >= 0 else ' - ') + f"{abs(entry.imag):.8f}j" if np.iscomplexobj(data) else f"{entry:.8f}" for entry in row])
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
