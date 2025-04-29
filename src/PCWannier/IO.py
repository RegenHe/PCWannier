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

        content = re.sub(r'#.*$', '', content)

        cells = re.split(r"(?=CELL[\[\(])", content.strip())

        for cell in cells:
            lines = cell.strip().splitlines()
            if not lines:
                continue
            header = lines[0]

            indices = list(map(int, re.findall(r"\d+", header)))

            if len(indices) > len(shape):
                raise ValueError(f"Parsed dimensions {len(indices)} are greater than expected shape dimensions {len(shape)}.")
            
            indices = indices[:len(shape)]

            indices = [i - 1 for i in indices]  

            matrix = []
            for line in lines[1:]:
                if not line.strip():
                    continue
                entries = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*[-+]\s*\d*\.?\d+(?:[eE][-+]?\d+)?j', line)
                row = [complex(eval(e.replace(' ', ''))) for e in entries]
                matrix.append(row)

            idx = tuple(indices)
            data[idx] = np.array(matrix, dtype=complex)

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