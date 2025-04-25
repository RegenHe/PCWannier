import logging
import numpy as np
import re
from typing import Dict, Tuple

class IO:
    @staticmethod
    def load_cell_matrix(filename: str, shape) -> np.ndarray:
        data = np.empty(shape, dtype=object)

        with open(filename, "r") as f:
            content = f.read()

        cells = re.split(r"\n(?=CELL\[)", content.strip())

        for cell in cells:
            lines = cell.strip().splitlines()
            if not lines:
                continue
            header = lines[0]
            i, j, k = map(int, re.findall(r"\d+", header))
            i, j, k = i - 1, j - 1, k - 1

            matrix = []
            for line in lines[1:]:
                if not line.strip():
                    continue
                entries = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*[-+]\s*\d*\.?\d+(?:[eE][-+]?\d+)?j', line)
                row = [complex(eval(e.replace(' ', ''))) for e in entries]
                matrix.append(row)

            data[i, j, k] = np.array(matrix, dtype=complex)
        return data

    @staticmethod
    def save_to_txt(filename: str, data: np.ndarray, shape: tuple) -> None:
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

            logging.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data to {filename}: {str(e)}")
            raise