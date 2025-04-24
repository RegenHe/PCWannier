import logging
import numpy as np
import re
from typing import Dict, Tuple

class IO:
    @staticmethod
    def load_cell_matrix(filename: str, shape) -> Dict[Tuple[int, int, int], np.ndarray]:
        with open(filename, "r") as f:
            content = f.read()

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

    @staticmethod
    def save_to_txt(filename: str, data: np.ndarray) -> None:
        try:
            with open(filename, 'w') as f:
                shape_info = f"Shape of the data array: {data.shape}"

                f.write(f"# {shape_info}\n")

                if np.iscomplexobj(data):
                    for idx in np.ndindex(data.shape):
                        matrix = data[idx]
                        f.write(f"Matrix at index {idx}:\n")
                        for row in matrix:
                            row_str = ' '.join([f"{entry.real:.8f} + {entry.imag:.8f}j" for entry in row])
                            f.write(row_str + '\n')
                else:
                    for idx in np.ndindex(data.shape):
                        matrix = data[idx]
                        f.write(f"Matrix at index {idx}:\n")
                        for row in matrix:
                            row_str = ' '.join([f"{entry:.8f}" for entry in row])
                            f.write(row_str + '\n')

            logging.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data to {filename}: {str(e)}")
            raise
        return data