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
            i, j, k = i - 1, j - 1, k - 1  # MATLAB 是 1-based，Python 是 0-based

            matrix = []
            for line in lines[1:]:
                if not line.strip():
                    continue
                entries = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*[-+]\s*\d*\.?\d+(?:[eE][-+]?\d+)?j', line)
                row = [complex(eval(e.replace(' ', ''))) for e in entries]
                matrix.append(row)

            data[i, j, k] = np.array(matrix, dtype=complex)

        return data