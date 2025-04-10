import numpy as np

from .utils import Mesh, RawData
from typing import List, Tuple


def load_comsol_mesh(filename: str) -> Mesh:
    ncoords = 0
    coords = None
    in_vertex_block = False

    nelems = 0
    in_elems_block = False
    lines_read = 0
    elements = None

    tri_elements_block = False

    with open(filename, "r") as f:
        for line in f:
            line_str = line.strip()

            if "# number of mesh vertices" in line_str:
                tokens = line_str.split()
                if len(tokens) >= 1:
                    try:
                        ncoords = int(tokens[0])
                        coords = np.empty((ncoords, 2), dtype=float)
                    except Exception as e:
                        raise RuntimeError(f"Failed to parse the number of coords: {line_str}") from e
                continue

            if line_str.startswith("# Mesh vertex coordinates"):
                in_vertex_block = True
                lines_read = 0
                continue

            if in_vertex_block and line_str:
                if lines_read < ncoords:
                    tokens = line_str.split()
                    if len(tokens) >= 2:
                        try:
                            coords[lines_read, 0] = float(tokens[0])
                            coords[lines_read, 1] = float(tokens[1])
                        except Exception as e:
                            raise RuntimeError("Error handling: invalid line in vertex block") from e
                    lines_read += 1
                else:
                    in_vertex_block = False

            if line_str.startswith("3 tri"):
                tri_elements_block = True
                continue

            if "# number of elements" in line_str and tri_elements_block:
                tokens = line_str.split()
                if len(tokens) >= 1:
                    try:
                        nelems = int(tokens[0])
                        elements = np.empty((nelems, 3), dtype=int)
                    except Exception as e:
                        raise RuntimeError(f"Failed to parse the number of elements: {line_str}") from e
                continue

            if line_str.startswith("# Elements"):
                in_elems_block = True
                lines_read = 0
                continue

            if in_elems_block and tri_elements_block and line_str:
                if lines_read < nelems:
                    tokens = line_str.split()
                    if len(tokens) >= 3:
                        try:
                            elements[lines_read, 0] = int(tokens[0])
                            elements[lines_read, 1] = int(tokens[1])
                            elements[lines_read, 2] = int(tokens[2])
                        except Exception as e:
                            raise RuntimeError(f"Failed to parse triangle indices: {line_str}") from e
                    else:
                        raise RuntimeError(f"Not enough integers in the line: {line_str}")
                    lines_read += 1
                else:
                    in_elems_block = False


    if coords is None or elements is None:
        raise RuntimeError("Failed to load mesh: coords or elements data is missing.")

    return Mesh(coords, elements)

def load_comsol_data(filename: str) -> RawData:
    points = []
    values = []

    with open(filename, "r") as f:
        for line in f:
            line_str: str = line.strip()
            if line_str.startswith("%"):
                continue

            if line_str:
                tokens = line_str.split()
                if len(tokens) >= 3:
                    try:
                        point = [float(token) for token in tokens[0:2]]
                        value = [complex(token.replace('i', 'j')) for token in tokens[2:]]
                        points.append(point)
                        values.append(value)
                    except Exception as e:
                        raise RuntimeError(f"Failed to parse data: {line_str}") from e
                else:
                    raise RuntimeError(f"Not enough data in the line: {line_str}")

    point_matrix = np.array(points, dtype=float)
    value_matrix = np.array(values, dtype=complex)
    
    return RawData(point_matrix, value_matrix)