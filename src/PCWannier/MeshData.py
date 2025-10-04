import numpy as np
from scipy.spatial import cKDTree

from typing import List, Tuple, Optional, NamedTuple
import warnings

from .Log import Logger
from .GlobalData import global_data
from .Utils import Mesh, RawData, StateCollection
from .IncarParser import EnergyWindow


def load_comsol_mesh(filename: str) -> Mesh:
    ncoords = 0
    coords = None
    in_vertex_block = False

    nelems = 0
    in_block = False

    lines_read = 0
    elements = None

    tri_elements_block = False

    ndege = 0
    edg_elements = None
    edg_elements_block = False

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

            if line_str.startswith("3 edg"):
                edg_elements_block = True
                continue
            if "number of elements" in line_str and edg_elements_block:
                tokens = line_str.split()
                if len(tokens) >= 1:
                    try:
                        ndege = int(tokens[0])
                        edg_elements = np.empty((ndege, 2), dtype=int)
                    except Exception as e:
                        raise RuntimeError(f"Failed to parse the number of elements: {line_str}") from e
                continue

            if line_str.startswith("# Elements") and (edg_elements_block or tri_elements_block):
                in_block = True
                lines_read = 0
                continue

            if in_block and edg_elements_block and line_str:
                if lines_read < ndege:
                    tokens = line_str.split()
                    if len(tokens) >= 2:
                        try:
                            edg_elements[lines_read, 0] = int(tokens[0])
                            edg_elements[lines_read, 1] = int(tokens[1])
                        except Exception as e:
                            raise RuntimeError(f"Failed to parse edge indices: {line_str}") from e
                    else:
                        raise RuntimeError(f"Not enough integers in the line: {line_str}")
                    lines_read += 1
                else:
                    in_block = False
                    edg_elements_block = False


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

            if in_block and tri_elements_block and line_str:
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
                    in_block = False
                    tri_elements_block = False


    if coords is None or elements is None:
        raise RuntimeError("Failed to load mesh: coords or elements data is missing.")

    return Mesh(coords, elements, edg_elements)

def load_comsol_data(filename: str) -> RawData:
    points = []
    values = []

    with open(filename, "r", errors="replace") as f:
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

# def match_data_to_mesh(mesh: Mesh, data: RawData) -> Tuple[np.ndarray, np.ndarray]:
#     if mesh.vertices.shape[1] != data.point_matrix.shape[1] or mesh.vertices.shape[0] != data.value_matrix.shape[0]:
#         Logger.warning("Mesh and data dimensions do not match.")

#     # tree = cKDTree(mesh.vertices)
#     # dists, idxs = tree.query(data.point_matrix, k=1)
    # tree = cKDTree(data.point_matrix)
    # dists, idxs = tree.query(mesh.vertices, k=1)

#     seen = set()
#     unique_idx = []
#     unique_dists = []
    
#     for i, idx in enumerate(idxs):
#         if idx not in seen:
#             seen.add(idx)
#             unique_idx.append(idx)
#             unique_dists.append(dists[i])

#     return unique_idx, unique_dists
def match_data_to_mesh(mesh: Mesh, data: RawData, *, value_col: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if mesh.vertices.shape[1] != data.point_matrix.shape[1] or mesh.vertices.shape[0] != data.value_matrix.shape[0]:
        Logger.warning("Mesh and data dimensions do not match.")
    data.value_matrix
    tree = cKDTree(mesh.vertices)
    dists, mesh_idxs = tree.query(data.point_matrix, k=1)

    buckets = [[] for _ in range(len(mesh.vertices))]
    for data_i, m_idx in enumerate(mesh_idxs):
        buckets[m_idx].append(data_i)

    comp_vals = (data.value_matrix[:] if data.value_matrix.ndim == 1 else data.value_matrix[:, value_col if value_col is not None else 0])

    mesh_to_data_idx = np.full(len(mesh.vertices), -1, dtype=int)
    mesh_dists = np.full(len(mesh.vertices), np.inf, dtype=float)

    for m_idx, lst in enumerate(buckets):
        if lst:
            avg_val = float(np.mean(comp_vals[lst]))
            best_data_i = max(lst, key=lambda i: comp_vals[i])
            comp_vals[best_data_i] = avg_val
            mesh_to_data_idx[m_idx] = best_data_i
            mesh_dists[m_idx] = np.linalg.norm(mesh.vertices[m_idx] - data.point_matrix[best_data_i])

    return mesh_to_data_idx, mesh_dists

def distribute_data(mesh: Mesh, data: RawData) -> StateCollection:
    if global_data.incar is None:
        raise RuntimeError("Incar data is not initialized.")
    
    bw = global_data.incar.band_window

    if isinstance(bw, EnergyWindow):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])
        Nv    = data.value_matrix.shape[0]

        d0, d1 = global_data.incar.dataset_order[0], global_data.incar.dataset_order[1]
        n0 = len(global_data.incar.k_points[0]) if d0 == "k1" else len(global_data.incar.k_points[1])
        n1 = len(global_data.incar.k_points[0]) if d1 == "k1" else len(global_data.incar.k_points[1])
        base = data.value_matrix.reshape((Nv, n0, n1, -1), order='C')

        pos = {dim: i for i, dim in enumerate(global_data.incar.dataset_order)}
        ax_k1 = pos["k1"] + 1
        ax_k2 = pos["k2"] + 1

        E_idx = getattr(global_data.state_collection, "E_idx", None)
        Eall = getattr(data, "energy_matrix", None)
        if Eall is None:
            Eall = getattr(global_data, "energy_matrix", None)

        fields = [[[] for _ in range(k2_sz)] for _ in range(k1_sz)]
        for i in range(k1_sz):
            for j in range(k2_sz):
                if E_idx is not None:
                    sel_list = E_idx[i][j]
                else:
                    sel_list = np.where((Eall[i, j] >= bw.emin) & (Eall[i, j] <= bw.emax))[0].tolist()
                if not sel_list:
                    continue
                if ax_k1 == 1 and ax_k2 == 2:
                    for b in sel_list:
                        fields[i][j].append(base[:, i, j, b])
                else:
                    for b in sel_list:
                        fields[i][j].append(base[:, j, i, b])

        global_data.state_collection.field = fields
        Logger.info("distribute data finished (energy-window, ragged E)")
    else:
        sizes = {
            "k1": len(global_data.incar.k_points[0]),
            "k2": len(global_data.incar.k_points[1]),
            "E": len(global_data.incar.band_window)
        }
        shape = tuple(sizes[dim] for dim in global_data.incar.dataset_order)

        fields = [
            [
                [np.zeros(data.value_matrix.shape[0], dtype=complex) for _ in range(shape[2])]
                for _ in range(shape[1])
            ]
            for _ in range(shape[0])
        ]
        t_fields = np.zeros((data.value_matrix.shape[0],) + shape, dtype=complex)

        for p in range(data.value_matrix.shape[0]):
            t_fields[p] = data.value_matrix[p].reshape((shape[0], shape[1], -1), order='C')[:, :, global_data.incar.band_window]

        desired_order = ["k1", "k2", "E"]
        indices = [global_data.incar.dataset_order.index(dim) for dim in desired_order]
        t_fields = np.transpose(t_fields, axes=(0, indices[0] + 1, indices[1] + 1, indices[2] + 1))

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    fields[i][j][k] = t_fields[:, i, j, k]

        global_data.state_collection.field = fields
        Logger.info("distribute data finished")
    return global_data.state_collection
