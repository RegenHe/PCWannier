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
                        Logger.error(f"Failed to parse the number of coords: {line_str}")
                        raise
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
                            Logger.error(f"Failed to parse vertex coordinates: {line_str}")
                            raise
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
                        Logger.error(f"Failed to parse the number of elements: {line_str}")
                        raise
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
                            Logger.error(f"Failed to parse edge indices: {line_str}")
                            raise
                    else:
                        Logger.error(f"Not enough integers in the line: {line_str}")
                        raise
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
                        Logger.error(f"Failed to parse the number of elements: {line_str}")
                        raise
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
                            Logger.error(f"Failed to parse triangle indices: {line_str}")
                            raise
                    else:
                        Logger.error(f"Not enough integers in the line: {line_str}")
                        raise
                    lines_read += 1
                else:
                    in_block = False
                    tri_elements_block = False


    if coords is None or elements is None:
        Logger.error("Failed to load mesh: coords or elements data is missing.")
        raise

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
                        Logger.error(f"Failed to parse data: {line_str}")
                        raise
                else:
                    Logger.error(f"Not enough data in the line: {line_str}")
                    raise

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
            avg_val = np.mean(comp_vals[lst])
            best_data_i = max(lst, key=lambda i: comp_vals[i])
            comp_vals[best_data_i] = avg_val
            mesh_to_data_idx[m_idx] = best_data_i
            mesh_dists[m_idx] = np.linalg.norm(mesh.vertices[m_idx] - data.point_matrix[best_data_i])

    return mesh_to_data_idx, mesh_dists

def distribute_data(mesh: Mesh, data: RawData, left: bool=False) -> StateCollection:
    if global_data.incar is None:
        Logger.error("Incar data is not initialized.")
        raise
    
    gd = global_data.incar
    bw = gd.band_window
    kdim = int(getattr(gd, "kdim", 2))

    Nk = [
        len(gd.k_points[0]) if kdim >= 1 else 1,
        len(gd.k_points[1]) if kdim >= 2 else 1,
        len(gd.k_points[2]) if kdim >= 3 else 1,
    ]
    Nv = data.value_matrix.shape[0]
    order = list(gd.dataset_order)

    if isinstance(bw, EnergyWindow):
        size_map = {"k1": Nk[0], "k2": Nk[1], "k3": Nk[2]}
        
        has_E = ("E" in order)
        if has_E:
            shape_in = (Nv,) + tuple(Nk[0] if d == "k1" else Nk[1] if d == "k2" else Nk[2] if d == "k3" else -1 for d in order)
            pos = {dim: (order.index(dim) + 1) for dim in order}
        else:
            shape_core = tuple(size_map[d] for d in order)
            shape_in = (Nv,) + shape_core + (-1,)
            pos = {dim: (order.index(dim) + 1) for dim in order}
            pos["E"] = 1 + len(order)

        base = data.value_matrix.reshape(shape_in, order='C')

        E_idx = getattr(global_data.state_collection, "E_idx", None)
        Eall = getattr(data, "energy_matrix", None)
        if Eall is None:
            Eall = getattr(global_data, "energy_matrix", None)
        if E_idx is None and Eall is None:
            Logger.error("energy_matrix is required for EnergyWindow but not found.")
            raise

        fields = [[[[] for _ in range(Nk[2])] for _ in range(Nk[1])] for _ in range(Nk[0])]

        for i in range(Nk[0]):
            for j in range(Nk[1]):
                for k in range(Nk[2]):
                    if E_idx is not None:
                        sel_list = E_idx[i][j][k]
                    else:
                        if Eall.ndim == 3:
                            eline = Eall[i, j, :]
                        elif Eall.ndim == 4:
                            eline = Eall[i, j, k, :]
                        else:
                            Logger.error(f"Unsupported energy_matrix ndim={Eall.ndim}")
                            raise
                        sel_list = np.where((eline >= bw.emin) & (eline <= bw.emax))[0].tolist()

                    if not sel_list:
                        continue

                    for b in sel_list:
                        idx = [slice(None)] * (1 + (len(order) if has_E else len(order) + 1))
                        if "k1" in pos: idx[pos["k1"]] = i
                        if "k2" in pos: idx[pos["k2"]] = j
                        if "k3" in pos: idx[pos["k3"]] = k
                        idx[pos["E"]] = b
                        vec = base[tuple(idx)]
                        if vec.ndim != 1 or vec.shape[0] != Nv:
                            Logger.error(f"Unexpected vec shape {vec.shape} at (i,j,k,b)=({i},{j},{k},{b}); expect ({Nv},)")
                            raise
                        fields[i][j][k].append(vec)

        if left:
            global_data.state_collection.Lfield = fields
        else:
            global_data.state_collection.Rfield = fields
        Logger.info("distribute data finished (energy-window, ragged E)")

    else:
        sizes = {"k1": Nk[0], "k2": Nk[1], "k3": Nk[2]}
        bw_idx = np.asarray(gd.band_window, dtype=int)

        has_E = ("E" in order)
        if has_E:
            shape_in = (Nv,) + tuple((sizes[d] if d in sizes else -1) for d in order)
            ax_E = order.index("E") + 1
        else:
            shape_in = (Nv,) + tuple(sizes[d] for d in order) + (-1,)
            ax_E = 1 + len(order)

        t_all = data.value_matrix.reshape(shape_in, order='C')
        t_sel = np.take(t_all, bw_idx, axis=ax_E)

        target = [d for d in ("k1", "k2", "k3") if d in order] + (["E"] if has_E else [])
        pos = {dim: (order.index(dim) + 1) for dim in order}
        if not has_E:
            pos["E"] = ax_E
        axes = (0,) + tuple(pos[d] for d in ("k1", "k2", "k3") if d in pos) + (pos["E"],)
        t_reordered = np.transpose(t_sel, axes=axes)

        t_reordered = np.reshape(t_reordered, (Nv, Nk[0], Nk[1], Nk[2], -1), order='C')

        Nb = t_reordered.shape[-1]
        fields = [
            [
                [
                    [np.zeros(Nv, dtype=complex) for _ in range(Nb)]
                    for _ in range(Nk[2])
                ]
                for _ in range(Nk[1])
            ]
            for _ in range(Nk[0])
        ]
        for i in range(Nk[0]):
            for j in range(Nk[1]):
                for k in range(Nk[2]):
                    for b in range(Nb):
                        fields[i][j][k][b] = t_reordered[:, i, j, k, b]

        if left:
            global_data.state_collection.Lfield = fields
        else:
            global_data.state_collection.Rfield = fields
        Logger.info("distribute data finished")

    return global_data.state_collection

