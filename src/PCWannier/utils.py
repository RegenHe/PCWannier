import os

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree

import copy

from typing import List, Tuple, Iterator, Callable, Any

from .IO import IO
from .Log import Logger
from .Timer import Timer, timer
from .GlobalData import global_data

from .IncarParser import IncarData

class Mesh:
    def __init__(self, vertices: np.ndarray, elements: np.ndarray, edge: np.ndarray=None) -> None:
        if vertices is None:
            self.vertices = None
            self.mindist = None
        else:
            self.vertices: np.ndarray = np.array(vertices)
            tree = cKDTree(self.vertices)
            dists, idxs = tree.query(self.vertices, k=2)
            self.mindist = np.min(dists[:, 1])
        
        if elements is None:
            self.elements = None
        else:
            self.elements: np.ndarray = np.array(elements)
        
        if edge is None:
            self.edge = None
        else:
            self.edge: np.ndarray = np.array(edge)
        # self.edge_index: np.ndarray = np.unique(self.edge.flatten())

        self.tri_weights = None
        self._precompute_tri_weights()

    def _precompute_tri_weights(self):
        elems, verts = self.elements, self.vertices
        v0, v1, v2 = verts[elems[:,0]], verts[elems[:,1]], verts[elems[:,2]]
        w = np.abs((v1[:,0]-v0[:,0])*(v2[:,1]-v0[:,1]) - (v2[:,0]-v0[:,0])*(v1[:,1]-v0[:,1])) / 6.0
        self.tri_weights = w


    def func(self, f, offset=[0, 0]):
        dx = self.vertices[:, 0] - offset[0]
        dy = self.vertices[:, 1] - offset[1]
        out = f(dx, dy)
        return np.asarray(out, dtype=np.complex128)
    

    def rfunc(self, f, offset=[0, 0], ang=0):
        dx = self.vertices[:, 0] - offset[0]
        dy = self.vertices[:, 1] - offset[1]
        out = f(np.hypot(dx, dy), np.arctan2(dy, dx) + np.deg2rad(ang))
        return np.asarray(out, dtype=np.complex128)
        

    def __repr__(self) -> str:
        return f"Mesh(vertices={self.vertices}, elements={self.elements})"
    
    def plot(self) -> None:
        fig, ax = plt.subplots()
        for element in self.elements:
            triangle_vertices = self.vertices[element]
            
            triangle = plt.Polygon(triangle_vertices, edgecolor='black', fill=None)
            ax.add_patch(triangle)

        if self.edge is not None:
            for line in self.edge:
                point1, point2 = self.vertices[line[0]], self.vertices[line[1]]
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='red', linewidth=2)

        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        min_x, max_x = np.min(self.vertices[:, 0]), np.max(self.vertices[:, 0])
        min_y, max_y = np.min(self.vertices[:, 1]), np.max(self.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)

        plt.show()
    def save_fig(self, filename):
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        fig, ax = plt.subplots()
        for element in self.elements:
            triangle_vertices = self.vertices[element]
            
            triangle = plt.Polygon(triangle_vertices, edgecolor='black', fill=None)
            ax.add_patch(triangle)

        if self.edge is not None:
            for line in self.edge:
                point1, point2 = self.vertices[line[0]], self.vertices[line[1]]
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='red', linewidth=2)

        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        min_x, max_x = np.min(self.vertices[:, 0]), np.max(self.vertices[:, 0])
        min_y, max_y = np.min(self.vertices[:, 1]), np.max(self.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    def extension(self, n: list, real_lattice_vectors=None, lattice_const=None) -> List:
        if real_lattice_vectors is None:
            real_lattice_vectors=global_data.incar.real_lattice_vectors
        if lattice_const is None:
            lattice_const=global_data.incar.lattice_const
        if n[0] < 1 or n[1] < 1:
            Logger.error("n must be greater than 1")
            raise
        if self.vertices is None:
            Logger.error("Mesh must be initialized")
            raise
        
        original_vertices = self.vertices.copy()
        original_elements = self.elements.copy()

        # TODO: using the edge to find the same vertices
        # original_index = self.edge_index.copy()
        
        index_map = None
        space_to_original_mapping = list(range(len(original_vertices)))

        for i in range(n[0]):
            for j in range(n[1]):
                if i == 0 and j == 0:
                    continue
                offset_x = (real_lattice_vectors[0][0] * i + real_lattice_vectors[1][0] * j) * lattice_const
                offset_y = (real_lattice_vectors[0][1] * i + real_lattice_vectors[1][1] * j) * lattice_const

                new_elements = original_elements + np.max(self.elements) + 1
                # new_vertices = original_vertices + np.array([offset_x + 1e-3 * i, offset_y + 1e-3 * j])
                new_vertices = original_vertices + np.array([offset_x, offset_y])
                vertex_idx_in_new_vertices, vertex_idx_in_vertices = self.match(new_vertices, self.vertices)
                for k in range(len(vertex_idx_in_new_vertices)):
                    new_elements[new_elements == (vertex_idx_in_new_vertices[k] + np.max(self.elements) + 1)] = vertex_idx_in_vertices[k]
                
                self.elements = np.vstack((self.elements, new_elements))
                self.vertices = np.vstack((self.vertices, new_vertices))

                space_to_original_mapping = np.hstack((space_to_original_mapping, np.arange(len(original_vertices))))

                index_map, space_to_original_mapping = self.rebuild_index(space_to_original_mapping)
        
        offset_x = (real_lattice_vectors[0][0] * np.floor((n[0] - 1) / 2) + real_lattice_vectors[1][0] * np.floor((n[1] - 1) / 2)) * lattice_const
        offset_y = (real_lattice_vectors[0][1] * np.floor((n[0] - 1) / 2) + real_lattice_vectors[1][1] * np.floor((n[1] - 1) / 2)) * lattice_const
        self.vertices = self.vertices - np.array([offset_x, offset_y])
        
        self._precompute_tri_weights()

        return space_to_original_mapping

    def match(self, new_vertices, vertices) -> Tuple[np.ndarray, np.ndarray]:
        tree = cKDTree(new_vertices)
        dists, idxs = tree.query(vertices, k=1)
        vertex_idx_in_vertices = np.where(dists < self.mindist * 0.5)[0]

        vertex_idx_in_new_vertices = []
        if vertex_idx_in_vertices.size > 0:
                t_tree = cKDTree(new_vertices)
                for i in vertex_idx_in_vertices:
                    t_dists, t_idxs = t_tree.query(vertices[i], k=1)
                    vertex_idx_in_new_vertices.append(t_idxs)
        else:
            Logger.warning(f"No points found")

        return np.array(vertex_idx_in_new_vertices), vertex_idx_in_vertices
    
    def rebuild_index(self, space_to_original_mapping=None):
        used_indices = set(self.elements.flatten())
        
        new_vertices = [self.vertices[i] for i in sorted(used_indices)]
        new_vertices = np.array(new_vertices)

        old_to_new_index = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_indices))}
        if space_to_original_mapping is None:
            space_to_original_mapping = list(range(len(self.vertices)))
        else:
            space_to_original_mapping = [space_to_original_mapping[i] for i in sorted(old_to_new_index.keys())]
        # space_to_original_mapping = list(sorted(used_indices))
        
        new_elements = []
        for element in self.elements:
            new_element = [old_to_new_index[idx] for idx in element]
            new_elements.append(new_element)
        
        self.vertices = new_vertices
        self.elements = np.array(new_elements)

        return old_to_new_index, space_to_original_mapping
    
    def __deepcopy__(self, memo=None):
        return Mesh(copy.deepcopy(self.vertices, memo), copy.deepcopy(self.elements, memo), copy.deepcopy(self.edge, memo))


class RawData:
    def __init__(self, point_matrix: np.ndarray, value_matrix: np.ndarray) -> None:
        self.point_matrix: np.ndarray = np.array(point_matrix)
        self.value_matrix: np.ndarray = np.array(value_matrix)

    def __repr__(self) -> str:
        return f"RawData(point_matrix={self.point_matrix}, value_matrix={self.value_matrix})"
    
class FieldData:
    def __init__(self, name: str, mesh: Mesh, value: np.ndarray) -> None:
        self.name: str = name
        self.mesh: Mesh = mesh
        self.field: np.ndarray = np.array(value)

    def __repr__(self) -> str:
        return f"FieldData(point_matrix={self.mesh}, value_matrix={self.field})"
    
    def plot(self, real=True) -> None:
        fig, ax = plt.subplots()
        triang = Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements)

        if real:
            plt.tricontourf(triang, np.real(self.field), levels=255, cmap='bwr')
        else:
            plt.tricontourf(triang, np.imag(self.field), levels=255, cmap='bwr')
        plt.clim(-max(np.abs(self.field)), max(np.abs(self.field)))
        if real:
            plt.colorbar(label='Real Part')
        else:
            plt.colorbar(label='Imaginary Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.mesh.vertices[:, 0]), np.max(self.mesh.vertices[:, 0])
        min_y, max_y = np.min(self.mesh.vertices[:, 1]), np.max(self.mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.show()
    def save_fig(self, filename, real=True):
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        fig, ax = plt.subplots()
        triang = Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements)

        if real:
            plt.tricontourf(triang, np.real(self.field), levels=255, cmap='bwr')
        else:
            plt.tricontourf(triang, np.imag(self.field), levels=255, cmap='bwr')
        plt.clim(-max(np.abs(self.field)), max(np.abs(self.field)))
        if real:
            plt.colorbar(label='Real Part')
        else:
            plt.colorbar(label='Imaginary Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.mesh.vertices[:, 0]), np.max(self.mesh.vertices[:, 0])
        min_y, max_y = np.min(self.mesh.vertices[:, 1]), np.max(self.mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin * (max_x - min_x), max_x + margin * (max_x - min_x))
        ax.set_ylim(min_y - margin * (max_y - min_y), max_y + margin * (max_y - min_y))
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        Logger.info(f"figure successfully saved to {filename}")


class StateCollection:
    def __init__(self, name: str, mesh: Mesh, kdim: int) -> None:
        self.name: str = name
        self.mesh: Mesh = mesh
        self.field: list = []
        self.epsilon: np.array = None
        self.normalization: float = 0.0

        self.transform: np.ndarray = None
        self.transform_: np.ndarray = None
        self.raw_field: list = None

        self.is_normalized = False
        self.is_bloch = False
        self.is_orthogonalized = False

        self.extention_mesh = None
        self.space_to_original_mapping = None

        self.extention_epsilon: np.array = None

        self.E: list = []
        self.E_idx: list = []
        self.inner_E_idx: list = []

        self.kdim = kdim
        self.k_shape = (len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]) if len(global_data.incar.k_points) > 1 else 1, len(global_data.incar.k_points[2]) if len(global_data.incar.k_points) > 2 else 1)

    def k_indices(self) -> Iterator[Tuple[int, int, int]]:
        Nk1, Nk2, Nk3 = self.k_shape
        if self.kdim not in (1, 2, 3):
            Logger.error(f"Unsupported k_dim = {self.kdim}; expected 1, 2 or 3.")
            raise
        shape = (Nk1, Nk2 if self.kdim >= 2 else 1, Nk3 if self.kdim >= 3 else 1)
        yield from np.ndindex(shape)
        
    def n_indices(self, *idx) -> Iterator[int]:
        i, j, k, _ = self._norm_ijkn(*(list(idx) + [0]))
        blk = self.field[i][j][k]
        if blk is None:
            return
        yield from range(len(blk))

    def get_k_num(self) -> int:
        Nkx, Nky, Nkz = self.k_shape
        if self.kdim == 1:
            return Nkx
        elif self.kdim == 2:
            return Nkx * Nky
        elif self.kdim == 3:
            return Nkx * Nky * Nkz
        else:
            Logger.error(f"Unsupported k_dim={self.kdim}; expected 1, 2 or 3.")
            raise

    def gen_matrix_on_kmesh(self, factory: Callable[..., Any]) -> np.ndarray:
        Nkx, Nky, Nkz = self.k_shape
        arr = np.empty((Nkx, Nky, Nkz), dtype=object)
        for idx in np.ndindex(Nkx, Nky, Nkz):
            arr[idx] = factory(*idx)
        return arr
    
    def gen_zeros_matrxi(self, dtype=complex) -> np.ndarray:
        return np.zeros(self.k_shape, dtype=dtype)

    def _norm_ijkn(self, *idx) -> tuple[int, int, int, int]:
        L = len(idx)
        if L < 1:
            Logger.error(f"_norm_ijkn expects at least 1 args, got {L}")
            raise

        n = int(idx[-1])
        coords = list(idx[:-1])

        expect = self.kdim
        if len(coords) < expect:
            coords += [0] * (expect - len(coords))
        else:
            coords = coords[:expect]

        coords += [0] * (3 - len(coords))
        i, j, k = int(coords[0]), int(coords[1]), int(coords[2])

        Nkx, Nky, Nkz = self.k_shape
        i = i % Nkx
        j = (j % Nky) if self.kdim >= 2 else 0
        k = (k % Nkz) if self.kdim >= 3 else 0

        return i, j, k, n
    
    def get(self, *idx):
        i, j, k, n = self._norm_ijkn(*idx)
        try:
            return self.field[i][j][k][n]
        except Exception as e:
            Logger.error(f"Failed to access field[{i}][{j}][{k}][{n}]: {e}")
            raise

    def get_block(self, *kidx):
        i, j, k, _ = self._norm_ijkn(*(kidx + (0,)))
        blk = self.field[i][j][k]
        if blk is None:
            Logger.error(f"empty block at k=({i},{j},{k})")
            raise

        if isinstance(blk, (list, tuple)):
            blk = np.vstack(blk)

        return np.asarray(blk, dtype=np.complex128)


    @timer("Orthogonalize - ")
    def orthogonalize(self, tol_rel: float = 1e-6, atol_abs: float = 1e-12) -> None:
        Logger.info("Orthogonalizing states...")
        if self.field is None:
            Logger.error("Field data is not initialized")
            raise

        if 'O' in global_data.incar.use_cached_data:
            Logger.info("using cache data - O")
            self.transform = IO.load_cell_matrix(global_data.incar.O_file, self.k_shape)
            self.is_orthogonalized = True
            self.is_normalized = True
            self.transform_ = self.gen_matrix_on_kmesh(lambda *_: None)
            for i, j, k in self.k_indices():
                T_diag = np.diag(np.diag(self.transform[i][j][k]))
                for n in range(len(self.field[i][j][k])):
                    if T_diag[n, n] == 0.0:
                        Logger.error(f"Normalization failed for field ({i}, {j}, {k}, {n})")
                        raise
                    self.field[i][j][k][n] *= T_diag[n, n]
                self.transform_[i][j][k] = np.linalg.inv(T_diag) @ self.transform[i][j][k]
            return
        
        Nv = self.mesh.vertices.shape[0]

        eps = np.asarray(self.epsilon)

        self.transform = self.gen_matrix_on_kmesh(lambda *_: None)
        self.transform_ = self.gen_matrix_on_kmesh(lambda *_: None)
        for i, j, k in self.k_indices():
            W = np.asarray(self.get_block(i, j, k))
            Nwin = len(W)
            Nv = self.mesh.vertices.shape[0]
            if W.ndim == 1:
                W = W[None, :]
            elif W.ndim == 2 and W.shape[1] != Nv:
                W = W.T
            if W.shape[1] != Nv:
                Logger.error(f"field shape {W.shape} incompatible with Nv={Nv}")
                raise

            S = np.empty((Nwin, Nwin), dtype=np.complex128)
            for a in self.n_indices(i, j, k):
                wa = self.get(i, j, k, a)

                right = np.conj(wa) * eps
                F = (right[:, None] * W.T).astype(np.complex128, copy=False)
                fd = FieldData("S", self.mesh, F)
                vals = WannierTools.integrate_over_mesh(fd, chunk_size=2048)

                # A = FieldData("A", self.mesh, np.broadcast_to(np.conj(wa[None, :]), (Nwin, Nv)).astype(np.complex128, copy=False))
                # B = FieldData("B", self.mesh, (W.T * eps[:, None]).astype(np.complex128, copy=False))
                # vals = WannierTools.integrate_over_mesh(A, other=B, chunk_size=2048)
                if vals.ndim == 0:
                    vals = np.asarray([vals])

                S[a, a:] = vals[a:]
                S[a:, a] = vals[a:].conjugate()

            S = 0.5*(S + S.conj().T)
            w, V = np.linalg.eigh(S)
            lam = w.real
            lam_max = np.max(lam)
            tau = max(tol_rel*lam_max, atol_abs)
            invsqrt = 1.0/np.sqrt(np.maximum(lam, tau))
            T_corr = V @ np.diag(invsqrt) @ V.conj().T
            T_corr = 0.5*(T_corr + T_corr.conj().T)
            # block orth then total orth maybe better
            T_full = T_corr

            T_diag = np.diag(np.diag(T_full))

            for n in range(len(self.field[i][j][k])):
                if T_diag[n, n] == 0.0:
                    Logger.error(f"Normalization failed for field ({i}, {j}, {k}, {n})")
                    raise
                self.field[i][j][k][n] *= T_diag[n, n]

            self.transform[i][j][k] = T_full
            self.transform_[i][j][k] = np.linalg.inv(T_diag) @ T_full
        self.is_orthogonalized = True
        self.is_normalized = True

        if not global_data.incar.O_file.lower == "false":
            IO.save_to_txt(global_data.incar.O_file, self.transform, self.k_shape)


    @timer("Orthogonality Check - ")
    def check_orthogonality(self) -> Tuple[np.ndarray, bool]:
        """
        report: 
            [0] herm_res     = ||S - S^†||_F
            [1] diag_max_err = max_i |S_{ii} - 1|
            [2] offdiag_max  = max_{i≠j} |S_{ij}|
            [3] frob_res     = ||S_H - I||_F,  S_H = (S + S^†)/2
            [4] lambda_min   = min eig(S_H)
            [5] cond         = cond(S_H) = lam_max / lam_min
        """
        if self.field is None:
            Logger.error("Field data is not initialized")
            raise
        
        Nv = self.mesh.vertices.shape[0]

        eps = np.asarray(self.epsilon)
        Nk1, Nk2, Nk3 = self.k_shape
        report = np.zeros((Nk1, Nk2, Nk3, 6), dtype=np.float64)
        need_orth = False
                    
        for i, j, k in self.k_indices():
            W = np.asarray(self.get_block(i, j, k))
            Nwin = len(W)
            Nv = self.mesh.vertices.shape[0]
            if W.ndim == 1:
                W = W[None, :]
            elif W.ndim == 2 and W.shape[1] != Nv:
                W = W.T
            if W.shape[1] != Nv:
                Logger.error(f"field shape {W.shape} incompatible with Nv={Nv}")
                raise

            S = np.empty((Nwin, Nwin), dtype=np.complex128)
            for a in self.n_indices(i, j, k):
                wa = self.get(i, j, k, a)

                right = np.conj(wa) * eps
                F = (right[:, None] * W.T).astype(np.complex128, copy=False)
                fd = FieldData("S", self.mesh, F)
                vals = WannierTools.integrate_over_mesh(fd, chunk_size=2048)
                if vals.ndim == 0:
                    vals = np.asarray([vals])
                # A = FieldData("A", self.mesh, np.broadcast_to(np.conj(wa[None, :]), (Nwin, Nv)).astype(np.complex128, copy=False))
                # B = FieldData("B", self.mesh, (W.T * eps[:, None]).astype(np.complex128, copy=False))
                # vals = WannierTools.integrate_over_mesh(A, other=B, chunk_size=2048)
                S[a, a:] = vals[a:]
                S[a:, a] = vals[a:].conjugate()
                # S[a, a] = np.real(S[a, a])
                
            if self.is_orthogonalized:
                S = self.transform_[i][j][k].conj().T @ S @ self.transform_[i][j][k]

            herm_res = float(np.linalg.norm(S - S.conj().T, ord='fro'))
            diag = np.real(np.diag(S))
            diag_max_err = float(np.max(np.abs(diag - 1.0))) if Nwin > 0 else 0.0
            offdiag = S - np.diag(np.diag(S))
            offdiag_max = float(np.max(np.abs(offdiag))) if Nwin > 1 else 0.0
            
            S_H = 0.5 * (S + S.conj().T)
            frob_res = float(np.linalg.norm(S_H - np.eye(Nwin), ord='fro'))

            evals = np.linalg.eigvalsh(S_H)
            lam_min = float(np.min(evals))
            lam_max = float(np.max(evals))

            lam_min_safe = max(lam_min, 1e-16)
            cond = float(lam_max / lam_min_safe) if lam_max > 0 else np.inf

            report[i, j, k, 0] = herm_res
            report[i, j, k, 1] = diag_max_err
            report[i, j, k, 2] = offdiag_max
            report[i, j, k, 3] = frob_res
            report[i, j, k, 4] = lam_min
            report[i, j, k, 5] = cond

            if herm_res > 1e-8 or diag_max_err > 1e-3 or offdiag_max > 1e-3 or lam_min < -1e-6:
                need_orth = True
                Logger.warning(f"[OrthChk] k=({i},{j},{k}): herm={herm_res:.3e}, "
                            f"diag_max_err={diag_max_err:.3e}, offdiag_max={offdiag_max:.3e}, "
                            f"frob={frob_res:.3e}, lam_min={lam_min:.3e}, cond={cond:.2e}")
        return report, need_orth
        
    def turn_to_Bloch(self) -> None:
        if self.field is None:
            Logger.error("Field data is not initialized")
            raise
        
        if self.is_bloch:
            Logger.info("Field data is already in Bloch form")
            return
        self.is_bloch = True

        for i, j, k in self.k_indices():
            phase = self.get_phase(i, j, k)
            for n in self.n_indices(i, j, k):
                self.field[i][j][k][n] = np.conj(phase) * self.field[i][j][k][n]
    
    def get_transform(self, zero=False) -> np.ndarray:
        if self.is_orthogonalized and self.transform_ is not None and not zero:
            return self.transform_
        else:
            return [[[np.identity(self.transform_[i][j][k].shape[0], dtype=self.transform_[i][j][k].dtype) for k in range(len(self.field[i][j]))] for j in range(len(self.field[i]))] for i in range(len(self.field))]
    
    def get_phase(self, i: int, j: int, k: int):
        if global_data.incar.dataset_type.lower() == 'comsol':
            sign = -1
        k_ = WannierTools.get_kxyz([i, j, k])[:global_data.incar.kdim]
        return np.exp(1j * sign * np.dot(self.mesh.vertices, k_))
    
    def get_phase_k(self, k: np.ndarray):
        if global_data.incar.dataset_type.lower() == 'comsol':
            sign = -1
        return np.exp(1j * sign * np.dot(self.mesh.vertices, k[:global_data.incar.kdim]))
    
    @staticmethod
    def get_phase_k_r(k: np.ndarray, r: np.ndarray):
        if global_data.incar.dataset_type.lower() == 'comsol':
            sign = -1
        return np.exp(1j * sign * np.dot(r, k))
    
    def get_extention_phase(self, i: int, j: int, k: int):
        if global_data.incar.dataset_type.lower() == 'comsol':
            sign = -1
        k = WannierTools.get_kxyz([i, j, k])
        return np.exp(1j * sign * np.dot(self.extention_mesh.vertices, k[:global_data.incar.kdim]))
    
    @timer("Extention - ")
    def extention(self, n: List) -> None:
        self.extention_mesh = copy.deepcopy(self.mesh)
        self.space_to_original_mapping = self.extention_mesh.extension(n)
        if self.epsilon is not None:
            self.get_extention_epsilon()
    
    def get_extention_field(self, *idx) -> List:
        i, j, k, n = self._norm_ijkn(*idx)
        if self.extention_mesh is None:
            Logger.error("The field has not been extended")
            raise
        return np.array([self.field[i][j][k][n][p] / np.sqrt(global_data.incar.extension[0] * global_data.incar.extension[1]) for p in self.space_to_original_mapping])
    
    def get_extention_epsilon(self) -> List:
        if self.extention_mesh is None:
            Logger.error("The field has not been extended")
            raise
        if self.extention_epsilon is None:
            self.extention_epsilon = np.array([self.epsilon[p] for p in self.space_to_original_mapping])
        return self.extention_epsilon

        
    def get_zero_field(self):
        return np.zeros(global_data.incar.kdim) * self.mesh.vertices.shape[0]
    
    def get_zero_extension_field(self):
        return np.zeros(global_data.incar.kdim) * self.extention_mesh.vertices.shape[0]
    
    def plot_field(self, *idx) -> None:
        i, j, k, n = self._norm_ijkn(*idx)

        fig, ax = plt.subplots()
        triang = Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements)

        plt.tricontourf(triang, np.real(self.field[i][j][k][n]), levels=255, cmap='jet')
        plt.colorbar(label='Real Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.mesh.vertices[:, 0]), np.max(self.mesh.vertices[:, 0])
        min_y, max_y = np.min(self.mesh.vertices[:, 1]), np.max(self.mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.show()
    
    def plot_extention_field(self, *idx) -> None:
        i, j, k, n = self._norm_ijkn(*idx)

        fig, ax = plt.subplots()
        triang = Triangulation(self.extention_mesh.vertices[:, 0], self.extention_mesh.vertices[:, 1], self.extention_mesh.elements)

        field = self.get_extention_field(i, j, k, n)
        plt.tricontourf(triang, np.real(field), levels=255, cmap='jet')
        plt.colorbar(label='Real Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.extention_mesh.vertices[:, 0]), np.max(self.extention_mesh.vertices[:, 0])
        min_y, max_y = np.min(self.extention_mesh.vertices[:, 1]), np.max(self.extention_mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.show()

    def plot_epsilon(self) -> None:
        fig, ax = plt.subplots()
        triang = Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements)

        plt.tricontourf(triang, np.real(self.epsilon), levels=255, cmap='jet')
        plt.colorbar(label='Real Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.mesh.vertices[:, 0]), np.max(self.mesh.vertices[:, 0])
        min_y, max_y = np.min(self.mesh.vertices[:, 1]), np.max(self.mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.show()
    
    def plot_extention_epsilon(self) -> None:
        fig, ax = plt.subplots()
        triang = Triangulation(self.extention_mesh.vertices[:, 0], self.extention_mesh.vertices[:, 1], self.extention_mesh.elements)

        plt.tricontourf(triang, np.real(self.extention_epsilon), levels=255, cmap='jet')
        plt.colorbar(label='Real Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.extention_mesh.vertices[:, 0]), np.max(self.extention_mesh.vertices[:, 0])
        min_y, max_y = np.min(self.extention_mesh.vertices[:, 1]), np.max(self.extention_mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.show()

    def __deepcopy__(self, memo=None):
        sc = StateCollection(copy.deepcopy(self.name, memo), copy.deepcopy(self.mesh, memo))
        sc.field = copy.deepcopy(self.field)
        sc.epsilon = copy.deepcopy(self.epsilon)
        sc.normalization = copy.deepcopy(self.normalization)

        sc.is_normalized = copy.deepcopy(self.is_normalized)
        sc.is_bloch = copy.deepcopy(self.is_bloch)
        return sc
    
    def __repr__(self) -> str:
        return f"StateCollection(point_matrix={self.mesh}, field_number={len(self.field)})"



class WannierTools:
    def __init__(self) -> None:
        self.init = False

    def preprocess(self) -> None:
        global_data.incar.kdim = len(global_data.incar.real_lattice_vectors)
        if np.allclose(np.asarray(global_data.incar.reciprocal_lattice_vectors), np.zeros((global_data.incar.kdim, global_data.incar.kdim))):
            v = np.linalg.inv(global_data.incar.real_lattice_vectors) @ np.eye(global_data.incar.kdim)
            Logger.info(f"reciprocal_lattice_vectors will be set to: {v.T}")
            global_data.incar.reciprocal_lattice_vectors = v.T
        
        if self.init is False:
            self.init = True
            global_data.incar.composition_of_b = [global_data.incar.composition_of_b, [[-v for v in sublist] for sublist in global_data.incar.composition_of_b]]
            global_data.incar.composition_of_b = [item for sublist in global_data.incar.composition_of_b for item in sublist]

            global_data.incar.b_vectors = []
            for i in range(len(global_data.incar.composition_of_b)):
                vec = np.zeros_like(global_data.incar.reciprocal_lattice_vectors[0], dtype=float)
                for ax in range(global_data.incar.kdim):
                    vec += global_data.incar.composition_of_b[i][ax] * global_data.incar.reciprocal_lattice_vectors[ax] / len(global_data.incar.k_points[ax]) * 2 * np.pi / global_data.incar.lattice_const
                global_data.incar.b_vectors.append(vec.tolist())
                
            global_data.incar.b_vectors = np.array(global_data.incar.b_vectors) * global_data.incar.lattice_const
            mat_a = np.eye(global_data.incar.kdim).reshape(-1, 1)
            mat_b = np.zeros((global_data.incar.kdim ** global_data.incar.kdim, len(global_data.incar.composition_of_b)))
            for i in range(global_data.incar.kdim):
                for j in range(global_data.incar.kdim):
                    for k in range(len(global_data.incar.composition_of_b)):
                        mat_b[i * global_data.incar.kdim + j][k] = global_data.incar.b_vectors[k][i] * global_data.incar.b_vectors[k][j]
            
            global_data.incar.wb = (np.linalg.pinv(mat_b) @ mat_a).flatten()
        
        global_data.incar.band_calc_num = 0
        for p in global_data.incar.projections:
            global_data.incar.band_calc_num += len(p['states'])

        global_data.incar.validate()

    def set_incar(self, incar_data: IncarData) -> None:
        global_data.incar = incar_data

    @staticmethod
    def neighbor_reciprocal_lattice_vectors(k: list, direction: int):
        gd = global_data.incar
        axes = int(gd.kdim)

        idxs = (list(k) + [0] * max(0, axes - len(k)))[:axes]
        comp = (list(gd.composition_of_b[direction]) + [0] * max(0, axes - len(gd.composition_of_b[direction])))[:axes]

        wrapped = [0, 0, 0]
        raw_list = []
        crossed = False

        for ax in range(axes):
            N = len(gd.k_points[ax])
            raw = idxs[ax] + comp[ax]
            w = int(np.mod(raw, N))
            wrapped[ax] = w
            raw_list.append(int(raw))
            if w != raw:
                crossed = True

        k1 = wrapped[0]
        k2 = wrapped[1] if axes >= 2 else 0
        k3 = wrapped[2] if axes >= 3 else 0

        k_raw3 = (raw_list + [0, 0, 0])[:3]
        k_ = k_raw3 if crossed else None

        return (k1, k2, k3), k_

    
    @staticmethod
    def get_kxyz(k: list) -> np.ndarray:
        kps = global_data.incar.k_points
        G = np.asarray(global_data.incar.reciprocal_lattice_vectors)

        axes = len(kps)
        idxs = list(k) + [0] * max(0, axes - len(k))
        idxs = idxs[:axes]

        # !!!!!!!!!!!! need to check !!!!!!!!!!!!
        if G.shape[0] < axes:
            G = np.pad(G, ((0, axes - G.shape[0]), (0, 0)), mode='constant')
        if G.shape[1] < 3:
            G = np.pad(G, ((0, 0), (0, 3 - G.shape[1])), mode='constant')

        vec = np.zeros(3, dtype=float)
        for ax in range(axes):
            arr = np.asarray(kps[ax], dtype=float)
            if arr.size == 0:
                Logger.error(f"empty k_points[{ax}]")
                raise
            if 0 <= idxs[ax] < arr.size:
                kval = arr[idxs[ax]]
            else:
                step = (arr[1] - arr[0]) if arr.size >= 2 else 0.0
                kval = arr[0] + step * idxs[ax]
            vec += kval * G[ax, :3]

        return (2.0 * np.pi / global_data.incar.lattice_const) * vec
    
    @staticmethod
    def integrate_over_triangle(vertices: np.ndarray, data_on_triangle: np.ndarray) -> complex:
        jacobian = np.array([[vertices[1, 0] - vertices[0, 0], vertices[2, 0] - vertices[0, 0]], [vertices[1, 1] - vertices[0, 1], vertices[2, 1] - vertices[0, 1]]])
        return np.sum(data_on_triangle) * np.abs(np.linalg.det(jacobian)) / 6.0
    
    
    @staticmethod
    @nb.njit(parallel=True, cache=True, fastmath=False)
    def _integrate_batch_numba(F, elems, w):
        Nt = elems.shape[0]
        K = F.shape[1]
        out_r = np.zeros(K, dtype=np.float64)
        out_i = np.zeros(K, dtype=np.float64)
        for k in nb.prange(K):
            sr = 0.0
            si = 0.0
            for t in range(Nt):
                i0, i1, i2 = elems[t, 0], elems[t, 1], elems[t, 2]
                wt = w[t]
                s = F[i0, k] + F[i1, k] + F[i2, k]
                sr += wt * s.real
                si += wt * s.imag
            out_r[k] = sr
            out_i[k] = si
        return out_r + 1j*out_i
    

    @nb.njit(parallel=True, cache=True, fastmath=False)
    def _integrate_prod_numba(A, B, elems, w, hermitian=False, block=1<<14):
        Nt = elems.shape[0]
        K  = A.shape[1]
        out_r = np.zeros(K, np.float64)
        out_i = np.zeros(K, np.float64)

        for k in range(K):
            nb_blocks = (Nt + block - 1) // block
            part_r = np.zeros(nb_blocks, np.float64)
            part_i = np.zeros(nb_blocks, np.float64)

            for b in nb.prange(nb_blocks):
                start = b * block
                end   = min(start + block, Nt)

                sr = 0.0; cr = 0.0
                si = 0.0; ci = 0.0

                for t in range(start, end):
                    i0 = elems[t,0]; i1 = elems[t,1]; i2 = elems[t,2]
                    a0 = A[i0,k]; a1 = A[i1,k]; a2 = A[i2,k]
                    b0 = B[i0,k]; b1 = B[i1,k]; b2 = B[i2,k]
                    if hermitian:
                        a0 = np.conj(a0); a1 = np.conj(a1); a2 = np.conj(a2)

                    z = 2.0*(a0*b0 + a1*b1 + a2*b2) + (a0*b1 + a1*b0 + a0*b2 + a2*b0 + a1*b2 + a2*b1)
                    s = 0.25 * w[t]

                    yr = s * z.real - cr
                    tmp = sr + yr
                    cr = (tmp - sr) - yr
                    sr = tmp

                    yi = s * z.imag - ci
                    tmp = si + yi
                    ci = (tmp - si) - yi
                    si = tmp

                part_r[b] = sr
                part_i[b] = si

            sr = 0.0; cr = 0.0
            for b in range(nb_blocks):
                y = part_r[b] - cr
                tmp = sr + y
                cr = (tmp - sr) - y
                sr = tmp

            si = 0.0; ci = 0.0
            for b in range(nb_blocks):
                y = part_i[b] - ci
                tmp = si + y
                ci = (tmp - si) - y
                si = tmp

            out_r[k] = sr
            out_i[k] = si

        return out_r + 1j*out_i

    @staticmethod
    def integrate_over_mesh(
        data: FieldData, *,
        other=None,
        hermitian=False,
        real_only=False,
        chunk_size=None
    ) -> complex | np.ndarray:
        mesh  = data.mesh
        elems = np.asarray(mesh.elements, dtype=np.intp)
        w = np.asarray(mesh.tri_weights, dtype=np.float64)
        Nv = mesh.vertices.shape[0]

        def _to_NV_K(arr):
            arr = np.asarray(arr)
            if arr.ndim == 1:
                out = arr.reshape(Nv, 1)
            elif arr.ndim == 2:
                out = arr if arr.shape[0] == Nv else arr.T
            else:
                Logger.error(f"field has invalid shape {arr.shape}")
                raise
            return out.astype(np.complex128, copy=False)

        A = _to_NV_K(data.field)

        if other is None:
            kernel = WannierTools._integrate_batch_numba
            if not hasattr(kernel, "_warmed"):
                _ = kernel(A[:, :1], elems, w)
                kernel._warmed = True
        else:
            B = _to_NV_K(other.field)
            if A.shape[1] != B.shape[1]:
                Logger.error("The number of columns in A and B must be the same")
                raise
            kernel = WannierTools._integrate_prod_numba
            if not hasattr(kernel, "_warmed"):
                _ = kernel(A[:, :1], B[:, :1], elems, w, hermitian)
                kernel._warmed = True

        K = A.shape[1]
        if chunk_size is not None and K > chunk_size:
            out = np.empty(K, dtype=np.complex128)
            s = 0
            while s < K:
                e = min(s + chunk_size, K)
                if other is None:
                    out[s:e] = WannierTools._integrate_batch_numba(A[:, s:e], elems, w)
                else:
                    out[s:e] = WannierTools._integrate_prod_numba(A[:, s:e], B[:, s:e], elems, w, hermitian)
                s = e
        else:
            if other is None:
                out = WannierTools._integrate_batch_numba(A, elems, w)
            else:
                out = WannierTools._integrate_prod_numba(A, B, elems, w, hermitian)
        
        if real_only:
            out = out.real

        return out[0] if (A.shape[1] == 1) else out
