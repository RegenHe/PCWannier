import numpy as np

import copy

from .Log import Logger
from .Timer import Timer, timer
from .IO import IO

from .GlobalData import global_data
from .IO import IO
from .Utils import FieldData, StateCollection, WannierTools

class MSet:
    def __init__(self):
        self.mM0 = None
        self.mMInitial = None
        self.mM = None

    @timer("M matrix initialize - ")
    def init_M0(self, state_collection: StateCollection):
        Logger.info('Starting to M Matrix initialization')
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])

        B = global_data.incar.band_calc_num
        E_idx = global_data.state_collection.E_idx
        b_sz_half = len(global_data.incar.composition_of_b) // 2

        if global_data.incar.M_in or 'M' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - M")
            self.mM0 = IO.load_cell_matrix(global_data.incar.M_file, shape=(k1_sz, k2_sz, b_sz_half))
            self.mMInitial = np.array([[[np.zeros((B, B), dtype=complex) for _ in range(b_sz_half)] for _ in range(k2_sz)] for _ in range(k1_sz)])
            self.mM = np.array([[[np.zeros((B, B), dtype=complex) for _ in range(b_sz_half)] for _ in range(k2_sz)] for _ in range(k1_sz)])
            global_data.state_collection.turn_to_Bloch()
            return
        global_data.state_collection.turn_to_Bloch()
        self.mM0 = np.empty((k1_sz, k2_sz, b_sz_half), dtype=object)
        for i in range(k1_sz):
            for j in range(k2_sz):
                for b in range(b_sz_half):
                    n_k1_idx, n_k2_idx, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                    self.mM0[i, j, b] = np.zeros((len(E_idx[i][j]), len(E_idx[n_k1_idx][n_k2_idx])), dtype=complex)
        
        for i in range(k1_sz):
            for j in range(k2_sz):
                Nv = state_collection.mesh.vertices.shape[0]

                left_arr = np.asarray(global_data.state_collection.field[i][j])
                if left_arr.ndim == 1:
                    left_arr = left_arr[None, :]
                elif left_arr.shape[1] != Nv:
                    left_arr = left_arr.T
                Nwin = left_arr.shape[0]

                for b in range(b_sz_half):
                    n_k1_idx, n_k2_idx, k_ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                    right_arr = np.asarray(global_data.state_collection.field[n_k1_idx][n_k2_idx])
                    if right_arr.ndim == 1:
                        right_arr = right_arr[None, :]
                    elif right_arr.shape[1] != Nv:
                        right_arr = right_arr.T

                    if k_ is not None:
                        phase1 = global_data.state_collection.get_phase(n_k1_idx, n_k2_idx)
                        phase2 = global_data.state_collection.get_phase(k_[0], k_[1])
                        right_arr = right_arr * (phase1 * np.conj(phase2))[None, :]

                    for m in range(Nwin):
                        base = np.conj(left_arr[m]) * state_collection.epsilon
                        F = (right_arr * base[None, :]).T.astype(np.complex128, copy=False)
                        fd = FieldData("M0", state_collection.mesh, F)
                        vals = WannierTools.integrate_over_mesh(fd, chunk_size=2048)
                        # A = FieldData("A", state_collection.mesh, np.broadcast_to(np.conj(left_arr[m][None, :]), (Nwin, Nv)).astype(np.complex128, copy=False))
                        # B = FieldData("B", state_collection.mesh, (right_arr.T * state_collection.epsilon[:, None]).astype(np.complex128, copy=False))
                        # vals = WannierTools.integrate_over_mesh(A, other=B, chunk_size=2048)
                        v = np.asarray(vals).ravel().astype(self.mM0.dtype, copy=False)
                        self.mM0[i, j, b][m, :v.size] = vals
        Logger.info("M0 initialized")

        self.mMInitial = np.array([[[np.zeros((B, B), dtype=complex) for _ in range(b_sz_half)] for _ in range(k2_sz)] for _ in range(k1_sz)])
        self.mM = np.array([[[np.zeros((B, B), dtype=complex) for _ in range(b_sz_half)] for _ in range(k2_sz)] for _ in range(k1_sz)])

        Logger.info('M Matrix initialization completed')
    
    def get_M0(self, i: int, j: int, b: int):
        if b < len(global_data.incar.composition_of_b) // 2:
            M = self.mM0[i, j, b]
        else:
            n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
            M = np.conj(self.mM0[n_k1, n_k2, b - len(global_data.incar.composition_of_b) // 2]).T
        
        T = global_data.state_collection.get_transform()
        T_k = T[i][j]

        n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
        T_k_b = T[n_k1][n_k2]
        
        M_transformed = T_k @ M @ T_k_b

        return M_transformed
    
    def get(self, i: int, j: int, b: int):
        if b < len(global_data.incar.composition_of_b) // 2:
            return self.mM[i, j, b]
        else:
            n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
            return np.conj(self.mM[n_k1, n_k2, b - len(global_data.incar.composition_of_b) // 2]).T
        
    def initial(self, V):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])
        b_sz_half = len(global_data.incar.composition_of_b) // 2
        for i in range(k1_sz):
            for j in range(k2_sz):
                for b in range(b_sz_half):
                    n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                    self.mMInitial[i, j, b] = np.conj(V[i][j]).T @ self.get_M0(i, j, b) @ V[n_k1][n_k2]

    def update(self, U):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])
        b_sz_half = len(global_data.incar.composition_of_b) // 2
        for i in range(k1_sz):
            for j in range(k2_sz):
                for b in range(b_sz_half):
                    n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                    self.mM[i, j, b] = np.conj(U[i][j]).T @ self.mMInitial[i, j, b] @ U[n_k1][n_k2]

    def save_as(self, filename):
        IO.save_to_txt(filename, self.mM0, (len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.composition_of_b) // 2))

