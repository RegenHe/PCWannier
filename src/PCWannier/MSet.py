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

        B = global_data.incar.band_calc_num
        E_idx = global_data.state_collection.E_idx
        b_sz_half = len(global_data.incar.composition_of_b) // 2

        if global_data.incar.M_in or 'M' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - M")
            self.mM0 = IO.load_cell_matrix(global_data.incar.M_file, global_data.state_collection.k_shape + (b_sz_half, ))
            self.mMInitial = np.array(global_data.state_collection.gen_matrix_on_kmesh(lambda *_: [np.zeros((B, B), dtype=complex) for _ in range(b_sz_half)]))
            self.mM = np.array(global_data.state_collection.gen_matrix_on_kmesh(lambda *_: [np.zeros((B, B), dtype=complex) for _ in range(b_sz_half)]))
            global_data.state_collection.turn_to_Bloch()
            return
        global_data.state_collection.turn_to_Bloch()
        self.mM0 = global_data.state_collection.gen_matrix_on_kmesh(lambda *_: np.empty(b_sz_half, dtype=object))
        for i, j, k in global_data.state_collection.k_indices():
            for b in range(b_sz_half):
                ik, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j, k], b)
                self.mM0[i, j, k][b] = np.zeros((len(E_idx[i][j][k]), len(E_idx[ik[0]][ik[1]][ik[2]])), dtype=complex)
        
        for i, j, k in global_data.state_collection.k_indices():
            Nv = state_collection.mesh.vertices.shape[0]

            left_arr = np.asarray(global_data.state_collection.get_block(i, j, k, left=True))
            if left_arr.ndim == 1:
                left_arr = left_arr[None, :]
            elif left_arr.shape[1] != Nv:
                left_arr = left_arr.T
            Nwin = left_arr.shape[0]

            for b in range(b_sz_half):
                ik, k_ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j, k], b)
                right_arr = np.asarray(global_data.state_collection.get_block(ik[0], ik[1], ik[2]))
                if right_arr.ndim == 1:
                    right_arr = right_arr[None, :]
                elif right_arr.shape[1] != Nv:
                    right_arr = right_arr.T

                if k_ is not None:
                    phase1 = global_data.state_collection.get_phase(ik[0], ik[1], ik[2])
                    phase2 = global_data.state_collection.get_phase(k_[0], k_[1], k_[2])
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
                    self.mM0[i, j, k][b][m, :v.size] = vals
        Logger.info("M0 initialized")

        self.mMInitial = global_data.state_collection.gen_matrix_on_kmesh(lambda *_: [np.zeros((B, B), dtype=complex) for _ in range(b_sz_half)])
        self.mM = global_data.state_collection.gen_matrix_on_kmesh(lambda *_: [np.zeros((B, B), dtype=complex) for _ in range(b_sz_half)])

        Logger.info('M Matrix initialization completed')
    
    def get_M0(self, i: int, j: int, k: int, b: int):
        if b < len(global_data.incar.composition_of_b) // 2:
            M = self.mM0[i, j, k][b]
        else:
            ik, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j, k], b)
            M = np.conj(self.mM0[ik[0], ik[1], ik[2]][b - len(global_data.incar.composition_of_b) // 2]).T
        
        T = global_data.state_collection.get_transform()
        T_k = T[i][j][k]

        ik, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j, k], b)
        T_k_b = T[ik[0]][ik[1]][ik[2]]
        
        M_transformed = T_k @ M @ T_k_b

        return M_transformed
    
    def get(self, i: int, j: int, k: int, b: int):
        if b < len(global_data.incar.composition_of_b) // 2:
            return self.mM[i, j, k][b]
        else:
            ik, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j, k], b)
            return np.conj(self.mM[ik[0], ik[1], ik[2]][b - len(global_data.incar.composition_of_b) // 2]).T
        
    def initial(self, V):
        b_sz_half = len(global_data.incar.composition_of_b) // 2
        for i, j, k in global_data.state_collection.k_indices():
            for b in range(b_sz_half):
                ik, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j, k], b)
                self.mMInitial[i, j, k][b] = np.conj(V[i][j][k]).T @ self.get_M0(i, j, k, b) @ V[ik[0]][ik[1]][ik[2]]

    def update(self, U):
        b_sz_half = len(global_data.incar.composition_of_b) // 2
        for i, j, k in global_data.state_collection.k_indices():
            for b in range(b_sz_half):
                ik, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j, k], b)
                self.mM[i, j, k][b] = np.conj(U[i][j][k]).T @ self.mMInitial[i, j, k][b] @ U[ik[0]][ik[1]][ik[2]]

    def save_as(self, filename):
        IO.save_to_txt(filename, self.mM0, self.mM0.shape + (len(global_data.incar.composition_of_b) // 2, ))

