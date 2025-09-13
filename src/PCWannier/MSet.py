import numpy as np

from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager

from threadpoolctl import threadpool_limits

import copy

from .Log import Logger
from .Timer import Timer, timer
from .IO import IO

from .GlobalData import global_data
from .CallableWrapper import CallableWrapper
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

        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.composition_of_b) // 2, len(global_data.incar.band_window)]
        if global_data.incar.M_in or 'M' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - M")
            self.mM0 = IO.load_cell_matrix(global_data.incar.M_file, shape=(shape[0], shape[1], shape[2]))
            self.mMInitial = np.array([[[np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex) for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])
            self.mM = np.array([[[np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex) for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])
            global_data.state_collection.turn_to_Bloch()
            return
        global_data.state_collection.turn_to_Bloch()
        self.mM0 = np.array([[[np.zeros((shape[3], shape[3]), dtype=complex) for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                Nv = state_collection.mesh.vertices.shape[0]

                left_arr = np.asarray(global_data.state_collection.field[i][j])
                if left_arr.ndim == 1:
                    left_arr = left_arr[None, :]
                elif left_arr.shape[1] != Nv:
                    left_arr = left_arr.T
                Nm = left_arr.shape[0]

                for b in range(shape[2]):
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

                    for m in range(Nm):
                        base = np.conj(left_arr[m]) * state_collection.epsilon
                        F = (right_arr * base[None, :]).T.astype(np.complex128, copy=False)
                        fd = FieldData("M0", state_collection.mesh, F)
                        vals = WannierTools.integrate_over_mesh(fd, chunk_size=2048)
                        self.mM0[i, j, b][m, :vals.shape[0]] = vals
        Logger.info("M0 initialized")

        self.mMInitial = np.array([[[np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex) for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])
        self.mM = np.array([[[np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex) for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])

        Logger.info('M Matrix initialization completed')
    
    def get_M0(self, i: int, j: int, b: int):
        if b < len(global_data.incar.composition_of_b) // 2:
            return self.mM0[i, j, b]
        else:
            n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
            return np.conj(self.mM0[n_k1, n_k2, b - len(global_data.incar.composition_of_b) // 2]).T
    
    def get(self, i: int, j: int, b: int):
        if b < len(global_data.incar.composition_of_b) // 2:
            return self.mM[i, j, b]
        else:
            n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
            return np.conj(self.mM[n_k1, n_k2, b - len(global_data.incar.composition_of_b) // 2]).T
        
    def initial(self, V):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.composition_of_b) // 2]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for b in range(shape[2]):
                    n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                    self.mMInitial[i, j, b] = np.conj(V[i][j]).T @ self.mM0[i, j, b] @ V[n_k1][n_k2]

    def update(self, U):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.composition_of_b) // 2]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for b in range(shape[2]):
                    n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                    self.mM[i, j, b] = np.conj(U[i][j]).T @ self.mMInitial[i, j, b] @ U[n_k1][n_k2]

    def save_as(self, filename):
        IO.save_to_txt(filename, self.mM0, (len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.composition_of_b) // 2))

