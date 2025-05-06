import numpy as np

from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager

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

        futures = []
        result_queue = Manager().Queue()

        def process_batch(i, j, m_range, n_range, b_range, result_queue):
            try:
                for m in m_range:
                    for n in n_range:
                        for b in b_range:
                            l_psi = global_data.state_collection.field[i][j][m]
                            n_k1_idx, n_k2_idx, k_ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                            if k_ is not None:
                                phase1 = global_data.state_collection.get_phase(n_k1_idx, n_k2_idx)
                                phase2 = global_data.state_collection.get_phase(k_[0], k_[1])
                                r_psi = global_data.state_collection.field[n_k1_idx][n_k2_idx][n] * phase1 * np.conj(phase2)
                            else:
                                r_psi = global_data.state_collection.field[n_k1_idx][n_k2_idx][n]
                            fd = FieldData("M0", state_collection.mesh, np.conj(l_psi) * state_collection.epsilon * r_psi)
                            result_queue.put((i, j, m, n, b, WannierTools.integrate_over_mesh(fd)))
            except Exception as e:
                    raise e

        with ProcessPoolExecutor(max_workers=global_data.threads) as executor:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    futures.append(
                            executor.submit(
                                CallableWrapper(process_batch), i, j, range(shape[3]), range(shape[3]), range(shape[2]), result_queue
                                ))
            wait(futures, return_when='ALL_COMPLETED')
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    raise e
        while not result_queue.empty():
            i, j, m, n, b, result = result_queue.get()
            self.mM0[i, j, b][m, n] = result
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

