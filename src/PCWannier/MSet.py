import numpy as np

from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager

import copy

from PCWannier.Timer import Timer, timer

from .GlobalData import global_data
from .CallableWrapper import CallableWrapper
from .Utils import FieldData, StateCollection, WannierTools

class MSet:
    def __init__(self):
        self.M0 = None
        self.M = None

    @timer
    def init_M0(self, state_collection: StateCollection):
        global_data.state_collection.turn_to_Bloch()
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.composition_of_b) // 2, len(global_data.incar.band_window)]
        self.M0 = np.array([[[np.zeros((shape[3], shape[3]), dtype=complex) for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])

        futures = []
        result_queue = Manager().Queue()
        def process_batch(i, j, m_range, n_range, b_range, result_queue):
            for m in m_range:
                for n in n_range:
                    for b in b_range:
                        l_psi = global_data.state_collection.field[i][j][m]
                        n_k1_idx, n_k2_idx = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                        r_psi = global_data.state_collection.field[n_k1_idx][n_k2_idx][n]
                        fd = FieldData("M0", state_collection.mesh, np.conj(l_psi) * state_collection.epsilon * r_psi)
                        result_queue.put((i, j, m, n, b, WannierTools.integrate_over_mesh(fd)))

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
            self.M0[i, j, b][m, n] = result
        print("M0 initialized")

        self.M = copy.deepcopy(self.M0)
    
    def get_M0(self, i: int, j: int, b: int):
        if b < len(global_data.incar.composition_of_b) // 2:
            return self.M0[i, j, b]
        else:
            n_k1, n_k2 = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
            return np.conj(self.M0[n_k1, n_k2, b - len(global_data.incar.composition_of_b) // 2]).T
    
    def get(self, i: int, j: int, b: int):
        if b < len(global_data.incar.composition_of_b) // 2:
            return self.M[i, j, b]
        else:
            n_k1, n_k2 = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
            return np.conj(self.M[n_k1, n_k2, b - len(global_data.incar.composition_of_b) // 2]).T
        
    def update(self, U):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.composition_of_b) // 2]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for b in range(shape[2]):
                    n_k1, n_k2 = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                    self.M[i, j, b] = np.conj(U[i][j]).T @ self.M0[i, j, b] @ U[n_k1][n_k2]
