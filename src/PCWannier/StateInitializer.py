import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager

from PCWannier.Timer import Timer, timer
from .GlobalData import global_data
from .CallableWrapper import CallableWrapper

from .Utils import FieldData, StateCollection, WannierTools

class StateInitializer:
    def __init__(self):
        self.matC = None
        self.matV = None

    def iter(self, err_diff: float, max_iter: int):
        pass

    @timer
    def projection(self):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.band_window), len(global_data.incar.band_calc)]
        self.matC = [[np.zeros((shape[2], shape[3]), dtype=complex) for _ in range(shape[1])]for _ in range(shape[0])]

        matA = [[np.zeros((shape[2], shape[3]), dtype=complex) for _ in range(shape[1])]for _ in range(shape[0])]
        matS = [[None for _ in range(shape[1])]for _ in range(shape[0])]

        g = []
        for p in global_data.incar.projections:
            for state in p['states']:
                f = lambda r, phi: StateBases.Radial(state[0])(r, state[2]) * StateBases.Angular(state[1])(phi)
                g.append(global_data.state_collection.extention_mesh.rfunc(f, p['position'], p['xaxis_angluar']))
                # fd = FieldData('', global_data.state_collection.extention_mesh, g[-1])
                # fd.plot()

        futures = []
        result_queue = Manager().Queue()

        @timer
        def process_batch(i, j, m_range, n_range, g, result_queue):
            phase = global_data.state_collection.get_extention_phase(i, j)
            for m in m_range:
                field = global_data.state_collection.get_extention_field(i, j, m)
                for n in n_range:
                    int_field = np.array(global_data.state_collection.extention_epsilon) * np.conj(phase * field) * g[n]
                    result_queue.put((i, j, m, n, WannierTools.integrate_over_mesh(FieldData('', global_data.state_collection.extention_mesh, int_field))))

        with ProcessPoolExecutor(max_workers=global_data.threads) as executor:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    futures.append(
                                executor.submit(
                                    CallableWrapper(process_batch), i, j, range(shape[2]), range(shape[3]), g, result_queue
                                    ))
            wait(futures, return_when='ALL_COMPLETED')
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    raise e
        while not result_queue.empty():
            i, j, m, n, result = result_queue.get()
            matA[i][j][m, n] = result
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                U, S, V = np.linalg.svd(matA[i][j])
                self.matC[i][j] = U @ np.eye(shape[2], shape[3]) @ np.conj(V).T
        print('projection compeleted')

    def update_Z(self):
        pass

class StateBases:
    @staticmethod
    def Angular(name: str):
        if name == 's':
            return StateBases.s
        elif name == 'px':
            return StateBases.px
        elif name == 'py':
            return StateBases.py
        elif name == 'dx2-y2':
            return StateBases.dx2_y2
        elif name == 'dxy':
            return StateBases.dxy
        elif name == 'fx(x2-3y2)':
            return StateBases.fxx2_3y2
        elif name == 'fy(3x2-y2)':
            return StateBases.fy3x2_y2
        else:
            raise ValueError(f"Invalid Angular state: '{name}'")
    
    @staticmethod
    def s(phi: float):
        return 1/np.sqrt(4 * np.pi)
    @staticmethod
    def px(phi: float):
        return np.sqrt(3 / (4 * np.pi)) * np.cos(phi)
    @staticmethod
    def py(phi: float):
        return np.sqrt(3 / (4 * np.pi)) * np.sin(phi)
    @staticmethod
    def dx2_y2(phi: float):
        return np.sqrt(15 / (16 * np.pi)) * np.cos(2 * phi)
    @staticmethod
    def dxy(phi: float):
        return np.sqrt(15 / (16 * np.pi)) * np.sin(2 * phi)
    @staticmethod
    def fxx2_3y2(phi: float):
        return np.sqrt(35 / (32 * np.pi)) * (np.cos(phi) ** 2 - 3 * np.sin(phi) ** 2) * np.cos(phi)
    @staticmethod
    def fy3x2_y2(phi: float):
        return np.sqrt(35 / (32 * np.pi)) * (3 * np.cos(phi) ** 2 - np.sin(phi) ** 2) * np.sin(phi)
    

    @staticmethod
    def Radial(n: int, l: int=None):
        if l is None:
            l = n - 1
        radial_functions = {
            (1, 0): StateBases.r10,
            (2, 0): StateBases.r20,
            (2, 1): StateBases.r21,
            (3, 0): StateBases.r30,
            (3, 1): StateBases.r31,
            (3, 2): StateBases.r32,
        }
        try:
            return radial_functions[(n, l)]
        except KeyError:
            raise ValueError(f"Invalid (n, l) combination: ({n}, {l})")

    @staticmethod
    def r10(r: float, alpha: float=1.0):
        return 2 * alpha ** (3 / 2) * np.exp(-alpha * r)
    
    @staticmethod
    def r20(r: float, alpha: float=1.0):
        return 1 / np.sqrt(2) * alpha ** (3 / 2) * (1 - 0.5 * alpha * r) * np.exp(-alpha * r / 2)
    @staticmethod
    def r21(r: float, alpha: float=1.0):
        return 1 / (2 * np.sqrt(6)) * alpha ** (3 / 2) * alpha * r * np.exp(-alpha * r / 2)
    
    @staticmethod
    def r30(r: float, alpha: float=1.0):
        return np.sqrt(4 / 27) * alpha ** (3 / 2) * (1 - 2 * alpha * r / 3 + 2 * alpha ** 2 * r ** 2 / 27) * np.exp(-alpha * r / 3)
    @staticmethod
    def r31(r: float, alpha: float=1.0):
        return 8 / (27 * np.sqrt(6)) * alpha ** (3 / 2) * (1 - alpha * r / 6) * alpha * r * np.exp(-alpha * r / 3)
    @staticmethod
    def r32(r: float, alpha: float=1.0):
        return 4 / (81 * np.sqrt(30)) * alpha ** (3 / 2) * alpha ** 2 * r ** 2 * np.exp(-alpha * r / 3)