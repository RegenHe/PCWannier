import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager

from PCWannier.Timer import Timer, timer
from PCWannier.IO import IO

from .GlobalData import global_data
from .CallableWrapper import CallableWrapper

from .Utils import FieldData, StateCollection, WannierTools

class StateInitializer:
    def __init__(self):
        self.matC = None
        self.matV = None
        self.matZ = None
        self.last_matZ = None

        self.lambda_ = None
        self.alpha = 0.5

    def iter(self, err_diff: float, max_iter: int):
        self.projection()
        self.matV = self.matC

        if len(global_data.incar.band_window) == global_data.incar.band_calc_num:
            global_data.m_set.initial(self.matV)
            return
        
        last_omega_I = 1e6
        for i in range(max_iter):
            self.update_Z()
            self.sort_Z()
            omega = self.get_omega_I()
            if i != 0:
                print(f"initializer iter: n = {i},\t omega = {np.abs(omega)},\t err_diff = {abs(omega - last_omega_I)}")
            else:
                print(f"initializer iter: n = {i},\t omega = {np.abs(omega)}")
            if abs(omega - last_omega_I) < err_diff:
                print(f"Convergence criterion met, err_diff = {abs(omega - last_omega_I)}, total iterations: {i + 1}")
                break
            last_omega_I = omega
        global_data.incar.m_set.initial(self.matV)


    @timer
    def projection(self):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.band_window), global_data.incar.band_calc_num]
        self.matC = [[np.zeros((shape[2], shape[3]), dtype=complex) for _ in range(shape[1])]for _ in range(shape[0])]
        self.matZ = [[np.zeros((shape[2], shape[2]), dtype=complex) for _ in range(shape[1])]for _ in range(shape[0])]
        self.last_matZ = [[None for _ in range(shape[1])]for _ in range(shape[0])]
        self.lambda_ = [[None for _ in range(shape[1])]for _ in range(shape[0])]

        matA = [[np.zeros((shape[2], shape[3]), dtype=complex) for _ in range(shape[1])]for _ in range(shape[0])]
        matS = [[None for _ in range(shape[1])]for _ in range(shape[0])]

        g = []
        for p in global_data.incar.projections:
            for state in p['states']:
                f = lambda r, phi: StateBases.Radial(state[0])(r, state[2]) * StateBases.Angular(state[1])(phi)
                h = global_data.state_collection.extention_mesh.rfunc(f, p['position'], p['xaxis_angluar'])
                # f = StateBases.temp(state[1], 0)
                # h = global_data.state_collection.extention_mesh.rfunc(f, p['position'], p['xaxis_angluar'])
                g.append(h / np.sqrt(WannierTools.integrate_over_mesh(FieldData('', global_data.state_collection.extention_mesh, global_data.state_collection.extention_epsilon * np.abs(h) ** 2))))
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
                    int_field = global_data.state_collection.extention_epsilon * np.conj(phase * field) * g[n]
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
                mU, mS, mVh = np.linalg.svd(matA[i][j])
                self.matC[i][j] = mU @ np.eye(shape[2], shape[3]) @ mVh
        print('projection compeleted')

    def update_Z(self):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), int(len(global_data.incar.composition_of_b)), len(global_data.incar.band_window), global_data.incar.band_calc_num]
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.matZ[i][j] = np.zeros((shape[3], shape[3]), dtype=complex)
                for b in range(shape[2]):
                    mM = global_data.m_set.get_M0(i, j, b)
                    n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                    self.matZ[i][j] += global_data.incar.wb[b] * mM @ (self.matV[n_k1][n_k2] @ np.conj(self.matV[n_k1][n_k2]).T) @ np.conj(mM).T
                if self.last_matZ[i][j] is None:
                    self.last_matZ[i][j] = self.matZ[i][j]
                else:
                    self.matZ[i][j] = self.alpha * self.matZ[i][j] + (1 - self.alpha) * self.last_matZ[i][j]
                    self.last_matZ[i][j] = self.matZ[i][j]
    
    def sort_Z(self):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.band_window), global_data.incar.band_calc_num]
        for i in range(shape[0]):
            for j in range(shape[1]):
                D, V = np.linalg.eig(self.matZ[i][j])
                sort_D = np.sort(D)[::-1]
                idx = np.argsort(D)[::-1]

                self.lambda_[i][j] = sort_D[:shape[3]]
                sort_V = V[:, idx]
                self.matV[i][j] = sort_V[:, :shape[3]]
    
    def get_omega_I(self):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.band_window), global_data.incar.band_calc_num]
        res = 0
        s_N_wb = shape[3] * np.sum(global_data.incar.wb)

        for i in range(shape[0]):
            for j in range(shape[1]):
                res += s_N_wb - np.sum(self.lambda_[i][j])
        
        return res / (shape[0] * shape[1])
    
    def save_as(self, filename):
        IO.save_to_txt(filename, self.matV, (len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])))

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
    
    # just for test
    def temp(name, n):
        if name == 's':
            return StateBases.s1
        elif name == 'px':
            return StateBases.p1
        elif name == 'py':
            return StateBases.p2
        elif name == 'dx2-y2':
            return StateBases.d1
        elif name == 'dxy':
            return StateBases.d2
    @staticmethod
    def s1(r, phi):
        r = 10 * r
        return np.exp(-r ** 2 / 20)
    @staticmethod
    def p1(r, phi):
        r = 10 * r
        return np.real(np.exp(-r ** 2 / 40) * np.exp(1j * phi) * r)
    @staticmethod
    def p2(r, phi):
        r = 10 * r
        return np.real(np.exp(-r ** 2 / 40) * np.exp(1j * phi) * r * np.exp(1j * np.pi * 0.5))
    @staticmethod
    def d1(r, phi):
        r = 10 * r
        return np.real(np.exp(-r ** 2 / 40) * np.exp(1j * 2 * phi) * r)
    @staticmethod
    def d2(r, phi):
        r = 10 * r
        return np.real(np.exp(-r ** 2 / 40) * np.exp(1j * 2 * phi) * r * np.exp(1j * np.pi * 0.5))