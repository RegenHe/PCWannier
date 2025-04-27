import os

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager

from .Log import Logger
from .Timer import Timer, timer
from .IO import IO

from .GlobalData import global_data
from .CallableWrapper import CallableWrapper

from .Utils import FieldData, StateCollection, WannierTools

class StateInitializer:
    def __init__(self):
        self.matC = None
        self.matV = None
        self.matZ = None
        self.last_matZ = None
        self.matA = None

        self.lambda_ = None
        self.alpha = 0.5

    @timer("State Initialize iter - ")
    def iter(self, err_diff: float, max_iter: int):
        Logger.info('Starting state initialization iteration')
        if 'V' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - V")
            self.matV = IO.load_cell_matrix(global_data.incar.V_file, shape=(len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])))
        else:
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
                Logger.info(f"initializer iter: n = {i},\t omega = {np.abs(omega)},\t err_diff = {abs(omega - last_omega_I)}")
            else:
                Logger.info(f"initializer iter: n = {i},\t omega = {np.abs(omega)}")
            if abs(omega - last_omega_I) < err_diff:
                Logger.info(f"Convergence criterion met, err_diff = {abs(omega - last_omega_I)}, total iterations: {i + 1}")
                break
            last_omega_I = omega
        if abs(omega - last_omega_I) > err_diff:
            Logger.warning(f"Convergence criteria not met, iteration limit reached, err_diff = {abs(omega - last_omega_I)}, total iterations: {i + 1}")
        
        if self.matA is None:
            self.projection()
        for i in range(len(global_data.incar.k_points[0])):
                for j in range(len(global_data.incar.k_points[1])):
                    t_ = np.conj(self.matV[i][j]).T @ self.matA[i][j]
                    mU, mS, mVh = np.linalg.svd(t_)
                    self.matV[i][j] = mU @ np.eye(global_data.incar.band_calc_num, global_data.incar.band_calc_num) @ mVh

        global_data.incar.m_set.initial(self.matV)
        Logger.info('State initialization iteration compeleted')


    @timer("State Projection iter - ")
    def projection(self):
        Logger.info('Starting to initialize projection')
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.band_window), global_data.incar.band_calc_num]
        self.matC = [[np.zeros((shape[2], shape[3]), dtype=complex) for _ in range(shape[1])]for _ in range(shape[0])]
        self.matZ = [[np.zeros((shape[2], shape[2]), dtype=complex) for _ in range(shape[1])]for _ in range(shape[0])]
        self.last_matZ = [[None for _ in range(shape[1])]for _ in range(shape[0])]
        self.lambda_ = [[None for _ in range(shape[1])]for _ in range(shape[0])]

        self.matA = [[np.zeros((shape[2], shape[3]), dtype=complex) for _ in range(shape[1])]for _ in range(shape[0])]
        # matS = [[None for _ in range(shape[1])]for _ in range(shape[0])]

        g = []
        for p in global_data.incar.projections:
            for state in p['states']:
                f = lambda r, phi: StateBases.Radial(state[0], state[1])(r, state[2]) * StateBases.Angular(state[1])(phi)
                h = global_data.state_collection.extention_mesh.rfunc(f, p['position'], p['xaxis_angluar'])
                # f = StateBases.temp(state[1], 0)
                # h = global_data.state_collection.extention_mesh.rfunc(f, p['position'], p['xaxis_angluar'])
                g.append(h / np.sqrt(WannierTools.integrate_over_mesh(FieldData('', global_data.state_collection.extention_mesh, global_data.state_collection.extention_epsilon * np.abs(h) ** 2))))
                # fd = FieldData('', global_data.state_collection.extention_mesh, g[-1])
                # fd.plot()

        futures = []
        result_queue = Manager().Queue()

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
            self.matA[i][j][m, n] = result
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                mU, mS, mVh = np.linalg.svd(self.matA[i][j])
                self.matC[i][j] = mU @ np.eye(shape[2], shape[3]) @ mVh
        Logger.info('Projection compeleted')

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
    def Angular(l: int=0):
        if l == 0:
            return StateBases.s
        elif l == 1:
            return StateBases.px
        elif l == -1:
            return StateBases.py
        elif l == 2:
            return StateBases.dx2_y2
        elif l == -2:
            return StateBases.dxy
        elif l == 3:
            return StateBases.fxx2_3y2
        elif l == -3:
            return StateBases.fy3x2_y2
        else:
            raise ValueError(f"Invalid Angular state: '{l}'")
    
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
        name = f"r{n}{abs(l)}"
        try:
            fn = getattr(StateBases, name)
        except AttributeError:
            err_msg = f"No radial function defined for (n={n}, l={l})"
            Logger.error(err_msg)
            raise ValueError(err_msg)
        return fn
        
    @staticmethod
    def r10(r: float, alpha: float=1.0):
        r = r / global_data.incar.lattice_const
        return 2 * alpha ** (3 / 2) * np.exp(-alpha * r / 2)
    
    @staticmethod
    def r20(r: float, alpha: float=1.0):
        r = r / global_data.incar.lattice_const
        return 1 / np.sqrt(2) * alpha ** (3 / 2) * (1 - 0.5 * alpha * r) * np.exp(-alpha * r / 2)
    @staticmethod
    def r21(r: float, alpha: float=1.0):
        r = r / global_data.incar.lattice_const
        return 1 / (2 * np.sqrt(6)) * alpha ** (3 / 2) * alpha * r * np.exp(-alpha * r / 2)
    
    @staticmethod
    def r30(r: float, alpha: float=1.0):
        r = r / global_data.incar.lattice_const
        return np.sqrt(4 / 27) * alpha ** (3 / 2) * (1 - 2 * alpha * r / 3 + 2 * alpha ** 2 * r ** 2 / 27) * np.exp(-alpha * r / 3)
    @staticmethod
    def r31(r: float, alpha: float=1.0):
        r = r / global_data.incar.lattice_const
        return 8 / (27 * np.sqrt(6)) * alpha ** (3 / 2) * (1 - alpha * r / 6) * alpha * r * np.exp(-alpha * r / 3)
    @staticmethod
    def r32(r: float, alpha: float=1.0):
        r = r / global_data.incar.lattice_const
        return 4 / (81 * np.sqrt(30)) * alpha ** (3 / 2) * alpha ** 2 * r ** 2 * np.exp(-alpha * r / 3)
    
    @staticmethod
    def r40(r: float, alpha: float=1.0):
        r = r / global_data.incar.lattice_const
        return 1 / 4 * alpha ** (3 / 2) * (1 - 3 / 4 * alpha * r + 1 / 8 * alpha ** 2 * r ** 2 - 1 / 192 * alpha ** 3 * r ** 3) * np.exp(-alpha * r / 4)
    @staticmethod
    def r41(r: float, alpha: float=1.0):
        r = r / global_data.incar.lattice_const
        return 5 / (16 * np.sqrt(15)) * alpha ** (3 / 2) * (1 - 1 / 4 * alpha * r + 1 / 80 * alpha ** 2 * r ** 2) * alpha * r * np.exp(-alpha * r / 4)
    @staticmethod
    def r42(r: float, alpha: float=1.0):
        r = r / global_data.incar.lattice_const
        return 1 / (64 * np.sqrt(5)) * alpha ** (3 / 2) * (1 - 1 / 12 * alpha * r) * alpha ** 2 * r ** 2 * np.exp(-alpha * r / 4)
    @staticmethod
    def r43(r: float, alpha: float=1.0):
        r = r / global_data.incar.lattice_const
        return 1 / (768 * np.sqrt(35)) * alpha ** (3 / 2) * alpha ** 3 * r ** 3 * np.exp(-alpha * r / 4)
    
    @staticmethod
    def plot_all(filename: str='./base/'):
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        max_n = 4

        x = np.linspace(-40 * global_data.incar.lattice_const, 40 * global_data.incar.lattice_const, 400)
        y = np.linspace(-40 * global_data.incar.lattice_const, 40 * global_data.incar.lattice_const, 400)
        X, Y = np.meshgrid(x, y, indexing='xy')
        for n in range(max_n + 1):
            for l in range(n):
                if l == 0:
                    r = np.sqrt(np.power(X, 2) + np.power(Y, 2))
                    phi = np.atan2(X, Y)
                    f = lambda r, phi: StateBases.Radial(n, l)(r) * StateBases.Angular(l)(phi)
                    Z = f(r, phi)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    cs = plt.contourf(X, Y, Z, 256, cmap="bwr")
                    plt.clim(-np.max(np.abs(Z)), np.max(np.abs(Z)))
                    plt.axis('equal')
                    plt.title(f'({n}, {l})')
                    plt.tight_layout()
                    plt.savefig(f"{filename}/base-({n}-{l})", dpi=300, bbox_inches='tight')
                else:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    r = np.sqrt(np.power(X, 2) + np.power(Y, 2))
                    phi = np.atan2(X, Y)
                    f = lambda r, phi: StateBases.Radial(n, l)(r) * StateBases.Angular(l)(phi)
                    Z = f(r, phi)
                    plt.subplot(1, 2, 1)
                    plt.contourf(X, Y, Z, 256, cmap="bwr")
                    plt.clim(-np.max(np.abs(Z)), np.max(np.abs(Z)))
                    plt.axis('equal')
                    plt.title(f'({n}, {l})')
                    plt.tight_layout()

                    f = lambda r, phi: StateBases.Radial(n, -l)(r) * StateBases.Angular(-l)(phi)
                    Z = f(r, phi)
                    plt.subplot(1, 2, 2)
                    plt.contourf(X, Y, Z, 256, cmap="bwr")
                    plt.clim(-np.max(np.abs(Z)), np.max(np.abs(Z)))
                    plt.axis('equal')
                    plt.title(f'({n}, {-l})')
                    plt.tight_layout()
                    plt.savefig(f"{filename}/base-({n}-{l})", dpi=300, bbox_inches='tight')