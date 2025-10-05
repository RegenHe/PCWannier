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
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])

        E_idx = global_data.state_collection.E_idx
        B = global_data.incar.band_calc_num

        self.matC = [[np.zeros((len(E_idx[i][j]), B), dtype=complex) for j in range(k2_sz)] for i in range(k1_sz)]
        self.matZ = [[np.zeros((len(E_idx[i][j]), B), dtype=complex) for j in range(k2_sz)] for i in range(k1_sz)]
        self.last_matZ = [[None for _ in range(k2_sz)] for _ in range(k1_sz)]
        self.lambda_ = [[None for _ in range(k2_sz)]for _ in range(k1_sz)]

        self.matA = [[np.zeros((len(E_idx[i][j]), B), dtype=complex) for j in range(k2_sz)]for i in range(k1_sz)]
        # matS = [[None for _ in range(shape[1])]for _ in range(shape[0])]
        self.matV = None
        self.alpha = 0.5

    @timer("State Initialize iter - ")
    def iter(self, err_diff: float, max_iter: int):
        Logger.info('Starting state initialization iteration')
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])
        E_idx = global_data.state_collection.E_idx
        B = global_data.incar.band_calc_num

        if 'V' in global_data.incar.use_cached_data and 'A' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - A")
            self.matA = IO.load_cell_matrix(global_data.incar.A_file, shape=(k1_sz, k2_sz))
            Logger.info(f"using cache data - V")
            self.matV = IO.load_cell_matrix(global_data.incar.V_file, shape=(k1_sz, k2_sz))
        elif 'V' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - V")
            self.matV = IO.load_cell_matrix(global_data.incar.V_file, shape=(k1_sz, k2_sz))
        elif 'A' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - A")
            self.matA = IO.load_cell_matrix(global_data.incar.A_file, shape=(k1_sz, k2_sz))
            for i in range(k1_sz):
                for j in range(k2_sz):
                    mU, mS, mVh = np.linalg.svd(self.matA[i][j])
                    self.matC[i][j] = mU @ np.eye(len(E_idx[i][j]), B) @ mVh
            self.matV = self.matC
        else:
            self.projection()
            self.matV = self.matC

        if global_data.incar.proj_iter is None:
            global_data.incar.proj_iter = True
        max_len = max(len(sel) for row in E_idx for sel in row)
        min_len = min(len(sel) for row in E_idx for sel in row)

        if min_len == global_data.incar.band_calc_num == max_len or (not global_data.incar.proj_iter):
            global_data.m_set.initial(self.matV)
            return
        
        last_omega_I = +np.inf
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
        if (omega - last_omega_I) > err_diff:
            Logger.warning(f"Convergence criteria not met, err_diff = {abs(omega - last_omega_I)}, total iterations: {i + 1}")
        
        if self.matA is None:
            self.projection()
        for i in range(k1_sz):
                for j in range(k2_sz):
                    t_ = np.conj(self.matV[i][j]).T @ self.matA[i][j]
                    mU, mS, mVh = np.linalg.svd(t_)
                    self.matV[i][j] = self.matV[i][j] @ (mU @ np.eye(B, B) @ mVh)

        global_data.m_set.initial(self.matV)
        Logger.info('State initialization iteration compeleted')


    @timer("State Projection iter - ")
    def projection(self):
        Logger.info('Starting to initialize projection')
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])

        B = global_data.incar.band_calc_num
        E_idx = global_data.state_collection.E_idx
        min_len = min(len(sel) for row in E_idx for sel in row)
        if B > min_len:
            err_msg = f"The number of calculated bands cannot exceed the band window, {B} > {min_len}"
            Logger.error(err_msg)
            raise

        a1 = np.array(global_data.incar.real_lattice_vectors[0], dtype=float) * global_data.incar.lattice_const
        a2 = np.array(global_data.incar.real_lattice_vectors[1], dtype=float) * global_data.incar.lattice_const
        ns = np.mgrid[-(k1_sz-1)//2:k1_sz//2, -(k2_sz-1)//2:k2_sz//2].reshape(2, -1).T
        T_list = [n1 * a1 + n2 * a2 for (n1, n2) in ns]

        H_list = []
        for p in global_data.incar.projections:
            for state in p['states']:
                if isinstance(state, dict) and 'lc_states' in state:
                    lc_states = state['lc_states']
                    lc_coeffs = state['lc_coeffs']

                    def f(r, phi, _lc_states=lc_states, _lc_coeffs=lc_coeffs):
                        s = 0.0 + 0.0j
                        for (n, l, z), c in zip(_lc_states, _lc_coeffs):
                            s += c * StateBases.Radial(n, l)(r, z) * StateBases.Angular(l)(phi)
                        return s
                else:
                    n, l, z = state
                    def f(r, phi, _n=n, _l=l, _z=z):
                        return StateBases.Radial(_n, _l)(r, _z) * StateBases.Angular(_l)(phi)

                cart_position = (p['frac_position'][0] * np.array(global_data.incar.real_lattice_vectors[0]) +
                                p['frac_position'][1] * np.array(global_data.incar.real_lattice_vectors[1]) +
                                np.array(global_data.incar.origin)) * global_data.incar.lattice_const
                
                col = np.zeros(len(global_data.state_collection.extention_mesh.vertices), dtype=np.complex128)
                for T in T_list:
                    col += global_data.state_collection.extention_mesh.rfunc(f, T + cart_position, p['xaxis_angluar'])

                H_list.append(np.asarray(col, dtype=np.complex128, copy=False))

        H = np.column_stack(H_list)

        eps = global_data.state_collection.extention_epsilon
        abs2 = np.abs(H)**2
        if np.isscalar(eps):
            F = (abs2 * float(eps)).astype(np.complex128, copy=False)
        else:
            eps_arr = np.asarray(eps, dtype=np.complex128)
            F = (abs2 * eps_arr[:, None]).astype(np.complex128, copy=False)

        fd = FieldData('', global_data.state_collection.extention_mesh, F)
        norms = WannierTools.integrate_over_mesh(fd, chunk_size=2048)
        norms = np.where(norms == 0, 1.0, norms)

        G = H / np.sqrt(norms)[None, :]

        g = [G[:, k] for k in range(G.shape[1])]

        for i in range(k1_sz):
            for j in range(k2_sz):
                Nv = global_data.state_collection.extention_mesh.vertices.shape[0]

                G = np.column_stack(g).astype(np.complex128, copy=False)
                if G.shape[0] != Nv:
                    G = G.T

                for m in range(len(E_idx[i][j])):
                    field = global_data.state_collection.get_extention_field(i, j, m)
                    
                    base = global_data.state_collection.extention_epsilon * np.conj(field)

                    F = (base[:, None] * G)
                    fd = FieldData('', global_data.state_collection.extention_mesh, F)
                    vals = WannierTools.integrate_over_mesh(fd, chunk_size=2048)
                    # base = global_data.state_collection.extention_epsilon * phase * field
                    # A = FieldData("A", global_data.state_collection.extention_mesh, np.broadcast_to(np.conj(base[None, :]), (shape[3], Nv)).astype(np.complex128, copy=False))
                    # B = FieldData("B", global_data.state_collection.extention_mesh, (G.T).astype(np.complex128, copy=False))
                    # vals = WannierTools.integrate_over_mesh(A, other=B, chunk_size=2048)
                    self.matA[i][j][m, :vals.shape[0]] = vals

                if global_data.incar.proj_binarize:
                    self.matA[i][j] = self.binarize(self.matA[i][j])
        if global_data.incar.proj_binarize:
            Logger.info("Projection binarization completed")
                
        for i in range(k1_sz):
            for j in range(k2_sz):
                mU, mS, mVh = np.linalg.svd(self.matA[i][j])
                self.matC[i][j] = mU @ np.eye(len(E_idx[i][j]), B) @ mVh
        Logger.info('Projection compeleted')

    def update_Z(self):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])

        E_idx = global_data.state_collection.E_idx
        b_sz = len(global_data.incar.composition_of_b)
        for i in range(k1_sz):
            for j in range(k2_sz):
                self.matZ[i][j] = np.zeros((len(E_idx[i][j]), len(E_idx[i][j])), dtype=complex)
                for b in range(b_sz):
                    mM = global_data.m_set.get_M0(i, j, b)
                    n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                    self.matZ[i][j] += global_data.incar.wb[b] * mM @ (self.matV[n_k1][n_k2] @ np.conj(self.matV[n_k1][n_k2]).T) @ np.conj(mM).T
                if self.last_matZ[i][j] is None:
                    self.last_matZ[i][j] = self.matZ[i][j]
                else:
                    self.matZ[i][j] = self.alpha * self.matZ[i][j] + (1 - self.alpha) * self.last_matZ[i][j]
                    self.last_matZ[i][j] = self.matZ[i][j]
    
    def sort_Z(self):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])

        B = global_data.incar.band_calc_num
        for i in range(k1_sz):
            for j in range(k2_sz):
                D, V = np.linalg.eig(self.matZ[i][j])
                sort_D = np.sort(D)[::-1]
                idx = np.argsort(D)[::-1]

                self.lambda_[i][j] = sort_D[:B]
                sort_V = V[:, idx]
                self.matV[i][j] = sort_V[:, :B]
    
    def get_omega_I(self):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])

        B = global_data.incar.band_calc_num
        res = 0
        s_N_wb = B * np.sum(global_data.incar.wb)

        for i in range(k1_sz):
            for j in range(k2_sz):
                res += s_N_wb - np.sum(self.lambda_[i][j])
        
        return res / (k1_sz * k2_sz)
    
    def save_as(self, filenameV: str, filenameA: str):
        if not filenameV.lower() == "false":
            IO.save_to_txt(filenameV, self.matV, (len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])))
        if not filenameA.lower() == "false":
            IO.save_to_txt(filenameA, self.matA, (len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])))
    
    @staticmethod
    def binarize(A):
        S = np.abs(np.asarray(A)).astype(np.float64, copy=False)
        n, m = S.shape
        S = np.where(np.isnan(S), -np.inf, S)

        order = np.argsort(S.ravel(), kind='mergesort')[::-1]
        rows, cols = np.unravel_index(order, (n, m))

        used_r = np.zeros(n, dtype=bool)
        used_c = np.zeros(m, dtype=bool)
        M = np.zeros((n, m), dtype=np.complex128)

        picked = 0
        for i, j in zip(rows, cols):
            if not used_r[i] and not used_c[j] and np.isfinite(S[i, j]):
                M[i, j] = 1.0
                used_r[i] = True
                used_c[j] = True
                picked += 1
                if picked == min(n, m) or used_r.all() or used_c.all():
                    break
        return M

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
        directory = os.path.dirname(filename + '/')
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        idx = 0
        for p in global_data.incar.projections:
            for state in p['states']:
                f = lambda r, phi: StateBases.Radial(state[0], state[1])(r, state[2]) * StateBases.Angular(state[1])(phi)
                cart_position = (p['frac_position'][0] * np.array(global_data.incar.real_lattice_vectors[0]) + p['frac_position'][1] * np.array(global_data.incar.real_lattice_vectors[1]) + np.array(global_data.incar.origin)) * global_data.incar.lattice_const
                h = global_data.state_collection.extention_mesh.rfunc(f, cart_position, p['xaxis_angluar'])
                field = FieldData('', global_data.state_collection.extention_mesh, h)
                field.save_fig(f"{os.path.dirname(filename)}/base-{idx}")
                idx += 1