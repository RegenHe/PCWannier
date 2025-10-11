import os

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

from .Log import Logger
from .Timer import Timer, timer
from .IO import IO

from .GlobalData import global_data

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
        self.diagII_sum = [[0.0 for _ in range(k2_sz)]for _ in range(k1_sz)]

        self.matA = [[np.zeros((len(E_idx[i][j]), B), dtype=complex) for j in range(k2_sz)]for i in range(k1_sz)]
        # matS = [[None for _ in range(shape[1])]for _ in range(shape[0])]
        self.matV = [[np.zeros((len(E_idx[i][j]), B), dtype=complex) for j in range(k2_sz)] for i in range(k1_sz)]
        self.alpha = 0.5

        self.I_idx = [[None for _ in range(k2_sz)]for _ in range(k1_sz)]
        self.O_idx = [[None for _ in range(k2_sz)]for _ in range(k1_sz)]

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

            E_idx = global_data.state_collection.E_idx
            inner_idx = global_data.state_collection.inner_E_idx
            for i in range(k1_sz):
                for j in range(k2_sz):
                    N_k = len(E_idx[i][j])
                    self.I_idx[i][j] = self.map_inner_to_local(E_idx[i][j], inner_idx[i][j])
                    self.O_idx[i][j] = np.setdiff1d(np.arange(N_k), self.I_idx[i][j], assume_unique=True)

        elif 'V' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - V")
            self.matV = IO.load_cell_matrix(global_data.incar.V_file, shape=(k1_sz, k2_sz))
            E_idx = global_data.state_collection.E_idx
            inner_idx = global_data.state_collection.inner_E_idx
            for i in range(k1_sz):
                for j in range(k2_sz):
                    N_k = len(E_idx[i][j])
                    self.I_idx[i][j] = self.map_inner_to_local(E_idx[i][j], inner_idx[i][j])
                    self.O_idx[i][j] = np.setdiff1d(np.arange(N_k), self.I_idx[i][j], assume_unique=True)

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
        if global_data.incar.v_proj:
            self.V_projection()

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
        if global_data.incar.inner_window is not False:
            Logger.info("Using inner window for projection")
            self.inner_projection()
        else:
            self.I_idx = [[np.empty((0,), dtype=np.intp) for j in range(k2_sz)] for i in range(k1_sz)]
            self.O_idx = [[np.setdiff1d(np.arange(len(E_idx[i][j])), self.I_idx[i][j], assume_unique=True) for j in range(k2_sz)] for i in range(k1_sz)]

        Logger.info('Projection compeleted')

    def inner_projection(self):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])

        E_idx = global_data.state_collection.E_idx
        N = global_data.incar.band_calc_num
        inner_idx = global_data.state_collection.inner_E_idx

        tol = 1e-10

        for i in range(k1_sz):
            for j in range(k2_sz):
                N_k = len(E_idx[i][j])
                self.I_idx[i][j] = self.map_inner_to_local(E_idx[i][j], inner_idx[i][j])
                self.O_idx[i][j] = np.setdiff1d(np.arange(N_k), self.I_idx[i][j], assume_unique=True)
                M_k = self.I_idx[i][j].size
                p = N - M_k
                if p == 0:
                    Logger
                    self.matV[i][j] = self.matC[i][j][:, :N]
                elif p < 0:
                    raise ValueError(f"(i={i}, j={j}) inner window M_k={M_k} > N={N}. Increase N or shrink inner window.")
                if p > self.O_idx[i][j].size:
                    raise ValueError(f"(i={i}, j={j}) outer window insufficient: N-M_k={p} > N_k-M_k={self.O_idx[i][j].size}.")
                
                # P_inner = np.zeros((N_k, N_k), dtype=np.complex128)
                # Q_inner = np.eye(N_k, dtype=np.complex128) - P_inner
                U, S, Vh = np.linalg.svd(self.matA[i][j], full_matrices=False)
                r = int(np.sum(S > tol))
                r = min(r, N)
                if r == 0:
                    P_G = np.zeros((N_k, N_k), dtype=np.complex128)
                else:
                    U_r = U[:, :r]
                    P_G = U_r @ U_r.conj().T

                # PG_II = 0.5*(P_G[np.ix_(I_idx, I_idx)] + P_G[np.ix_(I_idx, I_idx)].conj().T)
                # wI, WI = np.linalg.eigh(PG_II)
                # UI = np.zeros((N_k, M_k), dtype=complex)
                # UI[I_idx, :] = WI
                UI = np.eye(N_k, dtype=np.complex128)[:, self.I_idx[i][j]]

                P_G_OO = 0.5 * (P_G[np.ix_(self.O_idx[i][j], self.O_idx[i][j])] + P_G[np.ix_(self.O_idx[i][j], self.O_idx[i][j])].conj().T)
                wO, WO = np.linalg.eigh(P_G_OO)
                order = np.argsort(wO)[::-1]
                wO = wO[order]
                WO = WO[:, order]
                Vp = WO[:, :p]
                U_opt = np.zeros((N_k, p), dtype=np.complex128)
                U_opt[self.O_idx[i][j], :] = Vp
                
                U_init = np.concatenate([UI, U_opt], axis=1)
                self.matV[i][j] = U_init
        Logger.info('Inner projection compeleted')
    
    def V_projection(self):
        Logger.info('Starting V projection')
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])
        for i in range(k1_sz):
            for j in range(k2_sz):
                A = self.matV[i][j].conj().T @ self.matA[i][j]
                mU, mS, mVh = np.linalg.svd(A)
                self.matV[i][j] = self.matV[i][j] @ (mU @ np.eye(mU.shape[0], mVh.shape[0]) @ mVh)
        Logger.info('V projection compeleted')

    def update_Z(self):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])

        E_idx = global_data.state_collection.E_idx
        b_sz = len(global_data.incar.composition_of_b)
        for i in range(k1_sz):
            for j in range(k2_sz):
                self.matZ[i][j] = np.zeros((len(self.O_idx[i][j]), len(self.O_idx[i][j])), dtype=complex)
                diag_II = 0.0
                for b in range(b_sz):
                    mM = global_data.m_set.get_M0(i, j, b)
                    n_k1, n_k2, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j], b)
                    Cb = mM @ self.matV[n_k1][n_k2]
                    Cb_O = Cb[self.O_idx[i][j], :]
                    self.matZ[i][j] += global_data.incar.wb[b] * (Cb_O @ Cb_O.conj().T)

                    if self.I_idx[i][j].size > 0:
                        Ci = Cb[self.I_idx[i][j], :]
                        diag_II += global_data.incar.wb[b] * np.sum(np.abs(Ci)**2)
                self.diagII_sum[i][j] = diag_II

                self.matZ[i][j] = 0.5*(self.matZ[i][j] + self.matZ[i][j].conj().T)
                if self.last_matZ[i][j] is None:
                    self.last_matZ[i][j] = self.matZ[i][j]
                else:
                    self.matZ[i][j] = self.alpha * self.matZ[i][j] + (1 - self.alpha) * self.last_matZ[i][j]
                    self.last_matZ[i][j] = self.matZ[i][j]
    
    def sort_Z(self):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])

        B = global_data.incar.band_calc_num
        E_idx = global_data.state_collection.E_idx
        for i in range(k1_sz):
            for j in range(k2_sz):
                D, V = np.linalg.eigh(self.matZ[i][j])
                p = B - len(self.I_idx[i][j])
                Vp = V[:, -p:] if p > 0 else np.zeros((self.matZ[i][j].shape[0], 0), complex)

                self.lambda_[i][j] = float(np.sum(D[-p:])) if p>0 else 0.0
                N_k = len(E_idx[i][j])
                U_opt = np.zeros((N_k, p), dtype=complex)
                U_opt[ self.O_idx[i][j], :] = Vp
                E_I = np.eye(N_k, dtype=complex)[:, self.I_idx[i][j]]
                U_concat = np.concatenate([E_I, U_opt], axis=1)
                self.matV[i][j] = U_concat[:, :B]
    
    def get_omega_I(self):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])
        N = global_data.incar.band_calc_num
        s_N_wb = N * np.sum(global_data.incar.wb)
        total = 0.0
        for i in range(k1_sz):
            for j in range(k2_sz):
                total += s_N_wb - ( self.diagII_sum[i][j] + self.lambda_[i][j] )
        return total / (k1_sz * k2_sz)
        
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
    
    @staticmethod
    def map_inner_to_local(E_idx_ij, inner_idx_ij):
        E = np.asarray(E_idx_ij, dtype=int)
        inner = np.asarray(inner_idx_ij, dtype=int)

        E0 = E
        inner0 = inner

        pos = {band_id: pos for pos, band_id in enumerate(E0)}
        I = [pos[b] for b in inner0 if b in pos]
        I = np.unique(I) 
        return I

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