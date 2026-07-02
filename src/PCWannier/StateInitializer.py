import os

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Callable

from .Log import Logger
from .Timer import Timer, timer
from .IO import IO

from .GlobalData import global_data

from .Utils import FieldData, StateCollection, WannierTools

class StateInitializer:
    def __init__(self):
        E_idx = global_data.state_collection.E_idx
        B = global_data.incar.band_calc_num

        factory: Callable[[int, int, int], np.typing.NDArray[np.complex128]] = lambda i, j, k: np.zeros((len(E_idx[i][j][k]), B), dtype=np.complex128)

        self.matC = global_data.state_collection.gen_matrix_on_kmesh(factory)
        self.matZ = global_data.state_collection.gen_matrix_on_kmesh(factory)

        self.last_matZ = global_data.state_collection.gen_matrix_on_kmesh(lambda *_: None)
        self.lambda_ = global_data.state_collection.gen_matrix_on_kmesh(lambda *_: None)

        self.diagII_sum = global_data.state_collection.gen_matrix_on_kmesh(lambda *_: 0.0)

        self.matA = global_data.state_collection.gen_matrix_on_kmesh(factory)
        self.matV = global_data.state_collection.gen_matrix_on_kmesh(factory)

        self.I_idx = global_data.state_collection.gen_matrix_on_kmesh(lambda *_: None)
        self.O_idx = global_data.state_collection.gen_matrix_on_kmesh(lambda *_: None)

        self.alpha = 0.5


    @timer("State Initialize iter - ")
    def iter(self, err_diff: float, max_iter: int):
        Logger.info('Starting state initialization iteration')
        E_idx = global_data.state_collection.E_idx
        B = global_data.incar.band_calc_num

        inner_idx = global_data.state_collection.inner_E_idx
        if len(inner_idx) == 0:
            inner_idx = global_data.state_collection.gen_matrix_on_kmesh(lambda *_: [])

        if 'V' in global_data.incar.use_cached_data and 'A' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - A")
            self.matA = IO.load_cell_matrix(global_data.incar.A_file, global_data.state_collection.k_shape)
            Logger.info(f"using cache data - V")
            self.matV = IO.load_cell_matrix(global_data.incar.V_file, global_data.state_collection.k_shape)

            for i, j, k in global_data.state_collection.k_indices():
                N_k = len(E_idx[i][j][k])
                self.I_idx[i][j][k] = self.map_inner_to_local(E_idx[i][j][k], inner_idx[i][j][k])
                self.O_idx[i][j][k] = np.setdiff1d(np.arange(N_k), self.I_idx[i][j][k], assume_unique=True)

        elif 'V' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - V")
            self.matV = IO.load_cell_matrix(global_data.incar.V_file, global_data.state_collection.k_shape)
            for i, j, k in global_data.state_collection.k_indices():
                N_k = len(E_idx[i][j][k])
                self.I_idx[i][j][k] = self.map_inner_to_local(E_idx[i][j][k], inner_idx[i][j][k])
                self.O_idx[i][j][k] = np.setdiff1d(np.arange(N_k), self.I_idx[i][j][k], assume_unique=True)

        elif 'A' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - A")
            self.matA = IO.load_cell_matrix(global_data.incar.A_file, global_data.state_collection.k_shape)
            for i, j, k in global_data.state_collection.k_indices():
                mU, mS, mVh = np.linalg.svd(self.matA[i][j][k])
                self.matC[i][j][k] = mU @ np.eye(len(E_idx[i][j][k]), B) @ mVh
            self.matV = self.matC
            for i, j, k in global_data.state_collection.k_indices():
                N_k = len(E_idx[i][j][k])
                self.I_idx[i][j][k] = self.map_inner_to_local(E_idx[i][j][k], inner_idx[i][j][k])
                self.O_idx[i][j][k] = np.setdiff1d(np.arange(N_k), self.I_idx[i][j][k], assume_unique=True)
        else:
            self.projection()
            self.matV = self.matC

        if global_data.incar.proj_iter is None:
            global_data.incar.proj_iter = True

        min_len, max_len = self.get_min_max_len_idx(E_idx)

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
        for i, j, k in global_data.state_collection.k_indices():
            t_ = np.conj(self.matV[i][j][k]).T @ self.matA[i][j][k]
            mU, mS, mVh = np.linalg.svd(t_)
            self.matV[i][j][k] = self.matV[i][j][k] @ (mU @ np.eye(B, B) @ mVh)
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
        min_len, _ = self.get_min_max_len_idx(E_idx)
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
                
                # col = np.zeros(len(global_data.state_collection.extention_mesh.vertices), dtype=np.complex128)
                # for T in T_list:
                #     col += global_data.state_collection.extention_mesh.rfunc(f, T + cart_position, p['xaxis_angluar'])

                # H_list.append(np.asarray(col, dtype=np.complex128, copy=False))

                h = global_data.state_collection.extention_mesh.rfunc(f, cart_position, p['xaxis_angluar'])
                H_list.append(np.asarray(h, dtype=np.complex128))

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
        if norms.ndim == 0:
            norms = np.array([norms])
        G = H / np.sqrt(norms)[None, :]

        g = [G[:, k] for k in range(G.shape[1])]

        for i, j, k in global_data.state_collection.k_indices():
            phase = global_data.state_collection.get_extention_phase(i, j, k)
            Nv = global_data.state_collection.extention_mesh.vertices.shape[0]

            G = np.column_stack(g).astype(np.complex128, copy=False)
            if G.shape[0] != Nv:
                G = G.T

            for m in range(len(E_idx[i][j][k])):
                field = global_data.state_collection.get_extention_field(i, j, k, m, left=True)
                
                base = global_data.state_collection.extention_epsilon * np.conj(phase * field)

                F = (base[:, None] * G)
                fd = FieldData('', global_data.state_collection.extention_mesh, F)
                vals = WannierTools.integrate_over_mesh(fd, chunk_size=2048)
                # base = global_data.state_collection.extention_epsilon * phase * field
                # A = FieldData("A", global_data.state_collection.extention_mesh, np.broadcast_to(np.conj(base[None, :]), (shape[3], Nv)).astype(np.complex128, copy=False))
                # B = FieldData("B", global_data.state_collection.extention_mesh, (G.T).astype(np.complex128, copy=False))
                # vals = WannierTools.integrate_over_mesh(A, other=B, chunk_size=2048)
                if vals.ndim == 0:
                    vals = np.array([vals])
                self.matA[i][j][k][m, :vals.shape[0]] = vals

            if global_data.incar.proj_binarize:
                self.matA[i][j][k] = self.binarize(self.matA[i][j][k])
        if global_data.incar.proj_binarize:
            Logger.info("Projection binarization completed")
        
        for i, j, k in global_data.state_collection.k_indices():
            mU, mS, mVh = np.linalg.svd(self.matA[i][j][k])
            self.matC[i][j][k] = mU @ np.eye(len(E_idx[i][j][k]), B) @ mVh
        if global_data.incar.inner_window is not False:
            Logger.info("Using inner window for projection")
            self.inner_projection()
        else:
            self.I_idx = global_data.state_collection.gen_matrix_on_kmesh(lambda *_: np.empty((0,), dtype=np.intp))
            self.O_idx = global_data.state_collection.gen_matrix_on_kmesh(lambda i, j, k: np.setdiff1d(np.arange(len(E_idx[i][j][k])), self.I_idx[i][j][k], assume_unique=True))

        Logger.info('Projection compeleted')

    def inner_projection(self):
        E_idx = global_data.state_collection.E_idx
        N = global_data.incar.band_calc_num
        inner_idx = global_data.state_collection.inner_E_idx

        tol = 1e-10

        for i, j, k in global_data.state_collection.k_indices():
            N_k = len(E_idx[i][j][k])
            self.I_idx[i][j][k] = self.map_inner_to_local(E_idx[i][j][k], inner_idx[i][j][k])
            self.O_idx[i][j][k] = np.setdiff1d(np.arange(N_k), self.I_idx[i][j][k], assume_unique=True)
            M_k = self.I_idx[i][j][k].size
            p = N - M_k
            if p == 0:
                self.matV[i][j][k] = self.matC[i][j][k][:, :N]
            elif p < 0:
                Logger.error(f"(i={i}, j={j}, k={k}) inner window M_k={M_k} > N={N}. Increase N or shrink inner window.")
                raise
            if p > self.O_idx[i][j][k].size:
                Logger.error(f"(i={i}, j={j}, k={k}) outer window insufficient: N-M_k={p} > N_k-M_k={self.O_idx[i][j][k].size}.")
                raise
            
            # P_inner = np.zeros((N_k, N_k), dtype=np.complex128)
            # Q_inner = np.eye(N_k, dtype=np.complex128) - P_inner
            U, S, Vh = np.linalg.svd(self.matA[i][j][k], full_matrices=False)
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
            UI = np.eye(N_k, dtype=np.complex128)[:, self.I_idx[i][j][k]]

            P_G_OO = 0.5 * (P_G[np.ix_(self.O_idx[i][j][k], self.O_idx[i][j][k])] + P_G[np.ix_(self.O_idx[i][j][k], self.O_idx[i][j][k])].conj().T)
            wO, WO = np.linalg.eigh(P_G_OO)
            order = np.argsort(wO)[::-1]
            wO = wO[order]
            WO = WO[:, order]
            Vp = WO[:, :p]
            U_opt = np.zeros((N_k, p), dtype=np.complex128)
            U_opt[self.O_idx[i][j][k], :] = Vp
            
            U_init = np.concatenate([UI, U_opt], axis=1)
            self.matV[i][j][k] = U_init
        Logger.info('Inner projection compeleted')
    
    def V_projection(self):
        Logger.info('Starting V projection')
        for i, j, k in global_data.state_collection.k_indices():
            A = self.matV[i][j][k].conj().T @ self.matA[i][j][k]
            mU, mS, mVh = np.linalg.svd(A)
            self.matV[i][j][k] = self.matV[i][j][k] @ (mU @ np.eye(mU.shape[0], mVh.shape[0]) @ mVh)
        Logger.info('V projection compeleted')

    def update_Z(self):
        E_idx = global_data.state_collection.E_idx
        b_sz = len(global_data.incar.composition_of_b)
        for i, j, k in global_data.state_collection.k_indices():
            self.matZ[i][j][k] = np.zeros((len(self.O_idx[i][j][k]), len(self.O_idx[i][j][k])), dtype=complex)
            diag_II = 0.0
            for b in range(b_sz):
                mM = global_data.m_set.get_M0(i, j, k, b)
                ik, _ = WannierTools.neighbor_reciprocal_lattice_vectors([i, j, k], b)
                Cb = mM @ self.matV[ik[0]][ik[1]][ik[2]]
                Cb_O = Cb[self.O_idx[i][j][k], :]
                self.matZ[i][j][k] += global_data.incar.wb[b] * (Cb_O @ Cb_O.conj().T)

                if self.I_idx[i][j][k].size > 0:
                    Ci = Cb[self.I_idx[i][j][k], :]
                    diag_II += global_data.incar.wb[b] * np.sum(np.abs(Ci)**2)
            self.diagII_sum[i][j][k] = diag_II

            self.matZ[i][j][k] = 0.5*(self.matZ[i][j][k] + self.matZ[i][j][k].conj().T)
            if self.last_matZ[i][j][k] is None:
                self.last_matZ[i][j][k] = self.matZ[i][j][k]
            else:
                self.matZ[i][j][k] = self.alpha * self.matZ[i][j][k] + (1 - self.alpha) * self.last_matZ[i][j][k]
                self.last_matZ[i][j][k] = self.matZ[i][j][k]
    
    def sort_Z(self):
        B = global_data.incar.band_calc_num
        E_idx = global_data.state_collection.E_idx
        for i, j, k in global_data.state_collection.k_indices():
            D, V = np.linalg.eigh(self.matZ[i][j][k])
            p = B - len(self.I_idx[i][j][k])
            Vp = V[:, -p:] if p > 0 else np.zeros((self.matZ[i][j][k].shape[0], 0), complex)

            self.lambda_[i][j][k] = float(np.sum(D[-p:])) if p > 0 else 0.0
            N_k = len(E_idx[i][j][k])
            U_opt = np.zeros((N_k, p), dtype=complex)
            U_opt[ self.O_idx[i][j][k], :] = Vp
            if self.I_idx[i][j][k].size == 0:
                E_I = np.zeros((N_k, 0), dtype=complex)
            else:
                E_I = np.eye(N_k, dtype=complex)[:, self.I_idx[i][j][k]]
            U_concat = np.concatenate([E_I, U_opt], axis=1)
            self.matV[i][j][k] = U_concat[:, :B]
    
    def get_omega_I(self):
        N = global_data.incar.band_calc_num
        s_N_wb = N * np.sum(global_data.incar.wb)
        total = 0.0
        for i, j, k in global_data.state_collection.k_indices():
                total += s_N_wb - ( self.diagII_sum[i][j][k] + self.lambda_[i][j][k])
        return total / global_data.state_collection.get_k_num()
        
    def save_as(self, filenameV: str, filenameA: str):
        if not filenameV.lower() == "false":
            IO.save_to_txt(filenameV, self.matV, self.matV.shape)
        if not filenameA.lower() == "false":
            IO.save_to_txt(filenameA, self.matA, self.matA.shape)

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
    
    @staticmethod
    def get_min_max_len_idx(E_idx) -> Tuple[int, int]:
        it = (len(x) for x in (E_idx.flat if isinstance(E_idx, np.ndarray) else (li for plane in E_idx for row in plane for li in row)))
        min_len = None
        max_len = None
        for L in it:
            if min_len is None or L < min_len: min_len = L
            if max_len is None or L > max_len: max_len = L
        if min_len is None:
            Logger.error("E_idx is empty")
            raise
        return min_len, max_len

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
            Logger.error(f"Invalid Angular state: '{l}'")
            raise
    
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
            Logger.error(f"No radial function defined for (n={n}, l={l})")
            raise
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