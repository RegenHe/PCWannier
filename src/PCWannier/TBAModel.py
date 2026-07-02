import os

from itertools import product

import numpy as np
from math import factorial
import scipy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .Log import Logger
from .IO import IO
from .Timer import Timer, timer
from .Utils import global_data

from .Utils import WannierTools, FieldData
from .IncarParser import EnergyWindow

from .Finite import Finite2D


class TBAModal:
    def __init__(self):
        pass

    # @timer("Generate hoppings - ")
    def gen_hopping(self, r: list=[0, 0, 0]):
        kdim = global_data.incar.kdim

        a0 = global_data.incar.lattice_const
        avec = np.asarray(global_data.incar.real_lattice_vectors)
        D = avec.shape[1]
        r_use  = (list(r) + [0, 0, 0])[:kdim]
        r_cart = np.zeros(D, dtype=float)
        for a in range(kdim):
            r_cart += r_use[a] * avec[a, :]
        r_cart *= a0

        B = global_data.incar.band_calc_num

        hopping = np.zeros((B, B), dtype=complex)

        sign = -1 if str(global_data.incar.dataset_type).lower() == 'comsol' else 1
        if global_data.incar.disable_orth:
            # Logger.info("Disable orthogonalization as requested")
            T = global_data.state_collection.get_transform(True)
        else:
            T = global_data.state_collection.get_transform()

        for i, j, k in global_data.state_collection.k_indices():
                mU = T[i][j][k] @ global_data.state_initializer.matV[i][j][k] @ global_data.gradient.U[i][j][k]
                k_vec = WannierTools.get_kxyz([i, j, k])[:D]
                hopping += np.conj(mU).T @ np.diag(global_data.state_collection.E[i][j][k]) @ mU * np.exp(1j * (-sign) * np.dot(k_vec, r_cart))
        hopping = hopping / global_data.state_collection.get_k_num()
        return hopping

    def save_hoppings(self, filename: str):
        hoppings_dict = {}
        hoppings_dict[(0, 0, 0)] = self.gen_hopping()
        if global_data.incar.neighbor is None or len(global_data.incar.neighbor) == 0:
            global_data.incar.neighbor = self.R_half_rect(global_data.state_collection.k_shape)

        for idx, R in enumerate(global_data.incar.neighbor):
            R3 = (*R[:3], 0, 0, 0)[:3]
            hoppings_dict[(int(R3[0]), int(R3[1]), int(R3[2]))] = self.gen_hopping(R)
        
        IO.save_dict(filename, hoppings_dict)


    @timer("Generate High Symmetry Point Band Structure - ")
    def gen_hs_bands(self):
        kdim = global_data.incar.kdim
        if global_data.incar.band_figure.lower() == 'false':
            return
        H0 = self.gen_hopping()
        
        high_sym_points = []
        k_list = [np.array(global_data.incar.k_path[0]['point'])[:kdim]]
        total = 0
        for i in range(len(global_data.incar.k_path)):
            high_sym_points.append([global_data.incar.k_path[i]['name'], total])
            num = global_data.incar.k_path[i]['num']

            start = np.asarray(global_data.incar.k_path[i]['point'])[:kdim]
            stop =  np.asarray(global_data.incar.k_path[(i + 1) % len(global_data.incar.k_path)]['point'])[:kdim]

            if num > 0:
                seg = np.stack([np.linspace(start[a], stop[a], num + 1)[1:] for a in range(kdim)], axis=1)
                k_list.append(seg)

            total += num
        high_sym_points.append([global_data.incar.k_path[0]['name'], total])
        k_list = np.vstack(k_list)
        K = np.arange(0, total + 1)

        if global_data.incar.neighbor is None or len(global_data.incar.neighbor) == 0:
            global_data.incar.neighbor = self.R_half_rect(global_data.state_collection.k_shape)

        self.hoppings = []
        for p in global_data.incar.neighbor:
            self.hoppings.append(self.gen_hopping(p))
        
        a0 = float(global_data.incar.lattice_const)
        B = global_data.incar.band_calc_num
        avec = np.asarray(global_data.incar.real_lattice_vectors)
        G = np.asarray(global_data.incar.reciprocal_lattice_vectors, dtype=float)
        D = avec.shape[1]

        neigh = np.asarray(global_data.incar.neighbor, int)
        hops = np.asarray(self.hoppings, np.complex128)
        delta_R = (neigh[:, :D] @ avec) * a0
        nyq_mask = np.array([self.is_nyquist(R, global_data.state_collection.k_shape) for R in neigh], dtype=bool)

        H0 = np.asarray(self.gen_hopping([0, 0, 0]), np.complex128)

        def kfrac_to_kcart(kfrac: np.ndarray) -> np.ndarray:
            return (kfrac @ G) * (2.0 * np.pi / a0)

        def H_of_k_batch(k_cart: np.ndarray) -> np.ndarray:
            if neigh.size == 0:
                return np.broadcast_to(H0, (k_cart.shape[0], B, B)).copy()

            phase = np.exp(1j * (k_cart @ delta_R.T))
            if np.any(~nyq_mask):
                Hi = np.einsum('mn,nab->mab', phase[:, ~nyq_mask], hops[~nyq_mask])
            else:
                Hi = np.zeros((k_cart.shape[0], B, B), dtype=np.complex128)
            if np.any(nyq_mask):
                Hq = np.einsum('mn,nab->mab', phase[:,  nyq_mask], hops[ nyq_mask])
            else:
                Hq = 0.0

            return H0 + Hi + np.conjugate(np.swapaxes(Hi, -2, -1)) + Hq
        
        k_cart = kfrac_to_kcart(k_list)
        Hk = H_of_k_batch(k_cart)
        E = np.linalg.eigvals(Hk)
        E = np.sort(E)
        
        if global_data.incar.band_file.lower() != "false":
            IO.save_band(global_data.incar.band_file, E, k_list)

        if global_data.incar.DOS == 0:
            fig, ax = plt.subplots()
            for band in range(E.shape[1]):
                plt.plot(K, E[:, band], color='blue')

            for pos in [p[1] for p in high_sym_points]:
                plt.axvline(x=pos, color='black', linestyle='--', linewidth=0.5)
            plt.xticks([p[1] for p in high_sym_points], [p[0] for p in high_sym_points])
            plt.xlim(0, total)
            if isinstance(global_data.incar.band_window, EnergyWindow):
                plt.axhline(y=global_data.incar.band_window.emin, color='k', linestyle='--', linewidth=1, zorder=3)
                plt.axhline(y=global_data.incar.band_window.emax, color='k', linestyle='--', linewidth=1, zorder=3)
            if global_data.incar.inner_window is not False and isinstance(global_data.incar.inner_window, EnergyWindow):
                plt.axhline(y=global_data.incar.inner_window.emin, color='r', linestyle='--', linewidth=0.8, zorder=3)
                plt.axhline(y=global_data.incar.inner_window.emax, color='r', linestyle='--', linewidth=0.8, zorder=3)
            plt.title("Band Structure", fontsize=14)
            plt.ylabel("E", fontsize=12)
            plt.tight_layout()
            plt.savefig(global_data.incar.band_figure, dpi=300, bbox_inches='tight')
            plt.close(fig)
            Logger.info(f"figure successfully saved to {global_data.incar.band_figure}")
        elif global_data.incar.DOS == 1:
            E_list = np.linspace(np.min(E), np.max(E), global_data.incar.DOS_num)
            DOS = np.zeros((1, global_data.incar.DOS_num), dtype=complex)
            for i, e in enumerate(E_list):
                Gm = np.linalg.inv(Hk - (e - 1j * global_data.incar.DOS_eps) * np.eye(B, dtype=np.complex128))
                DOS[0, i] += np.real(-1 / np.pi * np.imag(np.diagonal(Gm, axis1=-2, axis2=-1))).sum()
            self.plot_hs_band_dos(K, E, high_sym_points, E_list, DOS, save_path=global_data.incar.band_figure, dos_title='PDOS')
            Logger.info(f"figure successfully saved to {global_data.incar.band_figure}")
        elif global_data.incar.DOS == 2:
            E_list = np.linspace(np.min(E), np.max(E), global_data.incar.DOS_num)
            DOS = np.zeros((global_data.incar.band_calc_num, global_data.incar.DOS_num), dtype=complex)
            for i, e in enumerate(E_list):
                Gm = np.linalg.inv(Hk - (e - 1j * global_data.incar.DOS_eps) * np.eye(B, dtype=np.complex128))
                DOS[:, i] += np.real(-1 / np.pi * np.imag(np.diagonal(Gm, axis1=-2, axis2=-1))).sum(axis=0)
            self.plot_hs_band_dos(K, E, high_sym_points, E_list, DOS, save_path=global_data.incar.band_figure, dos_title='PDOS')
            Logger.info(f"figure successfully saved to {global_data.incar.band_figure}")
        elif global_data.incar.DOS == 3:
            mesh = np.array(global_data.incar.DOS_Brillouin_mesh, int)[:kdim]
            k0_cart = WannierTools.get_kxyz([0, 0, 0])[:D]
            axes = [np.arange(m, dtype=float)/m for m in mesh]
            grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, kdim)
            k_cart_grid = k0_cart + (grid @ G) * (2 * np.pi / a0)
            
            Hk_grid = H_of_k_batch(k_cart_grid)
            DOS = np.zeros((1, global_data.incar.DOS_num), dtype=complex)
            E_list = np.linspace(np.min(E), np.max(E), global_data.incar.DOS_num)
            for i, e in enumerate(E_list):
                Gm = np.linalg.inv(Hk_grid - (e - 1j * global_data.incar.DOS_eps) * np.eye(B, dtype=np.complex128))
                diagG = np.diagonal(Gm, axis1=-2, axis2=-1)
                DOS[0, i] = (-1/np.pi) * np.imag(diagG).sum()
            self.plot_hs_band_dos(K, E, high_sym_points, E_list, DOS, save_path=global_data.incar.band_figure)
            Logger.info(f"figure successfully saved to {global_data.incar.band_figure}")
    
    @staticmethod
    def is_nyquist(R, kshape) -> bool:
        R = list(map(int, R))
        dims = len(kshape)
        all_zero = True
        for a in range(dims):
            N = int(kshape[a])
            ra = R[a] % N
            if ra != 0:
                all_zero = False
            if N % 2 == 0:
                if (2 * ra) % N != 0:
                    return False
            else:
                if ra != 0:
                    return False
        return not all_zero

    @staticmethod
    def R_half_rect(kshape) -> np.ndarray:
        dims = len(kshape)
        ranges = []
        for N in kshape:
            N = int(N)
            half = N // 2
            lo = -half + (1 if N % 2 == 0 else 0)
            hi = half
            ranges.append(range(lo, hi + 1))

        R = []
        for coords in product(*ranges):
            if all(c == 0 for c in coords):
                continue
            keep = False
            for c in coords:
                if c != 0:
                    keep = (c > 0)
                    break
            if keep:
                R.append(coords[:3] if dims >= 3 else (coords + (0, 0))[:3])
        return np.array(R, int)


    @staticmethod
    def plot_hs_band_dos(k_path: np.ndarray,
                        bands: np.ndarray,
                        high_sym_points: list,
                        dos_energy: np.ndarray,
                        dos_components: list,
                        dos_labels: list = None,
                        dos_colors: list = None,
                        dos_title: str = "DOS",
                        alpha: float = 0.3,
                        figsize=(8, 6),
                        save_path: str = None):
        if dos_labels is None:
            dos_labels = [f"DOS {i+1}" for i in range(len(dos_components))]
        if dos_colors is None:
            cmap = plt.get_cmap("tab10")
            dos_colors = [cmap(i) for i in range(len(dos_components))]

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)

        ax_band = fig.add_subplot(gs[0])
        for band in range(bands.shape[1]):
            ax_band.plot(k_path, np.real(bands[:, band]), color='blue')
        
        if isinstance(global_data.incar.band_window, EnergyWindow):
            plt.axhline(y=global_data.incar.band_window.emin, color='k', linestyle='--', linewidth=1, zorder=3)
            plt.axhline(y=global_data.incar.band_window.emax, color='k', linestyle='--', linewidth=1, zorder=3)
        if global_data.incar.inner_window is not False and isinstance(global_data.incar.inner_window, EnergyWindow):
            plt.axhline(y=global_data.incar.inner_window.emin, color='r', linestyle='--', linewidth=0.8, zorder=3)
            plt.axhline(y=global_data.incar.inner_window.emax, color='r', linestyle='--', linewidth=0.8, zorder=3)

        for pos in [p[1] for p in high_sym_points]:
            ax_band.axvline(x=pos, color='black', linestyle='--', linewidth=0.5)
        ax_band.set_xticks([p[1] for p in high_sym_points])
        ax_band.set_xticklabels([p[0] for p in high_sym_points])
        ax_band.set_xlim(0, k_path[-1])
        ax_band.set_title("Band Structure", fontsize=14)
        ax_band.set_ylabel("E", fontsize=12)

        ax_dos = fig.add_subplot(gs[1], sharey=ax_band)
        for i, dos in enumerate(dos_components):
            ax_dos.plot(np.real(dos), np.real(dos_energy), color=dos_colors[i], label=dos_labels[i])
            ax_dos.fill_betweenx(np.real(dos_energy), 0, np.real(dos), color=dos_colors[i], alpha=alpha)

        ax_dos.set_xlabel(dos_title)
        ax_dos.tick_params(labelleft=False)
        ax_dos.grid(True)
        ax_dos.legend(loc='upper right', fontsize='small')

        if save_path:
            fig.savefig(save_path, dpi=300)
            plt.close(fig)

    @timer("Generate Band Structure - ")
    def gen_band(self):
        kdim = global_data.incar.kdim
        a0 = float(global_data.incar.lattice_const)
        B = global_data.incar.band_calc_num

        k_num = np.array(global_data.incar.k_num, int)[:kdim]

        axes = [np.linspace(-0.5, 0.5, n, endpoint=False) for n in k_num]
        Kgrids = np.meshgrid(*axes, indexing="ij")
        kfrac = np.stack(Kgrids, axis=-1)
        Nkshape = kfrac.shape[:-1]
        Nk_tot = int(np.prod(Nkshape))
        
        if self.hoppings is None:
            self.hoppings = []
            for p in global_data.incar.neighbor:
                self.hoppings.append(self.gen_hopping(p))

        neigh = np.asarray(global_data.incar.neighbor, int)
        G = np.asarray(global_data.incar.reciprocal_lattice_vectors, float)[:kdim, :]
        avec = np.asarray(global_data.incar.real_lattice_vectors, float)[:kdim, :]
        D = G.shape[1]

        hops = np.asarray(self.hoppings, np.complex128)
        delta_R  = (neigh[:, :D] @ avec) * a0
        nyq_mask = np.array([self.is_nyquist(R, global_data.state_collection.k_shape) for R in neigh], bool)

        H0 = np.asarray(self.gen_hopping([0, 0, 0]), np.complex128)

        def kfrac_to_kcart(kf: np.ndarray) -> np.ndarray:
            return (kf @ G) * (2.0 * np.pi / a0)

        def H_of_k_batch(k_cart: np.ndarray) -> np.ndarray:
            M = k_cart.shape[0]
            if neigh.size == 0:
                return np.broadcast_to(H0, (M, B, B)).copy()

            phase = np.exp(1j * (k_cart @ delta_R.T))
            if np.any(~nyq_mask):
                Hi = np.einsum('mn,nab->mab', phase[:, ~nyq_mask], hops[~nyq_mask])
            else:
                Hi = np.zeros((M, B, B), dtype=np.complex128)
            if np.any(nyq_mask):
                Hq = np.einsum('mn,nab->mab', phase[:,  nyq_mask], hops[ nyq_mask])
            else:
                Hq = 0.0
            Hk = H0 + Hi + np.conjugate(np.swapaxes(Hi, -2, -1)) + Hq
            return Hk
        
        k_cart = kfrac_to_kcart(kfrac.reshape(Nk_tot, kdim))
        Hk = H_of_k_batch(k_cart)

        eigvals, eigvecs = np.linalg.eigh(Hk)
        self.eigvals = eigvals.reshape(*Nkshape, B)
        self.eigvecs = eigvecs.reshape(*Nkshape, B, B)

        self.groups = self.group_bands(self.eigvals, delta_rel=1e-2)
        for gid, g in enumerate(self.groups):
            Logger.info(f"group {gid}: bands {g}")

    @staticmethod
    def group_bands(E: np.ndarray, delta_rel=1e-3, delta_abs=None):
        E = np.asarray(E)
        if E.ndim < 2:
            Logger.error("E must have at least 2 dims: (..., Nb)")
            raise

        Nk = int(np.prod(E.shape[:-1]))
        Nb = int(E.shape[-1])
        Ek = E.reshape(Nk, Nb)

        span = float(Ek.max() - Ek.min())
        delta = float(delta_rel) * span
        if delta_abs is not None:
            delta = max(delta, float(delta_abs))
    
        mindiff = np.full((Nb, Nb), np.inf, dtype=float)

        denom = max(Nb * Nb, 1)
        k_block = max(1, int(64e6 // (8 * denom)))

        for s in range(0, Nk, k_block):
            t = min(s + k_block, Nk)
            X = Ek[s:t]
            local_min = np.min(np.abs(X[:, :, None] - X[:, None, :]), axis=0)
            np.minimum(mindiff, local_min, out=mindiff)

        adjacency = (mindiff <= delta)
        np.fill_diagonal(adjacency, True)
        adjacency |= adjacency.T

        graph = scipy.sparse.csr_matrix(adjacency)
        n_comp, labels = scipy.sparse.csgraph.connected_components(graph, directed=False)

        groups = [[] for _ in range(n_comp)]
        for b, lab in enumerate(labels):
            groups[lab].append(b)

        groups = [sorted(g) for g in groups if g]
        groups.sort(key=lambda g: g[0])
        return groups
    

    @timer("Generate Effective Hamiltonian - ")
    def effective_Hamiltonian(self):
        H0 = self.gen_hopping()

        if getattr(self, 'hoppings', None) is None or len(self.hoppings) == 0:
            self.hoppings = []
            for R in global_data.incar.neighbor:
                self.hoppings.append(self.gen_hopping(R))

        self.H_eff = {}

        Rlist = np.asarray(global_data.incar.neighbor, dtype=float)[:, 0:global_data.incar.kdim]
        Rint = Rlist.astype(int, copy=False)
        Tlist = np.stack(self.hoppings, axis=0).astype(np.complex128)

        D = global_data.incar.kdim
        axes = self._axis_names(D)

        Nks = [len(global_data.incar.k_points[d]) for d in range(D)]

        def is_nyquist_vectorized(Rint, Nks):
            mask = np.ones(Rint.shape[0], dtype=bool)
            for d in range(Rint.shape[1]):
                Nd = Nks[d]
                if Nd % 2 != 0:
                    mask &= False
                else:
                    mask &= ((2 * Rint[:, d]) % Nd == 0)
            return mask

        nyq_mask = is_nyquist_vectorized(Rint, Nks)
        int_mask = ~nyq_mask

        if np.any(nyq_mask):
            Tn = Tlist[nyq_mask]
            Tlist[nyq_mask] = 0.5 * (Tn + np.conjugate(Tn).transpose(0, 2, 1))

        phase = np.exp(1j * 2 * np.pi * (Rlist @ np.asarray(global_data.incar.eff_k, dtype=float)))

        WT  = phase[:, None, None] * Tlist
        WTd = np.conjugate(WT).transpose(0, 2, 1)

        H_const = H0.copy()
        if np.any(int_mask):
            H_const += WT[int_mask].sum(axis=0) + WTd[int_mask].sum(axis=0)
        if np.any(nyq_mask):
            H_const += WT[nyq_mask].sum(axis=0)
        self.H_eff['1'] = H_const

        for n in range(1, global_data.incar.eff_order + 1):
            monoms = self._monomials_of_order(D, n)
            pref_power = (1j) ** n
            sgn = -1 if (n % 2 == 1) else 1

            for m in monoms:
                prodRm = np.ones(Rlist.shape[0], dtype=float)
                denom = 1
                for a, e in enumerate(m):
                    if e:
                        prodRm *= Rlist[:, a] ** e
                        denom  *= factorial(e)

                term = np.zeros_like(H0, dtype=np.complex128)
                if np.any(int_mask):
                    w = prodRm[int_mask]
                    term += (np.tensordot(w, WT[int_mask],  axes=(0, 0)) + sgn * np.tensordot(w, WTd[int_mask], axes=(0, 0)))
                if np.any(nyq_mask):
                    w = prodRm[nyq_mask]
                    term += np.tensordot(w, WT[nyq_mask], axes=(0, 0))

                term *= (pref_power / denom)
                key = self._key_from_m(axes, m)
                self.H_eff[key] = term

        Logger.info(f"Effective Hamiltonian terms: {list(self.H_eff.keys())}")
        IO.save_dict(global_data.incar.eff_file, self.H_eff)

        if global_data.incar.decompose:
            decomp = self.decompose_effH_to_SU_N(self.H_eff)
            IO.save_dict(global_data.incar.decompose_file, decomp)
            Logger.info(f"Decomposed effective Hamiltonian terms saved to {global_data.incar.decompose_file}")

        return self.H_eff


    def _axis_names(self, dim):
        base = ['x','y','z']
        if dim <= 3:
            return base[:dim]
        return base + [f'k{i}' for i in range(dim-3)]

    def _monomials_of_order(self, dim, n):
        if n == 0:
            return [tuple([0]*dim)]
        out = []
        for m in product(range(n+1), repeat=dim):
            if sum(m) == n:
                out.append(m)
        return out

    def _key_from_m(self, axes, m):
        n = sum(m)
        if n == 0:
            return '1'
        var = ''.join(
            (axes[a] if e == 1 else axes[a]+str(e)) 
            for a, e in enumerate(m) if e > 0
        )
        denom = 1
        for e in m: denom *= factorial(e)
        coeff = factorial(n) // denom
        return (str(coeff) + var) if coeff > 1 else var
    
    @staticmethod
    def SU_N_generators(N: int):
        L = []
        for j in range(N):
            for k in range(j + 1, N):
                S = np.zeros((N, N), complex); S[j, k] = S[k, j] = 1.0
                A = np.zeros((N, N), complex); A[j, k] = -1j; A[k, j] = +1j
                L.append(S); L.append(A)
        for l in range(1, N):
            H = np.zeros((N, N), complex)
            H[range(l), range(l)] = 1.0
            H[l, l] = -float(l)
            H *= np.sqrt(2.0 / (l * (l + 1.0)))
            L.append(H)
        return np.eye(N, dtype=complex), L
    
    @staticmethod
    def SU_N_decompose(H: np.ndarray, L: list[np.ndarray]):
        N = H.shape[0]
        c0 = np.trace(H) / N
        c = 0.5 * np.array([np.trace(L @ H) for L in L], dtype=complex)
        if np.allclose(H, H.conj().T):
            c = c.real
        return c0, c

    @staticmethod
    def decompose_effH_to_SU_N(H_eff: dict):
        if not H_eff:
            return {}, {}

        any_key = next(iter(H_eff))
        N = H_eff[any_key].shape[0]

        I, L = TBAModal.SU_N_generators(N)
        decom = {1: I}
        for a, La in enumerate(L, start=2):
            decom[a] = La

        for key, M in H_eff.items():
            c0, c = TBAModal.SU_N_decompose(M, L)
            coeffs = np.concatenate(([c0], np.asarray(c)))
            decom[key] = [coeffs[i] for i in range(len(coeffs))]

        return decom
    
    @timer("Calculate Finite System Band Structure - ")
    def calc_finite(self):
        if global_data.incar.neighbor is None or len(global_data.incar.neighbor) == 0:
            global_data.incar.neighbor = self.R_half_rect(len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]))
        finite = Finite2D(global_data.incar.finite[0], global_data.incar.finite[1], self.gen_hopping, global_data.incar.neighbor)
        k_list = np.linspace(global_data.incar.finite_k[0], global_data.incar.finite_k[1], int(global_data.incar.finite_k[-1]), endpoint=True) * 2 * np.pi / global_data.incar.lattice_const
        k_list, E, V = finite.bands_stripe(k_list)
        fig, ax = plt.subplots()
        if k_list is not None:
            for band in range(E.shape[1]):
                plt.plot(k_list, E[:, band], color='blue')
            ax.set_xlim(np.min(k_list), np.max(k_list))
            plt.xlabel("k", fontsize=12)
        else:
            plt.scatter(np.linspace(0, E.size - 1, E.size), E, color='blue', s=3)
            plt.xlabel("solution number", fontsize=12)
        
        plt.title("Band Structure", fontsize=14)
        plt.ylabel("E", fontsize=12)
        plt.tight_layout()
        if global_data.incar.finite_band_figure.lower() != 'false':
            plt.savefig(global_data.incar.finite_band_figure, dpi=300, bbox_inches='tight')
            Logger.info(f"figure successfully saved to {global_data.incar.finite_band_figure}")
            plt.close(fig)
        
        if global_data.incar.finite_band_file.lower() != 'false':
            IO.save_band(global_data.incar.finite_band_file, E, k_list)
        
        if global_data.incar.finite_wavefunction_file.lower() != 'false':
            IO.save_to_txt(global_data.incar.finite_wavefunction_file, V, (V.shape[0], ))

        
        if global_data.incar.finite_DOS_num is not False:
            energy_range = np.linspace(E.min(), E.max(), global_data.incar.finite_DOS_num)
            klist, LDOS = finite.half_infinte_DOS(global_data.incar.finite_layer_num, k_list, global_data.incar.finite_DOS_eps, energy_range)

            if global_data.incar.finite_DOS_file.lower() != 'false':
                info = {'type': 'SDOS', 'energy_range': f"{E.min()}, {E.max()}", 'DOS_num': f"{global_data.incar.finite_DOS_num}"}
                IO.save_band(global_data.incar.finite_DOS_file, LDOS, klist, info)
            
            fig, ax = plt.subplots()
            K, E = np.meshgrid(klist, energy_range)
            pcm = ax.pcolormesh(K, E, LDOS.T, shading='auto', cmap='jet')
            fig.colorbar(pcm, ax=ax)
            
            plt.title("SDOS", fontsize=14)
            plt.xlabel("k", fontsize=12)
            plt.ylabel("E", fontsize=12)
            plt.tight_layout()
            
            if global_data.incar.finite_DOS_figure.lower() != 'false':
                plt.savefig(global_data.incar.finite_DOS_figure, dpi=300, bbox_inches='tight')
                Logger.info(f"figure successfully saved to {global_data.incar.finite_DOS_figure}")
                plt.close(fig)
