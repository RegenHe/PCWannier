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

from .Finite import Finite


class TBAModal:
    def __init__(self):
        pass

    # @timer("Generate hoppings - ")
    def gen_hopping(self, r: list=[0, 0]):
        r_ = [0, 0]
        r_[0] = (r[0] * global_data.incar.real_lattice_vectors[0][0] + r[1] * global_data.incar.real_lattice_vectors[1][0]) * global_data.incar.lattice_const
        r_[1] = (r[0] * global_data.incar.real_lattice_vectors[0][1] + r[1] * global_data.incar.real_lattice_vectors[1][1]) * global_data.incar.lattice_const
        # Logger.info(f"Generating hoppings - r = ({r[0]}, {r[1]})")

        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.band_window), global_data.incar.band_calc_num]
        hopping = np.zeros((shape[3], shape[3]), dtype=complex)

        if global_data.incar.disable_orth:
            # Logger.info("Disable orthogonalization as requested")
            T = global_data.state_collection.get_transform(True)
        else:
            T = global_data.state_collection.get_transform()

        for i in range(shape[0]):
            for j in range(shape[1]):
                mU = T[i][j] @ global_data.state_initializer.matV[i][j] @ global_data.gradient.U[i][j]
                if global_data.incar.dataset_type.lower() == 'comsol':
                    sign = -1
                kx, ky = WannierTools.get_kx_ky([i, j])
                hopping += np.conj(mU).T @ np.diag(global_data.state_collection.E[i][j]) @ mU * np.exp(1j * np.dot(-1 * sign * np.array([kx, ky]), r_))
        hopping = hopping / shape[0] / shape[1]
        return hopping

    def save_hoppings(self, filename: str):
        hoppings = [[None for _ in range(len(global_data.incar.hopping_state[1]))] for _ in range(len(global_data.incar.hopping_state[0]))]
        for i in range(len(global_data.incar.hopping_state[0])):
            for j in range(len(global_data.incar.hopping_state[1])):
                hoppings[i][j] = self.gen_hopping([global_data.incar.hopping_state[0][i], global_data.incar.hopping_state[1][j]])
        IO.save_to_txt(filename, hoppings, (len(global_data.incar.hopping_state[0]), len(global_data.incar.hopping_state[1])))

    @timer("Generate High Symmetry Point Band Structure - ")
    def gen_hs_bands(self):
        if global_data.incar.band_figure.lower() == 'false':
            return
        H0 = self.gen_hopping()
        
        high_sym_points = []
        k_list = np.array(global_data.incar.k_path[0]['point'])
        total = 0
        for i in range(len(global_data.incar.k_path)):
            high_sym_points.append([global_data.incar.k_path[i]['name'], total])
            total += global_data.incar.k_path[i]['num']

            start = global_data.incar.k_path[i]['point']
            stop = global_data.incar.k_path[(i + 1) % len(global_data.incar.k_path)]['point']
            kx_list = np.linspace(start[0], stop[0], global_data.incar.k_path[i]['num'] + 1)[1:]
            ky_list = np.linspace(start[1], stop[1], global_data.incar.k_path[i]['num'] + 1)[1:]
            k_list = np.vstack((k_list, (np.vstack((kx_list, ky_list))).T))
        high_sym_points.append([global_data.incar.k_path[0]['name'], total])
        K = np.arange(0, total + 1)

        if global_data.incar.neighbor == []:
            global_data.incar.neighbor = self.R_half_rect(len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]))

        self.hoppings = []
        for p in global_data.incar.neighbor:
            self.hoppings.append(self.gen_hopping(p))
        
        E = []
        for k_ in k_list:
            Hi = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
            Hq = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
            kx = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const
            ky = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const
            k = [kx, ky]
            for i in range(len(global_data.incar.neighbor)):
                r_ = [0, 0]
                r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][0] * global_data.incar.lattice_const)
                r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][1] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1] * global_data.incar.lattice_const)
                if self.is_nyquist(global_data.incar.neighbor[i][0], global_data.incar.neighbor[i][1], len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])):
                    Hq = self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                    continue
                Hi += self.hoppings[i] * np.exp(1j * np.dot(k, r_))
            Hi = Hi + np.conj(Hi).T
            H = H0 + Hi + Hq
            D, V = np.linalg.eig(H)
            E.append(np.sort(np.real(D)))
        E = np.array(E)
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
            for k_ in k_list:
                Hi = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
                Hq = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
                kx = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const
                ky = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const
                k = [kx, ky]
                for i in range(len(global_data.incar.neighbor)):
                    r_ = [0, 0]
                    r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][0]) * global_data.incar.lattice_const
                    r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][1] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1]) * global_data.incar.lattice_const
                    if self.is_nyquist(global_data.incar.neighbor[i][0], global_data.incar.neighbor[i][1], len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])):
                        Hq = self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                        continue
                    Hi += self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                Hi = Hi + np.conj(Hi).T
                H = H0 + Hi + Hq
                for i, e in enumerate(E_list):
                    G = np.linalg.inv(H - (e - 1j * global_data.incar.DOS_eps) * np.eye(H.shape[0], H.shape[1]))
                    DOS[0, i] += np.sum(np.real(-1 / np.pi * np.imag(np.diag(G))))
            self.plot_hs_band_dos(K, E, high_sym_points, E_list, DOS, save_path=global_data.incar.band_figure, dos_title='PDOS')
            Logger.info(f"figure successfully saved to {global_data.incar.band_figure}")
        elif global_data.incar.DOS == 2:
            E_list = np.linspace(np.min(E), np.max(E), global_data.incar.DOS_num)
            DOS = np.zeros((global_data.incar.band_calc_num, global_data.incar.DOS_num), dtype=complex)
            for k_ in k_list:
                Hi = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
                Hq = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
                kx = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const
                ky = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const
                k = [kx, ky]
                for i in range(len(global_data.incar.neighbor)):
                    r_ = [0, 0]
                    r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][0] * global_data.incar.lattice_const)
                    r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][1] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1] * global_data.incar.lattice_const)
                    if self.is_nyquist(global_data.incar.neighbor[i][0], global_data.incar.neighbor[i][1], len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])):
                        Hq = self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                        continue
                    Hi += self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                Hi = Hi + np.conj(Hi).T
                H = H0 + Hi + Hq
                for i, e in enumerate(E_list):
                    G = np.linalg.inv(H - (e - 1j * global_data.incar.DOS_eps) * np.eye(H.shape[0], H.shape[1]))
                    DOS[:, i] += np.real(-1 / np.pi * np.imag(np.diag(G)))
            self.plot_hs_band_dos(K, E, high_sym_points, E_list, DOS, save_path=global_data.incar.band_figure, dos_title='PDOS')
            Logger.info(f"figure successfully saved to {global_data.incar.band_figure}")
        elif global_data.incar.DOS == 3:
            E_list = np.linspace(np.min(E), np.max(E), global_data.incar.DOS_num)
            DOS = np.zeros((1, global_data.incar.DOS_num), dtype=complex)
            for k_ in k_list:
                Hi = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
                Hq = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
                kx = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const
                ky = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const
                k = [kx, ky]
                for i in range(len(global_data.incar.neighbor)):
                    r_ = [0, 0]
                    r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][0]) * global_data.incar.lattice_const
                    r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][1] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1]) * global_data.incar.lattice_const
                    if self.is_nyquist(global_data.incar.neighbor[i][0], global_data.incar.neighbor[i][1], len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])):
                        Hq = self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                        continue
                    Hi += self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                Hi = Hi + np.conj(Hi).T
                H = H0 + Hi + Hq
            for i in range(global_data.incar.DOS_Brillouin_mesh[0]):
                for j in range(global_data.incar.DOS_Brillouin_mesh[1]):
                    Hi = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
                    Hq = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
                    k0 = WannierTools.get_kx_ky([0, 0])
                    kx = k0[0] + i * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const / global_data.incar.DOS_Brillouin_mesh[0] + j * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const / global_data.incar.DOS_Brillouin_mesh[1]
                    ky = k0[1] + i * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const / global_data.incar.DOS_Brillouin_mesh[0] + j * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const / global_data.incar.DOS_Brillouin_mesh[1]
                    k = [kx, ky]
                    for i in range(len(global_data.incar.neighbor)):
                        r_ = [0, 0]
                        r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][0]) * global_data.incar.lattice_const
                        r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][1] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1]) * global_data.incar.lattice_const
                        if self.is_nyquist(global_data.incar.neighbor[i][0], global_data.incar.neighbor[i][1], len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])):
                            Hq = self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                            continue
                        Hi += self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                    Hi = Hi + np.conj(Hi).T
                    H = H0 + Hi + Hq
                    for i, e in enumerate(E_list):
                        G = np.linalg.inv(H - (e - 1j * global_data.incar.DOS_eps) * np.eye(H.shape[0], H.shape[1]))
                        DOS[0, i] += np.sum(np.real(-1 / np.pi * np.imag(np.diag(G))))
            self.plot_hs_band_dos(K, E, high_sym_points, E_list, DOS, save_path=global_data.incar.band_figure)
            Logger.info(f"figure successfully saved to {global_data.incar.band_figure}")
    
    @staticmethod
    def is_nyquist(Rx, Ry, Nx, Ny):
        return (Nx % 2 == 0 and (2*Rx) % Nx == 0) and (Ny % 2 == 0 and (2*Ry) % Ny == 0)

    @staticmethod
    def R_half_rect(Nx: int, Ny: int) -> np.ndarray:
        x_m = Nx // 2
        y_m = Ny // 2
        y_min = -y_m + (1 if Ny % 2 == 0 else 0)
        R = []
        Rmax = max(x_m, y_m)

        for r in range(1, Rmax + 1):
            if r <= y_m:
                xmax = min(r, x_m)
                for x in range(0, xmax + 1):
                    if not (x == 0 and r == 0):
                        R.append((x, r))
            if r <= x_m:
                y_top = min(r - 1, y_m)
                for y in range(y_top, y_min - 1, -1):
                    if not (r == 0 and y == 0):
                        R.append((r, y))
        return np.array(R, dtype=int)


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
        n1 = global_data.incar.k_num[0]
        n2 = global_data.incar.k_num[1]
        k1_grid = np.linspace(-0.5, 0.5, n1, endpoint=False)
        k2_grid = np.linspace(-0.5, 0.5, n2, endpoint=False)
        K1, K2 = np.meshgrid(k1_grid, k2_grid, indexing="ij")
        kvecs = (K1[..., None] * global_data.incar.reciprocal_lattice_vectors[0] + K2[..., None] * global_data.incar.reciprocal_lattice_vectors[1]) * 2 * np.pi / global_data.incar.lattice_const

        if self.hoppings is None:
            self.hoppings = []
            for p in global_data.incar.neighbor:
                self.hoppings.append(self.gen_hopping(p))

        neighbor = np.asarray(global_data.incar.neighbor)
        real_lattice = np.asarray(global_data.incar.real_lattice_vectors)
        delta_R = neighbor @ real_lattice * global_data.incar.lattice_const

        hoppings = np.asarray(self.hoppings)

        phase = np.exp(1j * np.einsum('...d,nd->...n', kvecs, delta_R))
        Hi = np.einsum('...n,nab->...ab', phase, hoppings)
        H0 = np.asarray(self.gen_hopping())
        Hk = H0 + Hi + np.conjugate(np.swapaxes(Hi, -2, -1))

        self.eigvals, self.eigvecs = np.linalg.eigh(Hk)
        del Hk, Hi, H0, phase

        self.groups = self.group_bands(self.eigvals, delta_rel=1e-2)
        for gid, g in enumerate(self.groups):
            Logger.info(f"group {gid}: bands {g}")

    @staticmethod
    def group_bands(E: np.ndarray, delta_rel=1e-3, delta_abs=None):
        E = np.asarray(E)
        if E.ndim < 2:
            raise ValueError("E must have at least 2 dims: (..., Nb)")

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

        Rlist = np.asarray(global_data.incar.neighbor, dtype=float)
        Rint  = Rlist.astype(int, copy=False)
        Tlist = np.stack(self.hoppings, axis=0).astype(np.complex128)
        axes  = self._axis_names(Rlist.shape[1])


        D = Rlist.shape[1]
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

        phase = np.exp(1j * 2 * np.pi * (Rlist @ np.asarray(global_data.incar.eff_k, dtype=float)))  # (NR,)

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
        if len(global_data.incar.neighbor) == 0:
            global_data.incar.neighbor = self.R_half_rect(len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]))
        finite = Finite(global_data.incar.finite[0], global_data.incar.finite[1], self.gen_hopping, global_data.incar.neighbor)
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
                info = {'type': 'LDOS', 'energy_range': f"{E.min()}, {E.max()}", 'DOS_num': f"{global_data.incar.finite_DOS_num}"}
                IO.save_band(global_data.incar.finite_DOS_file, LDOS, klist, info)
            
            fig, ax = plt.subplots()
            K, E = np.meshgrid(klist, energy_range)
            pcm = ax.pcolormesh(K, E, LDOS.T, shading='auto', cmap='jet')
            fig.colorbar(pcm, ax=ax)
            
            plt.title("LDOS", fontsize=14)
            plt.xlabel("k", fontsize=12)
            plt.ylabel("E", fontsize=12)
            plt.tight_layout()
            
            if global_data.incar.finite_DOS_figure.lower() != 'false':
                plt.savefig(global_data.incar.finite_DOS_figure, dpi=300, bbox_inches='tight')
                Logger.info(f"figure successfully saved to {global_data.incar.finite_DOS_figure}")
                plt.close(fig)
