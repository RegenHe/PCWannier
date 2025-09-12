import os

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from threadpoolctl import threadpool_limits

from .Log import Logger
from .IO import IO
from .Timer import Timer, timer
from .Utils import global_data

from .Utils import WannierTools, FieldData


class TBAModal:
    def __init__(self):
        pass

    @timer("Generate hoppings - ")
    def gen_hopping(self, r: list=[0, 0]):
        r_ = [0, 0]
        r_[0] = (r[0] * global_data.incar.real_lattice_vectors[0][0] + r[1] * global_data.incar.real_lattice_vectors[1][0]) * global_data.incar.lattice_const
        r_[1] = (r[0] * global_data.incar.real_lattice_vectors[0][1] + r[1] * global_data.incar.real_lattice_vectors[1][1]) * global_data.incar.lattice_const
        Logger.info(f"Generating hoppings - r = ({r[0]}, {r[1]})")

        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.band_window), global_data.incar.band_calc_num]
        hopping = np.zeros((shape[3], shape[3]), dtype=complex)
        for i in range(shape[0]):
            for j in range(shape[1]):
                mU = global_data.state_initializer.matV[i][j] @ global_data.gradient.U[i][j]
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

        self.hoppings = []
        for p in global_data.incar.neighbor:
            self.hoppings.append(self.gen_hopping(p))
        
        E = []
        for k_ in k_list:
            Hi = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
            kx = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const
            ky = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const
            k = [kx, ky]
            for i in range(len(global_data.incar.neighbor)):
                r_ = [0, 0]
                r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][0] * global_data.incar.lattice_const)
                r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][1] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1] * global_data.incar.lattice_const)
                Hi += self.hoppings[i] * np.exp(1j * np.dot(k, r_))
            Hi = Hi + np.conj(Hi).T
            H = H0 + Hi
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
            plt.title("Band Structure", fontsize=14)
            plt.ylabel("E", fontsize=12)
            plt.tight_layout()
            plt.savefig(global_data.incar.band_figure, dpi=300, bbox_inches='tight')
            Logger.info(f"figure successfully saved to {global_data.incar.band_figure}")
        elif global_data.incar.DOS == 1:
            E_list = np.linspace(np.min(E), np.max(E), global_data.incar.DOS_num)
            DOS = np.zeros((1, global_data.incar.DOS_num), dtype=complex)
            for k_ in k_list:
                Hi = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
                kx = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const
                ky = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const
                k = [kx, ky]
                for i in range(len(global_data.incar.neighbor)):
                    r_ = [0, 0]
                    r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][0]) * global_data.incar.lattice_const
                    r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][1] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1]) * global_data.incar.lattice_const
                    Hi += self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                Hi = Hi + np.conj(Hi).T
                H = H0 + Hi
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
                kx = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const
                ky = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const
                k = [kx, ky]
                for i in range(len(global_data.incar.neighbor)):
                    r_ = [0, 0]
                    r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][0] * global_data.incar.lattice_const)
                    r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][1] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1] * global_data.incar.lattice_const)
                    Hi += self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                Hi = Hi + np.conj(Hi).T
                H = H0 + Hi
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
                kx = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const
                ky = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const
                k = [kx, ky]
                for i in range(len(global_data.incar.neighbor)):
                    r_ = [0, 0]
                    r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][0]) * global_data.incar.lattice_const
                    r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][1] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1]) * global_data.incar.lattice_const
                    Hi += self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                Hi = Hi + np.conj(Hi).T
                H = H0 + Hi
            for i in range(global_data.incar.DOS_Brillouin_mesh[0]):
                for j in range(global_data.incar.DOS_Brillouin_mesh[1]):
                    Hi = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
                    k0 = WannierTools.get_kx_ky([0, 0])
                    kx = k0[0] + i * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const / global_data.incar.DOS_Brillouin_mesh[0] + j * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const / global_data.incar.DOS_Brillouin_mesh[1]
                    ky = k0[1] + i * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const / global_data.incar.DOS_Brillouin_mesh[0] + j * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const / global_data.incar.DOS_Brillouin_mesh[1]
                    k = [kx, ky]
                    for i in range(len(global_data.incar.neighbor)):
                        r_ = [0, 0]
                        r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][0]) * global_data.incar.lattice_const
                        r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][1] + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1]) * global_data.incar.lattice_const
                        Hi += self.hoppings[i] * np.exp(1j * np.dot(k, r_))
                    Hi = Hi + np.conj(Hi).T
                    H = H0 + Hi
                    for i, e in enumerate(E_list):
                        G = np.linalg.inv(H - (e - 1j * global_data.incar.DOS_eps) * np.eye(H.shape[0], H.shape[1]))
                        DOS[0, i] += np.sum(np.real(-1 / np.pi * np.imag(np.diag(G))))
            self.plot_hs_band_dos(K, E, high_sym_points, E_list, DOS, save_path=global_data.incar.band_figure)
            Logger.info(f"figure successfully saved to {global_data.incar.band_figure}")

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

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

        ax_band = fig.add_subplot(gs[0])
        for band in range(bands.shape[1]):
            ax_band.plot(k_path, np.real(bands[:, band]), color='blue')

        for pos in [p[1] for p in high_sym_points]:
            ax_band.axvline(x=pos, color='black', linestyle='--', linewidth=0.5)
        ax_band.set_xticks([p[1] for p in high_sym_points])
        ax_band.set_xticklabels([p[0] for p in high_sym_points])
        ax_band.set_xlim(0, k_path[-1])
        ax_band.set_title("Band Structure", fontsize=14)
        ax_band.set_ylabel("E", fontsize=12)

        ax_dos = fig.add_subplot(gs[1], sharey=ax_band)
        for i, dos in enumerate(dos_components):
            ax_dos.plot(dos, np.real(dos_energy), color=dos_colors[i], label=dos_labels[i])
            ax_dos.fill_betweenx(np.real(dos_energy), 0, dos, color=dos_colors[i], alpha=alpha)
        ax_dos.set_xlabel(dos_title)
        ax_dos.tick_params(labelleft=False)
        ax_dos.grid(True)
        ax_dos.legend(loc='upper right', fontsize='small')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

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

        with threadpool_limits(limits=global_data.threads):
            self.eigvals, self.eigvecs = np.linalg.eigh(Hk)
        del Hk, Hi, H0, phase

        self.groups = self.group_bands(self.eigvals, delta_rel=1e-2)
        for gid, g in enumerate(self.groups):
            Logger.info(f"group {gid}: bands {g}")

    @staticmethod
    def group_bands(E: np.ndarray, delta_rel=1e-2, delta_abs=None):
        E_glb_min = E.min()
        E_glb_max = E.max()
        span = E_glb_max - E_glb_min

        delta = delta_rel * span
        if delta_abs is not None:
            delta = max(delta, delta_abs) 
        E_min = E.min(axis=(0, 1))
        E_max = E.max(axis=(0, 1))

        overlap = (E_max[:, None] >= E_min[None, :] - delta) & (E_min[:, None] <= E_max[None, :] + delta)

        graph = scipy.sparse.csr_matrix(overlap)
        n_comp, labels = scipy.sparse.csgraph.connected_components(graph)

        groups = [[] for _ in range(n_comp)]
        for band, lab in enumerate(labels):
            groups[lab].append(band)

        groups = [sorted(g) for g in groups]
        return groups