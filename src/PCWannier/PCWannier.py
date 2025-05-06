import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .Log import Logger
from .IO import IO
from .Utils import global_data
from .Utils import WannierTools, FieldData
from .Timer import Timer, timer
from .IncarParser import IncarParser
import PCWannier.MeshData as MeshData
import PCWannier.MSet as MSet
import PCWannier.StateInitializer as StateInitializer
import PCWannier.Gradient as Gradient

class PCWannier:
    def __init__(self):
        self.wanniers: list = []

    def run(self, args):
        time = Timer("PCWannier run - ")
        self.logger = Logger(args.log)

        Logger.info('=========  PCWannier v0.1.0  =========')

        global_data.threads = args.threads
        Logger.info(f"Running with {args.threads} threads")

        parser = IncarParser(args.input)
        wtools = WannierTools()
        wtools.set_incar(parser.parse_file())
        wtools.preprocess()
        Logger.info(global_data.incar)

        if args.base:
            StateInitializer.StateBases.plot_all()
            return
        
        if global_data.incar.dataset_type.lower() == "comsol":
            mesh = MeshData.load_comsol_mesh(global_data.incar.mesh_file)
            raw_data = MeshData.load_comsol_data(global_data.incar.dataset_file)
            epsilon = MeshData.load_comsol_data(global_data.incar.dielectric_file)
            if global_data.incar.E_file.lower() != 'false':
                E_raw_data = MeshData.load_comsol_data(global_data.incar.E_file)
                    
        else:
            error_msg = f"Don't support {global_data.incar.dataset_type.lower()} type dataset"
            Logger.error(error_msg)
            raise ValueError(error_msg)

        idxs, dists = MeshData.match_data_to_mesh(mesh, raw_data)
        raw_data.value_matrix = raw_data.value_matrix[idxs]

        idxs, dists = MeshData.match_data_to_mesh(mesh, epsilon)

        MeshData.distribute_data(mesh, raw_data)
        global_data.state_collection.epsilon = epsilon.value_matrix[idxs].flatten()

        global_data.state_collection.normalize()

        global_data.state_collection.turn_to_Bloch()

        if global_data.incar.E_is_real:
            sizes = {"k1": len(global_data.incar.k_points[0]),"k2": len(global_data.incar.k_points[1]),"E": len(global_data.incar.band_window)}
            shape = tuple(sizes[dim] for dim in global_data.incar.dataset_order)
            t_ = E_raw_data.value_matrix[0].reshape((shape[0], shape[1], -1), order='C')[:,:, global_data.incar.band_window]
            desired_order = ["k1", "k2", "E"]
            indices = [global_data.incar.dataset_order.index(dim) for dim in desired_order]
            global_data.state_collection.E = np.real(np.transpose(t_, axes=(indices[0], indices[1], indices[2])))
        else:
            sizes = {"k1": len(global_data.incar.k_points[0]),"k2": len(global_data.incar.k_points[1]),"E": len(global_data.incar.band_window)}
            shape = tuple(sizes[dim] for dim in global_data.incar.dataset_order)
            t_ = E_raw_data.value_matrix[0].reshape((shape[0], shape[1], -1), order='C')[:,:, global_data.incar.band_window]
            desired_order = ["k1", "k2", "E"]
            indices = [global_data.incar.dataset_order.index(dim) for dim in desired_order]
            global_data.state_collection.E = np.transpose(t_, axes=(indices[0], indices[1], indices[2]))

        global_data.state_collection.extention(global_data.incar.extension)

        global_data.push_m_set(MSet.MSet())
        global_data.m_set.init_M0(global_data.state_collection)

        global_data.push_state_initializer(StateInitializer.StateInitializer())
        global_data.state_initializer.iter(global_data.incar.err_diff, global_data.incar.max_iter)

        global_data.push_gradient(Gradient.Gradient())
        global_data.gradient.iter(global_data.incar.err_diff, global_data.incar.max_iter)

        self.gen_wannier()

        if not global_data.incar.M_in:
            global_data.m_set.save_as(global_data.incar.M_file)
        
        global_data.state_initializer.save_as(global_data.incar.V_file)
        global_data.gradient.save_as(global_data.incar.U_file)

        if global_data.incar.hopping_file.lower() != "false":
            self.save_hoppings(global_data.incar.hopping_file)

        self.gen_band()

    @timer("Generate Wannier - ")
    def gen_wannier(self, r: list=[0, 0]):
        r_ = [0, 0]
        r_[0] = (r[0] * global_data.incar.real_lattice_vectors[0][0] * global_data.incar.lattice_const + r[1] * global_data.incar.real_lattice_vectors[0][1] * global_data.incar.lattice_const)
        r_[1] = (r[0] * global_data.incar.real_lattice_vectors[1][0] * global_data.incar.lattice_const + r[1] * global_data.incar.real_lattice_vectors[1][1] * global_data.incar.lattice_const)
        Logger.info(f"Generating Wannier Functions - r = ({r[0]}, {r[1]})")

        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.band_window), global_data.incar.band_calc_num]

        extention_field = np.array([[[None for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])
        for n in range(shape[2]):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    extention_field[i, j, n] = global_data.state_collection.get_extention_field(i, j, n)
        ubloch = np.array([[[None for _ in range(shape[3])] for _ in range(shape[1])] for _ in range(shape[0])])
        for n in range(shape[3]):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    t_ = global_data.state_collection.get_zero_extension_field()
                    mV = global_data.state_initializer.matV[i][j]
                    mU = global_data.gradient.U[i][j]
                    for m in range(shape[2]):
                        t_ += (mV @ mU)[m, n] * extention_field[i, j, m]
                    ubloch[i, j, n] = t_
        del extention_field

        phase_ = np.array([[None for _ in range(shape[1])] for _ in range(shape[0])])
        for i in range(shape[0]):
            for j in range(shape[1]):
                phase_[i, j] = global_data.state_collection.get_extention_phase(i, j)
        wannier = [global_data.state_collection.get_zero_extension_field() for _ in range(shape[2])]
        for n in range(shape[3]):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    kx, ky = WannierTools.get_kx_ky([i, j])
                    if global_data.incar.dataset_type.lower() == 'comsol':
                        sign = -1
                    phase = phase_[i, j] * np.exp(1j * np.dot(-1 * sign * np.array([kx, ky]), r_))
                    wannier[n] += phase * ubloch[i, j, n]
            wannier[n] /= np.sqrt(shape[0] * shape[1])
            norm = WannierTools.integrate_over_mesh(FieldData('wannier', global_data.state_collection.extention_mesh, np.abs(wannier[n]) ** 2 * global_data.state_collection.extention_epsilon))
            Logger.info(f"Check wannier function norm = {norm}")
            if not np.isclose(np.abs(norm), 1.0, atol=1e-3):
                warn = f"Normalization ({np.abs(norm)}) not equal to 1 in Wannier State - {n}, err = {np.abs(1 - np.abs(norm))}"
                Logger.warning(warn)
            if global_data.incar.wannier_figures.lower() != "false":
                fd = FieldData('wannier', global_data.state_collection.extention_mesh, wannier[n])
                fd.save_fig(global_data.incar.wannier_figures + f"/wannier-{n}.png")
    
    @timer("Generate hoppings - ")
    def gen_hopping(self, r: list=[0, 0]):
        r_ = [0, 0]
        r_[0] = (r[0] * global_data.incar.real_lattice_vectors[0][0] * global_data.incar.lattice_const + r[1] * global_data.incar.real_lattice_vectors[0][1] * global_data.incar.lattice_const)
        r_[1] = (r[0] * global_data.incar.real_lattice_vectors[1][0] * global_data.incar.lattice_const + r[1] * global_data.incar.real_lattice_vectors[1][1] * global_data.incar.lattice_const)
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

    @timer("Generate Band Structure - ")
    def gen_band(self):
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

        hoppings = []
        for p in global_data.incar.neighbor:
            hoppings.append(self.gen_hopping(p))
        
        E = []
        for k_ in k_list:
            Hi = np.zeros((global_data.incar.band_calc_num, global_data.incar.band_calc_num), dtype=complex)
            kx = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const
            ky = k_[0] * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const + k_[1] * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const
            k = [kx, ky]
            for i in range(len(global_data.incar.neighbor)):
                r_ = [0, 0]
                r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[0][1] * global_data.incar.lattice_const)
                r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[1][0] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1] * global_data.incar.lattice_const)
                Hi += hoppings[i] * np.exp(1j * np.dot(k, r_))
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
                    r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[0][1] * global_data.incar.lattice_const)
                    r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[1][0] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1] * global_data.incar.lattice_const)
                    Hi += hoppings[i] * np.exp(1j * np.dot(k, r_))
                Hi = Hi + np.conj(Hi).T
                H = H0 + Hi
                for i, e in enumerate(E_list):
                    G = np.linalg.inv(H - (e - 1j * global_data.incar.DOS_eps) * np.eye(H.shape[0], H.shape[1]))
                    DOS[0, i] += np.sum(np.real(-1 / np.pi * np.imag(np.diag(G))))
            self.plot_band_and_dos(K, E, high_sym_points, E_list, DOS, save_path=global_data.incar.band_figure)
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
                    r_[0] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[0][0] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[0][1] * global_data.incar.lattice_const)
                    r_[1] = (global_data.incar.neighbor[i][0] * global_data.incar.real_lattice_vectors[1][0] * global_data.incar.lattice_const + global_data.incar.neighbor[i][1] * global_data.incar.real_lattice_vectors[1][1] * global_data.incar.lattice_const)
                    Hi += hoppings[i] * np.exp(1j * np.dot(k, r_))
                Hi = Hi + np.conj(Hi).T
                H = H0 + Hi
                for i, e in enumerate(E_list):
                    G = np.linalg.inv(H - (e - 1j * global_data.incar.DOS_eps) * np.eye(H.shape[0], H.shape[1]))
                    DOS[:, i] += np.real(-1 / np.pi * np.imag(np.diag(G)))
            self.plot_band_and_dos(K, E, high_sym_points, E_list, DOS, save_path=global_data.incar.band_figure)
            Logger.info(f"figure successfully saved to {global_data.incar.band_figure}")

    @staticmethod
    def plot_band_and_dos(k_path: np.ndarray,
                        bands: np.ndarray,
                        high_sym_points: list,
                        dos_energy: np.ndarray,
                        dos_components: list,
                        dos_labels: list = None,
                        dos_colors: list = None,
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
            ax_band.plot(k_path, bands[:, band], color='blue')

        for pos in [p[1] for p in high_sym_points]:
            ax_band.axvline(x=pos, color='black', linestyle='--', linewidth=0.5)
        ax_band.set_xticks([p[1] for p in high_sym_points])
        ax_band.set_xticklabels([p[0] for p in high_sym_points])
        ax_band.set_xlim(0, k_path[-1])
        ax_band.set_title("Band Structure", fontsize=14)
        ax_band.set_ylabel("E", fontsize=12)

        ax_dos = fig.add_subplot(gs[1], sharey=ax_band)
        for i, dos in enumerate(dos_components):
            ax_dos.plot(dos, dos_energy, color=dos_colors[i], label=dos_labels[i])
            ax_dos.fill_betweenx(dos_energy, 0, dos, color=dos_colors[i], alpha=alpha)
        ax_dos.set_xlabel('DOS')
        ax_dos.tick_params(labelleft=False)
        ax_dos.grid(True)
        ax_dos.legend(loc='upper right', fontsize='small')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
