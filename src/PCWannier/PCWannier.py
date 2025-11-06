import os

from copy import copy

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .Log import Logger
from .IO import IO
from .Interpolator import Interpolator2D
from .Utils import global_data
from .Utils import WannierTools, FieldData, StateCollection
from .Timer import Timer, timer
from .IncarParser import IncarParser, EnergyWindow

from .Symmetry import Symmetry, Orthogonalizer, RotationOverlap, SymmetryAdapter

from . import MeshData
from . import MSet
from . import StateInitializer
from . import Gradient
from . import TBAModal
from . import Topo
from . import Fatband

class PCWannier:
    def __init__(self):
        self.wanniers: dict = {}

    @timer("PCWannier run - ")
    def run(self, args):
        self.logger = Logger(args.log)

        Logger.info('=========  PCWannier v0.1.1  =========')

        global_data.threads = args.threads
        nb.set_num_threads(global_data.threads)
        Logger.info(f"Running with {args.threads} threads")

        if args.fatband:
            Logger.info("Starting fatband calculation")
            fatband = Fatband.Fatband()
            fatband.parser(args.input)
            fatband.run()
            return


        self._parse_input(args)

        if args.cache:
            global_data.incar.use_cached_data = ['U', 'V', 'M', 'O', 'A']
        
        self._load_data()
        self._prepare_state_collection()

        if args.base:
            StateInitializer.StateBases.plot_all()
            return

        self._initialize_states()
        self._optimize_gradient()
        self._generate_output()

        self._topology_calculation()

        self._handle_interpolation(args.interp, args.interp_wannier, args.interp_epsilon)

    def _parse_input(self, args):
        parser = IncarParser(args.input)
        wtools = WannierTools()
        wtools.set_incar(parser.parse_file())
        wtools.preprocess()
        Logger.info(global_data.incar)

    def _load_data(self):
        if global_data.incar.dataset_type.lower() == "comsol":
            self.mesh = MeshData.load_comsol_mesh(global_data.incar.mesh_file)
            self.raw_data = MeshData.load_comsol_data(global_data.incar.dataset_file)
            self.epsilon = MeshData.load_comsol_data(global_data.incar.dielectric_file)
            self.epsilon.value_matrix = self.epsilon.value_matrix.flatten()
            if global_data.incar.E_file.lower() != 'false':
                self.E_raw_data = MeshData.load_comsol_data(global_data.incar.E_file)
        else:
            self._unsupported_dataset()

    def _unsupported_dataset(self):
        dtype = global_data.incar.dataset_type
        Logger.error(f"Don't support {dtype} type dataset")
        raise
    
    def _prepare_state_collection(self):
        global_data.push_state_collection(StateCollection("psi", self.mesh, global_data.incar.kdim))
        self._handle_energy_data()

        idxs, _ = MeshData.match_data_to_mesh(self.mesh, self.raw_data)
        self.raw_data.value_matrix = self.raw_data.value_matrix[idxs]
        
        idxs, _ = MeshData.match_data_to_mesh(self.mesh, self.epsilon)
        MeshData.distribute_data(self.mesh, self.raw_data)
        
        global_data.state_collection.epsilon = self.epsilon.value_matrix[idxs].flatten()
        global_data.state_collection.turn_to_Bloch()

        _, need_orth = global_data.state_collection.check_orthogonality()
        
        if need_orth:
            Logger.warning("Need to orthogonalize states")
            global_data.state_collection.orthogonalize()

            _, need_orth = global_data.state_collection.check_orthogonality()
            if need_orth:
                Logger.error("Orthogonalization failed")
                raise
        Logger.info("orthogonality check passed")

        global_data.state_collection.extention(global_data.incar.extension)

    def _handle_energy_data(self):
        gd = global_data.incar
        bw = gd.band_window
        kdim = int(gd.kdim)

        Nk = [
            len(gd.k_points[0]) if kdim >= 1 else 1,
            len(gd.k_points[1]) if kdim >= 2 else 1,
            len(gd.k_points[2]) if kdim >= 3 else 1,
        ]
        Nk1, Nk2, Nk3 = Nk

        raw = self.E_raw_data.value_matrix[0]
        n_total = raw.size
        n_k = Nk1 * Nk2 * Nk3
        if n_total % n_k != 0:
            Logger.error(f"[energy] size mismatch: total={n_total}, Nk product={n_k}. Check k_points/kdim/dataset_order.")
            raise
        nbands = n_total // n_k

        order = list(gd.dataset_order)
        size_map = {'k1': Nk1, 'k2': Nk2, 'k3': Nk3, 'E': nbands}

        try:
            shape_in = tuple(size_map[d] for d in order)
        except KeyError as e:
            Logger.error(f"dataset_order contains unknown dimension: {e}. Allowed values are 'k1', 'k2', 'k3', 'E'.")
            raise

        t_all = raw.reshape(shape_in, order='C')

        canon = ('k1', 'k2', 'k3', 'E')
        axes = [order.index(d) for d in canon if d in order]
        t_reordered = np.transpose(t_all, axes=axes)

        energy_matrix = t_reordered.reshape(Nk1, Nk2, Nk3, nbands)

        if getattr(gd, 'hermitian', False):
            energy_matrix = np.real(energy_matrix)

        if isinstance(bw, EnergyWindow):
            emin, emax = bw.emin, bw.emax

            fields = [[[[] for _ in range(Nk3)] for _ in range(Nk2)] for _ in range(Nk1)]
            idx_fields = [[[[] for _ in range(Nk3)] for _ in range(Nk2)] for _ in range(Nk1)]

            for i in range(Nk1):
                for j in range(Nk2):
                    for k in range(Nk3):
                        eline = energy_matrix[i, j, k, :]
                        sel = np.where((eline >= emin) & (eline <= emax))[0]
                        fields[i][j][k] = eline[sel].tolist()
                        idx_fields[i][j][k] = sel.tolist()

            self.E_raw_data.energy_matrix = energy_matrix
            global_data.energy_matrix = energy_matrix
            global_data.state_collection.E = fields
            global_data.state_collection.E_idx = idx_fields

            if gd.inner_window is not False:
                ibw = gd.inner_window
                if isinstance(ibw, EnergyWindow):
                    iemin, iemax = ibw.emin, ibw.emax
                    inner_idx_fields = [[[[] for _ in range(Nk3)] for _ in range(Nk2)] for _ in range(Nk1)]
                    for i in range(Nk1):
                        for j in range(Nk2):
                            for k in range(Nk3):
                                eline = energy_matrix[i, j, k, :]
                                isel = np.where((eline >= iemin) & (eline <= iemax))[0]
                                inner_idx_fields[i][j][k] = isel.tolist()
                    global_data.state_collection.inner_E_idx = inner_idx_fields
                else:
                    ibw_idx = np.asarray(ibw, dtype=int).tolist()
                    global_data.state_collection.inner_E_idx = [
                        [[ibw_idx.copy() for _ in range(Nk3)] for _ in range(Nk2)]
                        for _ in range(Nk1)
                    ]

        else:
            bw_idx = np.asarray(gd.band_window, dtype=int)
            E_out = energy_matrix[..., bw_idx]

            if getattr(gd, 'hermitian', False):
                E_out = np.real(E_out)

            global_data.state_collection.E = E_out
            bw_idx_list = bw_idx.tolist()
            global_data.state_collection.E_idx = [
                [[bw_idx_list.copy() for _ in range(Nk3)] for _ in range(Nk2)]
                for _ in range(Nk1)
            ]



    def _symmetry(self):
        Logger.info("Try to orthogonalize eigenvalues")
        self.orthogonalizer = Orthogonalizer()
        Transform = self.orthogonalizer.build_orthogonalization_matrix(global_data.state_collection)
        self.orthogonalizer.save_as(global_data.incar.O_file)
        global_data.state_collection.set_transform(Transform)

        seen = {}
        number = []
        pos = []

        for i, proj in enumerate(global_data.incar.projections):
            number.append(seen.setdefault(proj['atom'], i))
            pos.append(proj['frac_position'])

        self.symmetry = Symmetry(np.array(global_data.incar.real_lattice_vectors), np.array(pos), number)

        self.rotation_overlap = RotationOverlap(global_data.incar.real_lattice_vectors, global_data.state_collection)

        self.symmetry_adapter = SymmetryAdapter(self.symmetry, self.rotation_overlap)
        self.symmetry_adapter.initial_symmetrization(global_data.state_initializer.matV)
        # R, v = self.symmetry.get_Rv(1)
        # res = self.rotation_overlap.build_d_matrix(R, v)
        # print(f"Rotation overlap matrix: {res}")

    def _initialize_states(self):
        global_data.push_m_set(MSet.MSet())
        global_data.m_set.init_M0(global_data.state_collection)
        
        global_data.push_state_initializer(StateInitializer.StateInitializer())
        global_data.state_initializer.iter(global_data.incar.err_diff, global_data.incar.max_iter)
        
        if global_data.incar.symmetry:
            self._symmetry()

    def _optimize_gradient(self):
        global_data.push_gradient(Gradient.Gradient())
        global_data.gradient.iter(global_data.incar.err_diff, global_data.incar.max_iter)
        r = global_data.gradient.generateRn()
        Logger.info(f"Gradient optimization completed, r = {r}")
        if global_data.incar.w_center is not False:
            Logger.info(f"Set Wannier center to {global_data.incar.w_center}")
            for i in range(10):
                self.c_phase = global_data.gradient.set_center(global_data.incar.w_center)
            # global_data.gradient.iter(global_data.incar.err_diff, global_data.incar.max_iter)
            r = global_data.gradient.generateRn()
            Logger.info(f"Gradient optimization completed, r = {r}")
        

    def _generate_output(self):
        self.gen_wannier()
        
        if not global_data.incar.M_in:
            global_data.m_set.save_as(global_data.incar.M_file)
            
        global_data.state_initializer.save_as(global_data.incar.V_file, global_data.incar.A_file)
        global_data.gradient.save_as(global_data.incar.U_file)

        self.TBA = TBAModal.TBAModal()
        
        if global_data.incar.hopping_file.lower() != "false":
            self.TBA.save_hoppings(global_data.incar.hopping_file)

        self.TBA.gen_hs_bands()

        if global_data.incar.eff_k is not False:
            Logger.info(f"Calculate effective Hamiltonian at k = {global_data.incar.eff_k}")
            self.TBA.effective_Hamiltonian()
        if global_data.incar.finite is not False:
            Logger.info(f"Calculate finite system band structure")
            self.TBA.calc_finite()
    
    def _topology_calculation(self):
        self.TBA.gen_band()

        Logger.info(f"Topology calculation")
        for gid, g in enumerate(self.TBA.groups):
            self.Topo = Topo.Topo2D()
            self.Topo.construct_parallel_transport(self.TBA.eigvecs[:, :, :, g[0]:(g[-1] + 1)])
            if global_data.incar.hybrid_Wilson_loop:
                Z2_0 = self.Topo.save_hybrid_Wilson_loop(os.path.join(global_data.incar.topo_output, f"Hybrid_Wilson_Loop-{gid}-d-0.png"), self.TBA.eigvecs[:, :, g[0]:(g[-1] + 1), g[0]:(g[-1] + 1)], direction=0)
                Z2_1 = self.Topo.save_hybrid_Wilson_loop(os.path.join(global_data.incar.topo_output, f"Hybrid_Wilson_Loop-{gid}-d-1.png"), self.TBA.eigvecs[:, :, g[0]:(g[-1] + 1), g[0]:(g[-1] + 1)], direction=1)
                Logger.info(f"Direction 0: Z2 for group {gid}-bands {g} = {Z2_0}")
                Logger.info(f"Direction 0: Z2 for group {gid}-bands {g} = {Z2_1}")
            
            if global_data.incar.Chern_number:
                C = self.Topo.Chern_number(self.TBA.eigvecs[:, :, :, g[0]:(g[-1] + 1)], os.path.join(global_data.incar.topo_output, f"Chern_Number-{gid}.png"))
                Logger.info(f"Chern number for group {gid}-bands {g} = {C}")
        self.Topo.construct_parallel_transport(self.TBA.eigvecs)
        if global_data.incar.Chern_number:
            C = self.Topo.Chern_number(self.TBA.eigvecs, os.path.join(global_data.incar.topo_output, f"Chern_Number-all.png"))
            Logger.info(f"Chern number for all bands = {C}")

    def _handle_interpolation(self, interp_path: str, interp_wannier: str, interp_epsilon: str):
        if interp_path is None:
            return
        if os.path.exists(interp_path):
            Logger.info(f"Found existing interpolation file at {interp_path}, loading...")
            mesh_point = IO.load_mesh_points(interp_path)
            vals = []
            for wannier in self.wanniers[(0, 0)]:
                interp_real = Interpolator2D(global_data.state_collection.extention_mesh.vertices, global_data.state_collection.extention_mesh.elements, np.real(wannier))
                interp_imag = Interpolator2D(global_data.state_collection.extention_mesh.vertices, global_data.state_collection.extention_mesh.elements, np.imag(wannier))
                real = interp_real.batch_evaluate(mesh_point)
                imag = interp_imag.batch_evaluate(mesh_point)
                vals.append(real)
                vals.append(imag)
            if interp_wannier is None:
                IO.save_points_with_values(f"{os.path.splitext(interp_path)[0]}-interp-wannier.txt", mesh_point, vals)
            else:
                if not os.path.exists(os.path.dirname(interp_path)):
                    os.makedirs(os.path.dirname(interp_path))
                IO.save_points_with_values(interp_wannier, mesh_point, vals)

            vals = []
            interp = Interpolator2D(global_data.state_collection.extention_mesh.vertices, global_data.state_collection.extention_mesh.elements, global_data.state_collection.extention_epsilon)
            vals.append(interp.batch_evaluate(mesh_point))
            if interp_epsilon is None:
                IO.save_points_with_values(f"{os.path.splitext(interp_path)[0]}-interp-epsilon.txt", mesh_point, vals)
            else:
                if not os.path.exists(os.path.dirname(interp_path)):
                    os.makedirs(os.path.dirname(interp_path))
                IO.save_points_with_values(interp_epsilon, mesh_point, vals)

    @timer("Generate Wannier - ")
    def gen_wannier(self, r: list=[0, 0, 0], out: bool=True):
        a0 = global_data.incar.lattice_const

        avec = np.asarray(global_data.incar.real_lattice_vectors)
        D = avec.shape[1]
        r_use = list(r) + [0, 0, 0]
        r_cart = np.zeros(D)
        for a in range(global_data.incar.kdim):
            r_cart += r_use[a] * avec[a, :]
        r_cart *= a0

        if out:
            Logger.info(f"Generating Wannier Functions - r = {tuple(r_use[:global_data.incar.kdim])}")

        B = global_data.incar.band_calc_num
        E_idx = global_data.state_collection.E_idx

        Nv = global_data.state_collection.extention_mesh.vertices.shape[0]

        Wsum = np.zeros((Nv, B), dtype=np.complex128)
        if global_data.incar.disable_orth:
            if out:
                Logger.info("Disable orthogonalization as requested")
            T = global_data.state_collection.get_transform(True)
        else:
            T = global_data.state_collection.get_transform()
        
        sign = -1 if global_data.incar.dataset_type.lower() == 'comsol' else 1
        for i, j, k in global_data.state_collection.k_indices():
            phase_vec = global_data.state_collection.get_extention_phase(i, j, k)
            k_vec = WannierTools.get_kxyz([i, j, k])[:D]
            phase_scalar = np.exp(1j * (-(sign) * np.dot(k_vec, r_cart)))
            P = phase_vec * phase_scalar

            fields_ijk = [global_data.state_collection.get_extention_field(i, j, k, m) for m in range(len(E_idx[i][j][k]))]
            E = np.column_stack(fields_ijk).astype(np.complex128, copy=False)
            C = (T[i][j][k] @ global_data.state_initializer.matV[i][j][k] @ global_data.gradient.U[i][j][k])

            U_ij = E @ C

            Wsum += U_ij * P[:, None]

        Wsum /= np.sqrt(float(global_data.state_collection.get_k_num()))

        self.wanniers[tuple(r_use[:global_data.incar.kdim])] = [Wsum[:, n] for n in range(B)]

        if out:
            Fnorm = (np.abs(Wsum) ** 2) * global_data.state_collection.extention_epsilon[:, None]
            fd = FieldData('wannier', global_data.state_collection.extention_mesh, Fnorm.astype(np.complex128, copy=False))
            norms = WannierTools.integrate_over_mesh(fd, chunk_size=2048)
            if norms.ndim == 0:
                norms = np.array([norms])
            for n in range(B):
                norm = norms[n]
                Logger.info(f"Check wannier function norm = {norm}")
                if not np.isclose(np.abs(norm), 1.0, atol=1e-3):
                    warn = (f"Normalization ({np.abs(norm)}) not equal to 1 in Wannier State - {n}, "
                            f"err = {np.abs(1 - np.abs(norm))}")
                    Logger.warning(warn)
                if global_data.incar.wannier_figures.lower() != "false":
                    fdn = FieldData('wannier', global_data.state_collection.extention_mesh, self.wanniers[(r[0], r[1])][n])
                    fdn.save_fig(global_data.incar.wannier_figures + f"/wannier-{n}-real.png", real=True)
                    fdn.save_fig(global_data.incar.wannier_figures + f"/wannier-{n}-imag.png", real=False)
    
    def get_wannier(self, r: list = [0, 0, 0]):
        kdim = global_data.incar.kdim
        r_use = tuple((list(r) + [0, 0, 0])[:kdim])

        if r_use not in self.wanniers:
            self.gen_wannier(list(r_use) + [0] * (3 - len(r_use)), out=False)
        return self.wanniers[r_use]

    
