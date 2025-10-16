import os

from copy import copy

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .Log import Logger
from .IO import IO
from .Interpolator import Interpolator
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
        self.wanniers: list = []

    @timer("PCWannier run - ")
    def run(self, args):
        self.logger = Logger(args.log)

        Logger.info('=========  PCWannier v0.1.0  =========')

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
        raise ValueError(f"Don't support {dtype} type dataset")
    
    def _prepare_state_collection(self):
        global_data.push_state_collection(StateCollection("psi", self.mesh))
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
                raise ValueError("Orthogonalization failed")
        Logger.info("orthogonality check passed")

        global_data.state_collection.extention(global_data.incar.extension)

    def _handle_energy_data(self):
        bw = global_data.incar.band_window
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])

        if isinstance(bw, EnergyWindow):
            d0, d1 = global_data.incar.dataset_order[0], global_data.incar.dataset_order[1]
            if set((d0, d1)) != {"k1", "k2"}:
                Logger.error("Energy-window mode expects dataset_order first two dims to be k1 and k2.")
                raise RuntimeError("Energy-window mode expects dataset_order first two dims to be k1 and k2.")
            n0 = len(global_data.incar.k_points[0]) if d0 == "k1" else len(global_data.incar.k_points[1])
            n1 = len(global_data.incar.k_points[0]) if d1 == "k1" else len(global_data.incar.k_points[1])

            t_all = self.E_raw_data.value_matrix[0].reshape((n0, n1, -1), order='C')

            indices = [global_data.incar.dataset_order.index(dim) for dim in ["k1", "k2", "E"]]
            transposed_all = np.transpose(t_all, axes=indices)

            energy_matrix = np.ascontiguousarray(np.real(transposed_all) if global_data.incar.E_is_real else transposed_all)
            setattr(self.E_raw_data, "energy_matrix", energy_matrix)
            global_data.energy_matrix = energy_matrix

            fields = [[[] for _ in range(k2_sz)] for _ in range(k1_sz)]
            idx_fields = [[[] for _ in range(k2_sz)] for _ in range(k1_sz)]

            for i in range(k1_sz):
                for j in range(k2_sz):
                    eline = energy_matrix[i, j, :]
                    sel = np.where((eline >= bw.emin) & (eline <= bw.emax))[0]
                    fields[i][j] = eline[sel].tolist()
                    idx_fields[i][j] = sel.tolist()

            global_data.state_collection.E = fields
            global_data.state_collection.E_idx = idx_fields
            if global_data.incar.inner_window is not False:
                ibw = global_data.incar.inner_window
                if isinstance(ibw, EnergyWindow):
                    inner_idx_fields = [[[] for _ in range(k2_sz)] for _ in range(k1_sz)]
                    for i in range(k1_sz):
                        for j in range(k2_sz):
                            isel = np.where((eline >= ibw.emin) & (eline <= ibw.emax))[0]
                            inner_idx_fields[i][j] = isel.tolist()
                    global_data.state_collection.inner_E_idx = inner_idx_fields
                else:
                    ibw_idx = np.asarray(ibw, dtype=int).tolist()
                    global_data.state_collection.inner_E_idx = [[ibw_idx.copy() for _ in range(len(global_data.incar.k_points[1]))] for _ in range(len(global_data.incar.k_points[0]))]
        else:
            sizes = {
                "k1": len(global_data.incar.k_points[0]),
                "k2": len(global_data.incar.k_points[1]),
                "E": len(global_data.incar.band_window)
            }
            shape = tuple(sizes[dim] for dim in global_data.incar.dataset_order)
            t_ = self.E_raw_data.value_matrix[0].reshape((shape[0], shape[1], -1), order='C')[:,:, global_data.incar.band_window]
            indices = [global_data.incar.dataset_order.index(dim) for dim in ["k1", "k2", "E"]]
            transposed = np.transpose(t_, axes=indices)
            global_data.state_collection.E = np.real(transposed) if global_data.incar.E_is_real else transposed

            bw_idx = np.asarray(global_data.incar.band_window, dtype=int).tolist()
            global_data.state_collection.E_idx = [[bw_idx.copy() for _ in range(len(global_data.incar.k_points[1]))] for _ in range(len(global_data.incar.k_points[0]))]


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
    
    def _topology_calculation(self):
        self.TBA.gen_band()

        Logger.info(f"Topology calculation")
        for gid, g in enumerate(self.TBA.groups):
            self.Topo = Topo.Topo()
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
            for wannier in self.wanniers:
                interp_real = Interpolator(global_data.state_collection.extention_mesh.vertices, global_data.state_collection.extention_mesh.elements, np.real(wannier))
                interp_imag = Interpolator(global_data.state_collection.extention_mesh.vertices, global_data.state_collection.extention_mesh.elements, np.imag(wannier))
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
            interp = Interpolator(global_data.state_collection.extention_mesh.vertices, global_data.state_collection.extention_mesh.elements, global_data.state_collection.extention_epsilon)
            vals.append(interp.batch_evaluate(mesh_point))
            if interp_epsilon is None:
                IO.save_points_with_values(f"{os.path.splitext(interp_path)[0]}-interp-epsilon.txt", mesh_point, vals)
            else:
                if not os.path.exists(os.path.dirname(interp_path)):
                    os.makedirs(os.path.dirname(interp_path))
                IO.save_points_with_values(interp_epsilon, mesh_point, vals)

    @timer("Generate Wannier - ")
    def gen_wannier(self, r: list=[0, 0]):
        r_ = [0, 0]
        r_[0] = (r[0] * global_data.incar.real_lattice_vectors[0][0] + r[1] * global_data.incar.real_lattice_vectors[1][0]) * global_data.incar.lattice_const
        r_[1] = (r[0] * global_data.incar.real_lattice_vectors[0][1] + r[1] * global_data.incar.real_lattice_vectors[1][1]) * global_data.incar.lattice_const
        Logger.info(f"Generating Wannier Functions - r = ({r[0]}, {r[1]})")

        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])
        B = global_data.incar.band_calc_num
        E_idx = global_data.state_collection.E_idx

        Nv = global_data.state_collection.extention_mesh.vertices.shape[0]

        Wsum = np.zeros((Nv, B), dtype=np.complex128)
        if global_data.incar.disable_orth:
            Logger.info("Disable orthogonalization as requested")
            T = global_data.state_collection.get_transform(True)
        else:
            T = global_data.state_collection.get_transform()

        for i in range(k1_sz):
            for j in range(k2_sz):
                phase_vec = global_data.state_collection.get_extention_phase(i, j)
                kx, ky = WannierTools.get_kx_ky([i, j])
                sign = -1 if global_data.incar.dataset_type.lower() == 'comsol' else 1
                phase_scalar = np.exp(1j * (-(sign) * (kx * r_[0] + ky * r_[1])))
                P = phase_vec * phase_scalar

                fields_ij = [global_data.state_collection.get_extention_field(i, j, m) for m in range(len(E_idx[i][j]))]
                E = np.column_stack(fields_ij).astype(np.complex128, copy=False)
                C = (T[i][j] @ global_data.state_initializer.matV[i][j] @ global_data.gradient.U[i][j])

                U_ij = E @ C

                Wsum += U_ij * P[:, None]

        Wsum /= np.sqrt(k1_sz * k2_sz)

        self.wanniers = [Wsum[:, n] for n in range(B)]

        Fnorm = (np.abs(Wsum) ** 2) * global_data.state_collection.extention_epsilon[:, None]
        fd = FieldData('wannier', global_data.state_collection.extention_mesh, Fnorm.astype(np.complex128, copy=False))
        norms = WannierTools.integrate_over_mesh(fd, chunk_size=2048)

        for n in range(B):
            norm = norms[n]
            Logger.info(f"Check wannier function norm = {norm}")
            if not np.isclose(np.abs(norm), 1.0, atol=1e-3):
                warn = (f"Normalization ({np.abs(norm)}) not equal to 1 in Wannier State - {n}, "
                        f"err = {np.abs(1 - np.abs(norm))}")
                Logger.warning(warn)
            if global_data.incar.wannier_figures.lower() != "false":
                fdn = FieldData('wannier', global_data.state_collection.extention_mesh, self.wanniers[n])
                fdn.save_fig(global_data.incar.wannier_figures + f"/wannier-{n}-real.png", real=True)
                fdn.save_fig(global_data.incar.wannier_figures + f"/wannier-{n}-imag.png", real=False)
            

    
