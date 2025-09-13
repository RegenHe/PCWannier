import os

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .Log import Logger
from .IO import IO
from .Interpolator import Interpolator
from .Utils import global_data
from .Utils import WannierTools, FieldData
from .Timer import Timer, timer
from .IncarParser import IncarParser

from .Symmetry import Symmetry, Orthogonalizer, RotationOverlap, SymmetryAdapter

from . import MeshData
from . import MSet
from . import StateInitializer
from . import Gradient
from . import TBAModal
from . import Topo

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

        self._parse_input(args)
        
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
        idxs, _ = MeshData.match_data_to_mesh(self.mesh, self.raw_data)
        self.raw_data.value_matrix = self.raw_data.value_matrix[idxs]
        
        idxs, _ = MeshData.match_data_to_mesh(self.mesh, self.epsilon)
        MeshData.distribute_data(self.mesh, self.raw_data)
        
        global_data.state_collection.epsilon = self.epsilon.value_matrix[idxs].flatten()
        global_data.state_collection.normalize()
        global_data.state_collection.turn_to_Bloch()
        self._handle_energy_data()

    def _handle_energy_data(self):
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

        global_data.state_collection.extention(global_data.incar.extension)

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

    def _handle_interpolation(self, interp_path: str, interp_wannier: str, interp_epsilon: str):
        if interp_path is None:
            return
        if os.path.exists(interp_path):
            Logger.info(f"Found existing interpolation file at {interp_path}, loading...")
            mesh_point = IO.load_mesh_points(interp_path)
            vals = []
            for wannier in self.wanniers:
                interp = Interpolator(global_data.state_collection.extention_mesh.vertices, global_data.state_collection.extention_mesh.elements, wannier)
                res = interp.batch_evaluate(mesh_point)
                vals.append(res)
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
        self.wanniers = [global_data.state_collection.get_zero_extension_field() for _ in range(shape[2])]
        for n in range(shape[3]):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    kx, ky = WannierTools.get_kx_ky([i, j])
                    sign = 1
                    if global_data.incar.dataset_type.lower() == 'comsol':
                        sign = -1
                    phase = phase_[i, j] * np.exp(1j * np.dot(-1 * sign * np.array([kx, ky]), r_))
                    self.wanniers[n] += phase * ubloch[i, j, n]
            self.wanniers[n] /= np.sqrt(shape[0] * shape[1])
            norm = WannierTools.integrate_over_mesh(FieldData('wannier', global_data.state_collection.extention_mesh, np.abs(self.wanniers[n]) ** 2 * global_data.state_collection.extention_epsilon))
            Logger.info(f"Check wannier function norm = {norm}")
            if not np.isclose(np.abs(norm), 1.0, atol=1e-3):
                warn = f"Normalization ({np.abs(norm)}) not equal to 1 in Wannier State - {n}, err = {np.abs(1 - np.abs(norm))}"
                Logger.warning(warn)
            if global_data.incar.wannier_figures.lower() != "false":
                fd = FieldData('wannier', global_data.state_collection.extention_mesh, self.wanniers[n])
                fd.save_fig(global_data.incar.wannier_figures + f"/wannier-{n}.png")

    
