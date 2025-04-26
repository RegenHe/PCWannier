import numpy as np

import warnings

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
        global_data.threads = args.threads
        print(f"Running with {args.threads} threads")

        parser = IncarParser(args.input)
        wtools = WannierTools()
        wtools.set_incar(parser.parse_file())
        wtools.preprocess()
        print(global_data.incar)

        if global_data.incar.dataset_type.lower() == "comsol":
            mesh = MeshData.load_comsol_mesh(global_data.incar.mesh_file)
            raw_data = MeshData.load_comsol_data(global_data.incar.dataset_file)
            epsilon = MeshData.load_comsol_data(global_data.incar.dielectric_file)
            if global_data.incar.E_file.lower() != 'false':
                E_raw_data = MeshData.load_comsol_data(global_data.incar.E_file)
                    
        else:
            raise f"Don't support {global_data.incar.dataset_type.lower()} type dataset"

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
            t_ = E_raw_data.value_matrix[0].reshape(shape, order='C')
            desired_order = ["k1", "k2", "E"]
            indices = [global_data.incar.dataset_order.index(dim) for dim in desired_order]
            global_data.state_collection.E = np.real(np.transpose(t_, axes=(indices[0], indices[1], indices[2])))
        else:
            sizes = {"k1": len(global_data.incar.k_points[0]),"k2": len(global_data.incar.k_points[1]),"E": len(global_data.incar.band_window)}
            shape = tuple(sizes[dim] for dim in global_data.incar.dataset_order)
            t_ = E_raw_data.value_matrix[0].reshape(shape, order='C')
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



    def gen_wannier(self, r: list=[0, 0]):
        r_ = [0, 0]
        r_[0] = (r[0] * global_data.incar.real_lattice_vectors[0][0] * global_data.incar.lattice_const + r[1] * global_data.incar.real_lattice_vectors[0][1] * global_data.incar.lattice_const)
        r_[1] = (r[0] * global_data.incar.real_lattice_vectors[1][0] * global_data.incar.lattice_const + r[1] * global_data.incar.real_lattice_vectors[1][1] * global_data.incar.lattice_const)
        print(f"Generating Wannier Functions - r = ({r[0]}, {r[1]})")

        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), len(global_data.incar.band_window), global_data.incar.band_calc_num]

        ubloch = np.array([[[None for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])
        for n in range(shape[2]):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    t_ = global_data.state_collection.get_zero_extension_field()
                    mV = global_data.state_initializer.matV[i][j]
                    mU = global_data.gradient.U[i][j]
                    for m in range(shape[3]):
                        t_ += (mV @ mU)[m, n] * global_data.state_collection.get_extention_field(i, j, m)
                    ubloch[i, j, n] = t_
        
        wannier = [global_data.state_collection.get_zero_extension_field() for _ in range(shape[2])]
        for n in range(shape[2]):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    kx, ky = WannierTools.get_kx_ky([i, j])
                    if global_data.incar.dataset_type.lower() == 'comsol':
                        sign = -1
                    phase = global_data.state_collection.get_extention_phase(i, j) * np.exp(1j * np.dot(-1 * sign * np.array([kx, ky]), r_))
                    wannier[n] += phase * ubloch[i, j, n]
            wannier[n] /= np.sqrt(shape[0] * shape[1])
            norm = WannierTools.integrate_over_mesh(FieldData('wannier', global_data.state_collection.extention_mesh, np.abs(wannier[n]) ** 2 * global_data.state_collection.extention_epsilon))
            print(f"Check wannier function norm = {norm}")
            if not np.isclose(np.abs(norm), 1.0, atol=1e-3):
                warn = f"Normalization ({np.abs(norm)}) not equal to 1 in Wannier State - {n}, err = {np.abs(1 - np.abs(norm))}"
                warnings.warn(warn)
            if global_data.incar.wannier_figures.lower() != "false":
                fd = FieldData('wannier', global_data.state_collection.extention_mesh, wannier[n])
                fd.save_fig(global_data.incar.wannier_figures + f"/wannier-{n}.png")
    
    def gen_hopping(self, r: list=[0, 0]):
        r_ = [0, 0]
        r_[0] = (r[0] * global_data.incar.real_lattice_vectors[0][0] * global_data.incar.lattice_const + r[1] * global_data.incar.real_lattice_vectors[0][1] * global_data.incar.lattice_const)
        r_[1] = (r[0] * global_data.incar.real_lattice_vectors[1][0] * global_data.incar.lattice_const + r[1] * global_data.incar.real_lattice_vectors[1][1] * global_data.incar.lattice_const)
        print(f"Generating hoppings - r = ({r[0]}, {r[1]})")

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
