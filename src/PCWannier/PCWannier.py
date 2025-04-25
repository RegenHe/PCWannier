from PCWannier.Utils import global_data
from PCWannier.Utils import WannierTools
from PCWannier.Timer import Timer, timer
from PCWannier.IncarParser import IncarParser
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
        else:
            raise f"Don't support {global_data.incar.dataset_type.lower()} type dataset"

        idxs, dists = MeshData.match_data_to_mesh(mesh, raw_data)
        raw_data.value_matrix = raw_data.value_matrix[idxs]

        idxs, dists = MeshData.match_data_to_mesh(mesh, epsilon)

        MeshData.distribute_data(mesh, raw_data)
        global_data.state_collection.epsilon = epsilon.value_matrix[idxs].flatten()

        global_data.state_collection.normalize()

        global_data.state_collection.turn_to_Bloch()
        global_data.state_collection.extention(global_data.incar.extension)

        global_data.push_m_set(MSet.MSet())
        global_data.m_set.init_M0(global_data.state_collection)

        global_data.push_state_initializer(StateInitializer.StateInitializer())
        global_data.state_initializer.iter(global_data.incar.err_diff, global_data.incar.max_iter)

        global_data.push_gradient(Gradient.Gradient())
        global_data.gradient.iter(global_data.incar.err_diff, global_data.incar.max_iter)

        if not global_data.incar.M_in:
            global_data.m_set.save_as(global_data.incar.M_file)
        
        global_data.state_initializer.save_as(global_data.incar.V_file)
        global_data.gradient.save_as(global_data.incar.U_file)

    def gen_wannier(self, n: int, R: list=[0, 0]):
        pass
