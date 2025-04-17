from PCWannier import Utils
from PCWannier import IncarParser
from PCWannier import MeshData
from PCWannier import StateInitializer
from PCWannier import GlobalData

class TestStateInitializer:
    def test_base_generator(self):
        parser = IncarParser.IncarParser('examples/incar')
        wtools = Utils.WannierTools()
        wtools.set_incar(parser.parse_file())
        wtools.preprocess()
        print(GlobalData.global_data.incar)
        mesh = MeshData.load_comsol_mesh(GlobalData.global_data.incar.mesh_file)
        # mesh.plot_mesh()
        raw_data = MeshData.load_comsol_data(GlobalData.global_data.incar.dataset_file)
        idxs, dists = MeshData.match_data_to_mesh(mesh, raw_data)
        raw_data.value_matrix = raw_data.value_matrix[idxs]

        MeshData.distribute_data(mesh, raw_data)

        GlobalData.global_data.state_collection.extention([4, 4])
        GlobalData.global_data.state_initializer = StateInitializer.StateInitializer()
        states = GlobalData.global_data.incar.projections[0]
        f = lambda r, phi: StateInitializer.StateBases.Radial(states['states'][2][0])(r, states['states'][2][2]) * StateInitializer.StateBases.Angular(states['states'][2][1])(phi)
        field = GlobalData.global_data.state_collection.extention_mesh.rfunc(f, states['position'], states['xaxis_angluar'])
        fd = Utils.FieldData('', GlobalData.global_data.state_collection.extention_mesh, field)

if __name__ == "__main__":
    test = TestStateInitializer()
    test.test_base_generator()