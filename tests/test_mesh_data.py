
from PCWannier import MeshData

class TestMeshData():
    def test_load_comsol_mesh(self):
        mesh = MeshData.load_comsol_mesh("examples/Test.mphtxt")

        print(mesh.vertices.shape)
        print(mesh.elements.shape)

        assert (mesh.vertices.shape)[1] == 2, "Mesh vertices should have 2 columns"
        assert (mesh.elements.shape)[1] == 3, "Mesh elements should have 3 columns"