from PCWannier import MeshData

class TestMeshData():
    def test_load_comsol_mesh(self):
        mesh = MeshData.load_comsol_mesh("examples/Test.mphtxt")

        assert mesh.vertices.shape[1] == 2, "Mesh vertices should have 2 columns"
        assert mesh.elements.shape[1] == 3, "Mesh elements should have 3 columns"

    def test_load_comsol_data(self):
        raw_data = MeshData.load_comsol_data("examples/test.txt")

        assert raw_data.point_matrix.shape[1] == 2, "Point matrix should have 2 columns"
        assert raw_data.value_matrix.shape[0] == raw_data.point_matrix.shape[0], "Value matrix should have the same number of rows as point matrix"
    
    def test_match_data_to_mesh(self):
        mesh = MeshData.load_comsol_mesh("examples/Test.mphtxt")
        raw_data = MeshData.load_comsol_data("examples/test.txt")

        idxs, dists = MeshData.match_data_to_mesh(mesh, raw_data)
        print(idxs)

        assert idxs.shape[0] == mesh.vertices.shape[0], "Matched points should match mesh vertices"
        assert dists.shape[0] == mesh.vertices.shape[0], "Matched dists should match mesh elements"
        assert min(abs(dists)) <= 1e-6, "Minimum distance should be close to zero"

if __name__ == "__main__":
    test = TestMeshData()
    test.test_load_comsol_mesh()
    test.test_load_comsol_data()
    test.test_match_data_to_mesh()