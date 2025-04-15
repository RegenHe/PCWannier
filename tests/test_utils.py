import numpy as np

import copy

from PCWannier import Utils
from PCWannier import MeshData

class TestUtils:
    def test_integrate_over_mesh(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([2.0, 0.0])
        v3 = np.array([1.0, 1.0])
        v4 = np.array([2.0, 1.0])

        vertices = np.vstack([v1, v2, v3, v4]) + np.random.rand(1, 2)
        elements = np.array([[0, 1, 2], [1, 3, 2]])
        edge = np.array([[0, 1], [1, 3]])

        mesh = Utils.Mesh(vertices, elements, edge)

        value = np.array([
            1.0 + 6.0j, 1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 3.0j
        ])

        data = Utils.FieldData("test", mesh, value)

        result = Utils.integrate_over_mesh(data)
        assert np.isclose(result.real, 5/3), f"Expected real part 5/3, got {result.real}"
        assert np.isclose(result.imag, 1.5), f"Expected imaginary part 1.5, got {result.imag}"
    
    def test_match_and_rebuild(self):
        vertices = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]])
        elements = np.array([[0, 1, 2], [1, 2, 4]])

        mesh = Utils.Mesh(vertices, elements)
        new_vertices = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        vertices = np.array([[2.0, 2.0], [1.0, 0.0], [1.00001, 1.0001], [0.0, 0.0], [5.0, 5.0]])

        vertex_idx_in_new_vertices, vertex_idx_in_vertices = mesh.match(new_vertices, vertices)
        assert np.array_equal(vertex_idx_in_new_vertices, np.array([2, 1, 0])), "Matching indices are not correct."
        assert np.array_equal(vertex_idx_in_vertices, np.array([0, 2, 3])), "Matching indices are not correct."

        mesh.rebuild_index()
        assert np.array_equal(mesh.vertices, np.array([[0, 0], [1, 0], [0, 1], [2, 2]])), "Elements are not set correctly."
        assert np.array_equal(mesh.elements, np.array([[0, 1, 2], [1, 2, 3]])), "Elements are not set correctly."
    
    def test_wannier_tools(self):
        from PCWannier import IncarParser
        parser_data = IncarParser.IncarParser("examples/incar")
        wtools = Utils.WannierTools()
        wtools.set_incar(parser_data.parse_file())
        wtools.preprocess()
        print(Utils.global_data.incar)
        assert np.array_equal(Utils.global_data.incar.reciprocal_lattice_vectors, np.array([[1, 0], [0, 1]])), "Reciprocal lattice vectors are not set correctly."

    def test_state_collection(self):
        mesh = MeshData.load_comsol_mesh("examples/Test.mphtxt")
        raw_data = MeshData.load_comsol_data("examples/Ez.txt")

        epsilon = MeshData.load_comsol_data("examples/epsilon.txt")
        
        idxs, dists = MeshData.match_data_to_mesh(mesh, raw_data)
        raw_data.value_matrix = raw_data.value_matrix[idxs]
        state_collection = MeshData.distribute_data(mesh, raw_data)

        idxs, dists = MeshData.match_data_to_mesh(mesh, epsilon)

        state_collection.epsilon = epsilon.value_matrix[idxs].flatten()

        state_collection.normalize()

        assert np.isclose(state_collection.normalization[0][0][0], 9.4978), f"Expected normalization 9.4979, got {state_collection.normalization[0][0][0]}"
        assert np.isclose(state_collection.normalization[0][0][1], 5.2087), f"Expected normalization 5.2087, got {state_collection.normalization[0][0][1]}"
        assert np.isclose(state_collection.normalization[2][3][1], 4.9224), f"Expected normalization 4.9224, got {state_collection.normalization[2][3][1]}"
    
    def test_neighbor_reciprocal_lattice_vectors(self):
        from PCWannier import IncarParser
        parser_data = IncarParser.IncarParser("examples/incar")
        wtools = Utils.WannierTools()
        wtools.set_incar(parser_data.parse_file())
        wtools.preprocess()

        assert np.array_equal(wtools.neighbor_reciprocal_lattice_vectors([1, 1], 0), np.array([2, 1])), "neighbor_reciprocal_lattice_vectors functions are not set correctly."
        assert np.array_equal(wtools.neighbor_reciprocal_lattice_vectors([1, 2], 1), np.array([1, 3])), "neighbor_reciprocal_lattice_vectors functions are not set correctly."
        assert np.array_equal(wtools.neighbor_reciprocal_lattice_vectors([3, 3], 0), np.array([0, 3])), "neighbor_reciprocal_lattice_vectors functions are not set correctly."

if __name__ == "__main__":
    test = TestUtils()
    test.test_match_and_rebuild()
    test.test_wannier_tools()
    # test.test_neighbor_reciprocal_lattice_vectors()
    test.test_state_collection()
