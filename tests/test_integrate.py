import numpy as np

from PCWannier import Utils
from PCWannier import MeshData


class TestIntegrate:
    def test_meshdata_integrate(self):
        mesh = MeshData.load_comsol_mesh("examples/Test.mphtxt")
        raw_data = MeshData.load_comsol_data("examples/test.txt")

        idxs, dists = MeshData.match_data_to_mesh(mesh, raw_data)
        value = [raw_data.value_matrix[idx][0] for idx in idxs]

        data = Utils.OneStateData("test", mesh, value)
        result = Utils.integrate_over_mesh(data)

        assert np.isclose(result.real, 0.76136), f"Expected real part 0.76136, got {result.real}"

    def test_normalize_integrate(self):
        mesh = MeshData.load_comsol_mesh("examples/Test.mphtxt")
        raw_data = MeshData.load_comsol_data("examples/test.txt")

        epsilon = MeshData.load_comsol_data("examples/epsilon.txt")
        
        idxs, dists = MeshData.match_data_to_mesh(mesh, raw_data)
        value = np.array([raw_data.value_matrix[idx][0] for idx in idxs])

        idxs, dists = MeshData.match_data_to_mesh(mesh, epsilon)
        eps = epsilon.value_matrix[idxs].flatten()

        norm_psi = np.abs(value) ** 2 * eps

        data = Utils.OneStateData("test", mesh, norm_psi)
        result = Utils.integrate_over_mesh(data)

        assert np.isclose(result.real, 9.148508), f"Expected real part 9.148, got {result.real}"

if __name__ == "__main__":
    test = TestIntegrate()
    test.test_meshdata_integrate()
    test.test_normalize_integrate()
