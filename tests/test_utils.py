import numpy as np

from PCWannier import Utils

class TestUtils:
    def test_integrate_over_mesh(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([2.0, 0.0])
        v3 = np.array([1.0, 1.0])
        v4 = np.array([2.0, 1.0])

        vertices = np.vstack([v1, v2, v3, v4]) + np.random.rand(1, 2)
        elements = np.array([[0, 1, 2], [1, 3, 2]])

        mesh = Utils.Mesh(vertices, elements)

        value = np.array([
            1.0 + 6.0j, 1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 3.0j
        ])

        data = Utils.OneStateData("test", mesh, value)

        result = Utils.integrate_over_mesh(data)
        assert np.isclose(result.real, 5/3), f"Expected real part 5/3, got {result.real}"
        assert np.isclose(result.imag, 1.5), f"Expected imaginary part 1.5, got {result.imag}"

if __name__ == "__main__":
    test = TestUtils()
    test.test_integrate_over_mesh()