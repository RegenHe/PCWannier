import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    def __init__(self, vertices: np.ndarray, elements: np.ndarray) -> None:
        self.vertices: np.ndarray = np.array(vertices)
        self.elements: np.ndarray = np.array(elements)

    def __repr__(self) -> str:
        return f"Mesh(vertices={self.vertices}, elements={self.elements})"
    
    def plot_mesh(self) -> None:
        fig, ax = plt.subplots()
        for element in self.elements:
            triangle_vertices = self.vertices[element]
            
            triangle = plt.Polygon(triangle_vertices, edgecolor='black', fill=None)
            ax.add_patch(triangle)

            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
        min_x, max_x = np.min(self.vertices[:, 0]), np.max(self.vertices[:, 0])
        min_y, max_y = np.min(self.vertices[:, 1]), np.max(self.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)

        plt.show()


class RawData:
    def __init__(self, point_matrix: np.ndarray, value_matrix: np.ndarray) -> None:
        self.point_matrix: np.ndarray = np.array(point_matrix)
        self.value_matrix: np.ndarray = np.array(value_matrix)

    def __repr__(self) -> str:
        return f"RawData(point_matrix={self.point_matrix}, value_matrix={self.value_matrix})"
    
class OneStateData:
    def __init__(self, name: str, mesh: Mesh, value: np.ndarray) -> None:
        self.name: str = name
        self.mesh: Mesh = mesh
        self.value: np.ndarray = np.array(value)

    def __repr__(self) -> str:
        return f"OneStateData(point_matrix={self.mesh}, value_matrix={self.value})"


def integrate_over_triangle(vertices: np.ndarray, data_on_triangle: np.ndarray) -> complex:
    jacobian = np.array([[vertices[1, 0] - vertices[0, 0], vertices[2, 0] - vertices[0, 0]], [vertices[1, 1] - vertices[0, 1], vertices[2, 1] - vertices[0, 1]]])
    return np.sum(data_on_triangle) * np.abs(np.linalg.det(jacobian)) / 6.0

def integrate_over_mesh(data: OneStateData) -> complex:
    total_integral = 0.0 + 0.0j
    for idx in range(len(data.mesh.elements)):
        element = data.mesh.elements[idx]
        vertices = data.mesh.vertices[element, :]
        data_on_triangle = data.value[element]
        total_integral += integrate_over_triangle(vertices, data_on_triangle)
    return total_integral

class IncarData:
    def __init__(self):
        self.name = None
        self.lattice_const = None
        self.real_lattice_vectors = None
        self.reciprocal_lattice_vectors = None
        self.k_points = None
        self.dataset_type = None
        self.dataset_file = None
        self.dataset_order = None
        self.dielectric_file = None
        self.U_file = None
        self.hopping_file = None
        self.wannier_file = None
        self.wannier_figure = None

        self.b_vectors = None
        self.composition_of_b = None
        self.wb = None

        self.band_window = None
        self.band_calc = None

    def __repr__(self):
        return (
            f"IncarData =>\n"
            f"  name={self.name},\n"
            f"  lattice_const={self.lattice_const},\n"
            f"  real_lattice_vectors={self.real_lattice_vectors},\n"
            f"  reciprocal_lattice_vectors={self.reciprocal_lattice_vectors},\n"
            f"  k_points={self.k_points},\n"
            f"  dataset_type={self.dataset_type},\n"
            f"  dataset_file={self.dataset_file},\n"
            f"  dataset_order={self.dataset_order},\n"
            f"  dielectric_file={self.dielectric_file},\n"
            f"  U_file={self.U_file},\n"
            f"  hopping_file={self.hopping_file},\n"
            f"  wannier_file={self.wannier_file},\n"
            f"  wannier_figure={self.wannier_figure}\n"
            f"  b_vectors={self.b_vectors},\n"
            f"  composition_of_b={self.composition_of_b}\n"
            f"  wb={self.wb},\n"
            f"  band_window={self.band_window},\n"
            f"  band_calc={self.band_calc}\n"
        )

class wannier_tools:
    def __init__(self) -> None:
        self.init = False

    def preprocess(self) -> None:
        if np.array_equal(self.incar.reciprocal_lattice_vectors, np.array([[0, 0], [0, 0]])):
            print(self.incar.real_lattice_vectors)
            v = np.linalg.inv(self.incar.real_lattice_vectors) * np.eye(len(self.incar.real_lattice_vectors))
            print("reciprocal_lattice_vectors will be set to: ", v)
            self.incar.reciprocal_lattice_vectors = v
        
        if self.init is False:
            self.init = True
            self.incar.composition_of_b = [self.incar.composition_of_b, [[-v for v in sublist] for sublist in self.incar.composition_of_b]]
            self.incar.composition_of_b = [item for sublist in self.incar.composition_of_b for item in sublist]

            self.incar.b_vectors = []
            for i in range(len(self.incar.composition_of_b)):
                self.incar.b_vectors.append((self.incar.composition_of_b[i][0] * self.incar.reciprocal_lattice_vectors[0] / len(self.incar.k_points[0]) * 2 * np.pi / self.incar.lattice_const[0]
                                            + self.incar.composition_of_b[i][1] * self.incar.reciprocal_lattice_vectors[1] / len(self.incar.k_points[1]) * 2 * np.pi / self.incar.lattice_const[1]).tolist())

            mat_a = np.eye(2).reshape(-1, 1)
            mat_b = np.zeros((2 ** 2, len(self.incar.composition_of_b)))
            for i in range(2):
                for j in range(2):
                    for k in range(len(self.incar.composition_of_b)):
                        mat_b[i * 2 + j][k] = self.incar.b_vectors[k][i] * self.incar.b_vectors[k][j]
            
            self.incar.wb = np.linalg.pinv(mat_b) @ mat_a
            self.incar.wb = [item for sublist in self.incar.wb for item in sublist]

    def set_incar(self, incar_data: IncarData) -> None:
        self.incar = incar_data

    def neighbor_reciprocal_lattice_vectors(self, lattice_vector_idxs: np.ndarray, direction: int) -> np.ndarray:
        pass

if __name__ == "__main__":
    from PCWannier import IncarParser
    parser_data = IncarParser.IncarParser("examples/incar")
    wtools = wannier_tools()
    wtools.set_incar(parser_data.parse_file())
    wtools.preprocess()
    print(wtools.incar)