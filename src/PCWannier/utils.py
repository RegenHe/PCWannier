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



def jacobian_triangle(vertices: np.ndarray) -> np.ndarray:
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    jacobian = np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
    return jacobian

def integrate_over_triangle(vertices: np.ndarray, data_on_triangle: np.ndarray) -> complex:
    jacobian = jacobian_triangle(vertices)
    det_jacobian = np.abs(np.linalg.det(jacobian))
    return np.sum(data_on_triangle) * det_jacobian / 6.0

def integrate_over_mesh(data: OneStateData) -> complex:
    total_integral = 0.0 + 0.0j
    for idx in range(len(data.mesh.elements)):
        element = data.mesh.elements[idx]
        vertices = data.mesh.vertices[element, :]
        data_on_triangle = data.value[element]
        total_integral += integrate_over_triangle(vertices, data_on_triangle)
    return total_integral