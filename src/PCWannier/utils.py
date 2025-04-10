import numpy as np


class Mesh:
    def __init__(self, vertices: np.ndarray, elements: np.ndarray) -> None:
        self.vertices = vertices
        self.elements = elements

    def __repr__(self) -> str:
        return f"Mesh(vertices={self.vertices}, elements={self.elements})"


class RawData:
    def __init__(self, point_matrix: np.ndarray, value_matrix: np.ndarray) -> None:
        self.point_matrix: np.ndarray = point_matrix
        self.value_matrix: np.ndarray = value_matrix

    def __repr__(self) -> str:
        return f"RawData(point_matrix={self.point_matrix}, value_matrix={self.value_matrix})"