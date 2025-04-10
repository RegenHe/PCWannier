import numpy as np


class RawMesh:
    def __init__(self, vertices: np.ndarray, elements: np.ndarray) -> None:
        self.vertices = vertices
        self.elements = elements

    def __repr__(self) -> str:
        return f"Mesh(vertices={self.vertices}, elements={self.elements})"

