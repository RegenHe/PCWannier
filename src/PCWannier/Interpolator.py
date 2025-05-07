import numpy as np
import matplotlib.tri as mtri

class Interpolator:
    def __init__(self, points: np.ndarray, triangles: np.ndarray, values: np.ndarray):
        self.points = points
        self.triangles = triangles
        self.values = values

        self.triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        self.interpolator = mtri.LinearTriInterpolator(self.triangulation, values)

    def evaluate(self, x: float, y: float) -> float:
        val = self.interpolator(x, y)
        return val.item() if val is not None else np.nan

    def batch_evaluate(self, query_points: np.ndarray) -> np.ndarray:
        x = query_points[:, 0]
        y = query_points[:, 1]
        result = self.interpolator(x, y)
        return result.filled(np.nan) if isinstance(result, np.ma.MaskedArray) else result
