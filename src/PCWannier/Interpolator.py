from typing import Any, Tuple

import numpy as np
import matplotlib.tri as mtri

from .Log import Logger

class Interpolator2D:
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

class CachedInterpolator2D(Interpolator2D):
    def __init__(self, points: np.ndarray, triangles: np.ndarray):
        super().__init__(points, triangles, np.zeros(len(points)))
        self.cache = {}

    def weights(self, xy_wrapped: np.ndarray, key: Any):
        if key in self.cache:
            return self.cache[key]

        xy = np.asarray(xy_wrapped, float)

        tri_id = self.triangulation.get_trifinder()(xy[:, 0], xy[:, 1])

        mask_out = tri_id == -1
        if np.any(mask_out):
            if not hasattr(self, "_coord2vert"):
                self._coord2vert = {
                    tuple(np.round(p, 12)): i
                    for i, p in enumerate(self.points)
                }
                self._vert2tri = {}
                for t, tri in enumerate(self.triangles):
                    for v in tri:
                        self._vert2tri.setdefault(v, t)

            for idx in np.where(mask_out)[0]:
                key = tuple(np.round(xy[idx], 12))
                try:
                    v_id = self._coord2vert[key]
                    tri_id[idx] = self._vert2tri[v_id]
                except KeyError:
                    Logger.error(f"Point {key} is not within the triangulation domain")
                    raise

        verts = self.triangles[tri_id]
        A = self.points[verts]
        v0 = A[:, 1] - A[:, 0]
        v1 = A[:, 2] - A[:, 0]
        vp = xy - A[:, 0]

        den = v0[:, 0]*v1[:, 1] - v0[:, 1]*v1[:, 0]
        if np.any(np.isclose(den, 0.0)):
            bad = np.where(np.isclose(den, 0.0))[0]
            Logger.error(f"{bad.size} triangles are degenerate with area close to 0. Please check mesh quality.")
            raise

        a = ( vp[:, 0]*v1[:, 1] - vp[:, 1]*v1[:, 0]) / den
        b = ( v0[:, 0]*vp[:, 1] - v0[:, 1]*vp[:, 0]) / den
        bary = np.c_[a, b, 1.0 - a - b]

        self.cache[key] = (verts, bary)
        return verts, bary
