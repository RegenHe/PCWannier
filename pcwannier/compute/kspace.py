from __future__ import annotations

import numpy as np

from ..config import IncarConfig


def neighbor_reciprocal_lattice_vectors(config: IncarConfig, k: list[int], direction: int):
    axes = int(config.kdim)
    idxs = (list(k) + [0] * axes)[:axes]
    comp = (list(config.composition_of_b[direction]) + [0] * axes)[:axes]
    wrapped = [0, 0, 0]
    raw_list = []
    crossed = False
    for axis in range(axes):
        n_axis = len(config.k_points[axis])
        raw = int(idxs[axis] + comp[axis])
        w = int(np.mod(raw, n_axis))
        wrapped[axis] = w
        raw_list.append(raw)
        if w != raw:
            crossed = True
    return (wrapped[0], wrapped[1] if axes >= 2 else 0, wrapped[2] if axes >= 3 else 0), (
        raw_list + [0, 0, 0]
    )[:3] if crossed else None


def get_kxyz(config: IncarConfig, k: list[int]) -> np.ndarray:
    kps = config.k_points
    reciprocal = np.asarray(config.reciprocal_lattice_vectors, dtype=float)
    axes = len(kps)
    idxs = (list(k) + [0] * axes)[:axes]
    if reciprocal.shape[1] < 3:
        reciprocal = np.pad(reciprocal, ((0, 0), (0, 3 - reciprocal.shape[1])), mode="constant")
    vec = np.zeros(3, dtype=float)
    for axis in range(axes):
        arr = np.asarray(kps[axis], dtype=float)
        if 0 <= idxs[axis] < arr.size:
            kval = arr[idxs[axis]]
        else:
            step = (arr[1] - arr[0]) if arr.size >= 2 else 0.0
            kval = arr[0] + step * idxs[axis]
        vec += kval * reciprocal[axis, :3]
    return (2.0 * np.pi / float(config.lattice_const)) * vec
