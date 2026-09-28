from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def sample_fractional_band_path(
    k_path: Sequence[dict[str, Any]],
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, list[list[Any]]]:
    """Sample an incar-style fractional k path with PCWannier's closure rule."""

    points = tuple(k_path)
    if not points:
        raise ValueError("k_path must contain at least one point.")
    kdim = int(dimension)
    if kdim <= 0:
        raise ValueError("Band-path dimension must be positive.")
    first = _path_point(points[0], kdim, 0)
    last = _path_point(points[-1], kdim, len(points) - 1)
    closes_explicitly = periodically_equivalent_kpoint(first, last)
    path_parts = [first]
    high_symmetry_points: list[list[Any]] = []
    total = 0
    for index, entry in enumerate(points):
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"k_path point {index} has no name.")
        high_symmetry_points.append([name, total])
        if index == len(points) - 1 and closes_explicitly:
            break
        count = int(entry.get("num", 0))
        if count <= 0 or count != entry.get("num"):
            raise ValueError(f"k_path point {index} must have a positive integer num.")
        start = _path_point(entry, kdim, index)
        stop = _path_point(points[(index + 1) % len(points)], kdim, (index + 1) % len(points))
        path_parts.append(
            np.stack(
                [
                    np.linspace(start[axis], stop[axis], count + 1)[1:]
                    for axis in range(kdim)
                ],
                axis=1,
            )
        )
        total += count
    if not closes_explicitly:
        high_symmetry_points.append([str(points[0]["name"]), total])
    return (
        np.vstack(path_parts),
        np.arange(total + 1),
        high_symmetry_points,
    )


def _path_point(entry: dict[str, Any], dimension: int, index: int) -> np.ndarray:
    point = np.asarray(entry.get("point"), dtype=float)
    if point.shape != (dimension,) or not np.all(np.isfinite(point)):
        raise ValueError(
            f"k_path point {index} must have finite shape {(dimension,)}."
        )
    return point


def periodically_equivalent_kpoint(first: np.ndarray, last: np.ndarray) -> bool:
    """Return whether two fractional k points differ by a reciprocal vector."""

    difference = np.asarray(last, dtype=float) - np.asarray(first, dtype=float)
    return bool(np.allclose(difference, np.rint(difference), rtol=0.0, atol=1.0e-10))
