from __future__ import annotations

import numpy as np

from .group import SpaceGroupOperation
from .specs import FieldKind


def cartesian_field_matrix(
    operation: SpaceGroupOperation,
    real_lattice_vectors,
    field_kind: FieldKind,
    tolerance: float = 1.0e-8,
) -> np.ndarray:
    """Return the Cartesian component action of a fractional-space rotation."""

    lattice = np.asarray(real_lattice_vectors, dtype=float)
    expected = (operation.dimension, operation.dimension)
    if lattice.shape != expected:
        raise ValueError(f"real_lattice_vectors must have shape {expected}.")
    rotation = lattice.T @ operation.rotation @ np.linalg.inv(lattice.T)
    residual = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(operation.dimension), ord="fro")
    )
    if residual > tolerance:
        raise ValueError(
            "Fractional rotation is not an isometry of the supplied lattice "
            f"(residual={residual:.6g})."
        )
    if field_kind in {FieldKind.SCALAR, FieldKind.ELECTRIC_Z}:
        return np.ones((1, 1), dtype=float)
    if field_kind == FieldKind.MAGNETIC_AXIAL_Z:
        return np.asarray([[float(np.linalg.det(rotation))]], dtype=float)
    if field_kind == FieldKind.ELECTRIC_POLAR_VECTOR:
        return rotation
    if field_kind == FieldKind.MAGNETIC_AXIAL_VECTOR:
        return float(np.linalg.det(rotation)) * rotation
    raise ValueError(f"Unsupported field kind: {field_kind!r}.")

