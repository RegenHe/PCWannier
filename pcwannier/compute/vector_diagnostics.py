from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import InputBundle, PeriodicGrid


@dataclass(frozen=True)
class VectorFieldDifferentialDiagnostics:
    """Fourier-space differential residual for periodic vector fields."""

    quantity: str
    max_residual: float
    mean_residual: float
    worst_k_index: tuple[int, int, int]
    worst_band_index: int
    residuals: np.ndarray


def diagnose_bundle_vector_fields(
    bundle: InputBundle,
    *,
    quantity: str,
) -> VectorFieldDifferentialDiagnostics:
    """Evaluate divergence or curl without modifying the loaded MPB fields."""

    if not isinstance(bundle.mesh, PeriodicGrid) or bundle.mesh.dimension != 3:
        raise ValueError("Vector differential diagnostics require a 3D periodic grid.")
    if quantity not in {"longitudinal", "curl"}:
        raise ValueError("quantity must be 'longitudinal' or 'curl'.")

    residual_grid = np.empty(bundle.fields.shape, dtype=object)
    values: list[float] = []
    worst_value = -1.0
    worst_k = (0, 0, 0)
    worst_band = -1
    dimension = bundle.mesh.dimension
    bloch_sign = int(bundle.bloch_convention.sign)

    for index in np.ndindex(bundle.fields.shape):
        block = np.asarray(bundle.fields[index], dtype=np.complex128)
        if block.ndim != 3 or block.shape[1:] != (bundle.mesh.point_count, dimension):
            raise ValueError(
                f"Vector field block at k={index} has shape {block.shape}; expected "
                f"(bands, {bundle.mesh.point_count}, {dimension})."
            )
        k_fractional = np.array(
            [bundle.config.k_points[axis][index[axis]] for axis in range(dimension)],
            dtype=np.float64,
        )
        divergence, curl = periodic_vector_field_residuals(
            block,
            bundle.mesh.shape,
            bundle.mesh.lattice_vectors,
            k_fractional,
            bloch_sign=bloch_sign,
        )
        local = divergence if quantity == "longitudinal" else curl
        residual_grid[index] = local
        actual_bands = np.asarray(bundle.band_indices[index], dtype=int)
        for local_band, value in enumerate(local):
            scalar = float(value)
            values.append(scalar)
            if scalar > worst_value:
                worst_value = scalar
                worst_k = tuple(int(component) for component in index)
                worst_band = int(actual_bands[local_band])

    return VectorFieldDifferentialDiagnostics(
        quantity=quantity,
        max_residual=max(worst_value, 0.0),
        mean_residual=float(np.mean(values)) if values else 0.0,
        worst_k_index=worst_k,
        worst_band_index=worst_band,
        residuals=residual_grid,
    )


def periodic_vector_field_residuals(
    fields: np.ndarray,
    grid_shape: tuple[int, int, int],
    lattice_vectors: np.ndarray,
    k_fractional: np.ndarray,
    *,
    bloch_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return relative divergence and curl residuals for periodic Bloch fields.

    The periodic Fourier mode ``m`` belongs to the complete Bloch wavevector
    ``m + sign*k``. Parseval factors cancel in the reported relative norms.
    """

    shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("grid_shape must contain three positive integers.")
    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("lattice_vectors must have finite shape (3, 3).")
    if abs(float(np.linalg.det(lattice))) <= np.finfo(float).tiny:
        raise ValueError("lattice_vectors must be invertible.")
    k = np.asarray(k_fractional, dtype=np.float64)
    if k.shape != (3,) or not np.all(np.isfinite(k)):
        raise ValueError("k_fractional must contain three finite components.")
    sign = int(bloch_sign)
    if sign not in {-1, 1}:
        raise ValueError("bloch_sign must be +1 or -1.")

    array = np.asarray(fields, dtype=np.complex128)
    point_count = int(np.prod(shape, dtype=np.int64))
    if array.ndim != 3 or array.shape[1:] != (point_count, 3):
        raise ValueError(
            f"fields must have shape (bands, {point_count}, 3); got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("fields contains NaN or Inf.")

    modes = [np.fft.fftfreq(size) * size for size in shape]
    fractional_wavevectors = np.stack(
        np.meshgrid(*modes, indexing="ij"), axis=-1
    ) + sign * k
    cartesian_wavevectors = (
        2.0 * np.pi * fractional_wavevectors @ np.linalg.inv(lattice).T
    )
    q2 = np.sum(cartesian_wavevectors**2, axis=-1)

    reshaped = array.reshape((array.shape[0],) + shape + (3,))
    coefficients = np.fft.fftn(reshaped, axes=(1, 2, 3))
    divergence = np.sum(
        coefficients * cartesian_wavevectors[None, ...], axis=-1
    )
    curl = np.cross(
        cartesian_wavevectors[None, ...], coefficients, axis=-1
    )
    denominator = np.sum(
        q2[None, ...] * np.sum(np.abs(coefficients) ** 2, axis=-1),
        axis=(1, 2, 3),
        dtype=np.float64,
    )
    divergence_norm = np.sum(
        np.abs(divergence) ** 2,
        axis=(1, 2, 3),
        dtype=np.float64,
    )
    curl_norm = np.sum(
        np.abs(curl) ** 2,
        axis=(1, 2, 3, 4),
        dtype=np.float64,
    )

    scale = np.maximum(denominator, np.finfo(np.float64).tiny)
    divergence_residual = np.sqrt(divergence_norm / scale)
    curl_residual = np.sqrt(curl_norm / scale)
    zero_scale = denominator <= np.finfo(np.float64).tiny
    divergence_residual[zero_scale & (divergence_norm == 0.0)] = 0.0
    curl_residual[zero_scale & (curl_norm == 0.0)] = 0.0
    if not np.all(np.isfinite(divergence_residual)) or not np.all(
        np.isfinite(curl_residual)
    ):
        raise FloatingPointError("Vector differential diagnostics are non-finite.")
    return divergence_residual, curl_residual
