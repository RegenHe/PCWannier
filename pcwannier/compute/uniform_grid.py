from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .backend import resolve_backend

if TYPE_CHECKING:
    from ..data import PeriodicGrid

from ..data import periodic_axis_coordinates


def _cell_volume(
    *,
    cell_volume: float | None = None,
    lattice_vectors: np.ndarray | None = None,
) -> float:
    if cell_volume is None and lattice_vectors is None:
        raise ValueError("cell_volume and lattice_vectors cannot both be omitted.")
    lattice_volume = None
    if lattice_vectors is not None:
        lattice = np.asarray(lattice_vectors, dtype=np.float64)
        if lattice.ndim != 2 or lattice.shape[0] != lattice.shape[1]:
            raise ValueError("lattice_vectors must be a square matrix.")
        if not np.all(np.isfinite(lattice)):
            raise ValueError("lattice_vectors contains NaN or Inf.")
        lattice_volume = abs(float(np.linalg.det(lattice)))
        if not np.isfinite(lattice_volume) or lattice_volume <= 0.0:
            raise ValueError("lattice_vectors must define a positive cell volume.")
    volume = lattice_volume if cell_volume is None else float(cell_volume)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("cell_volume must be positive and finite.")
    if lattice_volume is not None and cell_volume is not None and not np.isclose(
        volume, lattice_volume, rtol=1.0e-12, atol=1.0e-15
    ):
        raise ValueError(
            "cell_volume is inconsistent with abs(det(lattice_vectors))."
        )
    return volume


def integrate_scalar(
    values: np.ndarray,
    *,
    cell_volume: float | None = None,
    lattice_vectors: np.ndarray | None = None,
):
    """Integrate scalar samples on a half-open periodic uniform grid."""

    array = np.asarray(values)
    if array.ndim < 1 or array.ndim > 3 or any(size <= 0 for size in array.shape):
        raise ValueError("values must have a non-empty 1D, 2D, or 3D grid shape.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("values must contain numeric real or complex samples.")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contains NaN or Inf.")
    volume = _cell_volume(
        cell_volume=cell_volume,
        lattice_vectors=lattice_vectors,
    )
    dtype = np.complex128 if np.iscomplexobj(array) else np.float64
    total = np.sum(array, dtype=dtype)
    return total * (volume / int(array.size))


def integrate_components(
    values: np.ndarray,
    *,
    cell_volume: float | None = None,
    lattice_vectors: np.ndarray | None = None,
):
    """Sum pointwise components, then integrate on a periodic uniform grid."""

    array = np.asarray(values)
    if array.ndim < 2 or array.ndim > 4 or array.shape[-1] <= 0:
        raise ValueError(
            "values must have shape (N1[, N2[, N3]], components)."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("values must contain numeric real or complex samples.")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contains NaN or Inf.")
    dtype = np.complex128 if np.iscomplexobj(array) else np.float64
    pointwise = np.sum(array, axis=-1, dtype=dtype)
    return integrate_scalar(
        pointwise,
        cell_volume=cell_volume,
        lattice_vectors=lattice_vectors,
    )


def periodic_grid_coordinates(
    shape,
    lattice_vectors: np.ndarray,
) -> np.ndarray:
    """Return Cartesian coordinates for u_i=i/N-1/2 on a periodic grid."""

    grid_shape = tuple(int(value) for value in shape)
    if not grid_shape or len(grid_shape) > 3 or any(value <= 0 for value in grid_shape):
        raise ValueError("shape must contain one to three positive integers.")
    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    dimension = len(grid_shape)
    if lattice.shape != (dimension, dimension) or not np.all(np.isfinite(lattice)):
        raise ValueError(
            f"lattice_vectors must have finite shape {(dimension, dimension)}."
        )
    if abs(float(np.linalg.det(lattice))) <= np.finfo(float).tiny:
        raise ValueError("lattice_vectors must define a positive cell volume.")
    axes = [periodic_axis_coordinates(size) for size in grid_shape]
    fractional = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return fractional @ lattice


class UniformGridInnerProduct:
    """Metric inner product using equal weights on periodic grid samples."""

    IMPLEMENTATION_VERSION = "uniform-grid-inner-product-v1"
    integration_family = "uniform_grid"
    uses_full_bloch_fields = False

    def __init__(
        self,
        grid: PeriodicGrid,
        metric: np.ndarray,
        *,
        backend: str | None = None,
        point_selector: np.ndarray | None = None,
    ) -> None:
        self.grid = grid
        self.backend = resolve_backend(backend)
        self.mode = "uniform"
        self.metric = np.asarray(metric, dtype=np.complex128).reshape(-1)
        expected = (grid.point_count,)
        if self.metric.shape != expected or not np.all(np.isfinite(self.metric)):
            raise ValueError(
                "Metric material must contain one finite value per grid point; "
                f"expected {expected}, got {self.metric.shape}."
            )
        if point_selector is None:
            self._point_indices = np.arange(grid.point_count, dtype=np.intp)
        else:
            selector = np.asarray(point_selector)
            if selector.dtype == bool:
                if selector.shape != expected:
                    raise ValueError(
                        f"Point selector must have shape {expected}; got {selector.shape}."
                    )
                self._point_indices = np.flatnonzero(selector)
            else:
                self._point_indices = np.asarray(
                    selector, dtype=np.intp
                ).reshape(-1)
                if (
                    self._point_indices.size
                    and (
                        np.min(self._point_indices) < 0
                        or np.max(self._point_indices) >= grid.point_count
                    )
                ):
                    raise ValueError("Point selector contains an out-of-range index.")
        self.point_weight = float(grid.cell_volume) / float(grid.point_count)

    def overlap(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        conjugate_left: bool = True,
        phase_wavevector: np.ndarray | None = None,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        del chunk_size
        left_matrix = _to_field_rows(left, self.grid.point_count, "left")
        right_matrix = _to_field_rows(right, self.grid.point_count, "right")
        if left_matrix.shape[2] != right_matrix.shape[2]:
            raise ValueError("left and right fields have different component counts.")
        indices = self._point_indices
        left_selected = left_matrix[:, indices, :]
        right_selected = right_matrix[:, indices, :]
        if phase_wavevector is not None:
            wavevector = np.asarray(
                phase_wavevector, dtype=np.float64
            ).reshape(-1)
            if (
                wavevector.shape != (self.grid.dimension,)
                or not np.all(np.isfinite(wavevector))
            ):
                raise ValueError(
                    f"phase_wavevector must contain {self.grid.dimension} finite components."
                )
            phase = np.exp(1j * (self.grid.vertices[indices] @ wavevector))
            right_selected = right_selected * phase[None, :, None]
        weighted_right = right_selected * self.metric[indices][None, :, None]
        left_values = left_selected.conj() if conjugate_left else left_selected
        result = np.einsum(
            "mpc,npc->mn",
            left_values,
            weighted_right,
            optimize=True,
        )
        return np.asarray(result * self.point_weight, dtype=np.complex128)

    def norms(
        self,
        values: np.ndarray,
        *,
        chunk_size: int | None = None,
        name: str = "metric norms",
    ) -> np.ndarray:
        del chunk_size
        matrix = _to_field_columns(values, self.grid.point_count, "values")
        indices = self._point_indices
        metric = self.metric[indices]
        imag_scale = max(
            float(np.max(np.abs(metric.real), initial=0.0)), 1.0
        )
        imag_residual = float(np.max(np.abs(metric.imag), initial=0.0))
        if imag_residual > 1.0e-12 * imag_scale:
            raise FloatingPointError(
                f"{name} requires a real metric; imaginary residual={imag_residual:.6g}."
            )
        integrand = metric.real[:, None, None] * np.abs(matrix[indices]) ** 2
        result = (
            np.sum(integrand, axis=(0, 1), dtype=np.float64) * self.point_weight
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError(f"{name} contains non-finite values.")
        return np.asarray(result, dtype=np.float64)

    def norm(
        self,
        field: np.ndarray,
        *,
        name: str = "metric field norm",
    ) -> float:
        array = np.asarray(field)
        if array.ndim == 1:
            values = array.reshape(self.grid.point_count, 1, 1)
        elif array.ndim == 2 and array.shape[0] == self.grid.point_count:
            values = array[:, :, None]
        else:
            raise ValueError(
                "field must have shape (points,) or (points, components)."
            )
        return float(self.norms(values, name=name)[0])

    def restrict_points(self, selector) -> UniformGridInnerProduct:
        local = np.asarray(selector)
        if local.dtype == bool:
            if local.shape != (self.grid.point_count,):
                raise ValueError("Restricted point mask has an invalid shape.")
            selected = np.intersect1d(
                self._point_indices,
                np.flatnonzero(local),
                assume_unique=True,
            )
        else:
            selected = np.intersect1d(
                self._point_indices,
                np.asarray(local, dtype=np.intp).reshape(-1),
                assume_unique=False,
            )
        return UniformGridInnerProduct(
            self.grid,
            self.metric,
            backend=self.backend,
            point_selector=selected,
        )

    def restrict_elements(self, selector):
        raise TypeError(
            "Uniform-grid integration restricts sample points, not finite elements."
        )


def _to_field_rows(values, point_count: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        result = array.reshape(1, -1, 1)
    elif array.ndim == 2:
        scalar = array if array.shape[1] == point_count else array.T
        result = scalar[:, :, None]
    elif array.ndim == 3:
        result = array if array.shape[1] == point_count else np.swapaxes(array, 0, 1)
    else:
        raise ValueError(f"{name} has invalid shape {array.shape}.")
    if result.shape[1] != point_count or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must contain finite values with one axis of length {point_count}."
        )
    return np.asarray(result, dtype=np.complex128)


def _to_field_columns(values, point_count: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        result = array.reshape(-1, 1, 1)
    elif array.ndim == 2:
        scalar = array if array.shape[0] == point_count else array.T
        result = scalar[:, None, :]
    elif array.ndim == 3:
        result = array if array.shape[0] == point_count else np.swapaxes(array, 0, 1)
    else:
        raise ValueError(f"{name} has invalid shape {array.shape}.")
    if result.shape[0] != point_count or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must contain finite values with one axis of length {point_count}."
        )
    return np.asarray(result, dtype=np.complex128)
