from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from ..data import FieldData
from .backend import BACKEND_NUMBA, resolve_backend

_NUMBA_PARALLEL_COLUMN_THRESHOLD = 32
_NUMBA_PARALLEL_ALLOWED: ContextVar[bool] = ContextVar("pcwannier_numba_parallel_allowed", default=True)


@dataclass(frozen=True)
class MeshIntegralView:
    elements: np.ndarray
    tri_weights: np.ndarray
    nv: int
    vertices: np.ndarray | None = None


@contextmanager
def numba_parallel_policy(enabled: bool):
    token = _NUMBA_PARALLEL_ALLOWED.set(bool(enabled))
    try:
        yield
    finally:
        _NUMBA_PARALLEL_ALLOWED.reset(token)


def mesh_integral_view(mesh) -> MeshIntegralView:
    if isinstance(mesh, MeshIntegralView):
        return mesh
    return MeshIntegralView(
        elements=np.asarray(mesh.elements, dtype=np.intp),
        tri_weights=np.asarray(mesh.tri_weights, dtype=np.float64),
        nv=int(mesh.vertices.shape[0]),
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
    )


def build_phase_weighted_triangle_mass(
    mesh,
    weights_vector: np.ndarray,
    phase_wavevector: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-12,
    maximum_order: int = 16,
) -> np.ndarray:
    """Build local mass matrices for epsilon_h exp(i q.r).

    The complete Bloch FEM fields remain linear on each original triangle.  The
    exponential is therefore integrated inside the element instead of being
    interpolated from phase-shifted nodal values.
    """

    view = mesh_integral_view(mesh)
    if view.vertices is None:
        raise ValueError("Phase-aware integration requires mesh vertex coordinates.")
    vertices = np.asarray(view.vertices, dtype=np.float64)
    wavevector = np.asarray(phase_wavevector, dtype=np.float64).reshape(-1)
    if wavevector.shape != (vertices.shape[1],) or not np.all(np.isfinite(wavevector)):
        raise ValueError(
            f"phase_wavevector must contain {vertices.shape[1]} finite Cartesian components."
        )
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive and finite.")
    if maximum_order < 4:
        raise ValueError("maximum_order must be at least 4.")

    nodal_weights = np.asarray(weights_vector, dtype=np.complex128).reshape(view.nv)
    if not np.all(np.isfinite(nodal_weights)):
        raise ValueError("Phase-aware integration weights contain non-finite values.")
    if np.all(wavevector == 0.0):
        result = _weighted_triangle_mass(
            nodal_weights[view.elements],
            view.tri_weights,
        )
        result.setflags(write=False)
        return result

    previous = None
    for order in range(3, int(maximum_order) + 1):
        current = _duffy_phase_weighted_triangle_mass(
            vertices[view.elements],
            nodal_weights[view.elements],
            view.tri_weights,
            wavevector,
            order,
        )
        if previous is not None:
            difference = float(np.max(np.abs(current - previous), initial=0.0))
            scale = max(float(np.max(np.abs(current), initial=0.0)), np.finfo(float).tiny)
            if difference <= relative_tolerance * scale:
                current.setflags(write=False)
                return current
        previous = current
    raise FloatingPointError(
        "Phase-aware triangle integration did not converge: "
        f"q={wavevector.tolist()}, maximum_order={maximum_order}, "
        f"relative_tolerance={relative_tolerance:.3g}."
    )


def integrate_overlap_element_matrices(
    mesh,
    left: np.ndarray,
    right: np.ndarray,
    element_matrices: np.ndarray,
    *,
    conjugate_left: bool = True,
    chunk_size: int | None = None,
    backend: str | None = None,
) -> np.ndarray:
    """Contract FEM nodal fields with precomputed local element matrices."""

    selected_backend = resolve_backend(backend)
    view = mesh_integral_view(mesh)
    lmat = _to_k_nv(left, view.nv, "left")
    rmat = _to_k_nv(right, view.nv, "right")
    local = np.asarray(element_matrices, dtype=np.complex128)
    expected = (view.elements.shape[0], 3, 3)
    if local.shape != expected or not np.all(np.isfinite(local)):
        raise ValueError(f"element_matrices must have finite shape {expected}; got {local.shape}.")
    if selected_backend == BACKEND_NUMBA:
        return _integrate_overlap_element_matrices_numba(
            lmat,
            rmat,
            view.elements,
            local,
            conjugate_left,
        )
    return _integrate_overlap_element_matrices(
        lmat,
        rmat,
        view.elements,
        local,
        conjugate_left,
        chunk_size,
    )


def integrate_over_mesh(
    data: FieldData,
    *,
    other: FieldData | None = None,
    hermitian: bool = False,
    real_only: bool = False,
    chunk_size: int | None = None,
    backend: str | None = None,
) -> complex | np.ndarray:
    selected_backend = resolve_backend(backend)
    view = mesh_integral_view(data.mesh)

    a = _to_nv_k(data.field, view.nv, "field")
    if other is None:
        if selected_backend == BACKEND_NUMBA:
            out = _integrate_batch_numba(a, view.elements, view.tri_weights)
        else:
            out = _integrate_batch(a, view.elements, view.tri_weights, chunk_size)
    else:
        b = _to_nv_k(other.field, view.nv, "other field")
        if a.shape[1] != b.shape[1]:
            raise ValueError("A and B must have the same number of columns.")
        if selected_backend == BACKEND_NUMBA:
            out = _integrate_product_numba(a, b, view.elements, view.tri_weights, hermitian)
        else:
            out = _integrate_product(a, b, view.elements, view.tri_weights, hermitian, chunk_size)
    if real_only:
        out = out.real
    return out[0] if out.shape == (1,) else out


def integrate_weighted_columns(
    mesh,
    weights_vector: np.ndarray,
    values: np.ndarray,
    *,
    chunk_size: int | None = None,
    backend: str | None = None,
    mode: str = "nodal",
) -> np.ndarray:
    selected_backend = resolve_backend(backend)
    view = mesh_integral_view(mesh)
    left = np.asarray(weights_vector, dtype=np.complex128).reshape(view.nv)
    right = _to_nv_k(values, view.nv, "values")
    _validate_integration_mode(mode)
    if mode == "quadratic":
        repeated_left = np.broadcast_to(left[:, None], right.shape)
        if selected_backend == BACKEND_NUMBA:
            return _integrate_product_numba(repeated_left, right, view.elements, view.tri_weights, False)
        return _integrate_product(repeated_left, right, view.elements, view.tri_weights, False, chunk_size)
    if selected_backend == BACKEND_NUMBA:
        return _integrate_weighted_columns_numba(left, right, view.elements, view.tri_weights)
    return _integrate_weighted_columns(left, right, view.elements, view.tri_weights, chunk_size)


def integrate_weighted_abs2_columns(
    mesh,
    weights_vector: np.ndarray,
    values: np.ndarray,
    *,
    chunk_size: int | None = None,
    backend: str | None = None,
    mode: str = "nodal",
) -> np.ndarray:
    selected_backend = resolve_backend(backend)
    view = mesh_integral_view(mesh)
    left = np.asarray(weights_vector, dtype=np.complex128).reshape(view.nv)
    right = _to_nv_k(values, view.nv, "values")
    _validate_integration_mode(mode)
    if mode == "quadratic":
        if selected_backend == BACKEND_NUMBA:
            return _integrate_weighted_abs2_columns_quadratic_numba(
                left,
                right,
                view.elements,
                view.tri_weights,
            )
        return _integrate_weighted_abs2_columns_quadratic(
            left,
            right,
            view.elements,
            view.tri_weights,
            chunk_size,
        )
    if selected_backend == BACKEND_NUMBA:
        return _integrate_weighted_abs2_columns_numba(left, right, view.elements, view.tri_weights)
    return _integrate_weighted_abs2_columns(left, right, view.elements, view.tri_weights, chunk_size)


def integrate_overlap_matrix(
    mesh,
    left: np.ndarray,
    right: np.ndarray,
    weights_vector: np.ndarray | None = None,
    *,
    conjugate_left: bool = True,
    chunk_size: int | None = None,
    backend: str | None = None,
    mode: str = "nodal",
) -> np.ndarray:
    selected_backend = resolve_backend(backend)
    view = mesh_integral_view(mesh)
    lmat = _to_k_nv(left, view.nv, "left")
    rmat = _to_k_nv(right, view.nv, "right")
    weights_vector = np.ones(view.nv, dtype=np.complex128) if weights_vector is None else np.asarray(weights_vector, dtype=np.complex128).reshape(view.nv)
    _validate_integration_mode(mode)
    if mode == "quadratic":
        if selected_backend == BACKEND_NUMBA:
            return _integrate_overlap_matrix_quadratic_numba(
                lmat,
                rmat,
                weights_vector,
                view.elements,
                view.tri_weights,
                conjugate_left,
            )
        return _integrate_overlap_matrix_quadratic(
            lmat,
            rmat,
            weights_vector,
            view.elements,
            view.tri_weights,
            conjugate_left,
            chunk_size,
        )
    if selected_backend == BACKEND_NUMBA:
        return _integrate_overlap_matrix_numba(
            lmat,
            rmat,
            weights_vector,
            view.elements,
            view.tri_weights,
            conjugate_left,
        )
    return _integrate_overlap_matrix(lmat, rmat, weights_vector, view.elements, view.tri_weights, conjugate_left, chunk_size)


def _to_nv_k(arr, nv: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        out = arr.reshape(nv, 1)
    elif arr.ndim == 2:
        out = arr if arr.shape[0] == nv else arr.T
    else:
        raise ValueError(f"{name} has invalid shape {arr.shape}")
    if out.shape[0] != nv:
        raise ValueError(f"{name} has invalid shape {arr.shape}; expected one dimension to be {nv}.")
    return out.astype(np.complex128, copy=False)


def _validate_integration_mode(mode: str) -> None:
    if mode not in {"nodal", "quadratic"}:
        raise ValueError("integration mode must be 'nodal' or 'quadratic'.")


def validated_real(values, name: str, *, rtol: float = 1e-10, atol: float = 1e-12) -> np.ndarray:
    array = np.asarray(values)
    imag_max = float(np.max(np.abs(array.imag))) if np.iscomplexobj(array) and array.size else 0.0
    scale = max(float(np.max(np.abs(array.real))) if array.size else 0.0, 1.0)
    tolerance = atol + rtol * scale
    if not np.isfinite(imag_max) or imag_max > tolerance:
        raise FloatingPointError(
            f"{name} should be real, but its imaginary residual is {imag_max:.6g} "
            f"(tolerance={tolerance:.6g})."
        )
    real = np.asarray(array.real, dtype=np.float64)
    if not np.all(np.isfinite(real)):
        raise FloatingPointError(f"{name} contains non-finite values.")
    return real


def _to_k_nv(arr, nv: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        out = arr.reshape(1, nv)
    elif arr.ndim == 2:
        out = arr if arr.shape[1] == nv else arr.T
    else:
        raise ValueError(f"{name} has invalid shape {arr.shape}")
    if out.shape[1] != nv:
        raise ValueError(f"{name} has invalid shape {arr.shape}; expected one dimension to be {nv}.")
    return out.astype(np.complex128, copy=False)


def _integrate_batch_numba(values: np.ndarray, elems: np.ndarray, weights: np.ndarray) -> np.ndarray:
    from .numba_kernels import integrate_batch_numba, integrate_batch_numba_parallel

    if _NUMBA_PARALLEL_ALLOWED.get() and values.shape[1] >= _NUMBA_PARALLEL_COLUMN_THRESHOLD:
        return integrate_batch_numba_parallel(values, elems, weights)
    return integrate_batch_numba(values, elems, weights)


def _integrate_product_numba(
    a: np.ndarray,
    b: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    hermitian: bool,
) -> np.ndarray:
    from .numba_kernels import integrate_product_numba, integrate_product_numba_parallel

    if _NUMBA_PARALLEL_ALLOWED.get() and a.shape[1] >= _NUMBA_PARALLEL_COLUMN_THRESHOLD:
        return integrate_product_numba_parallel(a, b, elems, weights, hermitian)
    return integrate_product_numba(a, b, elems, weights, hermitian)


def _integrate_weighted_columns_numba(
    left: np.ndarray,
    right: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    from .numba_kernels import integrate_weighted_columns_numba, integrate_weighted_columns_numba_parallel

    if _NUMBA_PARALLEL_ALLOWED.get() and right.shape[1] >= _NUMBA_PARALLEL_COLUMN_THRESHOLD:
        return integrate_weighted_columns_numba_parallel(left, right, elems, weights)
    return integrate_weighted_columns_numba(left, right, elems, weights)


def _integrate_weighted_abs2_columns_numba(
    left: np.ndarray,
    right: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    from .numba_kernels import integrate_weighted_abs2_columns_numba, integrate_weighted_abs2_columns_numba_parallel

    if _NUMBA_PARALLEL_ALLOWED.get() and right.shape[1] >= _NUMBA_PARALLEL_COLUMN_THRESHOLD:
        return integrate_weighted_abs2_columns_numba_parallel(left, right, elems, weights)
    return integrate_weighted_abs2_columns_numba(left, right, elems, weights)


def _integrate_weighted_abs2_columns_quadratic_numba(
    left: np.ndarray,
    right: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    from .numba_kernels import (
        integrate_weighted_abs2_columns_quadratic_numba,
        integrate_weighted_abs2_columns_quadratic_numba_parallel,
    )

    if _NUMBA_PARALLEL_ALLOWED.get() and right.shape[1] >= _NUMBA_PARALLEL_COLUMN_THRESHOLD:
        return integrate_weighted_abs2_columns_quadratic_numba_parallel(left, right, elems, weights)
    return integrate_weighted_abs2_columns_quadratic_numba(left, right, elems, weights)


def _integrate_overlap_matrix_numba(
    left: np.ndarray,
    right: np.ndarray,
    weights_vector: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    conjugate_left: bool,
) -> np.ndarray:
    from .numba_kernels import integrate_overlap_matrix_numba, integrate_overlap_matrix_numba_parallel

    work_items = left.shape[0] * right.shape[0]
    if _NUMBA_PARALLEL_ALLOWED.get() and work_items >= _NUMBA_PARALLEL_COLUMN_THRESHOLD:
        return integrate_overlap_matrix_numba_parallel(left, right, weights_vector, elems, weights, conjugate_left)
    return integrate_overlap_matrix_numba(left, right, weights_vector, elems, weights, conjugate_left)


def _integrate_overlap_matrix_quadratic_numba(
    left: np.ndarray,
    right: np.ndarray,
    weights_vector: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    conjugate_left: bool,
) -> np.ndarray:
    from .numba_kernels import integrate_overlap_matrix_quadratic_numba, integrate_overlap_matrix_quadratic_numba_parallel

    work_items = left.shape[0] * right.shape[0]
    if _NUMBA_PARALLEL_ALLOWED.get() and work_items >= _NUMBA_PARALLEL_COLUMN_THRESHOLD:
        return integrate_overlap_matrix_quadratic_numba_parallel(
            left, right, weights_vector, elems, weights, conjugate_left
        )
    return integrate_overlap_matrix_quadratic_numba(left, right, weights_vector, elems, weights, conjugate_left)


def _integrate_overlap_element_matrices_numba(
    left: np.ndarray,
    right: np.ndarray,
    elems: np.ndarray,
    element_matrices: np.ndarray,
    conjugate_left: bool,
) -> np.ndarray:
    from .numba_kernels import (
        integrate_overlap_element_matrices_numba,
        integrate_overlap_element_matrices_numba_parallel,
    )

    work_items = left.shape[0] * right.shape[0]
    if _NUMBA_PARALLEL_ALLOWED.get() and work_items >= _NUMBA_PARALLEL_COLUMN_THRESHOLD:
        return integrate_overlap_element_matrices_numba_parallel(
            left, right, elems, element_matrices, conjugate_left
        )
    return integrate_overlap_element_matrices_numba(
        left, right, elems, element_matrices, conjugate_left
    )


def _integrate_batch(values: np.ndarray, elems: np.ndarray, weights: np.ndarray, chunk_size: int | None) -> np.ndarray:
    k_count = values.shape[1]
    if chunk_size is None:
        chunk_size = max(1, k_count)
    out = np.empty(k_count, dtype=np.complex128)
    for start in range(0, k_count, chunk_size):
        end = min(start + chunk_size, k_count)
        block = values[:, start:end]
        tri_sum = block[elems[:, 0]] + block[elems[:, 1]] + block[elems[:, 2]]
        out[start:end] = np.sum(tri_sum * weights[:, None], axis=0)
    return out


def _integrate_weighted_columns(
    left: np.ndarray,
    right: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    chunk_size: int | None,
) -> np.ndarray:
    k_count = right.shape[1]
    if chunk_size is None:
        chunk_size = max(1, k_count)
    out = np.empty(k_count, dtype=np.complex128)
    for start in range(0, k_count, chunk_size):
        end = min(start + chunk_size, k_count)
        block = right[:, start:end]
        tri_sum = (
            left[elems[:, 0], None] * block[elems[:, 0]]
            + left[elems[:, 1], None] * block[elems[:, 1]]
            + left[elems[:, 2], None] * block[elems[:, 2]]
        )
        out[start:end] = np.sum(tri_sum * weights[:, None], axis=0)
    return out


def _integrate_weighted_abs2_columns(
    left: np.ndarray,
    right: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    chunk_size: int | None,
) -> np.ndarray:
    k_count = right.shape[1]
    if chunk_size is None:
        chunk_size = max(1, k_count)
    out = np.empty(k_count, dtype=np.complex128)
    for start in range(0, k_count, chunk_size):
        end = min(start + chunk_size, k_count)
        block = right[:, start:end]
        mag2 = block.real * block.real + block.imag * block.imag
        tri_sum = (
            left[elems[:, 0], None] * mag2[elems[:, 0]]
            + left[elems[:, 1], None] * mag2[elems[:, 1]]
            + left[elems[:, 2], None] * mag2[elems[:, 2]]
        )
        out[start:end] = np.sum(tri_sum * weights[:, None], axis=0)
    return out


def _weighted_triangle_mass(
    nodal_weights: np.ndarray,
    triangle_weights: np.ndarray,
) -> np.ndarray:
    """Return the exact local mass matrix for a linearly interpolated weight."""

    values = np.asarray(nodal_weights, dtype=np.complex128)
    summed = np.sum(values, axis=1)
    mass = (
        np.asarray(triangle_weights, dtype=np.float64)[:, None, None]
        * (values[:, :, None] + values[:, None, :] + summed[:, None, None])
        / 20.0
    )
    diagonal = np.arange(3)
    mass[:, diagonal, diagonal] *= 2.0
    return mass


def _duffy_phase_weighted_triangle_mass(
    triangle_vertices: np.ndarray,
    nodal_weights: np.ndarray,
    triangle_weights: np.ndarray,
    phase_wavevector: np.ndarray,
    order: int,
) -> np.ndarray:
    nodes, gauss_weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    gauss_weights = 0.5 * gauss_weights
    u, v = np.meshgrid(nodes, nodes, indexing="ij")
    wu, wv = np.meshgrid(gauss_weights, gauss_weights, indexing="ij")
    u = u.reshape(-1)
    v = v.reshape(-1)
    quadrature_weights = (wu * wv).reshape(-1) * (1.0 - u)
    barycentric = np.column_stack(
        ((1.0 - u) * (1.0 - v), u, (1.0 - u) * v)
    )
    points = np.einsum("qa,tad->tqd", barycentric, triangle_vertices, optimize=True)
    metric = np.einsum("qa,ta->tq", barycentric, nodal_weights, optimize=True)
    phase = np.exp(1j * np.einsum("tqd,d->tq", points, phase_wavevector, optimize=True))
    factors = (
        6.0
        * np.asarray(triangle_weights, dtype=np.float64)[:, None]
        * quadrature_weights[None, :]
        * metric
        * phase
    )
    return np.einsum(
        "tq,qi,qj->tij",
        factors,
        barycentric,
        barycentric,
        optimize=True,
    )


def _integrate_weighted_abs2_columns_quadratic(
    left: np.ndarray,
    right: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    chunk_size: int | None,
) -> np.ndarray:
    k_count = right.shape[1]
    if chunk_size is None:
        chunk_size = max(1, k_count)
    out = np.empty(k_count, dtype=np.complex128)
    triangle_block = 4096
    for start in range(0, k_count, chunk_size):
        end = min(start + chunk_size, k_count)
        total = np.zeros(end - start, dtype=np.complex128)
        correction = np.zeros_like(total)
        block = right[:, start:end]
        for tri_start in range(0, elems.shape[0], triangle_block):
            tri_end = min(tri_start + triangle_block, elems.shape[0])
            ids = elems[tri_start:tri_end]
            values = block[ids]
            mass = _weighted_triangle_mass(
                left[ids],
                weights[tri_start:tri_end],
            )
            local = np.einsum(
                "tic,tij,tjc->c",
                np.conj(values),
                mass,
                values,
                optimize=True,
            )
            compensated = local - correction
            updated = total + compensated
            correction = (updated - total) - compensated
            total = updated
        out[start:end] = total
    return out


def _integrate_overlap_matrix(
    left: np.ndarray,
    right: np.ndarray,
    weights_vector: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    conjugate_left: bool,
    chunk_size: int | None,
) -> np.ndarray:
    left_count = left.shape[0]
    right_count = right.shape[0]
    if chunk_size is None:
        chunk_size = max(1, right_count)
    out = np.empty((left_count, right_count), dtype=np.complex128)
    weighted = weights_vector
    for start in range(0, right_count, chunk_size):
        end = min(start + chunk_size, right_count)
        acc = np.zeros((left_count, end - start), dtype=np.complex128)
        rblock = right[start:end]
        for corner in range(3):
            ids = elems[:, corner]
            lvals = left[:, ids]
            if conjugate_left:
                lvals = np.conj(lvals)
            acc += np.einsum(
                "at,bt,t,t->ab",
                lvals,
                rblock[:, ids],
                weighted[ids],
                weights,
                optimize=True,
            )
        out[:, start:end] = acc
    return out


def _integrate_overlap_matrix_quadratic(
    left: np.ndarray,
    right: np.ndarray,
    weights_vector: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    conjugate_left: bool,
    chunk_size: int | None,
) -> np.ndarray:
    left_count = left.shape[0]
    right_count = right.shape[0]
    if chunk_size is None:
        chunk_size = max(1, right_count)
    out = np.empty((left_count, right_count), dtype=np.complex128)
    triangle_block = 4096
    for start in range(0, right_count, chunk_size):
        end = min(start + chunk_size, right_count)
        total = np.zeros((left_count, end - start), dtype=np.complex128)
        correction = np.zeros_like(total)
        rblock = right[start:end]
        for tri_start in range(0, elems.shape[0], triangle_block):
            tri_end = min(tri_start + triangle_block, elems.shape[0])
            ids = elems[tri_start:tri_end]
            lvals = left[:, ids]
            if conjugate_left:
                lvals = np.conj(lvals)
            rvals = rblock[:, ids]
            mass = _weighted_triangle_mass(
                weights_vector[ids],
                weights[tri_start:tri_end],
            )
            local = np.einsum(
                "ati,tij,btj->ab",
                lvals,
                mass,
                rvals,
                optimize=True,
            )
            compensated = local - correction
            updated = total + compensated
            correction = (updated - total) - compensated
            total = updated
        out[:, start:end] = total
    return out


def _integrate_overlap_element_matrices(
    left: np.ndarray,
    right: np.ndarray,
    elems: np.ndarray,
    element_matrices: np.ndarray,
    conjugate_left: bool,
    chunk_size: int | None,
) -> np.ndarray:
    left_count = left.shape[0]
    right_count = right.shape[0]
    if chunk_size is None:
        chunk_size = max(1, right_count)
    out = np.empty((left_count, right_count), dtype=np.complex128)
    triangle_block = 4096
    for start in range(0, right_count, chunk_size):
        end = min(start + chunk_size, right_count)
        total = np.zeros((left_count, end - start), dtype=np.complex128)
        correction = np.zeros_like(total)
        rblock = right[start:end]
        for tri_start in range(0, elems.shape[0], triangle_block):
            tri_end = min(tri_start + triangle_block, elems.shape[0])
            ids = elems[tri_start:tri_end]
            lvals = left[:, ids]
            if conjugate_left:
                lvals = np.conj(lvals)
            rvals = rblock[:, ids]
            local = np.einsum(
                "ati,tij,btj->ab",
                lvals,
                element_matrices[tri_start:tri_end],
                rvals,
                optimize=True,
            )
            compensated = local - correction
            updated = total + compensated
            correction = (updated - total) - compensated
            total = updated
        out[:, start:end] = total
    return out


def _integrate_product(
    a: np.ndarray,
    b: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    hermitian: bool,
    chunk_size: int | None,
) -> np.ndarray:
    k_count = a.shape[1]
    if chunk_size is None:
        chunk_size = max(1, k_count)
    out = np.empty(k_count, dtype=np.complex128)
    for start in range(0, k_count, chunk_size):
        end = min(start + chunk_size, k_count)
        total = np.zeros(end - start, dtype=np.complex128)
        correction = np.zeros(end - start, dtype=np.complex128)
        for tri, (e0, e1, e2) in enumerate(elems):
            a0 = a[e0, start:end]
            a1 = a[e1, start:end]
            a2 = a[e2, start:end]
            if hermitian:
                a0 = np.conj(a0)
                a1 = np.conj(a1)
                a2 = np.conj(a2)
            b0 = b[e0, start:end]
            b1 = b[e1, start:end]
            b2 = b[e2, start:end]
            z = 2.0 * (a0 * b0 + a1 * b1 + a2 * b2)
            z += a0 * b1 + a1 * b0 + a0 * b2 + a2 * b0 + a1 * b2 + a2 * b1
            term = 0.25 * weights[tri] * z
            compensated = term - correction
            updated = total + compensated
            correction = (updated - total) - compensated
            total = updated
        out[start:end] = total
    return out
