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
) -> np.ndarray:
    selected_backend = resolve_backend(backend)
    view = mesh_integral_view(mesh)
    left = np.asarray(weights_vector, dtype=np.complex128).reshape(view.nv)
    right = _to_nv_k(values, view.nv, "values")
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
) -> np.ndarray:
    selected_backend = resolve_backend(backend)
    view = mesh_integral_view(mesh)
    left = np.asarray(weights_vector, dtype=np.complex128).reshape(view.nv)
    right = _to_nv_k(values, view.nv, "values")
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
) -> np.ndarray:
    selected_backend = resolve_backend(backend)
    view = mesh_integral_view(mesh)
    lmat = _to_k_nv(left, view.nv, "left")
    rmat = _to_k_nv(right, view.nv, "right")
    weights_vector = np.ones(view.nv, dtype=np.complex128) if weights_vector is None else np.asarray(weights_vector, dtype=np.complex128).reshape(view.nv)
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
        a0 = a[elems[:, 0], start:end]
        a1 = a[elems[:, 1], start:end]
        a2 = a[elems[:, 2], start:end]
        if hermitian:
            a0 = np.conj(a0)
            a1 = np.conj(a1)
            a2 = np.conj(a2)
        b0 = b[elems[:, 0], start:end]
        b1 = b[elems[:, 1], start:end]
        b2 = b[elems[:, 2], start:end]
        z = 2.0 * (a0 * b0 + a1 * b1 + a2 * b2)
        z += a0 * b1 + a1 * b0 + a0 * b2 + a2 * b0 + a1 * b2 + a2 * b1
        out[start:end] = np.sum(0.25 * weights[:, None] * z, axis=0)
    return out
