from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(nogil=True)
def integrate_batch_numba(values: np.ndarray, elems: np.ndarray, weights: np.ndarray) -> np.ndarray:
    k_count = values.shape[1]
    out = np.empty(k_count, dtype=np.complex128)
    for col in range(k_count):
        total = 0.0 + 0.0j
        for tri in range(elems.shape[0]):
            e0 = elems[tri, 0]
            e1 = elems[tri, 1]
            e2 = elems[tri, 2]
            total += (values[e0, col] + values[e1, col] + values[e2, col]) * weights[tri]
        out[col] = total
    return out


@njit(nogil=True, parallel=True)
def integrate_batch_numba_parallel(values: np.ndarray, elems: np.ndarray, weights: np.ndarray) -> np.ndarray:
    k_count = values.shape[1]
    out = np.empty(k_count, dtype=np.complex128)
    for col in prange(k_count):
        total = 0.0 + 0.0j
        for tri in range(elems.shape[0]):
            e0 = elems[tri, 0]
            e1 = elems[tri, 1]
            e2 = elems[tri, 2]
            total += (values[e0, col] + values[e1, col] + values[e2, col]) * weights[tri]
        out[col] = total
    return out


@njit(nogil=True)
def integrate_product_numba(
    a: np.ndarray,
    b: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    hermitian: bool,
) -> np.ndarray:
    k_count = a.shape[1]
    out = np.empty(k_count, dtype=np.complex128)
    for col in range(k_count):
        total = 0.0 + 0.0j
        for tri in range(elems.shape[0]):
            e0 = elems[tri, 0]
            e1 = elems[tri, 1]
            e2 = elems[tri, 2]

            a0 = a[e0, col]
            a1 = a[e1, col]
            a2 = a[e2, col]
            if hermitian:
                a0 = np.conj(a0)
                a1 = np.conj(a1)
                a2 = np.conj(a2)

            b0 = b[e0, col]
            b1 = b[e1, col]
            b2 = b[e2, col]
            z = 2.0 * (a0 * b0 + a1 * b1 + a2 * b2)
            z += a0 * b1 + a1 * b0 + a0 * b2 + a2 * b0 + a1 * b2 + a2 * b1
            total += 0.25 * weights[tri] * z
        out[col] = total
    return out


@njit(nogil=True, parallel=True)
def integrate_product_numba_parallel(
    a: np.ndarray,
    b: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    hermitian: bool,
) -> np.ndarray:
    k_count = a.shape[1]
    out = np.empty(k_count, dtype=np.complex128)
    for col in prange(k_count):
        total = 0.0 + 0.0j
        for tri in range(elems.shape[0]):
            e0 = elems[tri, 0]
            e1 = elems[tri, 1]
            e2 = elems[tri, 2]

            a0 = a[e0, col]
            a1 = a[e1, col]
            a2 = a[e2, col]
            if hermitian:
                a0 = np.conj(a0)
                a1 = np.conj(a1)
                a2 = np.conj(a2)

            b0 = b[e0, col]
            b1 = b[e1, col]
            b2 = b[e2, col]
            z = 2.0 * (a0 * b0 + a1 * b1 + a2 * b2)
            z += a0 * b1 + a1 * b0 + a0 * b2 + a2 * b0 + a1 * b2 + a2 * b1
            total += 0.25 * weights[tri] * z
        out[col] = total
    return out


@njit(nogil=True)
def integrate_weighted_columns_numba(
    left: np.ndarray,
    right: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    k_count = right.shape[1]
    out = np.empty(k_count, dtype=np.complex128)
    for col in range(k_count):
        total = 0.0 + 0.0j
        for tri in range(elems.shape[0]):
            e0 = elems[tri, 0]
            e1 = elems[tri, 1]
            e2 = elems[tri, 2]
            total += (
                left[e0] * right[e0, col]
                + left[e1] * right[e1, col]
                + left[e2] * right[e2, col]
            ) * weights[tri]
        out[col] = total
    return out


@njit(nogil=True, parallel=True)
def integrate_weighted_columns_numba_parallel(
    left: np.ndarray,
    right: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    k_count = right.shape[1]
    out = np.empty(k_count, dtype=np.complex128)
    for col in prange(k_count):
        total = 0.0 + 0.0j
        for tri in range(elems.shape[0]):
            e0 = elems[tri, 0]
            e1 = elems[tri, 1]
            e2 = elems[tri, 2]
            total += (
                left[e0] * right[e0, col]
                + left[e1] * right[e1, col]
                + left[e2] * right[e2, col]
            ) * weights[tri]
        out[col] = total
    return out
