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
        correction = 0.0 + 0.0j
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
            term = 0.25 * weights[tri] * z
            compensated = term - correction
            updated = total + compensated
            correction = (updated - total) - compensated
            total = updated
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
        correction = 0.0 + 0.0j
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
            term = 0.25 * weights[tri] * z
            compensated = term - correction
            updated = total + compensated
            correction = (updated - total) - compensated
            total = updated
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


@njit(nogil=True)
def integrate_weighted_abs2_columns_numba(
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
            v0 = right[e0, col]
            v1 = right[e1, col]
            v2 = right[e2, col]
            total += (
                left[e0] * (v0.real * v0.real + v0.imag * v0.imag)
                + left[e1] * (v1.real * v1.real + v1.imag * v1.imag)
                + left[e2] * (v2.real * v2.real + v2.imag * v2.imag)
            ) * weights[tri]
        out[col] = total
    return out


@njit(nogil=True, parallel=True)
def integrate_weighted_abs2_columns_numba_parallel(
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
            v0 = right[e0, col]
            v1 = right[e1, col]
            v2 = right[e2, col]
            total += (
                left[e0] * (v0.real * v0.real + v0.imag * v0.imag)
                + left[e1] * (v1.real * v1.real + v1.imag * v1.imag)
                + left[e2] * (v2.real * v2.real + v2.imag * v2.imag)
            ) * weights[tri]
        out[col] = total
    return out


@njit(nogil=True)
def integrate_overlap_matrix_numba(
    left: np.ndarray,
    right: np.ndarray,
    weights_vector: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    conjugate_left: bool,
) -> np.ndarray:
    left_count = left.shape[0]
    right_count = right.shape[0]
    out = np.empty((left_count, right_count), dtype=np.complex128)
    for row in range(left_count):
        for col in range(right_count):
            total = 0.0 + 0.0j
            for tri in range(elems.shape[0]):
                e0 = elems[tri, 0]
                e1 = elems[tri, 1]
                e2 = elems[tri, 2]
                l0 = left[row, e0]
                l1 = left[row, e1]
                l2 = left[row, e2]
                if conjugate_left:
                    l0 = np.conj(l0)
                    l1 = np.conj(l1)
                    l2 = np.conj(l2)
                total += (
                    l0 * weights_vector[e0] * right[col, e0]
                    + l1 * weights_vector[e1] * right[col, e1]
                    + l2 * weights_vector[e2] * right[col, e2]
                ) * weights[tri]
            out[row, col] = total
    return out


@njit(nogil=True, parallel=True)
def integrate_overlap_matrix_numba_parallel(
    left: np.ndarray,
    right: np.ndarray,
    weights_vector: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    conjugate_left: bool,
) -> np.ndarray:
    left_count = left.shape[0]
    right_count = right.shape[0]
    out = np.empty((left_count, right_count), dtype=np.complex128)
    for linear in prange(left_count * right_count):
        row = linear // right_count
        col = linear % right_count
        total = 0.0 + 0.0j
        for tri in range(elems.shape[0]):
            e0 = elems[tri, 0]
            e1 = elems[tri, 1]
            e2 = elems[tri, 2]
            l0 = left[row, e0]
            l1 = left[row, e1]
            l2 = left[row, e2]
            if conjugate_left:
                l0 = np.conj(l0)
                l1 = np.conj(l1)
                l2 = np.conj(l2)
            total += (
                l0 * weights_vector[e0] * right[col, e0]
                + l1 * weights_vector[e1] * right[col, e1]
                + l2 * weights_vector[e2] * right[col, e2]
            ) * weights[tri]
        out[row, col] = total
    return out


@njit(nogil=True)
def integrate_overlap_matrix_quadratic_numba(
    left: np.ndarray,
    right: np.ndarray,
    weights_vector: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    conjugate_left: bool,
) -> np.ndarray:
    left_count = left.shape[0]
    right_count = right.shape[0]
    out = np.empty((left_count, right_count), dtype=np.complex128)
    for row in range(left_count):
        for col in range(right_count):
            total = 0.0 + 0.0j
            correction = 0.0 + 0.0j
            for tri in range(elems.shape[0]):
                e0 = elems[tri, 0]
                e1 = elems[tri, 1]
                e2 = elems[tri, 2]
                l0 = left[row, e0]
                l1 = left[row, e1]
                l2 = left[row, e2]
                if conjugate_left:
                    l0 = np.conj(l0)
                    l1 = np.conj(l1)
                    l2 = np.conj(l2)
                l0 *= weights_vector[e0]
                l1 *= weights_vector[e1]
                l2 *= weights_vector[e2]
                r0 = right[col, e0]
                r1 = right[col, e1]
                r2 = right[col, e2]
                z = 2.0 * (l0 * r0 + l1 * r1 + l2 * r2)
                z += l0 * r1 + l1 * r0 + l0 * r2 + l2 * r0 + l1 * r2 + l2 * r1
                term = 0.25 * weights[tri] * z
                compensated = term - correction
                updated = total + compensated
                correction = (updated - total) - compensated
                total = updated
            out[row, col] = total
    return out


@njit(nogil=True, parallel=True)
def integrate_overlap_matrix_quadratic_numba_parallel(
    left: np.ndarray,
    right: np.ndarray,
    weights_vector: np.ndarray,
    elems: np.ndarray,
    weights: np.ndarray,
    conjugate_left: bool,
) -> np.ndarray:
    left_count = left.shape[0]
    right_count = right.shape[0]
    out = np.empty((left_count, right_count), dtype=np.complex128)
    for linear in prange(left_count * right_count):
        row = linear // right_count
        col = linear % right_count
        total = 0.0 + 0.0j
        correction = 0.0 + 0.0j
        for tri in range(elems.shape[0]):
            e0 = elems[tri, 0]
            e1 = elems[tri, 1]
            e2 = elems[tri, 2]
            l0 = left[row, e0]
            l1 = left[row, e1]
            l2 = left[row, e2]
            if conjugate_left:
                l0 = np.conj(l0)
                l1 = np.conj(l1)
                l2 = np.conj(l2)
            l0 *= weights_vector[e0]
            l1 *= weights_vector[e1]
            l2 *= weights_vector[e2]
            r0 = right[col, e0]
            r1 = right[col, e1]
            r2 = right[col, e2]
            z = 2.0 * (l0 * r0 + l1 * r1 + l2 * r2)
            z += l0 * r1 + l1 * r0 + l0 * r2 + l2 * r0 + l1 * r2 + l2 * r1
            term = 0.25 * weights[tri] * z
            compensated = term - correction
            updated = total + compensated
            correction = (updated - total) - compensated
            total = updated
        out[row, col] = total
    return out
