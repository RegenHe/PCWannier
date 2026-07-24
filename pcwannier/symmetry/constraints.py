from __future__ import annotations

import numpy as np


def semilinear_value(value, antiunitary: bool) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    return matrix.conj() if antiunitary else matrix


def propagate_physical_frame(
    sewing_matrix,
    source_frame,
    target_representation,
    *,
    antiunitary: bool,
) -> np.ndarray:
    sewing = np.asarray(sewing_matrix, dtype=np.complex128)
    source = semilinear_value(source_frame, antiunitary)
    target = np.asarray(target_representation, dtype=np.complex128)
    return sewing @ source @ target.conj().T


def propagate_projector(
    sewing_matrix,
    source_projector,
    *,
    antiunitary: bool,
) -> np.ndarray:
    sewing = np.asarray(sewing_matrix, dtype=np.complex128)
    source = semilinear_value(source_projector, antiunitary)
    return sewing @ source @ sewing.conj().T


def propagate_target_gauge(
    target_representation,
    source_gauge,
    *,
    antiunitary: bool,
) -> np.ndarray:
    target = np.asarray(target_representation, dtype=np.complex128)
    source = semilinear_value(source_gauge, antiunitary)
    return target @ source @ target.conj().T
