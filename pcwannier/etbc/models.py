from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..data import InputBundle


@dataclass(frozen=True)
class ETBCKPointResult:
    auxiliary_frame: np.ndarray = field(repr=False)
    nullspace_coefficients: np.ndarray = field(repr=False)
    singular_values: np.ndarray
    trial_gram_eigenvalues: np.ndarray
    transverse_rank: int
    transverse_orthonormality_error: float
    auxiliary_orthonormality_error: float
    transverse_auxiliary_overlap: float
    augmented_orthonormality_error: float


@dataclass(frozen=True)
class ETBCKPointDiagnostics:
    k_index: tuple[int, int, int]
    transverse_dimension: int
    auxiliary_dimension: int
    singular_values: tuple[float, ...]
    trial_gram_eigenvalues: tuple[float, ...]
    transverse_orthonormality_error: float
    auxiliary_orthonormality_error: float
    transverse_auxiliary_overlap: float
    augmented_orthonormality_error: float
    gamma_regularized: bool = False


@dataclass(frozen=True)
class ETBCCompletionResult:
    augmented_bundle: InputBundle = field(repr=False)
    transverse_dimension: int
    auxiliary_dimension: int
    wannier_dimension: int
    auxiliary_eigenvalue: float
    diagnostics: tuple[ETBCKPointDiagnostics, ...]
    nullspace_coefficients: np.ndarray = field(repr=False)
    trial_projection_matrices: np.ndarray = field(repr=False)
    gamma_regularized_indices: tuple[tuple[int, int, int], ...] = ()

    @property
    def minimum_nonzero_singular_value(self) -> float:
        values = [
            min(item.singular_values)
            for item in self.diagnostics
            if not item.gamma_regularized and item.singular_values
        ]
        return float(min(values)) if values else float("nan")

    @property
    def maximum_transverse_auxiliary_overlap(self) -> float:
        return float(
            max(
                (item.transverse_auxiliary_overlap for item in self.diagnostics),
                default=0.0,
            )
        )

    @property
    def maximum_augmented_orthonormality_error(self) -> float:
        return float(
            max(
                (item.augmented_orthonormality_error for item in self.diagnostics),
                default=0.0,
            )
        )
