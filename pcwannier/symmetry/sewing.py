from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .group import SpaceGroupOperation
from .specs import FieldKind


@dataclass(frozen=True)
class SewingMatrixRequest:
    operation_index: int
    operation: SpaceGroupOperation
    source_k_fractional: np.ndarray
    target_k_fractional: np.ndarray
    reciprocal_lattice_shift: tuple[int, ...]
    band_indices: tuple[int, ...]
    field_kind: FieldKind
    target_band_indices: tuple[int, ...] | None = None


@runtime_checkable
class SewingMatrixProvider(Protocol):
    def sewing_matrix(self, request: SewingMatrixRequest) -> np.ndarray:
        """Return target-band by source-band sewing coefficients."""

