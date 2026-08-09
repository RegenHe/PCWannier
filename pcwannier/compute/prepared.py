from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..data import InputBundle
    from ..etbc.models import ETBCCompletionResult
    from .state import StateCollection


@dataclass(frozen=True)
class ProjectionSeed:
    """Explicit run-scoped projection matrices supplied by a preparatory stage."""

    matrices: np.ndarray = field(repr=False)
    source: str = "precomputed"

    def matrix(
        self,
        index: tuple[int, int, int],
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        value = np.asarray(self.matrices[index], dtype=np.complex128)
        if value.shape != expected_shape:
            raise ValueError(
                f"{self.source} projection at k={index} has shape {value.shape}; "
                f"expected {expected_shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{self.source} projection at k={index} is non-finite.")
        return value


@dataclass(frozen=True)
class PreparedRun:
    """Immutable ownership boundary for run-derived calculation inputs."""

    bundle: InputBundle
    state: StateCollection
    orthogonality_report: np.ndarray = field(repr=False)
    projection_seed: ProjectionSeed | None = field(default=None, repr=False)
    etbc: ETBCCompletionResult | None = None
    trial_covariance_diagnostics: tuple = ()
