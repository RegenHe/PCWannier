from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class BlochConvention:
    """Sign convention in psi_k = exp(sign * i k.r) u_k."""

    sign: int = 1

    def __post_init__(self) -> None:
        if self.sign not in {-1, 1}:
            raise ValueError("Bloch convention sign must be -1 or 1.")


class SpatialDiscretization(str, Enum):
    """Source-neutral spatial representation used by numerical algorithms."""

    TRIANGLE_FEM_P1 = "triangle_fem_p1"
    PERIODIC_FOURIER_COLLOCATION = "periodic_fourier_collocation"


class BlochFieldRepresentation(str, Enum):
    """Representation stored by an external field-data source."""

    FULL_BLOCH = "full_bloch"
    PERIODIC_PART = "periodic_part"
