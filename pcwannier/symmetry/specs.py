from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FieldKind(str, Enum):
    SCALAR = "scalar"
    ELECTRIC_POLAR_VECTOR = "electric_polar_vector"
    MAGNETIC_AXIAL_VECTOR = "magnetic_axial_vector"


@dataclass(frozen=True)
class DegeneracyTolerance:
    absolute: float = 1.0e-6
    relative: float = 1.0e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.absolute) or self.absolute < 0.0:
            raise ValueError("Absolute degeneracy tolerance must be finite and non-negative.")
        if not np.isfinite(self.relative) or self.relative < 0.0:
            raise ValueError("Relative degeneracy tolerance must be finite and non-negative.")

    def equivalent(self, left: complex, right: complex) -> bool:
        scale = max(abs(left), abs(right))
        return bool(abs(left - right) <= self.absolute + self.relative * scale)


@dataclass(frozen=True)
class IrrepCharacterSpec:
    name: str
    characters: dict[str, complex] = field(default_factory=dict)
    class_characters: dict[str, complex] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.characters) == bool(self.class_characters):
            raise ValueError(
                f"Irrep {self.name!r} must define exactly one of characters or class_characters."
            )


@dataclass(frozen=True)
class RepresentationPointSpec:
    name: str
    k_fractional: np.ndarray
    band_indices: tuple[int, ...] | None = None
    target_names: tuple[str, ...] | None = None
    degeneracy_tolerance: DegeneracyTolerance = field(default_factory=DegeneracyTolerance)
    conjugacy_classes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    irreps: tuple[IrrepCharacterSpec, ...] = ()

    def __post_init__(self) -> None:
        kpoint = np.asarray(self.k_fractional, dtype=float)
        if kpoint.ndim != 1 or not np.all(np.isfinite(kpoint)):
            raise ValueError("Representation-analysis k point must be a finite vector.")
        if self.band_indices is not None:
            if not self.band_indices or any(index < 0 for index in self.band_indices):
                raise ValueError("Analysis bands must contain non-negative actual band indices.")
            if len(set(self.band_indices)) != len(self.band_indices):
                raise ValueError("Analysis bands must be unique.")
        if self.target_names is not None and len(set(self.target_names)) != len(self.target_names):
            raise ValueError("Analysis target names must be unique.")
        kpoint = kpoint.copy()
        kpoint.setflags(write=False)
        object.__setattr__(self, "k_fractional", kpoint)


@dataclass(frozen=True)
class RepresentationAnalysisSpec:
    field_kind: FieldKind
    degeneracy_tolerance: DegeneracyTolerance
    points: tuple[RepresentationPointSpec, ...]


@dataclass(frozen=True)
class SymmetryGaugeSpec:
    enabled: bool = False
    tolerance: float = 1.0e-8
    max_iterations: int = 20
    svd_relative_tolerance: float = 1.0e-10
    validate_wannier: bool = True
    real_space_tolerance: float = 1.0e-6
    minimum_retained_norm: float = 0.99

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("Symmetry-gauge tolerance must be positive and finite.")
        if self.max_iterations <= 0:
            raise ValueError("Symmetry-gauge max_iterations must be positive.")
        if not np.isfinite(self.svd_relative_tolerance) or self.svd_relative_tolerance <= 0.0:
            raise ValueError("Symmetry-gauge SVD tolerance must be positive and finite.")
        if not np.isfinite(self.real_space_tolerance) or self.real_space_tolerance <= 0.0:
            raise ValueError("Wannier symmetry tolerance must be positive and finite.")
        if not 0.0 < self.minimum_retained_norm <= 1.0:
            raise ValueError("minimum_retained_norm must lie in (0, 1].")
