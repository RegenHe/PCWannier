from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..conventions import BlochConvention
from ..maxwell import FieldKind


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
class WannierTargetSpec:
    name: str
    center: np.ndarray
    site_irrep: str

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if not self.name or not self.site_irrep:
            raise ValueError("Wannier target name and site_irrep must not be empty.")
        if center.ndim != 1 or not np.all(np.isfinite(center)):
            raise ValueError("Wannier target center must be a finite vector.")
        center = center.copy()
        center.setflags(write=False)
        object.__setattr__(self, "center", center)


@dataclass(frozen=True)
class RepresentationPointSpec:
    name: str
    k_fractional: np.ndarray
    band_indices: tuple[int, ...] | None = None
    target_names: tuple[str, ...] | None = None
    degeneracy_tolerance: DegeneracyTolerance = field(default_factory=DegeneracyTolerance)

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
    leakage_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.leakage_tolerance) or self.leakage_tolerance <= 0.0:
            raise ValueError("Representation leakage tolerance must be positive and finite.")


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


@dataclass(frozen=True)
class SymmetryCalculationSpec:
    target_specs: tuple[WannierTargetSpec, ...] | None = None
    representation_analysis: RepresentationAnalysisSpec | None = None
    symmetry_gauge: SymmetryGaugeSpec | None = None
    bloch_convention: BlochConvention | None = None
    boundary_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.boundary_tolerance is not None and (
            not np.isfinite(self.boundary_tolerance) or self.boundary_tolerance <= 0.0
        ):
            raise ValueError("Symmetry boundary tolerance must be positive and finite.")
