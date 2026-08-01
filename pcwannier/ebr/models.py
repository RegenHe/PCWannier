from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _readonly_integer_array(values, *, ndim: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != ndim or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array.")
    rounded = np.rint(raw).astype(np.int64)
    if not np.array_equal(raw, rounded):
        raise ValueError(f"{name} must contain integers.")
    rounded.setflags(write=False)
    return rounded


@dataclass(frozen=True, order=True)
class SymmetryVectorKey:
    point_name: str
    irrep_name: str

    def __post_init__(self) -> None:
        if not str(self.point_name).strip() or not str(self.irrep_name).strip():
            raise ValueError("Symmetry-vector point and irrep names must not be empty.")

    @property
    def label(self) -> str:
        return f"{self.point_name}:{self.irrep_name}"


@dataclass(frozen=True)
class EBRKPoint:
    name: str
    k_fractional: np.ndarray

    def __post_init__(self) -> None:
        point = np.asarray(self.k_fractional, dtype=float)
        if not str(self.name).strip() or point.ndim != 1 or not np.all(np.isfinite(point)):
            raise ValueError("An EBR k point requires a name and a finite fractional vector.")
        point = point.copy()
        point.setflags(write=False)
        object.__setattr__(self, "k_fractional", point)


@dataclass(frozen=True)
class EBRDefinition:
    name: str
    wyckoff: str
    center: np.ndarray
    site_irrep: str

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if (
            not str(self.name).strip()
            or not str(self.wyckoff).strip()
            or not str(self.site_irrep).strip()
            or center.ndim != 1
            or not np.all(np.isfinite(center))
        ):
            raise ValueError("An EBR definition requires non-empty labels and a finite center.")
        center = center.copy()
        center.setflags(write=False)
        object.__setattr__(self, "center", center)


@dataclass(frozen=True)
class EBRCatalog:
    name: str
    space_group_number: int
    hall_number: int
    k_points: tuple[EBRKPoint, ...]
    ebrs: tuple[EBRDefinition, ...]
    source: str | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("EBR catalog name must not be empty.")
        if not 1 <= int(self.space_group_number) <= 230:
            raise ValueError("EBR catalog space_group_number must lie in [1, 230].")
        if not 1 <= int(self.hall_number) <= 530:
            raise ValueError("EBR catalog hall_number must lie in [1, 530].")
        if not self.k_points or not self.ebrs:
            raise ValueError("EBR catalog must define k_points and ebrs.")
        dimensions = {point.k_fractional.size for point in self.k_points}
        dimensions.update(ebr.center.size for ebr in self.ebrs)
        if len(dimensions) != 1:
            raise ValueError("EBR catalog k points and centers must use one common dimension.")
        point_names = [point.name for point in self.k_points]
        ebr_names = [ebr.name for ebr in self.ebrs]
        if len(point_names) != len(set(point_names)):
            raise ValueError("EBR catalog k-point names must be unique.")
        if len(ebr_names) != len(set(ebr_names)):
            raise ValueError("EBR catalog EBR names must be unique.")
        gamma = [
            point
            for point in self.k_points
            if np.max(np.abs(point.k_fractional - np.rint(point.k_fractional))) <= 1.0e-10
        ]
        if len(gamma) != 1:
            raise ValueError("EBR catalog must contain exactly one Gamma-equivalent k point.")

    @property
    def dimension(self) -> int:
        return int(self.k_points[0].k_fractional.size)

    @property
    def gamma_point(self) -> EBRKPoint:
        return next(
            point
            for point in self.k_points
            if np.max(np.abs(point.k_fractional - np.rint(point.k_fractional))) <= 1.0e-10
        )


@dataclass(frozen=True)
class BandSymmetryVector:
    row_keys: tuple[SymmetryVectorKey, ...]
    multiplicities: np.ndarray
    total_dimension: int

    def __post_init__(self) -> None:
        values = _readonly_integer_array(
            self.multiplicities, ndim=1, name="Band symmetry vector"
        )
        if values.shape != (len(self.row_keys),):
            raise ValueError("Band symmetry-vector values must match row_keys.")
        if len(self.row_keys) != len(set(self.row_keys)):
            raise ValueError("Band symmetry-vector row keys must be unique.")
        if int(self.total_dimension) <= 0:
            raise ValueError("Band symmetry-vector dimension must be positive.")
        object.__setattr__(self, "multiplicities", values)
        object.__setattr__(self, "total_dimension", int(self.total_dimension))

    def reordered(self, row_keys) -> "BandSymmetryVector":
        requested = tuple(row_keys)
        if set(requested) != set(self.row_keys):
            missing = sorted(key.label for key in set(requested) - set(self.row_keys))
            extra = sorted(key.label for key in set(self.row_keys) - set(requested))
            raise ValueError(
                f"Symmetry-vector rows do not match the EBR matrix; missing={missing}, extra={extra}."
            )
        positions = {key: index for index, key in enumerate(self.row_keys)}
        return BandSymmetryVector(
            requested,
            np.asarray([self.multiplicities[positions[key]] for key in requested]),
            self.total_dimension,
        )


@dataclass(frozen=True)
class EBRMatrix:
    row_keys: tuple[SymmetryVectorKey, ...]
    columns: tuple[EBRDefinition, ...]
    values: np.ndarray
    dimensions: np.ndarray

    def __post_init__(self) -> None:
        values = _readonly_integer_array(self.values, ndim=2, name="EBR matrix")
        dimensions = _readonly_integer_array(
            self.dimensions, ndim=1, name="EBR dimensions"
        )
        if values.shape != (len(self.row_keys), len(self.columns)):
            raise ValueError("EBR matrix shape does not match its row and column definitions.")
        if dimensions.shape != (len(self.columns),) or np.any(dimensions <= 0):
            raise ValueError("Every EBR matrix column must have a positive dimension.")
        if len(self.row_keys) != len(set(self.row_keys)):
            raise ValueError("EBR matrix row keys must be unique.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "dimensions", dimensions)


@dataclass(frozen=True)
class EBRDecomposition:
    multiplicities: np.ndarray

    def __post_init__(self) -> None:
        values = _readonly_integer_array(
            self.multiplicities, ndim=1, name="EBR decomposition"
        )
        if np.any(values < 0):
            raise ValueError("Ordinary EBR decompositions must be non-negative.")
        object.__setattr__(self, "multiplicities", values)


@dataclass(frozen=True)
class TETBSolution:
    n_t_plus_l: np.ndarray
    n_l: np.ndarray
    n_t: np.ndarray
    gamma_surrogate: np.ndarray
    auxiliary_dimension: int
    physical: bool
    physical_reason: str

    def __post_init__(self) -> None:
        n_tl = _readonly_integer_array(self.n_t_plus_l, ndim=1, name="n_T+L")
        n_l = _readonly_integer_array(self.n_l, ndim=1, name="n_L")
        n_t = _readonly_integer_array(self.n_t, ndim=1, name="n_T")
        surrogate = _readonly_integer_array(
            self.gamma_surrogate, ndim=1, name="Gamma surrogate"
        )
        if n_tl.shape != n_l.shape or n_t.shape != n_l.shape:
            raise ValueError("TETB EBR multiplicity vectors must have the same shape.")
        if np.any(n_tl < 0) or np.any(n_l < 0) or not np.array_equal(n_t, n_tl - n_l):
            raise ValueError("TETB multiplicities must satisfy n_T = n_T+L - n_L.")
        if int(self.auxiliary_dimension) < 0:
            raise ValueError("TETB auxiliary dimension must be non-negative.")
        object.__setattr__(self, "n_t_plus_l", n_tl)
        object.__setattr__(self, "n_l", n_l)
        object.__setattr__(self, "n_t", n_t)
        object.__setattr__(self, "gamma_surrogate", surrogate)
        object.__setattr__(self, "auxiliary_dimension", int(self.auxiliary_dimension))

    @property
    def composite(self) -> bool:
        return bool(np.any(np.minimum(self.n_t_plus_l, self.n_l) > 0))


@dataclass(frozen=True)
class EBRAnalysisResult:
    mode: str
    catalog: EBRCatalog
    symmetry_vector: BandSymmetryVector
    ebr_matrix: EBRMatrix
    regular_decompositions: tuple[EBRDecomposition, ...] = ()
    tetb_solutions: tuple[TETBSolution, ...] = ()
    optimal_auxiliary_dimension: int | None = None
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.mode not in {"regular", "transverse"}:
            raise ValueError("EBR analysis mode must be regular or transverse.")
        if self.symmetry_vector.row_keys != self.ebr_matrix.row_keys:
            raise ValueError("EBR result vector and matrix use different symmetry rows.")

    @property
    def physical_tetb_solutions(self) -> tuple[TETBSolution, ...]:
        return tuple(solution for solution in self.tetb_solutions if solution.physical)
