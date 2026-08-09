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
    space_group_number: int | None
    hall_number: int | None
    k_points: tuple[EBRKPoint, ...]
    ebrs: tuple[EBRDefinition, ...]
    source: str | None = None
    space_group_name: str | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("EBR catalog name must not be empty.")
        if not self.k_points or not self.ebrs:
            raise ValueError("EBR catalog must define k_points and ebrs.")
        dimensions = {point.k_fractional.size for point in self.k_points}
        dimensions.update(ebr.center.size for ebr in self.ebrs)
        if len(dimensions) != 1:
            raise ValueError("EBR catalog k points and centers must use one common dimension.")
        number = self.space_group_number
        hall = self.hall_number
        group_name = None if self.space_group_name is None else str(self.space_group_name).strip()
        if (number is None) != (hall is None):
            raise ValueError(
                "EBR catalog space_group_number and hall_number must be provided together."
            )
        if number is not None:
            if not 1 <= int(number) <= 230:
                raise ValueError("EBR catalog space_group_number must lie in [1, 230].")
            if not 1 <= int(hall) <= 530:
                raise ValueError("EBR catalog hall_number must lie in [1, 530].")
            object.__setattr__(self, "space_group_number", int(number))
            object.__setattr__(self, "hall_number", int(hall))
        elif not group_name:
            raise ValueError(
                "A non-Hall EBR catalog requires space_group_name."
            )
        object.__setattr__(self, "space_group_name", group_name)
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
class SignedEBRCombination:
    multiplicities: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "multiplicities",
            _readonly_integer_array(
                self.multiplicities, ndim=1, name="signed EBR combination"
            ),
        )


@dataclass(frozen=True)
class TETBCompletion:
    n_t_plus_l: np.ndarray
    n_l: np.ndarray
    auxiliary_dimension: int

    def __post_init__(self) -> None:
        n_tl = _readonly_integer_array(self.n_t_plus_l, ndim=1, name="n_T+L")
        n_l = _readonly_integer_array(self.n_l, ndim=1, name="n_L")
        if n_tl.shape != n_l.shape or np.any(n_tl < 0) or np.any(n_l < 0):
            raise ValueError("TETB completion multiplicities must be matching non-negative vectors.")
        if isinstance(self.auxiliary_dimension, bool) or int(self.auxiliary_dimension) < 0:
            raise ValueError("TETB completion auxiliary dimension must be non-negative.")
        object.__setattr__(self, "n_t_plus_l", n_tl)
        object.__setattr__(self, "n_l", n_l)
        object.__setattr__(self, "auxiliary_dimension", int(self.auxiliary_dimension))


@dataclass(frozen=True)
class EBRSubspacePointRepresentation:
    point_name: str
    irrep_multiplicities: tuple[tuple[str, int], ...]
    unresolved_dimension: int = 0
    alternative_count: int = 1
    channel_dimension_realizations: tuple[
        tuple[tuple[str, int], ...], ...
    ] = ()
    channel_assignment_complete: bool = True

    def __post_init__(self) -> None:
        if not str(self.point_name).strip():
            raise ValueError("A subspace point representation requires a point name.")
        multiplicities = tuple(
            (str(name), int(value)) for name, value in self.irrep_multiplicities
        )
        if any(not name.strip() or value <= 0 for name, value in multiplicities):
            raise ValueError("Physical irrep multiplicities must be positive.")
        if len({name for name, _ in multiplicities}) != len(multiplicities):
            raise ValueError("Physical irrep names must be unique at each k point.")
        if int(self.unresolved_dimension) < 0:
            raise ValueError("unresolved_dimension must be non-negative.")
        if int(self.alternative_count) <= 0:
            raise ValueError("alternative_count must be positive.")
        channel_realizations = tuple(
            tuple((str(name), int(value)) for name, value in realization)
            for realization in self.channel_dimension_realizations
        )
        for realization in channel_realizations:
            names = tuple(name for name, _ in realization)
            if any(not name.strip() or value < 0 for name, value in realization):
                raise ValueError("Channel dimensions must be named non-negative integers.")
            if len(names) != len(set(names)):
                raise ValueError("A channel realization may contain each channel only once.")
        if len(channel_realizations) != len(set(channel_realizations)):
            raise ValueError("Channel dimension realizations must be unique.")
        object.__setattr__(self, "irrep_multiplicities", multiplicities)
        object.__setattr__(self, "unresolved_dimension", int(self.unresolved_dimension))
        object.__setattr__(self, "alternative_count", int(self.alternative_count))
        object.__setattr__(
            self, "channel_dimension_realizations", channel_realizations
        )
        object.__setattr__(
            self, "channel_assignment_complete", bool(self.channel_assignment_complete)
        )


@dataclass(frozen=True)
class EBRSubspaceCandidate:
    solution: TETBSolution
    selected_symmetry_vector: BandSymmetryVector
    point_representations: tuple[EBRSubspacePointRepresentation, ...]
    includes_gamma_zero_modes: bool = False
    gamma_sector: str = "ordinary"

    def __post_init__(self) -> None:
        names = tuple(item.point_name for item in self.point_representations)
        if len(names) != len(set(names)):
            raise ValueError("A subspace candidate may contain each k point only once.")
        sector = str(self.gamma_sector).strip()
        if sector not in {
            "ordinary",
            "include_gamma_zero_modes",
            "exclude_gamma_zero_modes",
        }:
            raise ValueError(f"Unknown EBR Gamma sector {sector!r}.")
        object.__setattr__(self, "gamma_sector", sector)

    @property
    def block_realization_count(self) -> int:
        count = 1
        for item in self.point_representations:
            count *= item.alternative_count
        return count


@dataclass(frozen=True)
class EBRSearchStatistics:
    complete: bool
    gamma_sectors: tuple[str, ...] = ()
    auxiliary_dimensions_examined: tuple[int, ...] = ()
    weighted_vectors_generated: int = 0
    signed_combinations_tested: int = 0
    algebraic_solutions: int = 0
    unique_signed_solutions: int = 0
    completion_solutions: int = 0
    realizable_candidates: int = 0
    block_realization_count: int = 0
    search_limit: int = 0

    def __post_init__(self) -> None:
        counters = (
            self.weighted_vectors_generated,
            self.signed_combinations_tested,
            self.algebraic_solutions,
            self.unique_signed_solutions,
            self.completion_solutions,
            self.realizable_candidates,
            self.block_realization_count,
            self.search_limit,
        )
        if any(isinstance(value, bool) or int(value) < 0 for value in counters):
            raise ValueError("EBR search counters must be non-negative integers.")
        object.__setattr__(
            self,
            "gamma_sectors",
            tuple(str(value).strip() for value in self.gamma_sectors),
        )
        auxiliary = tuple(int(value) for value in self.auxiliary_dimensions_examined)
        if any(value < 0 for value in auxiliary) or len(auxiliary) != len(set(auxiliary)):
            raise ValueError("Examined auxiliary dimensions must be unique non-negative integers.")
        object.__setattr__(self, "auxiliary_dimensions_examined", auxiliary)


@dataclass(frozen=True)
class EBRSubspaceEnumeration:
    solutions: tuple[tuple[TETBSolution, BandSymmetryVector], ...]
    statistics: EBRSearchStatistics
    completion_groups: tuple[
        tuple[SignedEBRCombination, tuple[TETBCompletion, ...]], ...
    ] = ()


@dataclass(frozen=True)
class EBRAnalysisResult:
    mode: str
    catalog: EBRCatalog
    symmetry_vector: BandSymmetryVector
    ebr_matrix: EBRMatrix
    regular_decompositions: tuple[EBRDecomposition, ...] = ()
    tetb_solutions: tuple[TETBSolution, ...] = ()
    subspace_candidates: tuple[EBRSubspaceCandidate, ...] = ()
    optimal_auxiliary_dimension: int | None = None
    diagnostics: tuple[str, ...] = field(default_factory=tuple)
    subspace_fixed_band_indices: tuple[int, ...] = ()
    search_statistics: EBRSearchStatistics | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"regular", "transverse", "subspace"}:
            raise ValueError("EBR analysis mode must be regular, transverse, or subspace.")
        if self.symmetry_vector.row_keys != self.ebr_matrix.row_keys:
            raise ValueError("EBR result vector and matrix use different symmetry rows.")
        fixed = tuple(int(value) for value in self.subspace_fixed_band_indices)
        if any(value < 0 for value in fixed) or len(fixed) != len(set(fixed)):
            raise ValueError("Fixed EBR subspace bands must be unique non-negative indices.")
        object.__setattr__(self, "subspace_fixed_band_indices", fixed)

    @property
    def physical_tetb_solutions(self) -> tuple[TETBSolution, ...]:
        return tuple(solution for solution in self.tetb_solutions if solution.physical)
