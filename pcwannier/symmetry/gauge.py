from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from .bloch import StateBlochSymmetryProvider
from .representation import SymmetryContext, combined_target_matrix
from .stars import SymmetryStarPartition, build_symmetry_stars
from .twisted import TwistedRepresentation, build_little_group_twisted_pair

if TYPE_CHECKING:
    from .wannier_validation import WannierSymmetryValidation


@dataclass(frozen=True)
class IntertwinerSpace:
    basis: tuple[np.ndarray, ...]
    singular_values: np.ndarray
    constraint_rank: int
    dimension: int
    scalar_field: str = "complex"


@dataclass(frozen=True)
class ProjectedIntertwiner:
    matrix: np.ndarray
    iterations: int
    residual: float
    semiunitarity_error: float
    singular_values: np.ndarray


@dataclass(frozen=True)
class RepresentativeGaugeDiagnostics:
    star_index: int
    representative_index: tuple[int, ...]
    little_group_operation_indices: tuple[int, ...]
    hom_dimension: int
    target_commutant_dimension: int
    iterations: int
    projected_singular_values: np.ndarray
    residual: float
    semiunitarity_error: float


@dataclass(frozen=True)
class GaugeResidualReport:
    max_residual: float
    mean_residual: float
    residual_per_k: np.ndarray
    residual_per_operation: dict[str, np.ndarray]
    max_path_consistency: float
    max_semiunitarity_error: float


@dataclass(frozen=True)
class SymmetryGaugeResult:
    gauge: np.ndarray
    band_indices: tuple[int, ...] | None
    stars: SymmetryStarPartition
    representative_diagnostics: tuple[RepresentativeGaugeDiagnostics, ...]
    residuals: GaugeResidualReport
    band_indices_by_k: np.ndarray | None = None
    real_space_validation: WannierSymmetryValidation | None = None

    def bands_at(self, index) -> tuple[int, ...]:
        if self.band_indices_by_k is not None:
            return tuple(int(value) for value in np.asarray(self.band_indices_by_k[_state_index(index)]))
        if self.band_indices is None:
            raise ValueError("Symmetry gauge does not define its physical band window.")
        return self.band_indices


def solve_intertwiner_space(
    physical_matrices: Sequence[np.ndarray] | Mapping[object, np.ndarray],
    target_matrices: Sequence[np.ndarray] | Mapping[object, np.ndarray],
    *,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-12,
) -> IntertwinerSpace:
    """Solve d_g U = U D_g as a common column-major null space."""
    physical, target, antiunitary = _paired_representations(
        physical_matrices, target_matrices
    )
    m = physical[0].shape[0]
    n = target[0].shape[0]
    identity_m = np.eye(m, dtype=np.complex128)
    identity_n = np.eye(n, dtype=np.complex128)
    if any(antiunitary):
        constraints = []
        for dmat, dmat_target, is_antiunitary in zip(
            physical, target, antiunitary
        ):
            left = np.kron(identity_n, dmat)
            right = np.kron(dmat_target.T, identity_m)
            if is_antiunitary:
                coefficient_x = left - right
                coefficient_y = -1j * (left + right)
            else:
                coefficient_x = left - right
                coefficient_y = 1j * (left - right)
            constraints.append(
                np.block(
                    [
                        [coefficient_x.real, coefficient_y.real],
                        [coefficient_x.imag, coefficient_y.imag],
                    ]
                )
            )
        stacked = np.vstack(constraints)
        _, singular_values, vh = np.linalg.svd(stacked, full_matrices=True)
    else:
        constraints = [
            np.kron(identity_n, dmat) - np.kron(dmat_target.T, identity_m)
            for dmat, dmat_target in zip(physical, target)
        ]
        stacked = np.vstack(constraints)
        _, singular_values, vh = np.linalg.svd(stacked, full_matrices=True)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(float(absolute_tolerance), float(relative_tolerance) * largest)
    rank = int(np.sum(singular_values > threshold))
    null_vectors = vh[rank:].conj()
    if any(antiunitary):
        complex_size = m * n
        basis = tuple(
            (vector[:complex_size].real + 1j * vector[complex_size:].real).reshape(
                (m, n), order="F"
            )
            for vector in null_vectors
        )
        scalar_field = "real"
    else:
        basis = tuple(vector.reshape((m, n), order="F") for vector in null_vectors)
        scalar_field = "complex"
    frozen_singular = np.asarray(singular_values, dtype=float)
    frozen_singular.setflags(write=False)
    return IntertwinerSpace(basis, frozen_singular, rank, len(basis), scalar_field)


def project_intertwiner(
    initial: np.ndarray,
    physical_matrices: Sequence[np.ndarray] | Mapping[object, np.ndarray],
    target_matrices: Sequence[np.ndarray] | Mapping[object, np.ndarray],
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 20,
    svd_relative_tolerance: float = 1.0e-10,
) -> ProjectedIntertwiner:
    """Alternate finite-group projection and polar semiunitarization."""
    physical, target, antiunitary = _paired_representations(
        physical_matrices, target_matrices
    )
    matrix = np.asarray(initial, dtype=np.complex128)
    m = physical[0].shape[0]
    n = target[0].shape[0]
    if matrix.shape != (m, n):
        raise ValueError(f"Initial gauge has shape {matrix.shape}; expected {(m, n)}.")
    if m < n:
        raise ValueError("A semiunitary M x N_W gauge requires M >= N_W.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    latest_singular = np.empty(0, dtype=float)
    for iteration in range(1, int(max_iterations) + 1):
        projected = _average_intertwiner(matrix, physical, target, antiunitary)
        try:
            matrix, latest_singular = _polar_semiunitary(
                projected, svd_relative_tolerance
            )
        except RuntimeError:
            # Antiunitary projection is real-linear.  A perfectly valid trial
            # column can therefore be orthogonal to the required real form
            # solely because its phase is 1 instead of i.  Probe only column
            # phases before declaring the physical/target pair incompatible;
            # this preserves the trial subspace and its localization.
            if iteration != 1 or not any(antiunitary):
                raise
            phase_seed = _antiunitary_phase_seed(
                matrix,
                physical,
                target,
                antiunitary,
                svd_relative_tolerance,
            )
            if phase_seed is None:
                raise
            projected = _average_intertwiner(
                phase_seed, physical, target, antiunitary
            )
            matrix, latest_singular = _polar_semiunitary(
                projected, svd_relative_tolerance
            )
        residual = _fixed_k_residual(matrix, physical, target, antiunitary)
        semiunitarity = float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(n), ord="fro"))
        if residual <= tolerance and semiunitarity <= tolerance:
            return ProjectedIntertwiner(
                matrix,
                iteration,
                residual,
                semiunitarity,
                latest_singular,
            )
    raise RuntimeError(
        "Symmetry projection did not converge: "
        f"residual={residual:.6g}, semiunitarity={semiunitarity:.6g}, "
        f"iterations={max_iterations}."
    )


def _average_intertwiner(matrix, physical, target, antiunitary) -> np.ndarray:
    return sum(
        dmat
        @ (matrix.conj() if is_antiunitary else matrix)
        @ dmat_target.conj().T
        for dmat, dmat_target, is_antiunitary in zip(
            physical, target, antiunitary
        )
    ) / len(physical)


def _antiunitary_phase_seed(
    initial,
    physical,
    target,
    antiunitary,
    relative_tolerance: float,
) -> np.ndarray | None:
    """Find a full-rank real-form projection without changing the trial span."""
    column_count = initial.shape[1]
    if column_count <= 12:
        masks = range(1, 1 << column_count)
    else:
        patterns = []
        for column in range(column_count):
            pattern = np.zeros(column_count, dtype=bool)
            pattern[column] = True
            patterns.append(pattern)
        patterns.extend(
            (
                np.arange(column_count) % 2 == 0,
                np.arange(column_count) % 2 == 1,
                np.ones(column_count, dtype=bool),
            )
        )
        rng = np.random.default_rng(0)
        patterns.extend(
            rng.integers(0, 2, size=column_count, dtype=np.int8).astype(bool)
            for _ in range(64)
        )
        masks = patterns

    best_seed = None
    best_ratio = -np.inf
    for item in masks:
        if isinstance(item, (int, np.integer)):
            use_i = np.array(
                [bool(int(item) & (1 << column)) for column in range(column_count)]
            )
        else:
            use_i = np.asarray(item, dtype=bool)
        phases = np.where(use_i, 1.0j, 1.0)
        seed = initial * phases[None, :]
        projected = _average_intertwiner(seed, physical, target, antiunitary)
        singular = np.linalg.svd(projected, compute_uv=False)
        if singular.size != column_count or singular[0] == 0.0:
            continue
        ratio = float(singular[-1] / singular[0])
        if ratio > best_ratio:
            best_ratio = ratio
            best_seed = seed
    if best_seed is None or best_ratio <= float(relative_tolerance):
        return None
    return best_seed


def construct_symmetry_gauge(
    state,
    context: SymmetryContext,
    initial_gauge: np.ndarray,
    *,
    threads: int = 1,
    tolerance: float = 1.0e-8,
    max_iterations: int = 20,
    svd_relative_tolerance: float = 1.0e-10,
    provider: StateBlochSymmetryProvider | None = None,
) -> SymmetryGaugeResult:
    """Construct an isolated-band symmetry gauge on every k point."""
    del threads  # Sewing uses shared arrays; the small gauge algebra is deterministic and ordered.
    targets = context.model.targets
    if not targets:
        raise ValueError("Symmetry gauge construction requires at least one Wannier target.")
    wannier_dimension = sum(target.wannier_dimension for target in targets)
    band_indices_by_k, fixed_band_indices = _band_window(state)
    _validate_initial_gauge(state, initial_gauge, wannier_dimension)

    partition = build_symmetry_stars(context)
    provider = provider or StateBlochSymmetryProvider(state, context)
    gauge = np.empty(state.k_shape, dtype=object)
    representative_diagnostics = []
    path_residual = 0.0

    for star in partition.stars:
        representative_k = _fractional_at(context, star.representative_index)
        representative_state_index = _state_index(star.representative_index)
        representative_member = next(
            member for member in star.members if member.flat_index == star.representative_flat_index
        )
        little_paths = representative_member.paths
        physical = []
        target = []
        operation_indices = []
        for path in little_paths:
            source_bands = _bands_at_grid(band_indices_by_k, path.source_k_index)
            target_bands = _bands_at_grid(band_indices_by_k, path.target_k_index)
            physical.append(
                provider.sewing_matrix_between_mapping(path, source_bands, target_bands)
            )
            target.append(combined_target_matrix(targets, path.operation_index, representative_k))
            operation_indices.append(path.operation_index)

        physical_representation, target_representation = build_little_group_twisted_pair(
            context,
            representative_k,
            operation_indices,
            physical,
            target,
        )

        hom = solve_intertwiner_space(
            physical_representation,
            target_representation,
            relative_tolerance=svd_relative_tolerance,
        )
        if hom.dimension == 0:
            raise RuntimeError(
                f"Target and physical representations are incompatible at representative "
                f"k={star.representative_index}: dim Hom=0."
            )
        commutant = solve_intertwiner_space(
            target_representation,
            target_representation,
            relative_tolerance=svd_relative_tolerance,
        )
        try:
            projected = project_intertwiner(
                initial_gauge[representative_state_index],
                physical_representation,
                target_representation,
                tolerance=tolerance,
                max_iterations=max_iterations,
                svd_relative_tolerance=svd_relative_tolerance,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Cannot construct a full-rank symmetry gauge at representative "
                f"k={star.representative_index}: {exc}"
            ) from exc
        representative_gauge = projected.matrix
        representative_diagnostics.append(
            RepresentativeGaugeDiagnostics(
                star.index,
                star.representative_index,
                tuple(operation_indices),
                hom.dimension,
                commutant.dimension,
                projected.iterations,
                projected.singular_values,
                projected.residual,
                projected.semiunitarity_error,
            )
        )

        for member in star.members:
            candidates = []
            for path in member.paths:
                source_bands = _bands_at_grid(band_indices_by_k, path.source_k_index)
                target_bands = _bands_at_grid(band_indices_by_k, path.target_k_index)
                dmat = provider.sewing_matrix_between_mapping(
                    path, source_bands, target_bands
                )
                target_matrix = combined_target_matrix(targets, path.operation_index, representative_k)
                operation = context.model.group.operations[path.operation_index]
                source_matrix = (
                    representative_gauge.conj()
                    if operation.antiunitary
                    else representative_gauge
                )
                candidates.append(dmat @ source_matrix @ target_matrix.conj().T)
            canonical = candidates[0]
            for candidate in candidates[1:]:
                path_residual = max(
                    path_residual,
                    float(np.linalg.norm(candidate - canonical, ord="fro")),
                )
            gauge[_state_index(member.k_index)] = canonical

    report = evaluate_symmetry_gauge(
        state,
        context,
        provider,
        gauge,
        fixed_band_indices,
        path_residual,
        band_indices_by_k=band_indices_by_k,
    )
    if report.max_residual > tolerance:
        raise RuntimeError(
            f"Symmetry gauge intertwining residual {report.max_residual:.6g} exceeds {tolerance:.6g}."
        )
    if report.max_path_consistency > tolerance:
        raise RuntimeError(
            f"Symmetry gauge path-consistency residual {report.max_path_consistency:.6g} "
            f"exceeds {tolerance:.6g}."
        )
    if report.max_semiunitarity_error > tolerance:
        raise RuntimeError(
            f"Symmetry gauge semiunitarity residual {report.max_semiunitarity_error:.6g} "
            f"exceeds {tolerance:.6g}."
        )
    return SymmetryGaugeResult(
        gauge,
        fixed_band_indices,
        partition,
        tuple(representative_diagnostics),
        report,
        band_indices_by_k,
    )


def evaluate_symmetry_gauge(
    state,
    context: SymmetryContext,
    provider: StateBlochSymmetryProvider,
    gauge: np.ndarray,
    band_indices: tuple[int, ...] | None,
    path_residual: float = 0.0,
    *,
    band_indices_by_k: np.ndarray | None = None,
) -> GaugeResidualReport:
    """Evaluate the full-grid intertwining and semiunitarity residuals."""
    shape = tuple(len(axis) for axis in context.k_points)
    per_operation: dict[str, np.ndarray] = {}
    all_values = []
    residual_per_k = np.zeros(shape, dtype=float)
    targets = context.model.targets
    group = context.model.group
    for operation_index, operation in enumerate(group.operations):
        name = operation.name or f"g{operation_index}"
        values = np.zeros(shape, dtype=float)
        for source_index in np.ndindex(shape):
            mapping = provider.mapping(operation_index, source_index)
            source_bands = _bands_for_evaluation(
                band_indices, band_indices_by_k, mapping.source_k_index
            )
            target_bands = _bands_for_evaluation(
                band_indices, band_indices_by_k, mapping.target_k_index
            )
            dmat = provider.sewing_matrix_between_mapping(
                mapping, source_bands, target_bands
            )
            source_k = _fractional_at(context, source_index)
            target_matrix = combined_target_matrix(targets, operation_index, source_k)
            source_gauge = gauge[_state_index(source_index)]
            target_gauge = gauge[_state_index(mapping.target_k_index)]
            transformed_source = (
                source_gauge.conj() if operation.antiunitary else source_gauge
            )
            value = float(
                np.linalg.norm(
                    dmat @ transformed_source - target_gauge @ target_matrix,
                    ord="fro",
                )
            )
            values[source_index] = value
            residual_per_k[source_index] = max(residual_per_k[source_index], value)
            all_values.append(value)
        values.setflags(write=False)
        per_operation[name] = values

    semiunitarity = 0.0
    n = sum(target.wannier_dimension for target in targets)
    identity = np.eye(n)
    for index in np.ndindex(shape):
        matrix = gauge[_state_index(index)]
        semiunitarity = max(
            semiunitarity,
            float(np.linalg.norm(matrix.conj().T @ matrix - identity, ord="fro")),
        )
    residual_per_k.setflags(write=False)
    return GaugeResidualReport(
        max(all_values, default=0.0),
        float(np.mean(all_values)) if all_values else 0.0,
        residual_per_k,
        per_operation,
        float(path_residual),
        semiunitarity,
    )


def _paired_representations(physical_matrices, target_matrices):
    if isinstance(physical_matrices, TwistedRepresentation) or isinstance(
        target_matrices, TwistedRepresentation
    ):
        if not isinstance(physical_matrices, TwistedRepresentation) or not isinstance(
            target_matrices, TwistedRepresentation
        ):
            raise ValueError(
                "Physical and target representations must both be TwistedRepresentation objects."
            )
        physical_matrices.assert_compatible(target_matrices)
        physical_matrices.require_valid()
        target_matrices.require_valid()
        physical = physical_matrices.matrices
        target = target_matrices.matrices
        antiunitary = physical_matrices.antiunitary_flags
    elif isinstance(physical_matrices, Mapping) or isinstance(target_matrices, Mapping):
        if not isinstance(physical_matrices, Mapping) or not isinstance(target_matrices, Mapping):
            raise ValueError("Physical and target representations must use the same container type.")
        if tuple(physical_matrices) != tuple(target_matrices):
            raise ValueError("Physical and target representation keys must have the same order.")
        physical = tuple(np.asarray(physical_matrices[key], dtype=np.complex128) for key in physical_matrices)
        target = tuple(np.asarray(target_matrices[key], dtype=np.complex128) for key in target_matrices)
        antiunitary = (False,) * len(physical)
    else:
        physical = tuple(np.asarray(matrix, dtype=np.complex128) for matrix in physical_matrices)
        target = tuple(np.asarray(matrix, dtype=np.complex128) for matrix in target_matrices)
        antiunitary = (False,) * len(physical)
    if not physical or len(physical) != len(target):
        raise ValueError("Physical and target representations must contain the same non-zero number of matrices.")
    m = physical[0].shape[0]
    n = target[0].shape[0]
    if any(matrix.shape != (m, m) for matrix in physical):
        raise ValueError("All physical representation matrices must have the same square shape.")
    if any(matrix.shape != (n, n) for matrix in target):
        raise ValueError("All target representation matrices must have the same square shape.")
    if any(not np.all(np.isfinite(matrix)) for matrix in physical + target):
        raise ValueError("Representation matrices contain non-finite values.")
    return physical, target, antiunitary


def _polar_semiunitary(matrix: np.ndarray, relative_tolerance: float):
    left, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(np.finfo(float).eps, float(relative_tolerance) * largest)
    rank = int(np.sum(singular_values > threshold))
    if rank < matrix.shape[1]:
        raise RuntimeError(
            "Symmetry-projected initial gauge is rank deficient: "
            f"rank={rank}, required={matrix.shape[1]}, singular_values={singular_values.tolist()}."
        )
    return left @ vh, np.asarray(singular_values, dtype=float)


def _fixed_k_residual(matrix, physical, target, antiunitary) -> float:
    return max(
        (
            float(
                np.linalg.norm(
                    dmat @ (matrix.conj() if is_antiunitary else matrix)
                    - matrix @ dmat_target,
                    ord="fro",
                )
            )
            for dmat, dmat_target, is_antiunitary in zip(
                physical, target, antiunitary
            )
        ),
        default=0.0,
    )


def _band_window(state) -> tuple[np.ndarray, tuple[int, ...] | None]:
    grid = np.empty(state.k_shape, dtype=object)
    first: tuple[int, ...] | None = None
    fixed = True
    for index in state.k_indices():
        current = tuple(int(value) for value in np.asarray(state.E_idx[index]).reshape(-1))
        if not current:
            raise ValueError(f"The physical outer band window is empty at k={index}.")
        grid[index] = current
        if first is None:
            first = current
        elif current != first:
            fixed = False
    return grid, first if fixed else None


def _validate_initial_gauge(state, initial_gauge, n: int) -> None:
    if np.shape(initial_gauge) != state.k_shape:
        raise ValueError(
            f"Initial gauge k shape {np.shape(initial_gauge)} does not match state shape {state.k_shape}."
        )
    for index in state.k_indices():
        matrix = np.asarray(initial_gauge[index], dtype=np.complex128)
        m = len(state.E_idx[index])
        if m < n:
            raise ValueError(
                f"Physical outer space at k={index} has M={m}, smaller than N_W={n}."
            )
        if matrix.shape != (m, n):
            raise ValueError(f"Initial gauge at k={index} has shape {matrix.shape}; expected {(m, n)}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Initial gauge at k={index} contains non-finite values.")


def _bands_at_grid(grid: np.ndarray, index) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(grid[_state_index(index)]).reshape(-1))


def _bands_for_evaluation(fixed, grid, index) -> tuple[int, ...]:
    if grid is not None:
        return _bands_at_grid(grid, index)
    if fixed is None:
        raise ValueError("A fixed or per-k physical band window is required.")
    return tuple(int(value) for value in fixed)


def _fractional_at(context: SymmetryContext, index) -> np.ndarray:
    return np.asarray(
        [context.k_points[axis][index[axis]] for axis in range(context.model.dimension)],
        dtype=float,
    )


def _state_index(index) -> tuple[int, int, int]:
    values = list(index) + [0, 0, 0]
    return int(values[0]), int(values[1]), int(values[2])
