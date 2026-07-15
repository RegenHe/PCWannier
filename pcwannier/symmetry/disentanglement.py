from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from ..compute.kspace import neighbor_reciprocal_lattice_vectors
from .bloch import StateBlochSymmetryProvider
from .gauge import SymmetryGaugeResult, project_intertwiner
from .representation import SymmetryContext, combined_target_matrix
from .stars import SymmetryKStar, SymmetryStarPartition


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OuterWindowClosureReport:
    max_unitarity_error: float
    mean_unitarity_error: float
    max_leakage: float
    max_composition_residual: float
    matrix_count: int


@dataclass(frozen=True)
class ProjectorSymmetryReport:
    max_projector_residual: float
    mean_projector_residual: float
    max_intertwiner_residual: float
    mean_intertwiner_residual: float
    max_orthonormality_error: float
    max_frozen_residual: float
    max_path_consistency: float


@dataclass(frozen=True)
class SymmetryDisentanglementIteration:
    iteration: int
    omega_i: float
    projector_change: float
    max_projector_symmetry_residual: float
    mean_projector_symmetry_residual: float
    max_intertwiner_residual: float
    orthonormality_error: float
    frozen_window_residual: float
    path_consistency_residual: float
    mixing: float


@dataclass(frozen=True)
class SymmetryDisentanglementResult:
    optimal_frame: np.ndarray
    outer_band_indices: np.ndarray
    stars: SymmetryStarPartition
    closure: OuterWindowClosureReport
    iterations: tuple[SymmetryDisentanglementIteration, ...]
    diagnostics: ProjectorSymmetryReport
    representative_hom_dimensions: tuple[int, ...]
    converged: bool

    @property
    def final_omega_i(self) -> float:
        return self.iterations[-1].omega_i


def outer_band_grid(state) -> np.ndarray:
    grid = np.empty(state.k_shape, dtype=object)
    for index in state.k_indices():
        bands = tuple(int(value) for value in np.asarray(state.E_idx[index]).reshape(-1))
        if not bands:
            raise ValueError(f"Outer window is empty at k={index}.")
        grid[index] = bands
    return grid


def validate_outer_window_closure(
    state,
    context: SymmetryContext,
    provider: StateBlochSymmetryProvider,
    *,
    tolerance: float,
) -> OuterWindowClosureReport:
    bands = outer_band_grid(state)
    errors: list[float] = []
    leakages: list[float] = []
    for operation_index, mappings in enumerate(context.k_mappings):
        operation = context.model.group.operations[operation_index]
        for mapping in mappings:
            source_bands = _bands_at(bands, mapping.source_k_index)
            target_bands = _bands_at(bands, mapping.target_k_index)
            if len(source_bands) != len(target_bands):
                raise RuntimeError(
                    "Outer window is not symmetry closed: "
                    f"operation={operation.name or operation_index}, "
                    f"source_k={mapping.source_k_index} has M={len(source_bands)}, "
                    f"target_k={mapping.target_k_index} has M={len(target_bands)}."
                )
            matrix = provider.sewing_matrix_between_mapping(
                mapping, source_bands, target_bands
            )
            source_error = float(
                np.linalg.norm(matrix.conj().T @ matrix - np.eye(len(source_bands)), ord="fro")
            )
            target_error = float(
                np.linalg.norm(matrix @ matrix.conj().T - np.eye(len(target_bands)), ord="fro")
            )
            error = max(source_error, target_error)
            errors.append(error)
            leakages.append(float(np.sqrt(max(0.0, len(source_bands) - np.linalg.norm(matrix, ord="fro") ** 2))))
            if error > tolerance:
                raise RuntimeError(
                    "Outer window is not closed under symmetry: "
                    f"operation={operation.name or operation_index}, source_k={mapping.source_k_index}, "
                    f"target_k={mapping.target_k_index}, unitarity_residual={error:.6g}."
                )

    composition = _outer_composition_residual(state, context, provider, bands)
    if composition > tolerance:
        raise RuntimeError(
            f"Outer-window sewing composition residual {composition:.6g} exceeds {tolerance:.6g}."
        )
    return OuterWindowClosureReport(
        max(errors, default=0.0),
        float(np.mean(errors)) if errors else 0.0,
        max(leakages, default=0.0),
        composition,
        len(errors),
    )


def validate_frozen_window_covariance(
    initializer,
    context: SymmetryContext,
    provider: StateBlochSymmetryProvider,
    bands: np.ndarray,
    *,
    tolerance: float,
) -> float:
    state = initializer.state
    maximum = 0.0
    for index in state.k_indices():
        outer = set(_bands_at(bands, index))
        frozen_actual = tuple(
            int(value) for value in np.asarray(state.inner_E_idx[index]).reshape(-1)
        )
        missing = sorted(set(frozen_actual) - outer)
        if missing:
            raise RuntimeError(
                f"Frozen bands {missing} at k={index} are outside the configured outer window."
            )

    for operation_index, mappings in enumerate(context.k_mappings):
        operation = context.model.group.operations[operation_index]
        for mapping in mappings:
            source_index = _state_index(mapping.source_k_index)
            target_index = _state_index(mapping.target_k_index)
            source_frozen = _frozen_frame(initializer, source_index)
            target_frozen = _frozen_frame(initializer, target_index)
            if source_frozen.shape[1] != target_frozen.shape[1]:
                raise RuntimeError(
                    "Frozen window is not symmetry closed: "
                    f"operation={operation.name or operation_index}, source_k={mapping.source_k_index} "
                    f"has {source_frozen.shape[1]} states, target_k={mapping.target_k_index} "
                    f"has {target_frozen.shape[1]}."
                )
            dmat = provider.sewing_matrix_between_mapping(
                mapping,
                _bands_at(bands, mapping.source_k_index),
                _bands_at(bands, mapping.target_k_index),
            )
            source_projector = source_frozen @ source_frozen.conj().T
            target_projector = target_frozen @ target_frozen.conj().T
            residual = float(
                np.linalg.norm(
                    target_projector - dmat @ source_projector @ dmat.conj().T,
                    ord="fro",
                )
            )
            maximum = max(maximum, residual)
            if residual > tolerance:
                raise RuntimeError(
                    "Frozen window violates symmetry covariance: "
                    f"operation={operation.name or operation_index}, source_k={mapping.source_k_index}, "
                    f"target_k={mapping.target_k_index}, residual={residual:.6g}."
                )
    return maximum


def disentangle_symmetry_constrained(
    initializer,
    context: SymmetryContext,
    initial_gauge: SymmetryGaugeResult,
    provider: StateBlochSymmetryProvider,
    closure: OuterWindowClosureReport,
    *,
    err_diff: float,
    max_iter: int,
    mixing: float,
    tolerance: float,
    projection_max_iterations: int,
    svd_relative_tolerance: float,
    projector_tolerance: float | None = None,
) -> SymmetryDisentanglementResult:
    if max_iter < 0:
        raise ValueError("disentanglement max_iter must be non-negative.")
    if not np.isfinite(err_diff) or err_diff < 0.0:
        raise ValueError("disentanglement err_diff must be finite and non-negative.")
    if not np.isfinite(mixing) or not 0.0 < mixing <= 1.0:
        raise ValueError("disentanglement mixing must lie in (0, 1].")
    projector_tolerance = tolerance if projector_tolerance is None else projector_tolerance
    if not np.isfinite(projector_tolerance) or projector_tolerance <= 0.0:
        raise ValueError("disentanglement projector tolerance must be positive and finite.")

    state = initializer.state
    stars = initial_gauge.stars
    bands = initial_gauge.band_indices_by_k
    if bands is None:
        bands = outer_band_grid(state)
    frame = _copy_matrix_grid(initial_gauge.gauge)
    validate_frozen_window_covariance(
        initializer, context, provider, bands, tolerance=tolerance
    )
    path_residual = 0.0
    frame, path_residual = _restore_representative_constraints(
        initializer,
        context,
        stars,
        provider,
        bands,
        frame,
        tolerance=tolerance,
        max_iterations=projection_max_iterations,
        svd_relative_tolerance=svd_relative_tolerance,
    )
    report = evaluate_projector_symmetry(
        initializer,
        context,
        provider,
        bands,
        frame,
        path_residual=path_residual,
    )
    _validate_report(report, tolerance, iteration=0)
    omega = _omega_i(initializer, frame)
    history = [_iteration_record(0, omega, 0.0, report, mixing)]
    if max_iter == 0 or not initializer.config.proj_iter:
        return _result(initial_gauge, frame, bands, closure, history, report, True)

    previous_z: dict[int, np.ndarray] = {}
    last_omega = np.inf
    converged = False
    for iteration in range(1, max_iter + 1):
        representatives = []
        for star in stars.stars:
            zmat = _symmetrized_z(initializer, context, provider, bands, frame, star)
            if star.index in previous_z:
                zmat = mixing * zmat + (1.0 - mixing) * previous_z[star.index]
            previous_z[star.index] = zmat
            representatives.append(
                _updated_representative_frame(
                    initializer,
                    context,
                    provider,
                    bands,
                    frame,
                    star,
                    zmat,
                    tolerance=tolerance,
                    max_iterations=projection_max_iterations,
                    svd_relative_tolerance=svd_relative_tolerance,
                )
            )

        propagated, path_residual = _propagate_representatives(
            representatives, context, stars, provider, bands
        )
        change = _projector_change(frame, propagated)
        frame = propagated
        omega = _omega_i(initializer, frame)
        report = evaluate_projector_symmetry(
            initializer,
            context,
            provider,
            bands,
            frame,
            path_residual=path_residual,
        )
        _validate_report(report, tolerance, iteration)
        history.append(_iteration_record(iteration, omega, change, report, mixing))
        err = abs(last_omega - omega)
        LOGGER.info(
            "disentanglement iter %s omega_I=%s err=%s projector_change=%s "
            "projector_symmetry=%s intertwiner=%s orthonormality=%s frozen=%s path=%s mixing=%s",
            iteration,
            omega,
            err,
            change,
            report.max_projector_residual,
            report.max_intertwiner_residual,
            report.max_orthonormality_error,
            report.max_frozen_residual,
            report.max_path_consistency,
            mixing,
        )
        if err < err_diff and change < projector_tolerance:
            converged = True
            break
        last_omega = omega

    if not converged:
        LOGGER.warning("Symmetry disentanglement reached the iteration limit.")
    return _result(initial_gauge, frame, bands, closure, history, report, converged)


def evaluate_projector_symmetry(
    initializer,
    context: SymmetryContext,
    provider: StateBlochSymmetryProvider,
    bands: np.ndarray,
    frame: np.ndarray,
    *,
    path_residual: float = 0.0,
) -> ProjectorSymmetryReport:
    projector_values = []
    frame_values = []
    orthonormality = 0.0
    frozen_residual = 0.0
    targets = context.model.targets
    for index in initializer.state.k_indices():
        matrix = np.asarray(frame[index], dtype=np.complex128)
        projector = matrix @ matrix.conj().T
        orthonormality = max(
            orthonormality,
            float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(matrix.shape[1]), ord="fro")),
        )
        frozen = _frozen_frame(initializer, index)
        if frozen.shape[1]:
            frozen_residual = max(
                frozen_residual,
                float(np.linalg.norm((np.eye(matrix.shape[0]) - projector) @ frozen, ord="fro")),
            )

    for operation_index, mappings in enumerate(context.k_mappings):
        for mapping in mappings:
            source_index = _state_index(mapping.source_k_index)
            target_index = _state_index(mapping.target_k_index)
            source = np.asarray(frame[source_index])
            target = np.asarray(frame[target_index])
            dmat = provider.sewing_matrix_between_mapping(
                mapping,
                _bands_at(bands, mapping.source_k_index),
                _bands_at(bands, mapping.target_k_index),
            )
            source_k = _fractional_at(context, mapping.source_k_index)
            target_matrix = combined_target_matrix(targets, operation_index, source_k)
            frame_values.append(
                float(np.linalg.norm(dmat @ source - target @ target_matrix, ord="fro"))
            )
            projector_values.append(
                float(
                    np.linalg.norm(
                        target @ target.conj().T
                        - dmat @ source @ source.conj().T @ dmat.conj().T,
                        ord="fro",
                    )
                )
            )
    return ProjectorSymmetryReport(
        max(projector_values, default=0.0),
        float(np.mean(projector_values)) if projector_values else 0.0,
        max(frame_values, default=0.0),
        float(np.mean(frame_values)) if frame_values else 0.0,
        orthonormality,
        frozen_residual,
        float(path_residual),
    )


def _symmetrized_z(initializer, context, provider, bands, frame, star: SymmetryKStar):
    representative_member = _representative_member(star)
    dimension = len(_bands_at(bands, star.representative_index))
    accumulator = np.zeros((dimension, dimension), dtype=np.complex128)
    path_count = 0
    for member in star.members:
        raw = _raw_z(initializer, _state_index(member.k_index), frame)
        for path in member.paths:
            dmat = provider.sewing_matrix_between_mapping(
                path,
                _bands_at(bands, path.source_k_index),
                _bands_at(bands, path.target_k_index),
            )
            accumulator += dmat.conj().T @ raw @ dmat
            path_count += 1
    if path_count != len(context.model.group.operations):
        raise RuntimeError(
            f"Star {star.index} has {path_count} symmetry paths; expected "
            f"{len(context.model.group.operations)}."
        )
    little_count = len(representative_member.paths)
    if little_count == 0:
        raise RuntimeError(
            f"Star {star.index} representative has an empty little group."
        )
    # Sakuma Eq. (35) divides the full group-path sum by |G_k|. This leaves
    # one contribution from each distinct member of the symmetry star.
    zmat = accumulator / little_count
    little_accumulator = np.zeros_like(zmat)
    for path in representative_member.paths:
        dmat = provider.sewing_matrix_between_mapping(
            path,
            _bands_at(bands, path.source_k_index),
            _bands_at(bands, path.target_k_index),
        )
        little_accumulator += dmat.conj().T @ zmat @ dmat
    zmat = little_accumulator / len(representative_member.paths)
    return 0.5 * (zmat + zmat.conj().T)


def _raw_z(initializer, index, frame):
    m = len(initializer.state.E_idx[index])
    zmat = np.zeros((m, m), dtype=np.complex128)
    for b in range(len(initializer.config.composition_of_b)):
        neighbor, _ = neighbor_reciprocal_lattice_vectors(initializer.config, list(index), b)
        cb = initializer.mset.get_M0(*index, b) @ frame[neighbor]
        zmat += initializer.config.wb[b] * (cb @ cb.conj().T)
    return 0.5 * (zmat + zmat.conj().T)


def _updated_representative_frame(
    initializer,
    context,
    provider,
    bands,
    frame,
    star,
    zmat,
    *,
    tolerance,
    max_iterations,
    svd_relative_tolerance,
):
    index = _state_index(star.representative_index)
    current = np.asarray(frame[index], dtype=np.complex128)
    frozen = _frozen_frame(initializer, index)
    target_dimension = current.shape[1]
    remaining = target_dimension - frozen.shape[1]
    if remaining < 0:
        raise RuntimeError(f"Frozen window exceeds N_W at k={star.representative_index}.")
    projector_frozen = frozen @ frozen.conj().T
    complement_projector = np.eye(zmat.shape[0]) - projector_frozen
    little_paths = _representative_member(star).paths

    if len(little_paths) == 1:
        restricted = complement_projector @ zmat @ complement_projector
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
        complement = eigenvectors[:, np.argsort(eigenvalues)[::-1][:remaining]] if remaining else np.zeros((zmat.shape[0], 0), complex)
        complement = _orthogonal_complement_columns(complement, frozen, remaining)
        return np.column_stack((frozen, complement))

    candidate = projector_frozen @ current + complement_projector @ zmat @ complement_projector @ current
    representative_k = _fractional_at(context, star.representative_index)
    physical = []
    target = []
    for path in little_paths:
        physical.append(
            provider.sewing_matrix_between_mapping(
                path,
                _bands_at(bands, path.source_k_index),
                _bands_at(bands, path.target_k_index),
            )
        )
        target.append(combined_target_matrix(context.model.targets, path.operation_index, representative_k))

    projected = project_intertwiner(
        candidate,
        physical,
        target,
        tolerance=tolerance,
        max_iterations=max_iterations,
        svd_relative_tolerance=svd_relative_tolerance,
    ).matrix
    frozen_error = _frozen_containment(projected, frozen)
    if frozen_error > tolerance:
        raise RuntimeError(
            f"Frozen and target-representation constraints are incompatible at "
            f"k={star.representative_index}: residual={frozen_error:.6g}."
        )
    return projected


def _restore_representative_constraints(
    initializer,
    context,
    stars,
    provider,
    bands,
    frame,
    *,
    tolerance,
    max_iterations,
    svd_relative_tolerance,
):
    representatives = []
    for star in stars.stars:
        index = _state_index(star.representative_index)
        current = np.asarray(frame[index], dtype=np.complex128)
        frozen = _frozen_frame(initializer, index)
        if _frozen_containment(current, frozen) <= tolerance:
            representatives.append(current)
            continue
        remaining = current.shape[1] - frozen.shape[1]
        candidate = np.column_stack(
            (frozen, _orthogonal_complement_columns(current, frozen, remaining))
        )
        paths = _representative_member(star).paths
        representative_k = _fractional_at(context, star.representative_index)
        physical = [
            provider.sewing_matrix_between_mapping(
                path,
                _bands_at(bands, path.source_k_index),
                _bands_at(bands, path.target_k_index),
            )
            for path in paths
        ]
        target = [
            combined_target_matrix(context.model.targets, path.operation_index, representative_k)
            for path in paths
        ]
        projected = project_intertwiner(
            candidate,
            physical,
            target,
            tolerance=tolerance,
            max_iterations=max_iterations,
            svd_relative_tolerance=svd_relative_tolerance,
        ).matrix
        if _frozen_containment(projected, frozen) > tolerance:
            raise RuntimeError(
                f"Initial target frame does not contain the frozen subspace at k={star.representative_index}."
            )
        representatives.append(projected)
    return _propagate_representatives(representatives, context, stars, provider, bands)


def _propagate_representatives(representatives, context, stars, provider, bands):
    gauge = np.empty(_state_shape(stars.k_shape), dtype=object)
    path_residual = 0.0
    for star, representative in zip(stars.stars, representatives):
        representative_k = _fractional_at(context, star.representative_index)
        for member in star.members:
            candidates = []
            for path in member.paths:
                dmat = provider.sewing_matrix_between_mapping(
                    path,
                    _bands_at(bands, path.source_k_index),
                    _bands_at(bands, path.target_k_index),
                )
                target = combined_target_matrix(
                    context.model.targets, path.operation_index, representative_k
                )
                candidates.append(dmat @ representative @ target.conj().T)
            canonical = representative if member.flat_index == star.representative_flat_index else candidates[0]
            for candidate in candidates:
                path_residual = max(
                    path_residual,
                    float(np.linalg.norm(candidate - canonical, ord="fro")),
                )
            gauge[_state_index(member.k_index)] = canonical
    return gauge, path_residual


def _omega_i(initializer, frame) -> float:
    total = 0.0
    target_dimension = int(initializer.config.band_calc_num)
    for index in initializer.state.k_indices():
        for b, weight in enumerate(initializer.config.wb):
            neighbor, _ = neighbor_reciprocal_lattice_vectors(initializer.config, list(index), b)
            overlap = frame[index].conj().T @ initializer.mset.get_M0(*index, b) @ frame[neighbor]
            total += float(weight) * (target_dimension - np.linalg.norm(overlap, ord="fro") ** 2)
    value = total / initializer.state.get_k_num()
    if not np.isfinite(value):
        raise FloatingPointError("Symmetry disentanglement produced a non-finite Omega_I.")
    return float(np.real(value))


def _outer_composition_residual(state, context, provider, bands) -> float:
    group = context.model.group
    shape = tuple(len(axis) for axis in context.k_points)
    maximum = 0.0
    for source_index in np.ndindex(shape):
        source_k = _fractional_at(context, source_index)
        source_bands = _bands_at(bands, source_index)
        for right_index, right in enumerate(group.operations):
            right_mapping = provider.mapping(right_index, source_index)
            middle_bands = _bands_at(bands, right_mapping.target_k_index)
            right_matrix = provider.sewing_matrix_between_mapping(
                right_mapping, source_bands, middle_bands
            )
            right_k = right.act_reciprocal(source_k)
            middle_index = provider.find_k_index(right_k)
            for left_index, left in enumerate(group.operations):
                left_mapping = provider.mapping(left_index, middle_index)
                final_bands = _bands_at(bands, left_mapping.target_k_index)
                left_request = provider.request_for_mapping(
                    left_mapping,
                    middle_bands,
                    operation=left,
                    source_k_fractional=right_k,
                    target_band_indices=final_bands,
                )
                left_matrix = provider.sewing_matrix(left_request)
                product = left * right
                representative_index = group.operation_index(product)
                product_mapping = provider.mapping(representative_index, source_index)
                product_request = provider.request_for_mapping(
                    product_mapping,
                    source_bands,
                    operation=product,
                    source_k_fractional=source_k,
                    target_band_indices=final_bands,
                )
                product_matrix = provider.sewing_matrix(product_request)
                maximum = max(
                    maximum,
                    float(np.linalg.norm(left_matrix @ right_matrix - product_matrix, ord="fro")),
                )
    return maximum


def _orthogonal_complement_columns(values, frozen, count):
    if count == 0:
        return np.zeros((values.shape[0], 0), dtype=np.complex128)
    array = np.asarray(values, dtype=np.complex128)
    if frozen.shape[1]:
        array = array - frozen @ (frozen.conj().T @ array)
    left, singular, _ = np.linalg.svd(array, full_matrices=True)
    threshold = max(np.finfo(float).eps, (singular[0] if singular.size else 0.0) * 1.0e-12)
    rank = int(np.sum(singular > threshold))
    if rank < count:
        raise RuntimeError(
            f"Subspace update is rank deficient: rank={rank}, required={count}."
        )
    return left[:, :count]


def _frozen_frame(initializer, index):
    m = len(initializer.state.E_idx[index])
    frozen = np.asarray(initializer.I_idx[index], dtype=np.intp).reshape(-1)
    return np.eye(m, dtype=np.complex128)[:, frozen]


def _frozen_containment(frame, frozen):
    if frozen.shape[1] == 0:
        return 0.0
    projector = frame @ frame.conj().T
    return float(np.linalg.norm((np.eye(frame.shape[0]) - projector) @ frozen, ord="fro"))


def _projector_change(left, right):
    return max(
        (
            float(
                np.linalg.norm(
                    left[index] @ left[index].conj().T
                    - right[index] @ right[index].conj().T,
                    ord="fro",
                )
            )
            for index in np.ndindex(left.shape)
        ),
        default=0.0,
    )


def _validate_report(report, tolerance, iteration):
    values = (
        report.max_projector_residual,
        report.max_intertwiner_residual,
        report.max_orthonormality_error,
        report.max_frozen_residual,
        report.max_path_consistency,
    )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"Non-finite symmetry disentanglement diagnostic at iteration {iteration}.")
    if max(values) > tolerance:
        raise RuntimeError(
            f"Symmetry disentanglement residual exceeds {tolerance:.6g} at iteration {iteration}: "
            f"projector={values[0]:.6g}, intertwiner={values[1]:.6g}, "
            f"orthonormality={values[2]:.6g}, frozen={values[3]:.6g}, path={values[4]:.6g}."
        )


def _iteration_record(iteration, omega, change, report, mixing):
    return SymmetryDisentanglementIteration(
        iteration,
        omega,
        change,
        report.max_projector_residual,
        report.mean_projector_residual,
        report.max_intertwiner_residual,
        report.max_orthonormality_error,
        report.max_frozen_residual,
        report.max_path_consistency,
        mixing,
    )


def _result(initial_gauge, frame, bands, closure, history, report, converged):
    return SymmetryDisentanglementResult(
        frame,
        bands,
        initial_gauge.stars,
        closure,
        tuple(history),
        report,
        tuple(item.hom_dimension for item in initial_gauge.representative_diagnostics),
        converged,
    )


def _copy_matrix_grid(values):
    result = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        result[index] = np.asarray(values[index], dtype=np.complex128).copy()
    return result


def _representative_member(star):
    return next(member for member in star.members if member.flat_index == star.representative_flat_index)


def _fractional_at(context, index):
    return np.asarray(
        [context.k_points[axis][index[axis]] for axis in range(context.model.dimension)],
        dtype=float,
    )


def _bands_at(grid, index):
    return tuple(int(value) for value in np.asarray(grid[_state_index(index)]).reshape(-1))


def _state_shape(k_shape):
    return tuple((list(k_shape) + [1, 1, 1])[:3])


def _state_index(index):
    return (tuple(int(value) for value in index) + (0, 0, 0))[:3]
