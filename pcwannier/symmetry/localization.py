from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np
import scipy.linalg

from ..compute.mv_optimizer import (
    MVCandidate,
    copy_gauge,
    diagnose_diagonal_overlaps,
    protected_mv_line_search,
)
from ..logging_utils import should_log_progress
from .bloch import StateBlochSymmetryProvider
from .constraints import propagate_target_gauge as _propagate_target_matrix
from .gauge import GaugeResidualReport, SymmetryGaugeResult, evaluate_symmetry_gauge
from .representation import SymmetryContext
from .stars import (
    SymmetryStarPartition,
    fractional_at as _fractional_at,
    state_index as _state_index,
    state_shape as _state_shape,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetGaugeProjection:
    gauge: np.ndarray
    iterations: int
    max_path_consistency: float
    max_unitarity_error: float


@dataclass(frozen=True)
class TargetGaugePropagation:
    gauge: np.ndarray
    max_path_consistency: float


@dataclass(frozen=True)
class SymmetryLocalizationIteration:
    iteration: int
    omega: float
    omega_i: float
    omega_od: float
    omega_d: float
    gradient_norm: float
    max_intertwiner_residual: float
    mean_intertwiner_residual: float
    max_unitarity_error: float
    max_path_consistency: float
    epsilon: float
    min_abs_diagonal: float
    worst_diagonal_location: tuple[tuple[int, int, int], int, int]
    trial_step: float
    backtracking_steps: int


@dataclass(frozen=True)
class SymmetryLocalizationResult:
    iterations: tuple[SymmetryLocalizationIteration, ...]
    converged: bool
    final_gauge: np.ndarray
    residuals: GaugeResidualReport


def symmetrize_gradient(
    raw_gradient: np.ndarray,
    context: SymmetryContext,
    stars: SymmetryStarPartition,
) -> tuple[np.ndarray, ...]:
    """Pull the full-grid right-action gradient back to star representatives."""
    representative_gradients = []
    targets = context.model.targets
    for star in stars.stars:
        representative_k = _fractional_at(context, star.representative_index)
        little_member = star.representative_member
        little_count = len(little_member.paths)
        if little_count == 0:
            raise RuntimeError(f"Representative k={star.representative_index} has an empty little group.")

        representative_index = _state_index(star.representative_index)
        accumulator = np.zeros_like(np.asarray(raw_gradient[representative_index], dtype=np.complex128))
        path_count = 0
        for member in star.members:
            member_gradient = np.asarray(raw_gradient[_state_index(member.k_index)], dtype=np.complex128)
            for path in member.paths:
                dmat = context.target_matrix(
                    path.operation_index,
                    representative_k,
                    targets=targets,
                )
                operation = context.model.group.operations[path.operation_index]
                pulled = member_gradient.conj() if operation.antiunitary else member_gradient
                accumulator += dmat.conj().T @ pulled @ dmat
                path_count += 1
        if path_count != len(context.model.group.operations):
            raise RuntimeError(
                f"Symmetry star at k={star.representative_index} has {path_count} operation paths; "
                f"expected {len(context.model.group.operations)}."
            )

        # The denominator is |G_k|, so this is equivalently one contribution
        # from every distinct member of the star.
        constrained = accumulator / little_count
        little_projected = []
        for path in little_member.paths:
            dmat = context.target_matrix(
                path.operation_index,
                representative_k,
                targets=targets,
            )
            operation = context.model.group.operations[path.operation_index]
            pulled = constrained.conj() if operation.antiunitary else constrained
            little_projected.append(dmat.conj().T @ pulled @ dmat)
        constrained = sum(little_projected) / little_count
        constrained = 0.5 * (constrained - constrained.conj().T)
        if not np.all(np.isfinite(constrained)):
            raise FloatingPointError(
                f"Symmetry-constrained gradient is non-finite at k={star.representative_index}."
            )
        representative_gradients.append(constrained)
    return tuple(representative_gradients)


def propagate_target_gauge(
    representative_gauge: Sequence[np.ndarray],
    context: SymmetryContext,
    stars: SymmetryStarPartition,
) -> TargetGaugePropagation:
    """Propagate independent target-space gauges to every member of each star."""
    if len(representative_gauge) != len(stars.stars):
        raise ValueError("Representative gauge count does not match the symmetry-star count.")
    shape = _state_shape(stars.k_shape)
    gauge = np.empty(shape, dtype=object)
    max_path_consistency = 0.0
    targets = context.model.targets

    for star, representative in zip(stars.stars, representative_gauge):
        matrix = np.asarray(representative, dtype=np.complex128)
        representative_k = _fractional_at(context, star.representative_index)
        for member in star.members:
            candidates = []
            for path in member.paths:
                dmat = context.target_matrix(
                    path.operation_index,
                    representative_k,
                    targets=targets,
                )
                operation = context.model.group.operations[path.operation_index]
                candidates.append(
                    _propagate_target_matrix(
                        dmat,
                        matrix,
                        antiunitary=operation.antiunitary,
                    )
                )
            if not candidates:
                raise RuntimeError(f"Star member k={member.k_index} has no propagation path.")
            if member.flat_index == star.representative_flat_index:
                canonical = matrix
            else:
                canonical = candidates[0]
            for candidate in candidates:
                max_path_consistency = max(
                    max_path_consistency,
                    float(np.linalg.norm(candidate - canonical, ord="fro")),
                )
            gauge[_state_index(member.k_index)] = canonical
    return TargetGaugePropagation(gauge, max_path_consistency)


def project_target_gauge_to_stars(
    target_gauge: np.ndarray,
    context: SymmetryContext,
    stars: SymmetryStarPartition,
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 20,
    svd_relative_tolerance: float = 1.0e-10,
) -> TargetGaugeProjection:
    """Project a square target-space gauge onto the star and little-group constraints."""
    if np.shape(target_gauge) != _state_shape(stars.k_shape):
        raise ValueError(
            f"Target gauge k shape {np.shape(target_gauge)} does not match {_state_shape(stars.k_shape)}."
        )
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    representatives = []
    max_used_iterations = 0
    targets = context.model.targets
    for star in stars.stars:
        representative_k = _fractional_at(context, star.representative_index)
        pulled_back = []
        for member in star.members:
            member_matrix = np.asarray(target_gauge[_state_index(member.k_index)], dtype=np.complex128)
            if member_matrix.ndim != 2 or member_matrix.shape[0] != member_matrix.shape[1]:
                raise ValueError(
                    f"Target gauge at k={member.k_index} must be a square matrix; "
                    f"got {member_matrix.shape}."
                )
            for path in member.paths:
                dmat = context.target_matrix(
                    path.operation_index,
                    representative_k,
                    targets=targets,
                )
                operation = context.model.group.operations[path.operation_index]
                pulled = member_matrix.conj() if operation.antiunitary else member_matrix
                pulled_back.append(dmat.conj().T @ pulled @ dmat)
        matrix = sum(pulled_back) / len(pulled_back)
        little_paths = star.representative_member.paths
        for iteration in range(1, max_iterations + 1):
            projected_terms = []
            for path in little_paths:
                dmat = context.target_matrix(
                    path.operation_index,
                    representative_k,
                    targets=targets,
                )
                operation = context.model.group.operations[path.operation_index]
                source = matrix.conj() if operation.antiunitary else matrix
                projected_terms.append(dmat.conj().T @ source @ dmat)
            projected = sum(projected_terms) / len(little_paths)
            matrix = _polar_unitary(projected, svd_relative_tolerance)
            little_residual = max(
                (
                    float(
                        np.linalg.norm(
                            context.target_matrix(
                                path.operation_index,
                                representative_k,
                                targets=targets,
                            )
                            @ (
                                matrix.conj()
                                if context.model.group.operations[path.operation_index].antiunitary
                                else matrix
                            )
                            - matrix
                            @ context.target_matrix(
                                path.operation_index,
                                representative_k,
                                targets=targets,
                            ),
                            ord="fro",
                        )
                    )
                    for path in little_paths
                ),
                default=0.0,
            )
            unitarity = float(
                np.linalg.norm(matrix.conj().T @ matrix - np.eye(matrix.shape[0]), ord="fro")
            )
            if little_residual <= tolerance and unitarity <= tolerance:
                max_used_iterations = max(max_used_iterations, iteration)
                break
        else:
            raise RuntimeError(
                f"Target-gauge symmetry projection did not converge at k={star.representative_index}: "
                f"little_group_residual={little_residual:.6g}, unitarity={unitarity:.6g}."
            )
        representatives.append(matrix)

    propagated = propagate_target_gauge(representatives, context, stars)
    max_unitarity = max(
        (
            float(
                np.linalg.norm(
                    np.asarray(propagated.gauge[index]).conj().T @ np.asarray(propagated.gauge[index])
                    - np.eye(np.asarray(propagated.gauge[index]).shape[0]),
                    ord="fro",
                )
            )
            for index in np.ndindex(propagated.gauge.shape)
        ),
        default=0.0,
    )
    return TargetGaugeProjection(
        propagated.gauge,
        max_used_iterations,
        propagated.max_path_consistency,
        max_unitarity,
    )


def localize_symmetry_constrained(
    gradient,
    state,
    context: SymmetryContext,
    initial_gauge: SymmetryGaugeResult,
    provider: StateBlochSymmetryProvider,
    *,
    err_diff: float,
    max_iter: int,
    epsilon: float,
    tolerance: float,
    projection_max_iterations: int,
    svd_relative_tolerance: float,
) -> SymmetryLocalizationResult:
    """Minimize the existing MV spread on the symmetry-compatible gauge manifold."""
    target_dimension = sum(target.wannier_dimension for target in context.model.targets)
    for index in np.ndindex(initial_gauge.gauge.shape):
        matrix = np.asarray(initial_gauge.gauge[index])
        if matrix.ndim != 2 or matrix.shape[1] != target_dimension:
            raise ValueError(
                f"Selected Bloch frame at k={index} has shape {matrix.shape}; "
                f"expected M(k) x N_W with N_W={target_dimension}."
            )
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative.")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite.")
    if not np.isfinite(err_diff) or err_diff < 0.0:
        raise ValueError("err_diff must be finite and non-negative.")

    gradient.epsilon = float(epsilon)
    projected = project_target_gauge_to_stars(
        gradient.U,
        context,
        initial_gauge.stars,
        tolerance=tolerance,
        max_iterations=projection_max_iterations,
        svd_relative_tolerance=svd_relative_tolerance,
    )
    gradient.U = projected.gauge
    gradient.mset.update(gradient.U)
    diagonal_floor = float(getattr(gradient.config, "mv_diagonal_floor", 1.0e-8))
    diagnostics = diagnose_diagonal_overlaps(gradient.mset)
    if diagnostics.min_abs_diagonal < diagonal_floor:
        machine_floor = np.finfo(float).eps * 128.0
        if diagnostics.min_abs_diagonal > machine_floor:
            gradient.update()
        else:
            gradient.omega = np.array([np.nan, np.nan, np.inf], dtype=float)
        full_gauge = _compose_gauge(initial_gauge.gauge, gradient.U)
        report = evaluate_symmetry_gauge(
            state,
            context,
            provider,
            full_gauge,
            initial_gauge.band_indices,
            projected.max_path_consistency,
            band_indices_by_k=initial_gauge.band_indices_by_k,
        )
        LOGGER.warning(
            "Symmetry-constrained MV localization stopped before evaluating 1/M_nn: "
            "min|M_nn|=%.6g at %s is below mv_diagonal_floor=%.6g. "
            "The last symmetry-compatible gauge is preserved.",
            diagnostics.min_abs_diagonal,
            diagnostics.worst_location,
            diagonal_floor,
        )
        zero_gradients = tuple(
            np.zeros_like(np.asarray(gradient.U[_state_index(star.representative_index)]))
            for star in initial_gauge.stars.stars
        )
        history = (
            _iteration_record(
                0,
                gradient.omega,
                zero_gradients,
                report,
                0.0,
                diagnostics,
                trial_step=0.0,
                backtracking_steps=0,
            ),
        )
        return SymmetryLocalizationResult(history, False, full_gauge, report)
    gradient.update()
    if max_iter == 0:
        representative_gradients = tuple(
            np.zeros_like(np.asarray(gradient.U[_state_index(star.representative_index)]))
            for star in initial_gauge.stars.stars
        )
    else:
        gradient.calc(is_update=False)
        representative_gradients = symmetrize_gradient(gradient.G, context, initial_gauge.stars)

    full_gauge = _compose_gauge(initial_gauge.gauge, gradient.U)
    report = evaluate_symmetry_gauge(
        state,
        context,
        provider,
        full_gauge,
        initial_gauge.band_indices,
        projected.max_path_consistency,
        band_indices_by_k=initial_gauge.band_indices_by_k,
    )
    _validate_iteration(report, gradient.omega, representative_gradients, tolerance, 0)
    history = [
        _iteration_record(
            0,
            gradient.omega,
            representative_gradients,
            report,
            gradient.epsilon,
            diagnostics,
            trial_step=0.0,
            backtracking_steps=0,
        )
    ]
    if max_iter == 0:
        return SymmetryLocalizationResult(tuple(history), True, full_gauge, report)

    last_omega = float(np.sum(gradient.omega))
    err = np.inf
    gradient_tolerance = max(float(np.sqrt(max(err_diff, np.finfo(float).eps))), 1.0e-12)
    converged = False
    maximum_step = float(epsilon)
    max_line_search_steps = int(getattr(gradient.config, "mv_line_search_max_steps", 24))
    for iteration in range(1, max_iter + 1):
        baseline = copy_gauge(gradient.U)

        def build_candidate(step_epsilon: float) -> MVCandidate:
            representatives = []
            for star, constrained in zip(initial_gauge.stars.stars, representative_gradients):
                step = step_epsilon * constrained
                step_norm = float(np.linalg.norm(step, ord="fro"))
                if not np.isfinite(step_norm) or step_norm > 100.0:
                    raise FloatingPointError(
                        f"Symmetry-constrained gradient step is invalid at "
                        f"k={star.representative_index}: norm={step_norm:.6g}."
                    )
                current = np.asarray(baseline[_state_index(star.representative_index)])
                representatives.append(current @ scipy.linalg.expm(step))
            propagated = propagate_target_gauge(representatives, context, initial_gauge.stars)
            time_reversal_constraint = getattr(
                gradient, "time_reversal_constraint", None
            )
            if time_reversal_constraint is not None:
                path_consistency = propagated.max_path_consistency
                candidate_gauge = propagated.gauge
                for _ in range(4):
                    candidate_gauge = time_reversal_constraint.project(candidate_gauge).gauge
                    spatial = project_target_gauge_to_stars(
                        candidate_gauge,
                        context,
                        initial_gauge.stars,
                        tolerance=tolerance,
                        max_iterations=projection_max_iterations,
                        svd_relative_tolerance=svd_relative_tolerance,
                    )
                    candidate_gauge = spatial.gauge
                    path_consistency = max(
                        path_consistency,
                        spatial.max_path_consistency,
                    )
                propagated = TargetGaugePropagation(candidate_gauge, path_consistency)
            return MVCandidate(
                propagated.gauge,
                propagated.max_path_consistency,
                propagated,
            )

        allowed_path = max(tolerance, projected.max_path_consistency + tolerance)

        def validate_candidate(candidate: MVCandidate) -> tuple[bool, str | None]:
            if not np.isfinite(candidate.path_consistency):
                return False, "symmetry path-consistency residual is non-finite"
            if candidate.path_consistency > allowed_path:
                return (
                    False,
                    f"path-consistency residual {candidate.path_consistency:.6g} exceeds "
                    f"{allowed_path:.6g}",
                )
            time_reversal_constraint = getattr(
                gradient, "time_reversal_constraint", None
            )
            if time_reversal_constraint is not None:
                residual = time_reversal_constraint.residual(candidate.gauge)
                if residual > max(tolerance, 1.0e-8):
                    return (
                        False,
                        f"time-reversal gauge residual {residual:.6g} exceeds "
                        f"{max(tolerance, 1.0e-8):.6g}",
                    )
            return True, None

        line_search = protected_mv_line_search(
            gradient,
            build_candidate,
            initial_step=gradient.epsilon,
            diagonal_floor=diagonal_floor,
            max_steps=max_line_search_steps,
            candidate_validator=validate_candidate,
        )
        gradient.last_line_search = line_search
        if not line_search.accepted:
            LOGGER.warning(
                "Symmetry-constrained localization stopped after %s backtracking attempts "
                "at formal iteration %s: %s. The last valid gauge is preserved.",
                line_search.backtracking_steps,
                iteration,
                line_search.reason,
            )
            break

        propagated = line_search.metadata
        candidate_full_gauge = _compose_gauge(initial_gauge.gauge, gradient.U)
        candidate_report = evaluate_symmetry_gauge(
            state,
            context,
            provider,
            candidate_full_gauge,
            initial_gauge.band_indices,
            propagated.max_path_consistency,
            band_indices_by_k=initial_gauge.band_indices_by_k,
        )
        _validate_iteration(
            candidate_report,
            gradient.omega,
            representative_gradients,
            tolerance,
            iteration,
        )

        total = float(np.sum(gradient.omega))
        gradient_norm = max(
            (float(np.linalg.norm(matrix, ord="fro")) for matrix in representative_gradients),
            default=0.0,
        )
        full_gauge = candidate_full_gauge
        report = candidate_report
        err = abs(last_omega - total)
        history.append(
            _iteration_record(
                iteration,
                gradient.omega,
                representative_gradients,
                report,
                line_search.trial_step,
                line_search.diagnostics,
                trial_step=line_search.trial_step,
                backtracking_steps=line_search.backtracking_steps,
            )
        )
        finished = err <= err_diff and gradient_norm <= gradient_tolerance
        log = LOGGER.info if should_log_progress(
            iteration, total=max_iter, finished=finished
        ) else LOGGER.debug
        log(
            "gradient iter %s omega=%s omega_I=%s omega_OD=%s omega_D=%s err=%s "
            "max_gradient_norm=%s symmetry_max=%s symmetry_mean=%s unitarity=%s path=%s "
            "accepted_step=%s backtracks=%s min_abs_diagonal=%s worst_diagonal=%s",
            iteration,
            total,
            float(gradient.omega[0]),
            float(gradient.omega[1]),
            float(gradient.omega[2]),
            err,
            gradient_norm,
            report.max_residual,
            report.mean_residual,
            report.max_semiunitarity_error,
            report.max_path_consistency,
            line_search.trial_step,
            line_search.backtracking_steps,
            line_search.diagnostics.min_abs_diagonal,
            line_search.diagnostics.worst_location,
        )
        last_omega = total
        gradient.epsilon = min(maximum_step, line_search.trial_step * 1.5)
        if finished:
            converged = True
            break
        gradient.calc(is_update=False)
        representative_gradients = symmetrize_gradient(gradient.G, context, initial_gauge.stars)

    if not converged:
        LOGGER.warning(
            "Gradient iteration reached the limit with err=%s and max_gradient_norm=%s "
            "(required <= %s).",
            err,
            max(
                (float(np.linalg.norm(matrix, ord="fro")) for matrix in representative_gradients),
                default=0.0,
            ),
            gradient_tolerance,
        )
    return SymmetryLocalizationResult(tuple(history), converged, full_gauge, report)


def _iteration_record(
    iteration: int,
    omega: np.ndarray,
    gradients: Sequence[np.ndarray],
    report: GaugeResidualReport,
    epsilon: float,
    diagnostics,
    *,
    trial_step: float,
    backtracking_steps: int,
) -> SymmetryLocalizationIteration:
    gradient_norm = max(
        (float(np.linalg.norm(matrix, ord="fro")) for matrix in gradients),
        default=0.0,
    )
    return SymmetryLocalizationIteration(
        iteration,
        float(np.sum(omega)),
        float(omega[0]),
        float(omega[1]),
        float(omega[2]),
        gradient_norm,
        report.max_residual,
        report.mean_residual,
        report.max_semiunitarity_error,
        report.max_path_consistency,
        float(epsilon),
        float(diagnostics.min_abs_diagonal),
        diagnostics.worst_location,
        float(trial_step),
        int(backtracking_steps),
    )


def _validate_iteration(
    report: GaugeResidualReport,
    omega: np.ndarray,
    gradients: Sequence[np.ndarray],
    tolerance: float,
    iteration: int,
) -> None:
    if not np.all(np.isfinite(omega)) or any(not np.all(np.isfinite(matrix)) for matrix in gradients):
        raise FloatingPointError(f"Non-finite symmetry localization value at iteration {iteration}.")
    if report.max_residual > tolerance:
        LOGGER.debug(
            "Symmetry intertwining residual %.6g exceeds %.6g at iteration %s; "
            "continuing because the physical sewing input may be only approximately closed.",
            report.max_residual,
            tolerance,
            iteration,
        )
    if report.max_path_consistency > tolerance:
        LOGGER.debug(
            "Symmetry path-consistency residual %.6g exceeds %.6g at iteration %s; "
            "continuing with the canonical symmetry paths.",
            report.max_path_consistency,
            tolerance,
            iteration,
        )
    if report.max_semiunitarity_error > tolerance:
        raise RuntimeError(
            f"Gauge unitarity residual {report.max_semiunitarity_error:.6g} exceeds {tolerance:.6g} "
            f"at iteration {iteration}."
        )


def _compose_gauge(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if np.shape(left) != np.shape(right):
        raise ValueError(f"Gauge k shapes differ: {np.shape(left)} != {np.shape(right)}.")
    result = np.empty(np.shape(left), dtype=object)
    for index in np.ndindex(result.shape):
        result[index] = np.asarray(left[index]) @ np.asarray(right[index])
    return result


def _polar_unitary(matrix: np.ndarray, relative_tolerance: float) -> np.ndarray:
    left, singular_values, vh = np.linalg.svd(np.asarray(matrix, dtype=np.complex128), full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(np.finfo(float).eps, float(relative_tolerance) * largest)
    rank = int(np.sum(singular_values > threshold))
    if rank < matrix.shape[0]:
        raise RuntimeError(
            "Symmetry-projected target gauge is rank deficient: "
            f"rank={rank}, required={matrix.shape[0]}, singular_values={singular_values.tolist()}."
        )
    return left @ vh
