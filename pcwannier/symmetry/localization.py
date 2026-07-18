from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np
import scipy.linalg

from .bloch import StateBlochSymmetryProvider
from .gauge import GaugeResidualReport, SymmetryGaugeResult, evaluate_symmetry_gauge
from .representation import SymmetryContext, combined_target_matrix
from .stars import SymmetryKStar, SymmetryStarPartition


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
        little_member = _representative_member(star)
        little_count = len(little_member.paths)
        if little_count == 0:
            raise RuntimeError(f"Representative k={star.representative_index} has an empty little group.")

        representative_index = _state_index(star.representative_index)
        accumulator = np.zeros_like(np.asarray(raw_gradient[representative_index], dtype=np.complex128))
        path_count = 0
        for member in star.members:
            member_gradient = np.asarray(raw_gradient[_state_index(member.k_index)], dtype=np.complex128)
            for path in member.paths:
                dmat = combined_target_matrix(targets, path.operation_index, representative_k)
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
            dmat = combined_target_matrix(
                targets, path.operation_index, representative_k
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
                dmat = combined_target_matrix(targets, path.operation_index, representative_k)
                operation = context.model.group.operations[path.operation_index]
                source = matrix.conj() if operation.antiunitary else matrix
                candidates.append(dmat @ source @ dmat.conj().T)
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
                dmat = combined_target_matrix(targets, path.operation_index, representative_k)
                operation = context.model.group.operations[path.operation_index]
                pulled = member_matrix.conj() if operation.antiunitary else member_matrix
                pulled_back.append(dmat.conj().T @ pulled @ dmat)
        matrix = sum(pulled_back) / len(pulled_back)
        little_paths = _representative_member(star).paths
        for iteration in range(1, max_iterations + 1):
            projected_terms = []
            for path in little_paths:
                dmat = combined_target_matrix(
                    targets, path.operation_index, representative_k
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
                            combined_target_matrix(targets, path.operation_index, representative_k)
                            @ (
                                matrix.conj()
                                if context.model.group.operations[path.operation_index].antiunitary
                                else matrix
                            )
                            - matrix
                            @ combined_target_matrix(targets, path.operation_index, representative_k),
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
    gradient.update()
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
        )
    ]
    if max_iter == 0:
        return SymmetryLocalizationResult(tuple(history), True, full_gauge, report)

    last_omega = np.inf
    err = np.inf
    converged = False
    for iteration in range(1, max_iter + 1):
        step_epsilon = float(gradient.epsilon)
        representatives = []
        for star, constrained in zip(initial_gauge.stars.stars, representative_gradients):
            step = step_epsilon * constrained
            step_norm = float(np.linalg.norm(step, ord="fro"))
            if not np.isfinite(step_norm) or step_norm > 100.0:
                raise FloatingPointError(
                    f"Symmetry-constrained gradient step is invalid at k={star.representative_index}: "
                    f"norm={step_norm:.6g}. Reduce epsilon."
                )
            current = np.asarray(gradient.U[_state_index(star.representative_index)])
            representatives.append(current @ scipy.linalg.expm(step))

        propagated = propagate_target_gauge(representatives, context, initial_gauge.stars)
        gradient.U = propagated.gauge
        gradient.mset.update(gradient.U)
        gradient.update()
        full_gauge = _compose_gauge(initial_gauge.gauge, gradient.U)
        report = evaluate_symmetry_gauge(
            state,
            context,
            provider,
            full_gauge,
            initial_gauge.band_indices,
            propagated.max_path_consistency,
            band_indices_by_k=initial_gauge.band_indices_by_k,
        )
        _validate_iteration(report, gradient.omega, representative_gradients, tolerance, iteration)

        total = float(np.sum(gradient.omega))
        gradient_norm = max(
            (float(np.linalg.norm(matrix, ord="fro")) for matrix in representative_gradients),
            default=0.0,
        )
        err = abs(last_omega - total)
        history.append(
            _iteration_record(
                iteration,
                gradient.omega,
                representative_gradients,
                report,
                step_epsilon,
            )
        )
        LOGGER.info(
            "gradient iter %s omega=%s omega_I=%s omega_OD=%s omega_D=%s err=%s "
            "max_gradient_norm=%s symmetry_max=%s symmetry_mean=%s unitarity=%s path=%s",
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
        )
        if err < err_diff:
            converged = True
            break
        if np.isfinite(last_omega) and total > last_omega + max(err_diff, abs(last_omega) * 1.0e-12):
            gradient.epsilon *= 0.5
            LOGGER.warning("Omega increased; gradient step reduced to %s", gradient.epsilon)
        if err < gradient.epsilon * 1.0e-1:
            gradient.epsilon *= 0.1
        last_omega = total
        gradient.calc(is_update=False)
        representative_gradients = symmetrize_gradient(gradient.G, context, initial_gauge.stars)

    if not converged:
        LOGGER.warning("Gradient iteration reached the limit with err=%s", err)
    return SymmetryLocalizationResult(tuple(history), converged, full_gauge, report)


def _iteration_record(
    iteration: int,
    omega: np.ndarray,
    gradients: Sequence[np.ndarray],
    report: GaugeResidualReport,
    epsilon: float,
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
        raise RuntimeError(
            f"Symmetry intertwining residual {report.max_residual:.6g} exceeds {tolerance:.6g} "
            f"at iteration {iteration}."
        )
    if report.max_path_consistency > tolerance:
        raise RuntimeError(
            f"Symmetry path-consistency residual {report.max_path_consistency:.6g} exceeds "
            f"{tolerance:.6g} at iteration {iteration}."
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


def _representative_member(star: SymmetryKStar):
    return next(
        member for member in star.members if member.flat_index == star.representative_flat_index
    )


def _fractional_at(context: SymmetryContext, index) -> np.ndarray:
    return np.asarray(
        [context.k_points[axis][index[axis]] for axis in range(context.model.dimension)],
        dtype=float,
    )


def _state_shape(k_shape: tuple[int, ...]) -> tuple[int, int, int]:
    return tuple((list(k_shape) + [1, 1, 1])[:3])


def _state_index(index) -> tuple[int, int, int]:
    values = list(index) + [0, 0, 0]
    return int(values[0]), int(values[1]), int(values[2])
