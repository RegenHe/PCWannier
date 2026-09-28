from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class MVOverlapDiagnostics:
    min_abs_diagonal: float
    worst_k_index: tuple[int, int, int]
    worst_direction: int
    worst_band: int

    @property
    def worst_location(self) -> tuple[tuple[int, int, int], int, int]:
        return self.worst_k_index, self.worst_direction, self.worst_band


@dataclass(frozen=True)
class MVCandidate:
    gauge: np.ndarray
    path_consistency: float = 0.0
    metadata: Any = None


@dataclass(frozen=True)
class MVLineSearchResult:
    accepted: bool
    diagnostics: MVOverlapDiagnostics
    trial_step: float
    backtracking_steps: int
    metadata: Any = None
    reason: str | None = None


def copy_gauge(gauge: np.ndarray) -> np.ndarray:
    result = np.empty(np.shape(gauge), dtype=object)
    for index in np.ndindex(result.shape):
        result[index] = np.asarray(gauge[index], dtype=np.complex128).copy()
    return result


def diagnose_diagonal_overlaps(mset) -> MVOverlapDiagnostics:
    """Locate the smallest diagonal overlap used by the MV functional."""

    minimum = np.inf
    worst_k = (0, 0, 0)
    worst_direction = 0
    worst_band = 0
    direction_count = len(mset.config.composition_of_b)
    for index in mset.state.k_indices():
        storage_index = tuple(index)
        for direction in range(direction_count):
            diagonal = np.diag(np.asarray(mset.get(*storage_index, direction)))
            if diagonal.size == 0:
                continue
            magnitudes = np.abs(diagonal)
            band = int(np.argmin(magnitudes))
            value = float(magnitudes[band])
            if value < minimum:
                minimum = value
                worst_k = storage_index
                worst_direction = direction
                worst_band = band
    if not np.isfinite(minimum):
        minimum = 0.0
    return MVOverlapDiagnostics(minimum, worst_k, worst_direction, worst_band)


def max_unitarity_error(gauge: np.ndarray) -> float:
    maximum = 0.0
    for index in np.ndindex(np.shape(gauge)):
        matrix = np.asarray(gauge[index], dtype=np.complex128)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))
        ):
            return np.inf
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        maximum = max(
            maximum,
            float(np.linalg.norm(matrix.conj().T @ matrix - identity, ord="fro")),
        )
    return maximum


def protected_mv_line_search(
    gradient,
    candidate_builder: Callable[[float], MVCandidate],
    *,
    initial_step: float,
    diagonal_floor: float,
    max_steps: int,
    unitarity_tolerance: float = 1.0e-8,
    candidate_validator: Callable[[MVCandidate], tuple[bool, str | None]] | None = None,
) -> MVLineSearchResult:
    """Accept a unitary, nonsingular MV descent step from one fixed gauge.

    The MV gradient contains ``1 / M_nn``. Backtracking is part of a legal
    update: a trial is rejected when it crosses a diagonal-overlap zero, loses
    unitarity, violates an optional manifold constraint, or increases spread.
    """

    if not np.isfinite(initial_step) or initial_step <= 0.0:
        raise ValueError("MV line-search step must be positive and finite.")
    if not np.isfinite(diagonal_floor) or diagonal_floor <= 0.0:
        raise ValueError("MV diagonal floor must be positive and finite.")
    if max_steps <= 0:
        raise ValueError("MV line-search max_steps must be positive.")

    baseline_gauge = copy_gauge(gradient.U)
    baseline_omega = np.asarray(gradient.omega, dtype=float).copy()
    baseline_centers = (
        None
        if not hasattr(gradient, "rn")
        else np.asarray(gradient.rn, dtype=np.complex128).copy()
    )
    baseline_total = float(np.sum(baseline_omega))
    if not np.isfinite(baseline_total):
        raise FloatingPointError("MV line search requires a finite baseline spread.")

    gradient.mset.update(baseline_gauge)
    baseline_diagnostics = diagnose_diagonal_overlaps(gradient.mset)
    last_reason = "no candidate was evaluated"
    step = float(initial_step)

    for backtracking in range(max_steps):
        try:
            candidate = candidate_builder(step)
            unitarity = max_unitarity_error(candidate.gauge)
            if not np.isfinite(unitarity) or unitarity > unitarity_tolerance:
                last_reason = (
                    f"unitarity residual {unitarity:.6g} exceeds "
                    f"{unitarity_tolerance:.6g}"
                )
                step *= 0.5
                continue
            if candidate_validator is not None:
                valid, reason = candidate_validator(candidate)
                if not valid:
                    last_reason = reason or "candidate-specific validation failed"
                    step *= 0.5
                    continue

            gradient.U = candidate.gauge
            gradient.mset.update(gradient.U)
            diagnostics = diagnose_diagonal_overlaps(gradient.mset)
            if diagnostics.min_abs_diagonal < diagonal_floor:
                last_reason = (
                    f"min|M_nn|={diagnostics.min_abs_diagonal:.6g} is below "
                    f"the floor {diagonal_floor:.6g}"
                )
                step *= 0.5
                continue

            gradient.update()
            omega = np.asarray(gradient.omega, dtype=float).copy()
            total = float(np.sum(omega))
            increase_tolerance = max(abs(baseline_total) * 1.0e-12, 1.0e-14)
            if not np.all(np.isfinite(omega)) or total > baseline_total + increase_tolerance:
                last_reason = (
                    f"omega increased from {baseline_total:.12g} to {total:.12g}"
                    if np.isfinite(total)
                    else "candidate omega is non-finite"
                )
                step *= 0.5
                continue

            return MVLineSearchResult(
                accepted=True,
                diagnostics=diagnostics,
                trial_step=step,
                backtracking_steps=backtracking,
                metadata=candidate.metadata,
            )
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            last_reason = str(exc)
            step *= 0.5

    gradient.U = baseline_gauge
    gradient.omega = baseline_omega
    if baseline_centers is not None:
        gradient.rn = baseline_centers
    gradient.mset.update(gradient.U)
    return MVLineSearchResult(
        accepted=False,
        diagnostics=baseline_diagnostics,
        trial_step=step,
        backtracking_steps=max_steps,
        reason=last_reason,
    )
