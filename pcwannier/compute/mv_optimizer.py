from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable

import numpy as np
import scipy.linalg

from .kspace import neighbor_reciprocal_lattice_vectors


LOGGER = logging.getLogger(__name__)


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
    gauge: np.ndarray
    omega: np.ndarray
    diagnostics: MVOverlapDiagnostics
    trial_step: float
    backtracking_steps: int
    path_consistency: float = 0.0
    metadata: Any = None
    reason: str | None = None


@dataclass(frozen=True)
class MVGaugePreconditionResult:
    attempted: bool
    accepted: bool
    sweeps: int
    converged: bool
    initial_omega_tilde: float
    final_omega_tilde: float
    initial_diagnostics: MVOverlapDiagnostics
    final_diagnostics: MVOverlapDiagnostics
    max_gauge_change: float
    max_path_consistency: float
    max_unitarity_error: float
    reason: str | None = None
    max_time_reversal_residual: float = 0.0


@dataclass(frozen=True)
class MVTimeReversalDiagnostics:
    max_frame_unitarity_error: float
    max_square_residual: float


@dataclass(frozen=True)
class MVTimeReversalProjection:
    gauge: np.ndarray
    max_residual: float
    max_unitarity_error: float


@dataclass(frozen=True)
class MVTimeReversalConstraint:
    """Bosonic time-reversal structure of the selected initial Bloch frame.

    ``sewing[k]`` is defined by ``Theta Phi(k) = Phi(-k) sewing[k]``.  A
    right-acting MV gauge therefore has to obey
    ``W(-k) = sewing[k] W(k)* sewing[k]^dagger``.
    """

    partner_indices: np.ndarray
    sewing: np.ndarray
    diagnostics: MVTimeReversalDiagnostics
    tolerance: float = 1.0e-10

    def residual(self, gauge: np.ndarray) -> float:
        maximum = 0.0
        for index in np.ndindex(np.shape(gauge)):
            partner = tuple(self.partner_indices[index])
            transform = np.asarray(self.sewing[index], dtype=np.complex128)
            expected = transform @ np.asarray(gauge[index]).conj() @ transform.conj().T
            maximum = max(
                maximum,
                float(np.linalg.norm(np.asarray(gauge[partner]) - expected, ord="fro")),
            )
        return maximum

    def project(self, gauge: np.ndarray, *, max_iterations: int = 12) -> MVTimeReversalProjection:
        """Project a unitary k-grid onto its fixed bosonic TR structure."""

        projected = copy_gauge(gauge)
        visited: set[tuple[int, ...]] = set()
        for index in np.ndindex(projected.shape):
            if index in visited:
                continue
            partner = tuple(self.partner_indices[index])
            transform = np.asarray(self.sewing[index], dtype=np.complex128)
            if partner == index:
                current = np.asarray(projected[index], dtype=np.complex128)
                for _ in range(max_iterations):
                    image = transform @ current.conj() @ transform.conj().T
                    updated = _polar_unitary(0.5 * (current + image))
                    current = updated
                    image = transform @ current.conj() @ transform.conj().T
                    if np.linalg.norm(current - image, ord="fro") <= self.tolerance:
                        break
                projected[index] = current
                visited.add(index)
                continue

            source = np.asarray(projected[index], dtype=np.complex128)
            target = np.asarray(projected[partner], dtype=np.complex128)
            # Pull W(-k) back through the semilinear sewing before averaging.
            pulled_target = transform.T @ target.conj() @ transform.conj()
            try:
                source = _polar_unitary(0.5 * (source + pulled_target))
            except FloatingPointError:
                # Antipodal trial gauges can make the arithmetic mean singular.
                # Either endpoint is still a valid deterministic seed.
                source = _polar_unitary(source)
            projected[index] = source
            projected[partner] = transform @ source.conj() @ transform.conj().T
            visited.add(index)
            visited.add(partner)

        return MVTimeReversalProjection(
            projected,
            self.residual(projected),
            max_unitarity_error(projected),
        )


def copy_gauge(gauge: np.ndarray) -> np.ndarray:
    result = np.empty(np.shape(gauge), dtype=object)
    for index in np.ndindex(result.shape):
        result[index] = np.asarray(gauge[index], dtype=np.complex128).copy()
    return result


def diagnose_diagonal_overlaps(mset) -> MVOverlapDiagnostics:
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
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
            return np.inf
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        maximum = max(
            maximum,
            float(np.linalg.norm(matrix.conj().T @ matrix - identity, ord="fro")),
        )
    return maximum


def build_time_reversal_constraint(
    state,
    initial_frame: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> MVTimeReversalConstraint:
    """Resolve the bosonic TR sewing carried by an initial selected frame."""

    if getattr(state, "maxwell", None) is None:
        raise ValueError("Time-reversal gauge construction requires Maxwell field metadata.")
    frame = np.asarray(initial_frame, dtype=object)
    if frame.shape != state.k_shape:
        raise ValueError(
            f"Initial frame k-grid has shape {frame.shape}; expected {state.k_shape}."
        )
    partner_indices = _time_reversal_partner_grid(
        state.config.k_points,
        state.k_shape,
        int(state.config.kdim),
        tolerance=max(float(tolerance), 1.0e-10),
    )
    sewing = np.empty(state.k_shape, dtype=object)
    max_frame_unitarity = 0.0
    max_square = 0.0
    visited: set[tuple[int, ...]] = set()

    for index in state.k_indices():
        if index in visited:
            continue
        partner = tuple(partner_indices[index])
        source = _selected_full_bloch_frame(state, frame, index)
        target = _selected_full_bloch_frame(state, frame, partner)
        raw = np.asarray(
            state.inner_product.overlap(
                target,
                state.maxwell.apply_time_reversal(source),
            ),
            dtype=np.complex128,
        )
        if raw.shape[0] != raw.shape[1] or not np.all(np.isfinite(raw)):
            raise ValueError(
                f"Time-reversal sewing at k={index} must be a finite square matrix; "
                f"got {raw.shape}."
            )
        identity = np.eye(raw.shape[0], dtype=np.complex128)
        max_frame_unitarity = max(
            max_frame_unitarity,
            float(np.linalg.norm(raw.conj().T @ raw - identity, ord="fro")),
        )

        if partner == index:
            transform = _symmetric_unitary_polar(raw)
            raw_square = raw @ raw.conj()
            max_square = max(
                max_square,
                float(np.linalg.norm(raw_square - identity, ord="fro")),
            )
            sewing[index] = transform
            visited.add(index)
            continue

        reverse_raw = np.asarray(
            state.inner_product.overlap(
                source,
                state.maxwell.apply_time_reversal(target),
            ),
            dtype=np.complex128,
        )
        max_frame_unitarity = max(
            max_frame_unitarity,
            float(np.linalg.norm(reverse_raw.conj().T @ reverse_raw - identity, ord="fro")),
        )
        max_square = max(
            max_square,
            float(np.linalg.norm(reverse_raw @ raw.conj() - identity, ord="fro")),
        )
        transform = _polar_unitary(raw)
        sewing[index] = transform
        # This choice enforces Theta^2=+1 exactly for the projected structure.
        sewing[partner] = transform.T
        visited.add(index)
        visited.add(partner)

    return MVTimeReversalConstraint(
        partner_indices,
        sewing,
        MVTimeReversalDiagnostics(max_frame_unitarity, max_square),
        float(tolerance),
    )


def _selected_full_bloch_frame(state, frame: np.ndarray, index: tuple[int, ...]) -> np.ndarray:
    block = np.asarray(state.get_internal_block(*index, full_bloch=True), dtype=np.complex128)
    coefficients = np.asarray(frame[index], dtype=np.complex128)
    if coefficients.ndim != 2 or coefficients.shape[0] != block.shape[0]:
        raise ValueError(
            f"Selected-frame coefficients at k={index} have shape {coefficients.shape}; "
            f"expected ({block.shape[0]}, N_W)."
        )
    if block.ndim == 2:
        return np.asarray(block.T @ coefficients, dtype=np.complex128).T
    if block.ndim == 3:
        return np.einsum("npc,ni->ipc", block, coefficients, optimize=True)
    raise ValueError(f"Bloch fields at k={index} have unsupported shape {block.shape}.")


def _time_reversal_partner_grid(
    k_points,
    shape: tuple[int, ...],
    dimension: int,
    *,
    tolerance: float,
) -> np.ndarray:
    axes = tuple(np.asarray(k_points[axis], dtype=float).reshape(-1) for axis in range(dimension))
    partners = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        target = []
        for axis, values in enumerate(axes):
            point = -float(values[index[axis]])
            distances = np.abs((values - point + 0.5) % 1.0 - 0.5)
            order = np.argsort(distances)
            if distances[order[0]] > tolerance:
                raise ValueError(
                    f"The configured k mesh is not closed under time reversal on axis {axis}: "
                    f"-k={point:.12g} has nearest periodic distance {distances[order[0]]:.6g}."
                )
            if values.size > 1 and abs(distances[order[1]] - distances[order[0]]) <= tolerance:
                raise ValueError(
                    f"The configured k mesh has an ambiguous time-reversal partner on axis {axis}."
                )
            target.append(int(order[0]))
        target.extend([0] * (len(shape) - len(target)))
        partners[index] = tuple(target[: len(shape)])
    return partners


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
    """Try updates from one immutable baseline and retain only a legal descent."""
    if not np.isfinite(initial_step) or initial_step <= 0.0:
        raise ValueError("MV line-search step must be positive and finite.")
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
    gradient.mset.update(baseline_gauge)
    baseline_diagnostics = diagnose_diagonal_overlaps(gradient.mset)
    last_reason = "no candidate was evaluated"
    step = float(initial_step)

    for backtracking in range(max_steps):
        try:
            candidate = candidate_builder(step)
            unitarity = max_unitarity_error(candidate.gauge)
            if not np.isfinite(unitarity) or unitarity > unitarity_tolerance:
                last_reason = f"unitarity residual {unitarity:.6g} exceeds {unitarity_tolerance:.6g}"
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
                True,
                copy_gauge(gradient.U),
                omega,
                diagnostics,
                step,
                backtracking,
                float(candidate.path_consistency),
                candidate.metadata,
            )
        except (FloatingPointError, RuntimeError, ValueError, scipy.linalg.LinAlgError) as exc:
            last_reason = str(exc)
            step *= 0.5

    gradient.U = baseline_gauge
    gradient.omega = baseline_omega
    if baseline_centers is not None:
        gradient.rn = baseline_centers
    gradient.mset.update(gradient.U)
    return MVLineSearchResult(
        False,
        baseline_gauge,
        baseline_omega,
        baseline_diagnostics,
        step,
        max_steps,
        reason=last_reason,
    )


def expected_target_centers(context, current_centers: np.ndarray, config) -> np.ndarray:
    """Return target centers in the lattice images nearest the current centers."""
    fractional = []
    for target in context.model.targets:
        for orbit_point in target.orbit.points:
            for _ in range(target.site_irrep.dimension):
                fractional.append(np.asarray(orbit_point.position, dtype=float))
    if len(fractional) != current_centers.shape[1]:
        raise ValueError(
            f"Target center count {len(fractional)} does not match N_W={current_centers.shape[1]}."
        )
    lattice = np.asarray(config.real_lattice_vectors, dtype=float) * float(config.lattice_const)
    current_real = np.asarray(np.real_if_close(current_centers), dtype=float)
    output = np.empty_like(current_real)
    offsets = np.asarray(list(np.ndindex(*(3,) * lattice.shape[0])), dtype=int) - 1
    for band, center in enumerate(fractional):
        delta_fractional = current_real[:, band] @ np.linalg.inv(lattice)
        base = np.rint(delta_fractional - center).astype(int)
        candidates_fractional = center[None, :] + base[None, :] + offsets
        candidates = candidates_fractional @ lattice
        distance = np.linalg.norm(candidates - current_real[:, band][None, :], axis=1)
        output[:, band] = candidates[int(np.argmin(distance))]
    return output


def precondition_mv_gauge(
    gradient,
    centers: np.ndarray,
    *,
    diagonal_floor: float,
    max_sweeps: int = 50,
    convergence_tolerance: float = 1.0e-8,
    symmetry_projector: Callable[[np.ndarray], tuple[np.ndarray, float, float]] | None = None,
    time_reversal_constraint: MVTimeReversalConstraint | None = None,
) -> MVGaugePreconditionResult:
    """Synchronize the full U(N) frame on the periodic neighbor graph."""
    if max_sweeps <= 0:
        raise ValueError("MV preconditioner max_sweeps must be positive.")
    centers = np.asarray(centers, dtype=float)
    band_count = int(gradient.config.band_calc_num)
    if centers.shape != (gradient.config.kdim, band_count):
        raise ValueError(
            f"Expected centers shape {(gradient.config.kdim, band_count)}, got {centers.shape}."
        )

    initial_gauge = copy_gauge(gradient.U)
    initial_centers = np.asarray(gradient.rn, dtype=np.complex128).copy()
    gradient.mset.update(initial_gauge)
    initial_diagnostics = diagnose_diagonal_overlaps(gradient.mset)
    initial_omega_tilde = np.inf
    machine_floor = np.finfo(float).eps * 128.0
    if initial_diagnostics.min_abs_diagonal > machine_floor:
        try:
            gradient.update()
            initial_omega_tilde = float(gradient.omega[1] + gradient.omega[2])
        except FloatingPointError:
            pass

    b_half = len(gradient.config.composition_of_b) // 2
    phases = tuple(
        np.diag(np.exp(-1j * (np.asarray(gradient.config.b_vectors[b], dtype=float) @ centers)))
        for b in range(b_half)
    )
    current = copy_gauge(initial_gauge)
    best_gauge = None
    best_diagnostics = initial_diagnostics
    best_tilde = initial_omega_tilde
    best_path = 0.0
    best_unitarity = max_unitarity_error(initial_gauge)
    best_time_reversal = (
        0.0
        if time_reversal_constraint is None
        else time_reversal_constraint.residual(initial_gauge)
    )
    allowed_path = np.inf
    allowed_unitarity = 1.0e-8
    if symmetry_projector is not None:
        _, initial_path, initial_projected_unitarity = symmetry_projector(
            copy_gauge(initial_gauge)
        )
        allowed_path = max(1.0e-8, float(initial_path) + 1.0e-10)
        allowed_unitarity = max(
            1.0e-8, float(initial_projected_unitarity) + 1.0e-10
        )
    allowed_time_reversal = np.inf
    if time_reversal_constraint is not None:
        allowed_time_reversal = max(
            1.0e-8,
            time_reversal_constraint.residual(initial_gauge) + 1.0e-10,
        )
    maximum_change = 0.0
    converged = False
    completed_sweeps = 0
    failure_reason = None

    for sweep in range(1, max_sweeps + 1):
        try:
            proposal = np.empty(np.shape(current), dtype=object)
            for index in gradient.state.k_indices():
                accumulator = np.zeros((band_count, band_count), dtype=np.complex128)
                for direction in range(b_half):
                    target, _ = neighbor_reciprocal_lattice_vectors(
                        gradient.config, list(index), direction
                    )
                    outgoing = gradient.mset.get_initial(*index, direction)
                    accumulator += (
                        gradient.config.wb[direction]
                        * outgoing
                        @ np.asarray(current[target])
                        @ phases[direction].conj().T
                    )
                    source, _ = neighbor_reciprocal_lattice_vectors(
                        gradient.config, list(index), direction + b_half
                    )
                    incoming = gradient.mset.get_initial(*index, direction + b_half)
                    accumulator += (
                        gradient.config.wb[direction]
                        * incoming
                        @ np.asarray(current[source])
                        @ phases[direction]
                    )
                aligned = _polar_unitary(accumulator)
                proposal[index] = _polar_unitary(
                    0.5 * np.asarray(current[index]) + 0.5 * aligned
                )

            # Keep the synchronization workspace separate from the legal
            # candidate.  Feeding the anti-linear projection back into every
            # Jacobi sweep can trap a perfectly smooth real frame in a much
            # poorer local basin.  The scratch frame may use the full U(N)
            # freedom; only projected candidates can become the accepted MV
            # gauge.
            working = proposal
            working_change = max(
                (
                    float(
                        np.linalg.norm(
                            np.asarray(working[index]) - np.asarray(current[index]),
                            ord="fro",
                        )
                    )
                    for index in np.ndindex(working.shape)
                ),
                default=0.0,
            )
            if symmetry_projector is not None:
                working, _, _ = symmetry_projector(working)
            proposal = copy_gauge(working)
            path_consistency = 0.0
            projected_unitarity = max_unitarity_error(proposal)
            if time_reversal_constraint is not None:
                # Spatial and bosonic TR projectors commute in the exact
                # problem. Alternate them only on the candidate copy.
                for _ in range(4):
                    tr_projection = time_reversal_constraint.project(proposal)
                    proposal = tr_projection.gauge
                    projected_unitarity = max(
                        projected_unitarity,
                        tr_projection.max_unitarity_error,
                    )
                    if symmetry_projector is not None:
                        proposal, path_consistency, spatial_unitarity = symmetry_projector(
                            proposal
                        )
                        projected_unitarity = max(
                            projected_unitarity, spatial_unitarity
                        )
            elif symmetry_projector is not None:
                proposal, path_consistency, spatial_unitarity = symmetry_projector(proposal)
                projected_unitarity = max(projected_unitarity, spatial_unitarity)
            time_reversal_residual = (
                0.0
                if time_reversal_constraint is None
                else time_reversal_constraint.residual(proposal)
            )
        except (FloatingPointError, RuntimeError, ValueError, scipy.linalg.LinAlgError) as exc:
            failure_reason = str(exc)
            break
        maximum_change = max(maximum_change, working_change)
        current = copy_gauge(working)
        completed_sweeps = sweep

        gradient.U = proposal
        gradient.mset.update(proposal)
        diagnostics = diagnose_diagonal_overlaps(gradient.mset)
        unitarity = max(max_unitarity_error(proposal), float(projected_unitarity))
        if (
            diagnostics.min_abs_diagonal >= diagonal_floor
            and np.isfinite(unitarity)
            and unitarity <= allowed_unitarity
            and np.isfinite(path_consistency)
            and path_consistency <= allowed_path
            and np.isfinite(time_reversal_residual)
            and time_reversal_residual <= allowed_time_reversal
        ):
            try:
                gradient.update()
                candidate_tilde = float(gradient.omega[1] + gradient.omega[2])
            except FloatingPointError:
                candidate_tilde = np.inf
            if np.isfinite(candidate_tilde) and candidate_tilde < best_tilde:
                best_tilde = candidate_tilde
                best_gauge = copy_gauge(proposal)
                best_diagnostics = diagnostics
                best_path = float(path_consistency)
                best_unitarity = unitarity
                best_time_reversal = float(time_reversal_residual)
        if working_change < convergence_tolerance:
            converged = True
            break

    improvement_tolerance = max(abs(initial_omega_tilde) * 1.0e-12, 1.0e-14)
    accepted = best_gauge is not None and (
        not np.isfinite(initial_omega_tilde)
        or best_tilde < initial_omega_tilde - improvement_tolerance
    )
    if accepted:
        gradient.U = best_gauge
        gradient.mset.update(gradient.U)
        gradient.update()
        final_diagnostics = best_diagnostics
        final_tilde = best_tilde
        reason = None
    else:
        gradient.U = initial_gauge
        gradient.mset.update(gradient.U)
        final_diagnostics = initial_diagnostics
        final_tilde = initial_omega_tilde
        reason = failure_reason or "synchronization found no legal candidate with a lower exact MV spread"
        if np.isfinite(initial_omega_tilde):
            gradient.update()
        else:
            gradient.rn = initial_centers

    return MVGaugePreconditionResult(
        True,
        accepted,
        completed_sweeps,
        converged,
        initial_omega_tilde,
        final_tilde,
        initial_diagnostics,
        final_diagnostics,
        maximum_change,
        best_path if accepted else 0.0,
        best_unitarity if accepted else max_unitarity_error(initial_gauge),
        reason,
        best_time_reversal if accepted else (
            0.0
            if time_reversal_constraint is None
            else time_reversal_constraint.residual(initial_gauge)
        ),
    )


def _polar_unitary(matrix: np.ndarray) -> np.ndarray:
    left, singular_values, vh = np.linalg.svd(
        np.asarray(matrix, dtype=np.complex128), full_matrices=False
    )
    if not singular_values.size or float(singular_values[-1]) <= np.finfo(float).eps:
        raise FloatingPointError(
            f"U(N) synchronization polar factor is rank deficient: singular_values={singular_values.tolist()}."
        )
    return left @ vh


def _symmetric_unitary_polar(matrix: np.ndarray) -> np.ndarray:
    """Nearest symmetric unitary needed by a bosonic TRIM sewing matrix."""

    result = _polar_unitary(0.5 * (matrix + matrix.T))
    for _ in range(8):
        symmetric = 0.5 * (result + result.T)
        updated = _polar_unitary(symmetric)
        if np.linalg.norm(updated - result, ord="fro") <= 1.0e-13:
            result = updated
            break
        result = updated
    residual = float(np.linalg.norm(result - result.T, ord="fro"))
    if residual > 1.0e-10:
        raise FloatingPointError(
            f"Time-reversal sewing at a TRIM could not be made symmetric (residual={residual:.6g})."
        )
    return result
