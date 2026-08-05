from __future__ import annotations

from collections.abc import Callable
import logging

import numpy as np

from ..conventions import BlochFieldRepresentation
from ..compute.parallel import memory_limited_threads, parallel_map
from ..data import BandChannelReference, InputBundle
from .models import (
    ETBCCompletionResult,
    ETBCKPointDiagnostics,
    ETBCKPointResult,
)

LOGGER = logging.getLogger(__name__)


def _field_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"{name} must have shape (states, points, 3); got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _mix_fields(fields: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.einsum(
        "npc,ni->ipc",
        np.asarray(fields, dtype=np.complex128),
        np.asarray(coefficients, dtype=np.complex128),
        optimize=True,
    )


def _hermitian(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.complex128)
    return 0.5 * (matrix + matrix.conj().T)


def _lowdin_frame(
    fields: np.ndarray,
    inner_product,
    *,
    rank_tolerance: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _field_rows(fields, name=name)
    if rows.shape[0] == 0:
        return rows.copy(), np.zeros((0, 0), dtype=np.complex128), np.zeros(0)
    gram = _hermitian(inner_product.overlap(rows, rows, chunk_size=64))
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    scale = max(float(np.max(eigenvalues)), np.finfo(float).tiny)
    threshold = float(rank_tolerance) * scale
    if (
        not np.all(np.isfinite(eigenvalues))
        or float(np.min(eigenvalues)) <= threshold
    ):
        raise ValueError(
            f"{name} is rank deficient: eigenvalues={eigenvalues.tolist()}, "
            f"threshold={threshold:.6g}."
        )
    inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    return _mix_fields(rows, inverse_sqrt), inverse_sqrt, eigenvalues


def _canonicalize_nullspace(coefficients: np.ndarray) -> np.ndarray:
    result = np.asarray(coefficients, dtype=np.complex128).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        value = result[pivot, column]
        if abs(value) > np.finfo(float).tiny:
            result[:, column] *= np.exp(-1j * np.angle(value))
    return result


def construct_auxiliary_frame(
    transverse_frame: np.ndarray,
    trial_frame: np.ndarray,
    inner_product,
    *,
    rank_tolerance: float,
) -> ETBCKPointResult:
    """Construct the metric-orthogonal complement of a transverse frame."""

    transverse = _field_rows(transverse_frame, name="transverse frame")
    trials = _field_rows(trial_frame, name="trial frame")
    n_t = transverse.shape[0]
    n_w = trials.shape[0]
    if transverse.shape[1:] != trials.shape[1:]:
        raise ValueError("Transverse and trial frames use different spatial grids.")
    if n_t <= 0 or n_w <= n_t:
        raise ValueError(
            f"ETBC requires 0 < N_T < N_W; got N_T={n_t}, N_W={n_w}."
        )

    transverse_gram = _hermitian(
        inner_product.overlap(transverse, transverse, chunk_size=64)
    )
    transverse_error = float(
        np.linalg.norm(transverse_gram - np.eye(n_t), ord="fro")
    )
    algebra_tolerance = max(1.0e-8, 100.0 * float(rank_tolerance))
    if transverse_error > algebra_tolerance:
        raise ValueError(
            "ETBC requires an orthonormal physical transverse frame; "
            f"residual={transverse_error:.6g}."
        )

    orthogonal_trials, _, trial_eigenvalues = _lowdin_frame(
        trials,
        inner_product,
        rank_tolerance=rank_tolerance,
        name="ETBC trial frame",
    )
    overlap = inner_product.overlap(
        transverse, orthogonal_trials, chunk_size=64
    )
    _, singular_values, vh = np.linalg.svd(overlap, full_matrices=True)
    if singular_values.size != n_t or not np.all(np.isfinite(singular_values)):
        raise ValueError("ETBC overlap SVD returned invalid singular values.")
    largest = max(float(singular_values[0]), np.finfo(float).tiny)
    threshold = float(rank_tolerance) * largest
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank != n_t:
        raise ValueError(
            f"ETBC transverse/trial overlap has rank {rank}, expected {n_t}; "
            f"singular_values={singular_values.tolist()}, threshold={threshold:.6g}."
        )
    y_l = _canonicalize_nullspace(vh.conj().T[:, n_t:])
    auxiliary = _mix_fields(orthogonal_trials, y_l)
    auxiliary_gram = _hermitian(
        inner_product.overlap(auxiliary, auxiliary, chunk_size=64)
    )
    cross = inner_product.overlap(transverse, auxiliary, chunk_size=64)
    augmented = np.concatenate((transverse, auxiliary), axis=0)
    augmented_gram = _hermitian(
        inner_product.overlap(augmented, augmented, chunk_size=64)
    )
    auxiliary_error = float(
        np.linalg.norm(auxiliary_gram - np.eye(n_w - n_t), ord="fro")
    )
    cross_error = float(np.linalg.norm(cross, ord="fro"))
    augmented_error = float(
        np.linalg.norm(augmented_gram - np.eye(n_w), ord="fro")
    )
    if max(auxiliary_error, cross_error, augmented_error) > algebra_tolerance:
        raise ValueError(
            "ETBC constructed an invalid augmented frame: "
            f"auxiliary={auxiliary_error:.6g}, cross={cross_error:.6g}, "
            f"augmented={augmented_error:.6g}."
        )
    return ETBCKPointResult(
        auxiliary,
        y_l,
        singular_values,
        trial_eigenvalues,
        rank,
        transverse_error,
        auxiliary_error,
        cross_error,
        augmented_error,
    )


def _fractional_k(config, index: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(
        [config.k_points[axis][index[axis]] for axis in range(int(config.kdim))],
        dtype=float,
    )


def _is_gamma(config, index: tuple[int, int, int]) -> bool:
    k = _fractional_k(config, index)
    tolerance = max(float(config.gamma_zero_mode_tolerance), 1.0e-12)
    return bool(np.allclose(k - np.rint(k), 0.0, rtol=0.0, atol=tolerance))


def _constant_vector_frame(point_count: int) -> np.ndarray:
    output = np.zeros((3, point_count, 3), dtype=np.complex128)
    output[:, :, :] = np.eye(3, dtype=np.complex128)[:, None, :]
    return output


def _project_out(fields: np.ndarray, basis: np.ndarray, inner_product) -> np.ndarray:
    if fields.shape[0] == 0 or basis.shape[0] == 0:
        return np.asarray(fields, dtype=np.complex128).copy()
    coefficients = inner_product.overlap(basis, fields, chunk_size=64)
    return np.asarray(fields, dtype=np.complex128) - _mix_fields(basis, coefficients)


def _gamma_augmented_frame(
    transverse: np.ndarray,
    orthogonal_trials: np.ndarray,
    transverse_hamiltonian: np.ndarray,
    zero_positions: np.ndarray,
    inner_product,
    *,
    auxiliary_dimension: int,
    auxiliary_eigenvalue: float,
    rank_tolerance: float,
    zero_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    if zero_positions.size != 2:
        raise ValueError(
            "ETBC Gamma regularization requires exactly two selected zero-frequency "
            f"transverse modes; found {zero_positions.tolist()}."
        )
    constants, _, _ = _lowdin_frame(
        _constant_vector_frame(transverse.shape[1]),
        inner_product,
        rank_tolerance=rank_tolerance,
        name="Gamma constant-vector frame",
    )
    h_eigenvalues, h_eigenvectors = np.linalg.eigh(
        _hermitian(transverse_hamiltonian)
    )
    h_zero_positions = np.flatnonzero(np.abs(h_eigenvalues) <= zero_tolerance)
    if h_zero_positions.size != zero_positions.size:
        raise ValueError(
            "ETBC Gamma Hamiltonian zero-space dimension does not match the "
            f"selected zero modes: Hamiltonian={h_zero_positions.tolist()}, "
            f"bands={zero_positions.tolist()}."
        )
    h_positive_positions = np.setdiff1d(
        np.arange(transverse.shape[0]), h_zero_positions, assume_unique=True
    )
    energy_frame = _mix_fields(transverse, h_eigenvectors)
    positive_positions = np.setdiff1d(
        np.arange(transverse.shape[0]), zero_positions, assume_unique=True
    )
    positive_raw = energy_frame[h_positive_positions]
    positive_projected = _project_out(positive_raw, constants, inner_product)
    positive, positive_transform, _ = _lowdin_frame(
        positive_projected,
        inner_product,
        rank_tolerance=rank_tolerance,
        name="Gamma positive-frequency transverse frame",
    )

    physical = np.empty_like(transverse)
    physical[zero_positions[0]] = constants[0]
    physical[zero_positions[1]] = constants[1]
    physical[positive_positions] = positive

    extra_count = auxiliary_dimension - 1
    if extra_count > 0:
        fixed = np.concatenate((constants, positive), axis=0)
        residual_trials = _project_out(orthogonal_trials, fixed, inner_product)
        gram = _hermitian(
            inner_product.overlap(residual_trials, residual_trials, chunk_size=64)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        selected = order[:extra_count]
        scale = max(float(np.max(eigenvalues)), np.finfo(float).tiny)
        threshold = rank_tolerance * scale
        if selected.size != extra_count or np.any(eigenvalues[selected] <= threshold):
            raise ValueError(
                "ETBC Gamma trial complement cannot supply the remaining auxiliary "
                f"directions; eigenvalues={eigenvalues.tolist()}, need={extra_count}."
            )
        coefficients = eigenvectors[:, selected] / np.sqrt(eigenvalues[selected])[None, :]
        extra = _mix_fields(residual_trials, coefficients)
        auxiliary = np.concatenate((constants[2:3], extra), axis=0)
    else:
        auxiliary = constants[2:3]

    augmented = np.concatenate((physical, auxiliary), axis=0)
    gram = _hermitian(inner_product.overlap(augmented, augmented, chunk_size=64))
    augmented_error = float(
        np.linalg.norm(gram - np.eye(augmented.shape[0]), ord="fro")
    )
    cross_error = float(
        np.linalg.norm(
            inner_product.overlap(physical, auxiliary, chunk_size=64), ord="fro"
        )
    )
    auxiliary_error = float(
        np.linalg.norm(
            inner_product.overlap(auxiliary, auxiliary, chunk_size=64)
            - np.eye(auxiliary_dimension),
            ord="fro",
        )
    )
    tolerance = max(1.0e-8, 100.0 * rank_tolerance)
    if max(augmented_error, cross_error, auxiliary_error) > tolerance:
        raise ValueError(
            "Gamma ETBC regularization produced a non-orthonormal frame: "
            f"auxiliary={auxiliary_error:.6g}, cross={cross_error:.6g}, "
            f"augmented={augmented_error:.6g}."
        )

    h_physical = np.zeros(
        (transverse.shape[0], transverse.shape[0]), dtype=np.complex128
    )
    if positive_positions.size:
        h_positive = np.diag(h_eigenvalues[h_positive_positions])
        h_physical[np.ix_(positive_positions, positive_positions)] = (
            positive_transform.conj().T @ h_positive @ positive_transform
        )
    h_augmented = np.zeros(
        (augmented.shape[0], augmented.shape[0]), dtype=np.complex128
    )
    h_augmented[: transverse.shape[0], : transverse.shape[0]] = h_physical
    h_augmented[transverse.shape[0] :, transverse.shape[0] :] = (
        float(auxiliary_eigenvalue) * np.eye(auxiliary_dimension)
    )
    return augmented, h_augmented, (
        auxiliary_error,
        cross_error,
        augmented_error,
    )


def complete_transverse_bundle(
    physical_state,
    trial_provider: Callable[[tuple[int, int, int]], np.ndarray],
    *,
    auxiliary_eigenvalue: float = 0.0,
    rank_tolerance: float = 1.0e-10,
) -> ETBCCompletionResult:
    """Complete a fixed isolated transverse band group with trial-space modes."""

    config = physical_state.config
    counts = [len(physical_state.E_idx[index]) for index in physical_state.k_indices()]
    if not counts or min(counts) != max(counts):
        raise ValueError(
            "ETBC requires the same isolated transverse-band dimension at every k point."
        )
    n_t = int(counts[0])
    n_w = int(config.band_calc_num)
    n_l = n_w - n_t
    if n_t <= 0 or n_l <= 0:
        raise ValueError(
            f"ETBC requires N_W>N_T>0; got N_T={n_t}, N_W={n_w}."
        )
    if physical_state.get_block(0, 0, 0).ndim != 3:
        raise ValueError("ETBC requires a three-component vector field.")

    k_shape = physical_state.k_shape
    augmented_fields = np.empty(k_shape, dtype=object)
    augmented_energies = np.empty(k_shape, dtype=object)
    augmented_indices = np.empty(k_shape, dtype=object)
    augmented_inner = np.empty(k_shape, dtype=object)
    augmented_zero = np.empty(k_shape, dtype=object)
    base_hamiltonians = np.empty(k_shape, dtype=object)
    nullspaces = np.empty(k_shape, dtype=object)
    projection_matrices = np.empty(k_shape, dtype=object)
    diagnostics: list[ETBCKPointDiagnostics] = []
    gamma_regularized: list[tuple[int, int, int]] = []
    l_offset = int(np.asarray(physical_state.energy_matrix).shape[-1])
    indices = tuple(physical_state.k_indices())
    point_count = int(physical_state.mesh.vertices.shape[0])
    bytes_per_worker = n_w * point_count * 3 * np.dtype(np.complex128).itemsize * 5
    workers = memory_limited_threads(
        physical_state.configured_threads,
        bytes_per_worker,
    )
    LOGGER.info(
        "ETBC k-point construction: workers=%d requested=%d estimated_worker_memory=%.1f MB",
        workers,
        physical_state.configured_threads,
        bytes_per_worker / (1024.0**2),
    )

    def calc_index(index):
        transverse_full = physical_state.get_internal_block(
            *index, full_bloch=True
        )
        transform = np.asarray(
            physical_state.get_transform(False)[index], dtype=np.complex128
        )
        physical_energies = np.asarray(
            physical_state.E[index], dtype=float
        ).reshape(-1)
        h_transverse = transform.conj().T @ np.diag(physical_energies) @ transform
        trial_values = np.asarray(trial_provider(index), dtype=np.complex128)
        if trial_values.shape == (physical_state.mesh.vertices.shape[0], n_w, 3):
            trial_rows = np.swapaxes(trial_values, 0, 1)
        else:
            trial_rows = trial_values
        if trial_rows.shape != (n_w, physical_state.mesh.vertices.shape[0], 3):
            raise ValueError(
                f"ETBC trial frame at k={index} has shape {trial_rows.shape}; expected "
                f"{(n_w, physical_state.mesh.vertices.shape[0], 3)}."
            )

        zero_tolerance = float(config.gamma_zero_mode_tolerance)
        zero_positions = np.flatnonzero(np.abs(physical_energies) <= zero_tolerance)
        gamma_special = _is_gamma(config, index) and zero_positions.size > 0
        if gamma_special:
            orthogonal_trials, _, trial_eigenvalues = _lowdin_frame(
                trial_rows,
                physical_state.inner_product,
                rank_tolerance=rank_tolerance,
                name=f"ETBC trial frame at Gamma k={index}",
            )
            augmented_full, h_augmented, errors = _gamma_augmented_frame(
                transverse_full,
                orthogonal_trials,
                h_transverse,
                zero_positions,
                physical_state.inner_product,
                auxiliary_dimension=n_l,
                auxiliary_eigenvalue=auxiliary_eigenvalue,
                rank_tolerance=rank_tolerance,
                zero_tolerance=zero_tolerance,
            )
            singular_values: tuple[float, ...] = ()
            nullspace = None
            auxiliary_error, cross_error, augmented_error = errors
            diagnostic = ETBCKPointDiagnostics(
                tuple(int(value) for value in index),
                n_t,
                n_l,
                singular_values,
                tuple(float(value) for value in trial_eigenvalues),
                0.0,
                auxiliary_error,
                cross_error,
                augmented_error,
                True,
            )
        else:
            if _is_gamma(config, index) and zero_positions.size not in {0, 2}:
                raise ValueError(
                    "ETBC Gamma point contains an unsupported number of selected zero modes: "
                    f"{zero_positions.tolist()}."
                )
            result = construct_auxiliary_frame(
                transverse_full,
                trial_rows,
                physical_state.inner_product,
                rank_tolerance=rank_tolerance,
            )
            augmented_full = np.concatenate(
                (transverse_full, result.auxiliary_frame), axis=0
            )
            h_augmented = np.zeros((n_w, n_w), dtype=np.complex128)
            h_augmented[:n_t, :n_t] = h_transverse
            h_augmented[n_t:, n_t:] = float(auxiliary_eigenvalue) * np.eye(n_l)
            nullspace = result.nullspace_coefficients
            diagnostic = ETBCKPointDiagnostics(
                tuple(int(value) for value in index),
                n_t,
                n_l,
                tuple(float(value) for value in result.singular_values),
                tuple(float(value) for value in result.trial_gram_eigenvalues),
                result.transverse_orthonormality_error,
                result.auxiliary_orthonormality_error,
                result.transverse_auxiliary_overlap,
                result.augmented_orthonormality_error,
                False,
            )

        phase = physical_state.get_phase(*index)[None, :, None]
        periodic_fields = np.ascontiguousarray(
            augmented_full * np.conj(phase)
        )
        energies = np.concatenate(
            (
                physical_energies,
                np.full(n_l, float(auxiliary_eigenvalue), dtype=float),
            )
        )
        physical_ids = np.asarray(physical_state.E_idx[index], dtype=int)
        band_indices = np.concatenate(
            (physical_ids, l_offset + np.arange(n_l, dtype=int))
        ).tolist()
        zero_modes = np.abs(energies) <= float(
            config.gamma_zero_mode_tolerance
        )
        projection = physical_state.inner_product.overlap(
            augmented_full,
            trial_rows,
            chunk_size=64,
        )
        return (
            tuple(int(value) for value in index),
            periodic_fields,
            energies,
            band_indices,
            zero_modes,
            _hermitian(h_augmented),
            nullspace,
            diagnostic,
            gamma_special,
            projection,
        )

    for (
        index,
        periodic_fields,
        energies,
        band_indices,
        zero_modes,
        h_augmented,
        nullspace,
        diagnostic,
        gamma_special,
        projection,
    ) in parallel_map(indices, calc_index, workers):
        augmented_fields[index] = periodic_fields
        augmented_energies[index] = energies
        augmented_indices[index] = band_indices
        augmented_inner[index] = []
        augmented_zero[index] = zero_modes
        base_hamiltonians[index] = h_augmented
        nullspaces[index] = nullspace
        projection_matrices[index] = projection
        diagnostics.append(diagnostic)
        if gamma_special:
            gamma_regularized.append(index)

    energy_matrix = np.concatenate(
        (
            np.asarray(physical_state.energy_matrix, dtype=float),
            np.full(k_shape + (n_l,), float(auxiliary_eigenvalue), dtype=float),
        ),
        axis=-1,
    )
    channels = dict(physical_state.band_channels)
    channels.update(
        {
            l_offset + position: BandChannelReference("L", position)
            for position in range(n_l)
        }
    )
    longitudinal_zero = np.empty(k_shape, dtype=object)
    for index in np.ndindex(k_shape):
        longitudinal_zero[index] = (
            list(range(n_l))
            if abs(float(auxiliary_eigenvalue))
            <= float(config.gamma_zero_mode_tolerance)
            else []
        )
    bundle = InputBundle(
        config=config,
        maxwell=physical_state.maxwell,
        bloch_convention=physical_state.bloch_convention,
        mesh=physical_state.mesh,
        fields=augmented_fields,
        metric_material=physical_state.metric_material,
        energies=augmented_energies,
        band_indices=augmented_indices,
        inner_band_indices=augmented_inner,
        energy_matrix=energy_matrix,
        field_representation=BlochFieldRepresentation.PERIODIC_PART,
        symmetry=physical_state.symmetry,
        analysis_field_kind=physical_state.maxwell.symmetry_field_kind,
        zero_modes=augmented_zero,
        band_channels=channels,
        auxiliary_zero_mode_bands={"longitudinal": longitudinal_zero},
        base_hamiltonians=base_hamiltonians,
    )
    return ETBCCompletionResult(
        bundle,
        n_t,
        n_l,
        n_w,
        float(auxiliary_eigenvalue),
        tuple(diagnostics),
        nullspaces,
        projection_matrices,
        tuple(gamma_regularized),
    )
