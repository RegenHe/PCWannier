from __future__ import annotations

from collections.abc import Callable
import logging

import numpy as np

from ..conventions import BlochFieldRepresentation
from ..compute.parallel import memory_limited_threads, parallel_map
from ..compute.prepared import ProjectionSeed
from ..data import BandChannelReference, InputBundle
from .models import (
    ETBCCompletionResult,
    ETBCCompletionArtifacts,
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
        overlap = np.asarray(
            inner_product.overlap(fixed, orthogonal_trials, chunk_size=64),
            dtype=np.complex128,
        )
        _, singular_values, right_adjoint = np.linalg.svd(
            overlap, full_matrices=True
        )
        scale = max(float(np.max(singular_values, initial=0.0)), 1.0)
        rank = int(np.count_nonzero(singular_values > rank_tolerance * scale))
        expected_rank = fixed.shape[0]
        nullity = orthogonal_trials.shape[0] - rank
        if rank != expected_rank or nullity != extra_count:
            raise ValueError(
                "ETBC Gamma trial overlap does not contain the required fixed "
                "representations and complementary auxiliary space: "
                f"singular_values={singular_values.tolist()}, rank={rank}, "
                f"expected_rank={expected_rank}, nullity={nullity}, "
                f"need={extra_count}."
            )
        coefficients = right_adjoint.conj().T[:, rank:]
        extra = _mix_fields(orthogonal_trials, coefficients)
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


def _streaming_internal_chunk(physical_state, index, start: int, stop: int):
    block = physical_state.get_block(*index)[:, start:stop]
    transform = np.asarray(
        physical_state.get_transform(False)[index], dtype=np.complex128
    )
    if block.ndim != 3:
        raise ValueError("Streaming ETBC requires three-component vector fields.")
    periodic = np.einsum("npc,ni->ipc", block, transform, optimize=True)
    phase = physical_state.get_phase(*index)[start:stop]
    return periodic, periodic * phase[None, :, None]


def _complete_transverse_bundle_streaming(
    physical_state,
    trial_source,
    *,
    auxiliary_eigenvalue: float,
    rank_tolerance: float,
) -> ETBCCompletionArtifacts:
    """Two-pass ETBC completion without retaining the full trial k-grid."""

    config = physical_state.config
    indices = tuple(physical_state.k_indices())
    counts = [len(physical_state.E_idx[index]) for index in indices]
    if not counts or min(counts) != max(counts):
        raise ValueError(
            "ETBC requires the same isolated transverse-band dimension at every k point."
        )
    n_t = int(counts[0])
    n_w = int(trial_source.trial_count)
    n_l = n_w - n_t
    if n_w != int(config.band_calc_num) or n_t <= 0 or n_l <= 0:
        raise ValueError(f"ETBC requires N_W>N_T>0; got N_T={n_t}, N_W={n_w}.")
    inner = physical_state.inner_product
    if getattr(inner, "domain_kind", None) != "points":
        raise TypeError("Streaming ETBC currently requires a uniform-grid inner product.")
    metric = np.asarray(inner.metric, dtype=np.complex128).reshape(-1)
    if np.max(np.abs(metric.imag), initial=0.0) > 1.0e-12:
        raise ValueError("Streaming ETBC requires a real metric material.")
    point_weight = float(inner.point_weight)

    raw_grams = {
        index: np.zeros((n_w, n_w), dtype=np.complex128) for index in indices
    }
    raw_cross = {
        index: np.zeros((n_t, n_w), dtype=np.complex128) for index in indices
    }
    transverse_grams = {
        index: np.zeros((n_t, n_t), dtype=np.complex128) for index in indices
    }
    gamma_indices = {
        index
        for index in indices
        if _is_gamma(config, index)
        and np.any(
            np.abs(np.asarray(physical_state.E[index], dtype=float))
            <= float(config.gamma_zero_mode_tolerance)
        )
    }
    gamma_trials = {
        index: np.empty((n_w, trial_source.point_count, 3), dtype=np.complex128)
        for index in gamma_indices
    }

    for start, stop, trial_chunk in trial_source.iter_raw_chunks():
        weights = metric.real[start:stop] * point_weight
        for index in indices:
            trials = np.asarray(trial_chunk[index], dtype=np.complex128)
            _, transverse = _streaming_internal_chunk(
                physical_state, index, start, stop
            )
            raw_grams[index] += np.einsum(
                "pic,pjc,p->ij",
                trials.conj(),
                trials,
                weights,
                optimize=True,
            )
            raw_cross[index] += np.einsum(
                "tpc,pic,p->ti",
                transverse.conj(),
                trials,
                weights,
                optimize=True,
            )
            transverse_grams[index] += np.einsum(
                "tpc,upc,p->tu",
                transverse.conj(),
                transverse,
                weights,
                optimize=True,
            )
            if index in gamma_trials:
                gamma_trials[index][:, start:stop, :] = np.swapaxes(
                    trials, 0, 1
                )

    coefficients = {}
    projections = np.empty(physical_state.k_shape, dtype=object)
    singular_by_k = {}
    trial_eigenvalues_by_k = {}
    diagnostics_by_k = {}
    gamma_augmented = {}
    gamma_hamiltonians = {}
    gamma_regularized = []
    algebra_tolerance = max(1.0e-8, 100.0 * float(rank_tolerance))

    for index in indices:
        gram_raw = _hermitian(raw_grams[index])
        raw_norms = np.real(np.diag(gram_raw))
        if np.any(~np.isfinite(raw_norms)) or np.any(raw_norms <= 0.0):
            raise ValueError(
                f"ETBC trial frame at k={index} has invalid norms {raw_norms.tolist()}."
            )
        normalization = np.diag(1.0 / np.sqrt(raw_norms))
        gram = _hermitian(normalization @ gram_raw @ normalization)
        trial_eigenvalues, trial_vectors = np.linalg.eigh(gram)
        scale = max(float(np.max(trial_eigenvalues)), np.finfo(float).tiny)
        threshold = float(rank_tolerance) * scale
        if float(np.min(trial_eigenvalues)) <= threshold:
            raise ValueError(
                f"ETBC trial frame at k={index} is rank deficient: "
                f"eigenvalues={trial_eigenvalues.tolist()}, threshold={threshold:.6g}."
            )
        lowdin = trial_vectors @ np.diag(
            1.0 / np.sqrt(trial_eigenvalues)
        ) @ trial_vectors.conj().T
        orthogonalizer = normalization @ lowdin
        overlap = raw_cross[index] @ orthogonalizer
        _, singular_values, vh = np.linalg.svd(overlap, full_matrices=True)
        largest = max(float(singular_values[0]), np.finfo(float).tiny)
        rank = int(
            np.count_nonzero(singular_values > float(rank_tolerance) * largest)
        )
        if index not in gamma_indices and rank != n_t:
            raise ValueError(
                f"ETBC transverse/trial overlap at k={index} has rank {rank}, "
                f"expected {n_t}; singular_values={singular_values.tolist()}."
            )
        transverse_error = float(
            np.linalg.norm(
                _hermitian(transverse_grams[index]) - np.eye(n_t), ord="fro"
            )
        )
        if transverse_error > algebra_tolerance:
            raise ValueError(
                "ETBC requires an orthonormal physical transverse frame; "
                f"k={index}, residual={transverse_error:.6g}."
            )
        physical_energies = np.asarray(physical_state.E[index], dtype=float)
        transform = np.asarray(
            physical_state.get_transform(False)[index], dtype=np.complex128
        )
        h_transverse = transform.conj().T @ np.diag(physical_energies) @ transform
        zero_positions = np.flatnonzero(
            np.abs(physical_energies) <= float(config.gamma_zero_mode_tolerance)
        )
        if index in gamma_indices:
            normalized_trials = _mix_fields(
                gamma_trials[index], normalization
            )
            orthogonal_trials = _mix_fields(normalized_trials, lowdin)
            transverse_full = physical_state.get_internal_block(
                *index, full_bloch=True
            )
            augmented, h_augmented, errors = _gamma_augmented_frame(
                transverse_full,
                orthogonal_trials,
                h_transverse,
                zero_positions,
                inner,
                auxiliary_dimension=n_l,
                auxiliary_eigenvalue=auxiliary_eigenvalue,
                rank_tolerance=rank_tolerance,
                zero_tolerance=float(config.gamma_zero_mode_tolerance),
            )
            gamma_augmented[index] = augmented
            gamma_hamiltonians[index] = h_augmented
            projections[index] = inner.overlap(
                augmented, normalized_trials, chunk_size=64
            )
            gamma_regularized.append(index)
            auxiliary_error, cross_error, augmented_error = errors
            singular_tuple = ()
            coefficients[index] = None
        else:
            if _is_gamma(config, index) and zero_positions.size not in {0, 2}:
                raise ValueError(
                    "ETBC Gamma point contains an unsupported number of selected "
                    f"zero modes: {zero_positions.tolist()}."
                )
            y_l = _canonicalize_nullspace(vh.conj().T[:, n_t:])
            auxiliary_coefficients = orthogonalizer @ y_l
            coefficients[index] = auxiliary_coefficients
            auxiliary_gram = _hermitian(
                auxiliary_coefficients.conj().T
                @ gram_raw
                @ auxiliary_coefficients
            )
            cross = raw_cross[index] @ auxiliary_coefficients
            augmented_gram = np.block(
                [
                    [transverse_grams[index], cross],
                    [cross.conj().T, auxiliary_gram],
                ]
            )
            auxiliary_error = float(
                np.linalg.norm(auxiliary_gram - np.eye(n_l), ord="fro")
            )
            cross_error = float(np.linalg.norm(cross, ord="fro"))
            augmented_error = float(
                np.linalg.norm(augmented_gram - np.eye(n_w), ord="fro")
            )
            if max(auxiliary_error, cross_error, augmented_error) > algebra_tolerance:
                raise ValueError(
                    f"ETBC constructed an invalid frame at k={index}: "
                    f"auxiliary={auxiliary_error:.6g}, cross={cross_error:.6g}, "
                    f"augmented={augmented_error:.6g}."
                )
            normalized_right = gram_raw @ normalization
            projections[index] = np.concatenate(
                (
                    raw_cross[index] @ normalization,
                    auxiliary_coefficients.conj().T @ normalized_right,
                ),
                axis=0,
            )
            singular_tuple = tuple(float(value) for value in singular_values)
            gamma_hamiltonians[index] = np.block(
                [
                    [h_transverse, np.zeros((n_t, n_l), dtype=np.complex128)],
                    [
                        np.zeros((n_l, n_t), dtype=np.complex128),
                        float(auxiliary_eigenvalue)
                        * np.eye(n_l, dtype=np.complex128),
                    ],
                ]
            )
        singular_by_k[index] = singular_tuple
        trial_eigenvalues_by_k[index] = tuple(
            float(value) for value in trial_eigenvalues
        )
        diagnostics_by_k[index] = (
            transverse_error,
            auxiliary_error,
            cross_error,
            augmented_error,
        )

    augmented_fields = np.empty(physical_state.k_shape, dtype=object)
    for index in indices:
        if index in gamma_augmented:
            phase = physical_state.get_phase(*index)[None, :, None]
            augmented_fields[index] = np.ascontiguousarray(
                gamma_augmented[index] * np.conj(phase)
            )
        else:
            values = np.empty(
                (n_w, trial_source.point_count, 3), dtype=np.complex128
            )
            values[:n_t] = physical_state.get_internal_block(
                *index, full_bloch=False
            )
            augmented_fields[index] = values

    for start, stop, trial_chunk in trial_source.iter_raw_chunks():
        for index in indices:
            coefficient = coefficients[index]
            if coefficient is None:
                continue
            auxiliary_full = np.einsum(
                "pic,il->lpc",
                np.asarray(trial_chunk[index], dtype=np.complex128),
                coefficient,
                optimize=True,
            )
            phase = physical_state.get_phase(*index)[start:stop]
            augmented_fields[index][n_t:, start:stop, :] = (
                auxiliary_full * np.conj(phase)[None, :, None]
            )

    augmented_energies = np.empty(physical_state.k_shape, dtype=object)
    augmented_indices = np.empty(physical_state.k_shape, dtype=object)
    augmented_inner = np.empty(physical_state.k_shape, dtype=object)
    augmented_zero = np.empty(physical_state.k_shape, dtype=object)
    base_hamiltonians = np.empty(physical_state.k_shape, dtype=object)
    diagnostics = []
    l_offset = int(np.asarray(physical_state.energy_matrix).shape[-1])
    for index in indices:
        direct_gram = _hermitian(
            inner.overlap(
                augmented_fields[index], augmented_fields[index], chunk_size=64
            )
        )
        direct_error = float(
            np.linalg.norm(direct_gram - np.eye(n_w), ord="fro")
        )
        old = diagnostics_by_k[index]
        if direct_error > algebra_tolerance:
            raise ValueError(
                f"Streaming ETBC augmented frame at k={index} failed direct Gram "
                f"validation: residual={direct_error:.6g}."
            )
        diagnostics.append(
            ETBCKPointDiagnostics(
                tuple(int(value) for value in index),
                n_t,
                n_l,
                singular_by_k[index],
                trial_eigenvalues_by_k[index],
                old[0],
                old[1],
                old[2],
                max(old[3], direct_error),
                index in gamma_indices,
            )
        )
        physical_energies = np.asarray(physical_state.E[index], dtype=float)
        energies = np.concatenate(
            (
                physical_energies,
                np.full(n_l, float(auxiliary_eigenvalue), dtype=float),
            )
        )
        physical_ids = np.asarray(physical_state.E_idx[index], dtype=int)
        augmented_energies[index] = energies
        augmented_indices[index] = np.concatenate(
            (physical_ids, l_offset + np.arange(n_l, dtype=int))
        ).tolist()
        augmented_inner[index] = []
        augmented_zero[index] = np.abs(energies) <= float(
            config.gamma_zero_mode_tolerance
        )
        base_hamiltonians[index] = _hermitian(gamma_hamiltonians[index])

    energy_matrix = np.concatenate(
        (
            np.asarray(physical_state.energy_matrix, dtype=float),
            np.full(
                physical_state.k_shape + (n_l,),
                float(auxiliary_eigenvalue),
                dtype=float,
            ),
        ),
        axis=-1,
    )
    channels = dict(physical_state.band_channels)
    for index in indices:
        for actual_band in np.asarray(physical_state.E_idx[index], dtype=int):
            channels.setdefault(
                int(actual_band), BandChannelReference("H", int(actual_band))
            )
    channels.update(
        {
            l_offset + position: BandChannelReference("L", position)
            for position in range(n_l)
        }
    )
    longitudinal_zero = np.empty(physical_state.k_shape, dtype=object)
    for index in np.ndindex(physical_state.k_shape):
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
        zero_modes=augmented_zero,
        band_channels=channels,
        auxiliary_zero_mode_bands={"longitudinal": longitudinal_zero},
        base_hamiltonians=base_hamiltonians,
    )
    result = ETBCCompletionResult(
        n_t,
        n_l,
        n_w,
        float(auxiliary_eigenvalue),
        tuple(diagnostics),
        tuple(gamma_regularized),
    )
    return ETBCCompletionArtifacts(
        result=result,
        augmented_bundle=bundle,
        projection_seed=ProjectionSeed(projections, source="ETBC trial"),
    )


def prepare_transverse_bundle(
    physical_state,
    trial_provider: Callable[[tuple[int, int, int]], np.ndarray],
    *,
    auxiliary_eigenvalue: float = 0.0,
    rank_tolerance: float = 1.0e-10,
) -> ETBCCompletionArtifacts:
    """Build run-scoped ETBC artifacts for a fixed transverse band group."""

    if hasattr(trial_provider, "iter_raw_chunks"):
        return _complete_transverse_bundle_streaming(
            physical_state,
            trial_provider,
            auxiliary_eigenvalue=float(auxiliary_eigenvalue),
            rank_tolerance=float(rank_tolerance),
        )

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
    for index in physical_state.k_indices():
        for actual_band in np.asarray(
            physical_state.E_idx[index], dtype=int
        ).reshape(-1):
            channels.setdefault(
                int(actual_band), BandChannelReference("H", int(actual_band))
            )
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
        zero_modes=augmented_zero,
        band_channels=channels,
        auxiliary_zero_mode_bands={"longitudinal": longitudinal_zero},
        base_hamiltonians=base_hamiltonians,
    )
    result = ETBCCompletionResult(
        n_t,
        n_l,
        n_w,
        float(auxiliary_eigenvalue),
        tuple(diagnostics),
        tuple(gamma_regularized),
    )
    return ETBCCompletionArtifacts(
        result=result,
        augmented_bundle=bundle,
        projection_seed=ProjectionSeed(
            projection_matrices,
            source="ETBC trial",
        ),
    )


def complete_transverse_bundle(
    physical_state,
    trial_provider: Callable[[tuple[int, int, int]], np.ndarray],
    *,
    auxiliary_eigenvalue: float = 0.0,
    rank_tolerance: float = 1.0e-10,
) -> ETBCCompletionResult:
    """Complete a transverse bundle and return persistent diagnostics only.

    Augmented fields and the projection seed are run-scoped artifacts. The
    calculation runner imports :func:`prepare_transverse_bundle` from this
    concrete module and releases those large arrays after initialization.
    """

    return prepare_transverse_bundle(
        physical_state,
        trial_provider,
        auxiliary_eigenvalue=auxiliary_eigenvalue,
        rank_tolerance=rank_tolerance,
    ).result
