from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .kspace import neighbor_reciprocal_lattice_vectors
from .parallel import parallel_map


Index3D = tuple[int, int, int]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeighborSubspaceSmoothness:
    label: str
    link_count: int
    minimum_singular_value: float
    minimum_regular_singular_value: float
    median_minimum_singular_value: float
    worst_source_index: Index3D
    worst_direction: int
    worst_regular_source_index: Index3D
    worst_regular_direction: int


def _evaluate_neighbor_subspace(
    state,
    label: str,
    overlap_at: Callable[[Index3D, int], np.ndarray],
    basis_at: Callable[[Index3D], np.ndarray],
    *,
    zero_mode_at: Callable[[Index3D], bool],
) -> NeighborSubspaceSmoothness:
    b_count = len(state.config.composition_of_b) // 2
    links = tuple(
        (tuple(int(value) for value in raw_index), direction)
        for raw_index in state.k_indices()
        for direction in range(b_count)
    )

    def calculate(link) -> tuple[float, Index3D, int, bool]:
        index, direction = link
        source_basis = np.asarray(basis_at(index), dtype=np.complex128)
        target_raw, _ = neighbor_reciprocal_lattice_vectors(
            state.config, list(index), direction
        )
        target = tuple(int(value) for value in target_raw)
        target_basis = np.asarray(basis_at(target), dtype=np.complex128)
        overlap = np.asarray(overlap_at(index, direction), dtype=np.complex128)
        reduced = source_basis.conj().T @ overlap @ target_basis
        singular_values = np.linalg.svd(reduced, compute_uv=False)
        if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
            raise FloatingPointError(
                f"Invalid neighbor singular values for {label} at k={index}, "
                f"direction={direction}."
            )
        return (
            float(singular_values[-1]),
            index,
            direction,
            not (zero_mode_at(index) or zero_mode_at(target)),
        )

    threads = max(1, int(getattr(state, "configured_threads", 1)))
    rows = list(parallel_map(links, calculate, threads, ordered=True))
    if not rows:
        raise ValueError(f"No neighbor links are available for {label} diagnostics.")
    regular = [row for row in rows if row[3]]
    if not regular:
        regular = rows
    worst = min(rows, key=lambda row: row[0])
    worst_regular = min(regular, key=lambda row: row[0])
    return NeighborSubspaceSmoothness(
        label=label,
        link_count=len(rows),
        minimum_singular_value=worst[0],
        minimum_regular_singular_value=worst_regular[0],
        median_minimum_singular_value=float(np.median([row[0] for row in rows])),
        worst_source_index=worst[1],
        worst_direction=worst[2],
        worst_regular_source_index=worst_regular[1],
        worst_regular_direction=worst_regular[2],
    )


def outer_channel_smoothness(state, mset) -> tuple[NeighborSubspaceSmoothness, ...]:
    """Measure gauge-invariant neighbor overlap for each input T/L channel."""

    channels = getattr(state, "band_channels", {})
    if not channels:
        return ()
    positions: dict[str, dict[Index3D, np.ndarray]] = {}
    for raw_index in state.k_indices():
        index = tuple(int(value) for value in raw_index)
        grouped: dict[str, list[int]] = {}
        for position, actual_band in enumerate(state.E_idx[index]):
            reference = channels.get(int(actual_band))
            if reference is None:
                continue
            name = str(reference.channel).strip().upper()
            grouped.setdefault(name, []).append(position)
        for name, values in grouped.items():
            positions.setdefault(name, {})[index] = np.asarray(values, dtype=int)

    results = []
    zero_tolerance = float(state.config.gamma_zero_mode_tolerance)
    for name in sorted(positions):
        channel_positions = positions[name]
        dimensions = {values.size for values in channel_positions.values()}
        if len(channel_positions) != state.get_k_num() or len(dimensions) != 1:
            continue

        def basis_at(index: Index3D) -> np.ndarray:
            count = len(state.E_idx[index])
            return np.eye(count, dtype=np.complex128)[:, channel_positions[index]]

        def zero_mode_at(index: Index3D) -> bool:
            energies = np.asarray(state.E[index], dtype=float)
            return bool(
                np.any(np.abs(energies[channel_positions[index]]) <= zero_tolerance)
            )

        results.append(
            _evaluate_neighbor_subspace(
                state,
                f"outer {name}",
                lambda index, direction: mset.get_M0(*index, direction),
                basis_at,
                zero_mode_at=zero_mode_at,
            )
        )
    return tuple(results)


def selected_sector_smoothness(
    ctx,
    projectors: np.ndarray,
) -> tuple[NeighborSubspaceSmoothness, ...]:
    """Measure final selected T, L, and combined subspaces in the source Hilbert space."""

    state = ctx.state
    projector_grid = np.asarray(projectors, dtype=object)
    if projector_grid.shape != state.k_shape:
        raise ValueError(
            f"Transverse projector grid has shape {projector_grid.shape}; "
            f"expected {state.k_shape}."
        )
    coefficient_grid = state.gen_matrix_on_kmesh(
        lambda i, j, k: np.asarray(
            ctx.output_state_coefficients_at(i, j, k), dtype=np.complex128
        )
    )
    bases: dict[str, dict[Index3D, np.ndarray]] = {"selected T+L": {}}
    eigensystems: dict[Index3D, tuple[np.ndarray, np.ndarray]] = {}
    transverse_dimensions: set[int] = set()
    for raw_index in state.k_indices():
        index = tuple(int(value) for value in raw_index)
        projector = np.asarray(projector_grid[index], dtype=np.complex128)
        projector = 0.5 * (projector + projector.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(projector)
        transverse_dimension = int(np.count_nonzero(eigenvalues > 0.5))
        transverse_dimensions.add(transverse_dimension)
        eigensystems[index] = (eigenvalues, eigenvectors)
        bases["selected T+L"][index] = np.eye(
            projector.shape[0], dtype=np.complex128
        )

    wannier_dimension = next(iter(eigensystems.values()))[0].size
    split_available = (
        len(transverse_dimensions) == 1
        and 0 < next(iter(transverse_dimensions)) < wannier_dimension
    )
    labels = ["selected T+L"]
    if split_available:
        transverse_dimension = next(iter(transverse_dimensions))
        bases["selected T"] = {}
        bases["selected L"] = {}
        for index, (_, eigenvectors) in eigensystems.items():
            bases["selected T"][index] = eigenvectors[:, -transverse_dimension:]
            bases["selected L"][index] = eigenvectors[:, :-transverse_dimension]
        labels = ["selected T", "selected L", "selected T+L"]
    else:
        LOGGER.debug(
            "Selected T/L sector smoothness is unavailable because P_T does not "
            "define a fixed nontrivial split: transverse_dimensions=%s, "
            "wannier_dimension=%s. The complete selected-space diagnostic is retained.",
            sorted(transverse_dimensions),
            wannier_dimension,
        )

    def overlap_at(index: Index3D, direction: int) -> np.ndarray:
        target_raw, _ = neighbor_reciprocal_lattice_vectors(
            state.config, list(index), direction
        )
        target = tuple(int(value) for value in target_raw)
        return (
            coefficient_grid[index].conj().T
            @ np.asarray(ctx.mset.mM0[index][direction], dtype=np.complex128)
            @ coefficient_grid[target]
        )

    zero_tolerance = float(state.config.gamma_zero_mode_tolerance)

    def zero_mode_at(index: Index3D) -> bool:
        energies = np.asarray(state.E[index], dtype=float)
        channels = getattr(state, "band_channels", {})
        transverse = [
            position
            for position, actual_band in enumerate(state.E_idx[index])
            if str(channels[int(actual_band)].channel).strip().upper() in {"H", "T"}
        ]
        return bool(
            transverse
            and np.any(np.abs(energies[np.asarray(transverse)]) <= zero_tolerance)
        )

    return tuple(
        _evaluate_neighbor_subspace(
            state,
            label,
            overlap_at,
            lambda index, label=label: bases[label][index],
            zero_mode_at=zero_mode_at,
        )
        for label in labels
    )
