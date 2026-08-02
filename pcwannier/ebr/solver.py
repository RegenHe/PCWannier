from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np

from .models import (
    BandSymmetryVector,
    EBRDecomposition,
    EBRMatrix,
    EBRSearchStatistics,
    EBRSubspaceEnumeration,
    TETBSolution,
)


class EBRSearchLimitError(RuntimeError):
    pass


@dataclass
class _SearchBudget:
    limit: int
    visited: int = 0
    signed_combinations: int = 0

    def consume(self) -> None:
        self.visited += 1
        if self.visited > self.limit:
            raise EBRSearchLimitError(
                f"EBR enumeration exceeded ebr_max_states={self.limit}; "
                f"generated_vectors={self.visited}, "
                f"signed_combinations_tested={self.signed_combinations}. "
                "The search is incomplete; reduce the catalog or increase the explicit "
                "search limit."
            )

    def consume_signed_combination(self) -> None:
        self.signed_combinations += 1


def decompose_ebr(
    vector: BandSymmetryVector,
    matrix: EBRMatrix,
    *,
    max_states: int = 1_000_000,
) -> tuple[EBRDecomposition, ...]:
    """Enumerate all non-negative exact solutions of A n = v."""

    physical = vector.reordered(matrix.row_keys)
    budget = _validated_budget(max_states)
    candidates = _enumerate_weighted_vectors(
        matrix.dimensions,
        physical.total_dimension,
        budget,
    )
    solutions = [
        EBRDecomposition(candidate)
        for candidate in candidates
        if np.array_equal(matrix.values @ candidate, physical.multiplicities)
    ]
    solutions.sort(key=lambda item: tuple(int(value) for value in item.multiplicities))
    return tuple(solutions)


def enumerate_tetb_decompositions(
    vector: BandSymmetryVector,
    matrix: EBRMatrix,
    *,
    gamma_point_name: str,
    max_auxiliary_bands: int = 6,
    max_states: int = 1_000_000,
    required_auxiliary_gamma_rows: tuple[int, ...] = (),
    stop_at_first_physical: bool = True,
) -> tuple[TETBSolution, ...]:
    """Enumerate transverse n_T = n_T+L - n_L solutions by auxiliary dimension."""

    physical = vector.reordered(matrix.row_keys)
    if isinstance(max_auxiliary_bands, bool) or int(max_auxiliary_bands) < 0:
        raise ValueError("max_auxiliary_bands must be a non-negative integer.")
    budget = _validated_budget(max_states)
    gamma_rows = np.asarray(
        [
            index
            for index, key in enumerate(matrix.row_keys)
            if key.point_name == gamma_point_name
        ],
        dtype=np.int64,
    )
    if gamma_rows.size == 0:
        raise ValueError(f"EBR matrix has no Gamma rows named {gamma_point_name!r}.")
    non_gamma_rows = np.asarray(
        [index for index in range(len(matrix.row_keys)) if index not in set(gamma_rows)],
        dtype=np.int64,
    )
    required_rows = tuple(int(value) for value in required_auxiliary_gamma_rows)
    if any(value not in set(gamma_rows) for value in required_rows):
        raise ValueError("Required auxiliary-irrep rows must belong to Gamma.")

    reduced_matrix = matrix.values[non_gamma_rows]
    reduced_vector = physical.multiplicities[non_gamma_rows]
    all_solutions: list[TETBSolution] = []
    for auxiliary_dimension in range(int(max_auxiliary_bands) + 1):
        longitudinal = _enumerate_weighted_vectors(
            matrix.dimensions, auxiliary_dimension, budget
        )
        combined = _enumerate_weighted_vectors(
            matrix.dimensions,
            physical.total_dimension + auxiliary_dimension,
            budget,
        )
        combined_by_vector: dict[bytes, list[np.ndarray]] = {}
        for candidate in combined:
            key = np.ascontiguousarray(reduced_matrix @ candidate).tobytes()
            combined_by_vector.setdefault(key, []).append(candidate)
        level_solutions = []
        for n_l in longitudinal:
            target = reduced_vector + reduced_matrix @ n_l
            key = np.ascontiguousarray(target).tobytes()
            for n_t_plus_l in combined_by_vector.get(key, ()):
                full_combined = matrix.values @ n_t_plus_l
                full_longitudinal = matrix.values @ n_l
                surrogate = full_combined - full_longitudinal - physical.multiplicities
                if np.any(surrogate[non_gamma_rows] != 0):
                    raise RuntimeError("Internal TETB enumeration produced a non-Gamma residual.")
                available_at_gamma = full_combined[gamma_rows] - physical.multiplicities[gamma_rows]
                contains_negative_surrogate = bool(np.any(available_at_gamma < 0))
                auxiliary_kind_ok = (
                    auxiliary_dimension == 0
                    or not required_rows
                    or any(full_longitudinal[index] > 0 for index in required_rows)
                )
                physical_solution = not contains_negative_surrogate and auxiliary_kind_ok
                if contains_negative_surrogate:
                    reason = "negative Gamma surrogate is not contained in the auxiliary space"
                elif not auxiliary_kind_ok:
                    reason = "auxiliary bands contain neither the scalar nor pseudoscalar Gamma irrep"
                else:
                    reason = "physical"
                level_solutions.append(
                    TETBSolution(
                        n_t_plus_l,
                        n_l,
                        n_t_plus_l - n_l,
                        surrogate,
                        auxiliary_dimension,
                        physical_solution,
                        reason,
                    )
                )
        level_solutions.sort(
            key=lambda item: (
                tuple(int(value) for value in item.n_l),
                tuple(int(value) for value in item.n_t_plus_l),
            )
        )
        all_solutions.extend(level_solutions)
        if stop_at_first_physical and any(item.physical for item in level_solutions):
            break
    return tuple(all_solutions)


def enumerate_ebr_subspace_solutions(
    available: BandSymmetryVector,
    matrix: EBRMatrix,
    *,
    gamma_point_name: str,
    target_dimension: int,
    gamma_surrogate,
    max_auxiliary_bands: int = 6,
    max_states: int = 1_000_000,
    required_auxiliary_gamma_rows: tuple[int, ...] = (),
    gamma_sector: str = "requested",
) -> EBRSubspaceEnumeration:
    """Enumerate signed EBRs whose HSP irreps fit inside an outer-band inventory.

    ``gamma_surrogate`` is nonzero only when the requested subspace includes the
    two direction-dependent transverse Gamma zero modes.  Otherwise Gamma is
    matched as an ordinary positive-frequency representation point.
    """

    inventory = available.reordered(matrix.row_keys)
    if isinstance(target_dimension, bool) or int(target_dimension) < 2:
        raise ValueError("target_dimension must be an integer of at least two.")
    if isinstance(max_auxiliary_bands, bool) or int(max_auxiliary_bands) < 0:
        raise ValueError("max_auxiliary_bands must be a non-negative integer.")
    surrogate = _readonly_vector(gamma_surrogate, len(matrix.row_keys), "Gamma surrogate")
    gamma_mask = np.asarray(
        [key.point_name == gamma_point_name for key in matrix.row_keys], dtype=bool
    )
    if not np.any(gamma_mask):
        raise ValueError(f"EBR matrix has no Gamma rows named {gamma_point_name!r}.")
    if np.any(surrogate[~gamma_mask] != 0):
        raise ValueError("The transverse Gamma surrogate must vanish away from Gamma.")
    required_rows = tuple(int(value) for value in required_auxiliary_gamma_rows)
    gamma_indices = set(np.flatnonzero(gamma_mask).tolist())
    if any(value not in gamma_indices for value in required_rows):
        raise ValueError("Required auxiliary-irrep rows must belong to Gamma.")

    budget = _validated_budget(max_states)
    output: list[tuple[TETBSolution, BandSymmetryVector]] = []
    seen: set[tuple[int, ...]] = set()
    for auxiliary_dimension in range(int(max_auxiliary_bands) + 1):
        longitudinal = _enumerate_weighted_vectors(
            matrix.dimensions, auxiliary_dimension, budget
        )
        combined = _enumerate_weighted_vectors(
            matrix.dimensions, int(target_dimension) + auxiliary_dimension, budget
        )
        for n_l in longitudinal:
            full_longitudinal = matrix.values @ n_l
            auxiliary_kind_ok = (
                auxiliary_dimension == 0
                or not required_rows
                or any(full_longitudinal[index] > 0 for index in required_rows)
            )
            if not auxiliary_kind_ok:
                continue
            for n_t_plus_l in combined:
                budget.consume_signed_combination()
                n_t = n_t_plus_l - n_l
                key = tuple(int(value) for value in n_t)
                if key in seen:
                    continue
                formal = matrix.values @ n_t
                selected = formal - surrogate
                if np.any(selected < 0) or np.any(selected > inventory.multiplicities):
                    continue
                seen.add(key)
                solution = TETBSolution(
                    n_t_plus_l,
                    n_l,
                    n_t,
                    surrogate,
                    auxiliary_dimension,
                    True,
                    "fits the outer-window high-symmetry inventory",
                )
                output.append(
                    (
                        solution,
                        BandSymmetryVector(
                            matrix.row_keys,
                            selected,
                            int(target_dimension),
                        ),
                    )
                )
    output.sort(
        key=lambda item: (
            item[0].auxiliary_dimension,
            tuple(int(value) for value in item[0].n_t),
        )
    )
    solutions = tuple(output)
    return EBRSubspaceEnumeration(
        solutions,
        EBRSearchStatistics(
            complete=True,
            gamma_sectors=(gamma_sector,),
            auxiliary_dimensions_examined=tuple(
                range(int(max_auxiliary_bands) + 1)
            ),
            weighted_vectors_generated=budget.visited,
            signed_combinations_tested=budget.signed_combinations,
            algebraic_solutions=len(solutions),
            search_limit=budget.limit,
        ),
    )


def _validated_budget(max_states: int) -> _SearchBudget:
    if isinstance(max_states, bool) or int(max_states) <= 0:
        raise ValueError("max_states must be a positive integer.")
    return _SearchBudget(int(max_states))


def _readonly_vector(values, size: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (size,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be a finite vector with length {size}.")
    rounded = np.rint(raw).astype(np.int64)
    if not np.array_equal(raw, rounded):
        raise ValueError(f"{name} must contain integers.")
    return rounded


def _enumerate_weighted_vectors(
    dimensions: np.ndarray,
    total: int,
    budget: _SearchBudget,
) -> tuple[np.ndarray, ...]:
    weights = np.asarray(dimensions, dtype=np.int64)
    if weights.ndim != 1 or weights.size == 0 or np.any(weights <= 0):
        raise ValueError("EBR dimensions must be positive integers.")
    if isinstance(total, bool) or int(total) < 0:
        raise ValueError("Requested EBR dimension must be a non-negative integer.")
    total = int(total)
    suffix_gcd = np.empty(weights.size + 1, dtype=np.int64)
    suffix_gcd[-1] = 0
    for index in range(weights.size - 1, -1, -1):
        suffix_gcd[index] = gcd(int(weights[index]), int(suffix_gcd[index + 1]))
    current = np.zeros(weights.size, dtype=np.int64)
    output: list[np.ndarray] = []

    def search(index: int, remaining: int) -> None:
        if index == weights.size:
            if remaining == 0:
                budget.consume()
                output.append(current.copy())
            return
        divisor = int(suffix_gcd[index])
        if divisor and remaining % divisor:
            return
        weight = int(weights[index])
        for multiplicity in range(remaining // weight + 1):
            current[index] = multiplicity
            search(index + 1, remaining - multiplicity * weight)
        current[index] = 0

    search(0, total)
    return tuple(output)
