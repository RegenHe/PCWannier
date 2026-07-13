from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import numpy as np

from .group import SpaceGroup


@dataclass(frozen=True)
class ConjugacyClass:
    operation_indices: tuple[int, ...]


@dataclass(frozen=True)
class AutomaticIrrepCharacter:
    dimension: int
    characters: tuple[complex, ...]


@dataclass(frozen=True)
class GroupIrrep:
    name: str
    dimension: int
    table: "FiniteGroupTable"
    matrices: tuple[np.ndarray, ...] | None
    characters: tuple[complex, ...]

    def matrix_for_global_index(self, operation_index: int) -> np.ndarray:
        if self.matrices is None:
            raise ValueError(
                f"Irrep {self.name!r} has characters only and cannot be used as a Wannier site representation."
            )
        return self.matrices[self.table.local_index(operation_index)]

    def character_for_global_index(self, operation_index: int) -> complex:
        return self.characters[self.table.local_index(operation_index)]


class FiniteGroupTable:
    """Finite group data for the full group or a closed operation subset."""

    def __init__(
        self,
        group: SpaceGroup,
        operation_indices: Iterable[int] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        indices = (
            tuple(range(len(group.operations)))
            if operation_indices is None
            else tuple(int(index) for index in operation_indices)
        )
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("A finite-group table requires unique operation indices.")
        if any(index < 0 or index >= len(group.operations) for index in indices):
            raise IndexError("Finite-group operation index is out of range.")
        if group.identity_index not in indices:
            raise ValueError("A finite-group operation subset must contain the identity.")
        self.group = group
        self.operation_indices = indices
        self.name = name
        self._global_to_local = {global_index: local for local, global_index in enumerate(indices)}
        multiplication = np.empty((len(indices), len(indices)), dtype=np.int64)
        for left_local, left_global in enumerate(indices):
            for right_local, right_global in enumerate(indices):
                product_global = group.operation_index(
                    group.operations[left_global] * group.operations[right_global]
                )
                if product_global not in self._global_to_local:
                    raise ValueError("The supplied operation subset is not closed under multiplication.")
                multiplication[left_local, right_local] = self._global_to_local[product_global]
        multiplication.setflags(write=False)
        self.multiplication = multiplication
        self.identity_index = self._global_to_local[group.identity_index]
        inverse = np.empty(len(indices), dtype=np.int64)
        for local_index in range(len(indices)):
            candidates = np.flatnonzero(
                (multiplication[local_index] == self.identity_index)
                & (multiplication[:, local_index] == self.identity_index)
            )
            if candidates.size != 1:
                raise ValueError("Finite-group inverse is missing or non-unique.")
            inverse[local_index] = candidates[0]
        inverse.setflags(write=False)
        self.inverse = inverse

    @property
    def order(self) -> int:
        return len(self.operation_indices)

    @property
    def operation_names(self) -> tuple[str, ...]:
        names = tuple(self.group.operations[index].name for index in self.operation_indices)
        if any(name is None for name in names):
            raise ValueError("All operations used in representation analysis must have names.")
        return tuple(str(name) for name in names)

    def local_index(self, global_index: int) -> int:
        try:
            return self._global_to_local[int(global_index)]
        except KeyError as exc:
            raise ValueError("Operation does not belong to this finite-group table.") from exc

    @cached_property
    def conjugacy_classes(self) -> tuple[ConjugacyClass, ...]:
        remaining = set(range(self.order))
        output = []
        while remaining:
            representative = min(remaining)
            members = {
                int(
                    self.multiplication[
                        self.multiplication[conjugator, representative],
                        self.inverse[conjugator],
                    ]
                )
                for conjugator in range(self.order)
            }
            output.append(
                ConjugacyClass(
                    tuple(sorted(self.operation_indices[local_index] for local_index in members))
                )
            )
            remaining.difference_update(members)
        return tuple(output)

    @cached_property
    def regular_matrices(self) -> tuple[np.ndarray, ...]:
        matrices = []
        for operation in range(self.order):
            matrix = np.zeros((self.order, self.order), dtype=np.complex128)
            matrix[self.multiplication[operation], np.arange(self.order)] = 1.0
            matrix.setflags(write=False)
            matrices.append(matrix)
        return tuple(matrices)

    @cached_property
    def automatic_irreps(self) -> tuple[AutomaticIrrepCharacter, ...]:
        blocks = [np.eye(self.order, dtype=np.complex128)]
        central_operators = []
        for conjugacy_class in self.conjugacy_classes:
            class_sum = sum(
                self.regular_matrices[self.local_index(global_index)]
                for global_index in conjugacy_class.operation_indices
            )
            central_operators.extend(
                (
                    0.5 * (class_sum + class_sum.conj().T),
                    (class_sum - class_sum.conj().T) / (2.0j),
                )
            )
        for operator in central_operators:
            refined = []
            for basis in blocks:
                restricted = basis.conj().T @ operator @ basis
                values, vectors = np.linalg.eigh(0.5 * (restricted + restricted.conj().T))
                for cluster in _eigenvalue_clusters(values):
                    child = basis @ vectors[:, cluster]
                    child, _ = np.linalg.qr(child)
                    refined.append(child)
            blocks = refined

        irreps = []
        for basis in blocks:
            isotypic_dimension = basis.shape[1]
            dimension = int(round(np.sqrt(isotypic_dimension)))
            if dimension * dimension != isotypic_dimension:
                raise ValueError(
                    "Could not resolve finite-group irreducible characters from the regular representation."
                )
            characters = tuple(
                complex(np.trace(basis.conj().T @ matrix @ basis) / dimension)
                for matrix in self.regular_matrices
            )
            irreps.append(AutomaticIrrepCharacter(dimension, characters))
        irreps.sort(key=lambda item: (item.dimension, _character_sort_key(item.characters)))
        _validate_automatic_character_table(self, irreps)
        return tuple(irreps)

    def projective_factor_residual(self, k_fractional) -> float:
        """Return the largest phase from representative products differing by a lattice translation."""
        kpoint = np.asarray(k_fractional, dtype=float)
        if kpoint.shape != (self.group.dimension,):
            raise ValueError(f"k_fractional must have shape {(self.group.dimension,)}.")
        residual = 0.0
        for left_local, left_global in enumerate(self.operation_indices):
            for right_local, right_global in enumerate(self.operation_indices):
                product = self.group.operations[left_global] * self.group.operations[right_global]
                target_local = int(self.multiplication[left_local, right_local])
                representative = self.group.operations[self.operation_indices[target_local]]
                shift = product.translation - representative.translation
                rounded = np.rint(shift)
                if not np.allclose(shift, rounded, rtol=0.0, atol=self.group.tolerance):
                    raise ValueError("Space-group product differs from its representative by a non-lattice shift.")
                phase = np.exp(-2j * np.pi * np.dot(kpoint, rounded))
                residual = max(residual, float(abs(phase - 1.0)))
        return residual


def _eigenvalue_clusters(values: np.ndarray) -> tuple[np.ndarray, ...]:
    if values.size == 0:
        return ()
    scale = max(1.0, float(np.max(np.abs(values))))
    threshold = 1.0e-9 * scale
    clusters = []
    start = 0
    for index in range(1, values.size):
        if abs(values[index] - values[index - 1]) > threshold:
            clusters.append(np.arange(start, index))
            start = index
    clusters.append(np.arange(start, values.size))
    return tuple(clusters)


def _character_sort_key(characters: tuple[complex, ...]) -> tuple[float, ...]:
    values = []
    for value in characters:
        values.extend((round(value.real, 10), round(value.imag, 10)))
    return tuple(values)


def _validate_automatic_character_table(
    table: FiniteGroupTable,
    irreps: list[AutomaticIrrepCharacter],
) -> None:
    if sum(irrep.dimension**2 for irrep in irreps) != table.order:
        raise ValueError("Irreducible-character dimensions do not satisfy the finite-group sum rule.")
    for left_index, left in enumerate(irreps):
        for right_index, right in enumerate(irreps):
            inner = sum(
                np.conj(left.characters[index]) * right.characters[index]
                for index in range(table.order)
            ) / table.order
            expected = 1.0 if left_index == right_index else 0.0
            if abs(inner - expected) > 1.0e-7:
                raise ValueError("Automatically generated irreducible characters are not orthonormal.")
