from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import numpy as np

from .group import SpaceGroup


@dataclass(frozen=True)
class ConjugacyClass:
    element_indices: tuple[int, ...]


@dataclass(frozen=True)
class AutomaticIrrepCharacter:
    dimension: int
    characters: tuple[complex, ...]


class FiniteGroupTable:
    """A finite multiplication table independent of any space-group embedding."""

    def __init__(self, element_names: Iterable[str], multiplication, *, name: str | None = None):
        names = tuple(str(value) for value in element_names)
        if not names or len(names) != len(set(names)) or any(not value for value in names):
            raise ValueError("Finite-group element names must be unique and non-empty.")
        product = np.asarray(multiplication, dtype=np.int64)
        if product.shape != (len(names), len(names)):
            raise ValueError("Finite-group multiplication must be a square table matching its elements.")
        if np.any(product < 0) or np.any(product >= len(names)):
            raise ValueError("Finite-group multiplication contains an invalid element index.")
        identity_candidates = [
            index
            for index in range(len(names))
            if np.array_equal(product[index], np.arange(len(names)))
            and np.array_equal(product[:, index], np.arange(len(names)))
        ]
        if len(identity_candidates) != 1:
            raise ValueError("Finite-group identity is missing or non-unique.")
        self.element_names = names
        self.name = None if name is None else str(name)
        self.identity_index = identity_candidates[0]
        product = product.copy()
        product.setflags(write=False)
        self.multiplication = product
        inverse = np.empty(len(names), dtype=np.int64)
        for index in range(len(names)):
            matches = np.flatnonzero(
                (product[index] == self.identity_index)
                & (product[:, index] == self.identity_index)
            )
            if matches.size != 1:
                raise ValueError("Finite-group inverse is missing or non-unique.")
            inverse[index] = matches[0]
        inverse.setflags(write=False)
        self.inverse = inverse
        self._validate_associativity()

    @property
    def order(self) -> int:
        return len(self.element_names)

    @property
    def operation_names(self) -> tuple[str, ...]:
        return self.element_names

    def element_index(self, name: str) -> int:
        try:
            return self.element_names.index(str(name))
        except ValueError as exc:
            raise KeyError(f"Unknown finite-group element {name!r}.") from exc

    @cached_property
    def element_orders(self) -> tuple[int, ...]:
        output = []
        for element in range(self.order):
            value = self.identity_index
            for power in range(1, self.order + 1):
                value = int(self.multiplication[value, element])
                if value == self.identity_index:
                    output.append(power)
                    break
            else:
                raise ValueError("Finite-group element order exceeds the group order.")
        return tuple(output)

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
            output.append(ConjugacyClass(tuple(sorted(members))))
            remaining.difference_update(members)
        return tuple(output)

    @cached_property
    def regular_matrices(self) -> tuple[np.ndarray, ...]:
        matrices = []
        for element in range(self.order):
            matrix = np.zeros((self.order, self.order), dtype=np.complex128)
            matrix[self.multiplication[element], np.arange(self.order)] = 1.0
            matrix.setflags(write=False)
            matrices.append(matrix)
        return tuple(matrices)

    @cached_property
    def automatic_irreps(self) -> tuple[AutomaticIrrepCharacter, ...]:
        blocks = [np.eye(self.order, dtype=np.complex128)]
        central_operators = []
        for conjugacy_class in self.conjugacy_classes:
            class_sum = sum(
                self.regular_matrices[index] for index in conjugacy_class.element_indices
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

    def _validate_associativity(self) -> None:
        for left in range(self.order):
            for middle in range(self.order):
                for right in range(self.order):
                    lhs = self.multiplication[self.multiplication[left, middle], right]
                    rhs = self.multiplication[left, self.multiplication[middle, right]]
                    if lhs != rhs:
                        raise ValueError("Finite-group multiplication is not associative.")


@dataclass(frozen=True)
class ConcreteFiniteGroup:
    """A finite co-group together with its actual Seitz representatives."""

    group: SpaceGroup
    operation_indices: tuple[int, ...]
    table: FiniteGroupTable

    @classmethod
    def from_space_group(
        cls,
        group: SpaceGroup,
        operation_indices: Iterable[int] | None = None,
        *,
        name: str | None = None,
    ) -> "ConcreteFiniteGroup":
        indices = (
            tuple(range(len(group.operations)))
            if operation_indices is None
            else tuple(int(index) for index in operation_indices)
        )
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("A concrete finite group requires unique operation indices.")
        if group.identity_index not in indices:
            raise ValueError("A concrete finite group must contain the identity representative.")
        global_to_local = {global_index: local for local, global_index in enumerate(indices)}
        multiplication = np.empty((len(indices), len(indices)), dtype=np.int64)
        for left_local, left_global in enumerate(indices):
            for right_local, right_global in enumerate(indices):
                product_global = group.operation_index(
                    group.operations[left_global] * group.operations[right_global]
                )
                if product_global not in global_to_local:
                    raise ValueError("The supplied Seitz representative subset is not closed.")
                multiplication[left_local, right_local] = global_to_local[product_global]
        names = tuple(
            group.operations[index].name or f"g{index}" for index in indices
        )
        return cls(group, indices, FiniteGroupTable(names, multiplication, name=name))

    @property
    def order(self) -> int:
        return self.table.order

    def local_index(self, operation_index: int) -> int:
        try:
            return self.operation_indices.index(int(operation_index))
        except ValueError as exc:
            raise ValueError("Operation does not belong to this concrete finite group.") from exc

    def global_index(self, local_index: int) -> int:
        return self.operation_indices[int(local_index)]

    @property
    def rotations(self) -> tuple[np.ndarray, ...]:
        return tuple(self.group.operations[index].rotation for index in self.operation_indices)


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
