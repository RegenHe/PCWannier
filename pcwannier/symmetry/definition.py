from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .group import SpaceGroup
from .tables import FiniteGroupTable, GroupIrrep


@dataclass(frozen=True)
class StandardSubgroupDefinition:
    name: str
    table: FiniteGroupTable
    irreps: tuple[GroupIrrep, ...]

    def irrep(self, name: str) -> GroupIrrep:
        for irrep in self.irreps:
            if irrep.name == name:
                return irrep
        raise KeyError(f"Unknown irrep {name!r} in subgroup {self.name!r}.")


@dataclass(frozen=True)
class ResolvedLittleGroup:
    name: str
    table: FiniteGroupTable
    irreps: tuple[GroupIrrep, ...]


@dataclass(frozen=True)
class SymmetryGroupDefinition:
    name: str
    dimension: int
    tolerance: float
    group: SpaceGroup
    table: FiniteGroupTable
    irreps: tuple[GroupIrrep, ...]
    subgroups: tuple[StandardSubgroupDefinition, ...] = ()

    def __post_init__(self) -> None:
        if self.dimension != self.group.dimension:
            raise ValueError("Symmetry-group definition dimension does not match its operations.")
        operation_sets = [subgroup.table.operation_indices for subgroup in self.subgroups]
        if len(operation_sets) != len(set(operation_sets)):
            raise ValueError("Standard subgroup operation sets must be unique.")

    def subgroup_for_operations(
        self, operation_indices
    ) -> StandardSubgroupDefinition | None:
        key = tuple(sorted(int(index) for index in operation_indices))
        if key == tuple(sorted(self.table.operation_indices)):
            return StandardSubgroupDefinition(self.name, self.table, self.irreps)
        for subgroup in self.subgroups:
            if tuple(sorted(subgroup.table.operation_indices)) == key:
                return subgroup
        return None

    def resolve_little_group(self, operation_indices, k_fractional) -> ResolvedLittleGroup:
        indices = tuple(int(index) for index in operation_indices)
        declared = self.subgroup_for_operations(indices)
        if declared is not None:
            table = declared.table
            name = declared.name
            irreps = declared.irreps
        else:
            table = FiniteGroupTable(self.group, indices)
            name = "subgroup_" + "_".join(str(index) for index in sorted(indices))
            irreps = tuple(
                GroupIrrep(
                    f"{name}_irrep_{position + 1}",
                    automatic.dimension,
                    table,
                    None,
                    automatic.characters,
                )
                for position, automatic in enumerate(table.automatic_irreps)
            )
        factor_residual = table.projective_factor_residual(k_fractional)
        if factor_residual > max(self.tolerance, 1.0e-10):
            raise NotImplementedError(
                f"Little group {name!r} has a non-trivial projective factor at k={np.asarray(k_fractional).tolist()} "
                f"(residual={factor_residual:.6g}); nonsymmorphic projective irreps are not implemented."
            )
        return ResolvedLittleGroup(name, table, irreps)

    def site_irrep(self, operation_indices, name: str) -> GroupIrrep:
        indices = tuple(int(index) for index in operation_indices)
        declared = self.subgroup_for_operations(indices)
        if declared is not None:
            return declared.irrep(name)
        full = next((irrep for irrep in self.irreps if irrep.name == name), None)
        if full is None or full.matrices is None:
            available = sorted(
                irrep.name
                for subgroup in self.subgroups
                if set(subgroup.table.operation_indices) == set(indices)
                for irrep in subgroup.irreps
            )
            raise KeyError(
                f"No matrix representation named {name!r} is defined for site-group operations {indices}; "
                f"available={available}."
            )
        restricted_characters = np.asarray(
            [full.character_for_global_index(index) for index in indices], dtype=np.complex128
        )
        norm = float(np.vdot(restricted_characters, restricted_characters).real / len(indices))
        if abs(norm - 1.0) > 1.0e-7:
            raise ValueError(
                f"Full-group irrep {name!r} becomes reducible on the target site group; "
                "define and reference a standard subgroup irrep instead."
            )
        table = FiniteGroupTable(self.group, indices)
        matrices = tuple(full.matrix_for_global_index(index) for index in table.operation_indices)
        return GroupIrrep(name, full.dimension, table, matrices, tuple(restricted_characters))


def build_group_irrep(
    table: FiniteGroupTable,
    name: str,
    dimension: int,
    *,
    characters: Mapping[str, complex] | None = None,
    generators: Mapping[str, np.ndarray] | None = None,
    matrices: Mapping[str, np.ndarray] | None = None,
) -> GroupIrrep:
    if dimension <= 0:
        raise ValueError(f"Irrep {name!r} dimension must be positive.")
    supplied = sum(value is not None for value in (characters, generators, matrices))
    if supplied != 1:
        raise ValueError(
            f"Irrep {name!r} must define exactly one of characters, generators, or matrices."
        )
    operation_names = table.operation_names
    if characters is not None:
        if dimension != 1:
            raise ValueError(
                f"Irrep {name!r} has dimension {dimension}; characters alone only define one-dimensional matrices."
            )
        _require_exact_names(characters, operation_names, f"Irrep {name!r} characters")
        representation = tuple(
            np.asarray([[complex(characters[operation_name])]], dtype=np.complex128)
            for operation_name in operation_names
        )
    elif matrices is not None:
        _require_exact_names(matrices, operation_names, f"Irrep {name!r} matrices")
        representation = tuple(
            _validated_matrix(matrices[operation_name], dimension, f"Irrep {name!r} operation {operation_name!r}")
            for operation_name in operation_names
        )
    else:
        assert generators is not None
        representation = _generate_representation(table, name, dimension, generators)

    _validate_representation(table, name, dimension, representation)
    output = tuple(np.asarray(matrix, dtype=np.complex128).copy() for matrix in representation)
    for matrix in output:
        matrix.setflags(write=False)
    irrep = GroupIrrep(
        str(name),
        int(dimension),
        table,
        output,
        tuple(complex(np.trace(matrix)) for matrix in output),
    )
    _validate_class_characters(irrep)
    return irrep


def validate_irrep_table(table: FiniteGroupTable, irreps: tuple[GroupIrrep, ...]) -> None:
    names = [irrep.name for irrep in irreps]
    if not irreps or len(names) != len(set(names)):
        raise ValueError("Irrep names must be unique and the irrep table must be non-empty.")
    if any(irrep.table is not table for irrep in irreps):
        raise ValueError("All irreps must use the same finite-group table.")
    if sum(irrep.dimension**2 for irrep in irreps) != table.order:
        raise ValueError("Irrep dimensions do not satisfy sum(dim^2) = group order.")
    for left_index, left in enumerate(irreps):
        for right_index, right in enumerate(irreps):
            inner = sum(
                np.conj(left.characters[index]) * right.characters[index]
                for index in range(table.order)
            ) / table.order
            expected = 1.0 if left_index == right_index else 0.0
            if abs(inner - expected) > 1.0e-7:
                raise ValueError(
                    f"Irrep characters {left.name!r} and {right.name!r} violate character orthogonality."
                )
    automatic = list(table.automatic_irreps)
    for irrep in irreps:
        match = next(
            (
                position
                for position, candidate in enumerate(automatic)
                if candidate.dimension == irrep.dimension
                and np.allclose(candidate.characters, irrep.characters, rtol=0.0, atol=1.0e-7)
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Irrep {irrep.name!r} does not match the finite-group character table.")
        automatic.pop(match)
    if automatic:
        raise ValueError("The supplied irrep table is incomplete.")


def _generate_representation(
    table: FiniteGroupTable,
    name: str,
    dimension: int,
    generators: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, ...]:
    if not generators:
        raise ValueError(f"Irrep {name!r} generators must not be empty.")
    unknown = sorted(set(generators) - set(table.operation_names))
    if unknown:
        raise ValueError(f"Irrep {name!r} references unknown generator operations {unknown}.")
    known: dict[int, np.ndarray] = {table.identity_index: np.eye(dimension, dtype=np.complex128)}
    generator_items = []
    for operation_name, matrix in generators.items():
        local_index = table.operation_names.index(operation_name)
        value = _validated_matrix(matrix, dimension, f"Irrep {name!r} generator {operation_name!r}")
        existing = known.get(local_index)
        if existing is not None and not np.allclose(existing, value, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"Irrep {name!r} supplies a conflicting identity/generator matrix.")
        known[local_index] = value
        generator_items.append((local_index, value))
    queue = list(known)
    while queue:
        current = queue.pop(0)
        for generator_index, generator_matrix in generator_items:
            product = int(table.multiplication[current, generator_index])
            product_matrix = known[current] @ generator_matrix
            existing = known.get(product)
            if existing is None:
                known[product] = product_matrix
                queue.append(product)
            elif not np.allclose(existing, product_matrix, rtol=0.0, atol=1.0e-8):
                raise ValueError(f"Irrep {name!r} generators violate a group relation.")
    if len(known) != table.order:
        missing = [table.operation_names[index] for index in range(table.order) if index not in known]
        raise ValueError(f"Irrep {name!r} generators do not generate matrices for {missing}.")
    return tuple(known[index] for index in range(table.order))


def _validate_representation(
    table: FiniteGroupTable,
    name: str,
    dimension: int,
    matrices: tuple[np.ndarray, ...],
) -> None:
    identity = np.eye(dimension)
    for index, matrix in enumerate(matrices):
        if not np.allclose(matrix.conj().T @ matrix, identity, rtol=0.0, atol=1.0e-8):
            raise ValueError(
                f"Irrep {name!r} matrix for {table.operation_names[index]!r} is not unitary."
            )
    for left in range(table.order):
        for right in range(table.order):
            product = int(table.multiplication[left, right])
            if not np.allclose(
                matrices[product], matrices[left] @ matrices[right], rtol=0.0, atol=1.0e-8
            ):
                raise ValueError(
                    f"Irrep {name!r} violates multiplication for {table.operation_names[left]!r} "
                    f"and {table.operation_names[right]!r}."
                )


def _validate_class_characters(irrep: GroupIrrep) -> None:
    for conjugacy_class in irrep.table.conjugacy_classes:
        values = [
            irrep.character_for_global_index(index)
            for index in conjugacy_class.operation_indices
        ]
        if max((abs(value - values[0]) for value in values), default=0.0) > 1.0e-8:
            raise ValueError(f"Irrep {irrep.name!r} characters are not constant on a conjugacy class.")


def _validated_matrix(value, dimension: int, description: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{description} must be a finite {dimension} x {dimension} matrix.")
    return matrix.copy()


def _require_exact_names(mapping: Mapping[str, object], names: tuple[str, ...], description: str) -> None:
    if set(mapping) != set(names):
        missing = sorted(set(names) - set(mapping))
        extra = sorted(set(mapping) - set(names))
        raise ValueError(f"{description} must cover every operation exactly once; missing={missing}, extra={extra}.")
