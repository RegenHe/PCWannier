from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Mapping, Protocol

import numpy as np

from .group import SpaceGroup
from .tables import ConcreteFiniteGroup, FiniteGroupTable


@dataclass(frozen=True)
class GroupIrrep:
    name: str
    dimension: int
    table: FiniteGroupTable
    matrices: tuple[np.ndarray, ...]
    characters: tuple[complex, ...]

    def matrix(self, element_index: int) -> np.ndarray:
        return self.matrices[int(element_index)]


@dataclass(frozen=True)
class FiniteGroupDefinition:
    name: str
    dimension: int
    table: FiniteGroupTable
    point_actions: tuple[np.ndarray, ...] | None
    irreps: tuple[GroupIrrep, ...]

    def irrep(self, name: str) -> GroupIrrep:
        for irrep in self.irreps:
            if irrep.name == name:
                return irrep
        available = ", ".join(irrep.name for irrep in self.irreps)
        raise KeyError(f"Finite group {self.name!r} has no irrep {name!r}; available={available}.")


@dataclass(frozen=True)
class FiniteGroupIdentification:
    concrete: ConcreteFiniteGroup
    canonical: FiniteGroupDefinition
    actual_to_canonical: tuple[int, ...]
    canonical_to_actual: tuple[int, ...]
    mapping_method: str
    candidate_count: int

    def canonical_name_for_operation(self, operation_index: int) -> str:
        actual = self.concrete.local_index(operation_index)
        return self.canonical.table.element_names[self.actual_to_canonical[actual]]

    def resolved_irrep(self, name: str) -> "ResolvedIrrep":
        return ResolvedIrrep(self.canonical.irrep(name), self)


@dataclass(frozen=True)
class ResolvedIrrep:
    canonical_irrep: GroupIrrep
    identification: FiniteGroupIdentification

    @property
    def name(self) -> str:
        return self.canonical_irrep.name

    @property
    def dimension(self) -> int:
        return self.canonical_irrep.dimension

    @property
    def characters(self) -> tuple[complex, ...]:
        return tuple(
            self.canonical_irrep.characters[canonical]
            for canonical in self.identification.actual_to_canonical
        )

    def matrix_for_global_index(self, operation_index: int) -> np.ndarray:
        actual = self.identification.concrete.local_index(operation_index)
        canonical = self.identification.actual_to_canonical[actual]
        return self.canonical_irrep.matrix(canonical)

    def character_for_global_index(self, operation_index: int) -> complex:
        actual = self.identification.concrete.local_index(operation_index)
        canonical = self.identification.actual_to_canonical[actual]
        return self.canonical_irrep.characters[canonical]


class FiniteGroupLibrary:
    def __init__(self, definitions):
        items = tuple(definitions)
        names = [definition.name for definition in items]
        if not items or len(names) != len(set(names)):
            raise ValueError("Finite-group library names must be unique and non-empty.")
        self.definitions = items

    def identify(self, concrete: ConcreteFiniteGroup) -> FiniteGroupIdentification:
        matches = []
        for definition in self.definitions:
            if definition.table.order != concrete.table.order:
                continue
            isomorphisms = _group_isomorphisms(concrete.table, definition.table)
            if not isomorphisms:
                continue
            exact = [
                mapping
                for mapping in isomorphisms
                if _point_actions_match(concrete, definition, mapping)
            ]
            geometric = [
                mapping
                for mapping in isomorphisms
                if _point_action_invariants_match(concrete, definition, mapping)
            ]
            if definition.point_actions is not None and not geometric:
                continue
            candidates = exact or geometric or isomorphisms
            method = (
                "point_action"
                if exact
                else "point_action_invariants"
                if geometric
                else "abstract_deterministic"
            )
            selected = min(candidates, key=lambda value: _mapping_sort_key(concrete, value))
            inverse = [0] * len(selected)
            for actual, canonical in enumerate(selected):
                inverse[canonical] = actual
            matches.append(
                FiniteGroupIdentification(
                    concrete,
                    definition,
                    tuple(selected),
                    tuple(inverse),
                    method,
                    len(isomorphisms),
                )
            )
        if not matches:
            signature = (
                concrete.table.order,
                tuple(sorted(concrete.table.element_orders)),
                tuple(sorted(len(value.element_indices) for value in concrete.table.conjugacy_classes)),
            )
            raise ValueError(f"No canonical finite group matches actual group signature {signature}.")
        exact = [match for match in matches if match.mapping_method == "point_action"]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            names = [match.canonical.name for match in exact]
            raise ValueError(f"Actual finite group geometrically matches multiple definitions: {names}.")
        geometric = [
            match for match in matches if match.mapping_method == "point_action_invariants"
        ]
        if len(geometric) == 1:
            return geometric[0]
        if len(matches) > 1:
            names = [match.canonical.name for match in matches]
            raise ValueError(f"Actual finite group abstractly matches multiple definitions: {names}.")
        return matches[0]


@dataclass(frozen=True)
class FactorSystem:
    lattice_shifts: np.ndarray
    phases: np.ndarray
    cocycle_residual: float
    trivializing_cochain: np.ndarray | None

    @property
    def is_trivial(self) -> bool:
        return self.trivializing_cochain is not None

    @property
    def phase_residual(self) -> float:
        return float(np.max(np.abs(self.phases - 1.0), initial=0.0))


class ProjectiveIrrepResolver(Protocol):
    def resolve(
        self,
        identification: FiniteGroupIdentification,
        factor_system: FactorSystem,
    ) -> tuple[ResolvedIrrep, ...]: ...


@dataclass(frozen=True)
class ResolvedLittleGroup:
    name: str
    concrete: ConcreteFiniteGroup
    identification: FiniteGroupIdentification
    factor_system: FactorSystem
    irreps: tuple[ResolvedIrrep, ...]

    @property
    def table(self) -> FiniteGroupTable:
        return self.concrete.table

    def require_irreps(self) -> tuple[ResolvedIrrep, ...]:
        if not self.factor_system.is_trivial:
            raise NotImplementedError(
                f"Little co-group {self.name!r} has a non-trivial factor system "
                f"(phase_residual={self.factor_system.phase_residual:.6g}, "
                f"cocycle_residual={self.factor_system.cocycle_residual:.6g}); "
                "projective irreps and small representations are not implemented."
            )
        return self.irreps


@dataclass(frozen=True)
class SpaceGroupDefinition:
    name: str
    dimension: int
    tolerance: float
    group: SpaceGroup
    finite_groups: FiniteGroupLibrary

    def __post_init__(self) -> None:
        if self.dimension != self.group.dimension:
            raise ValueError("Space-group definition dimension does not match its operations.")

    def identify_operations(self, operation_indices) -> FiniteGroupIdentification:
        concrete = ConcreteFiniteGroup.from_space_group(self.group, operation_indices)
        return self.finite_groups.identify(concrete)

    def site_irrep(self, operation_indices, name: str) -> ResolvedIrrep:
        return self.identify_operations(operation_indices).resolved_irrep(name)

    def resolve_little_group(
        self,
        operation_indices,
        k_fractional,
        *,
        projective_resolver: ProjectiveIrrepResolver | None = None,
    ) -> ResolvedLittleGroup:
        concrete = ConcreteFiniteGroup.from_space_group(self.group, operation_indices)
        identification = self.finite_groups.identify(concrete)
        factor = build_factor_system(concrete, k_fractional, self.tolerance)
        if factor.is_trivial:
            irreps = tuple(
                identification.resolved_irrep(irrep.name)
                for irrep in identification.canonical.irreps
            )
        elif projective_resolver is not None:
            irreps = tuple(projective_resolver.resolve(identification, factor))
        else:
            irreps = ()
        return ResolvedLittleGroup(
            identification.canonical.name,
            concrete,
            identification,
            factor,
            irreps,
        )


SymmetryGroupDefinition = SpaceGroupDefinition


def identify_finite_group(
    concrete_group: ConcreteFiniteGroup,
    library: FiniteGroupLibrary,
) -> FiniteGroupIdentification:
    return library.identify(concrete_group)


def resolve_little_group(
    definition: SpaceGroupDefinition,
    k_fractional,
    operation_indices=None,
) -> ResolvedLittleGroup:
    if operation_indices is None:
        kpoint = np.asarray(k_fractional, dtype=float)
        operation_indices = tuple(
            index
            for index, operation in enumerate(definition.group.operations)
            if np.allclose(
                operation.act_reciprocal(kpoint) - kpoint,
                np.rint(operation.act_reciprocal(kpoint) - kpoint),
                rtol=0.0,
                atol=definition.tolerance,
            )
        )
    return definition.resolve_little_group(operation_indices, k_fractional)


def build_factor_system(
    concrete: ConcreteFiniteGroup,
    k_fractional,
    tolerance: float,
) -> FactorSystem:
    kpoint = np.asarray(k_fractional, dtype=float)
    if kpoint.shape != (concrete.group.dimension,):
        raise ValueError(f"k_fractional must have shape {(concrete.group.dimension,)}.")
    shifts = np.empty(
        (concrete.order, concrete.order, concrete.group.dimension), dtype=np.int64
    )
    for left in range(concrete.order):
        for right in range(concrete.order):
            left_operation = concrete.group.operations[concrete.global_index(left)]
            right_operation = concrete.group.operations[concrete.global_index(right)]
            product_operation = left_operation * right_operation
            target = int(concrete.table.multiplication[left, right])
            representative = concrete.group.operations[concrete.global_index(target)]
            difference = product_operation.translation - representative.translation
            rounded = np.rint(difference).astype(np.int64)
            if not np.allclose(difference, rounded, rtol=0.0, atol=tolerance):
                raise ValueError("Seitz representatives differ by a non-lattice translation.")
            shifts[left, right] = rounded
    phases = np.exp(-2j * np.pi * np.einsum("abd,d->ab", shifts, kpoint))
    cocycle_residual = 0.0
    for left in range(concrete.order):
        for middle in range(concrete.order):
            for right in range(concrete.order):
                lm = int(concrete.table.multiplication[left, middle])
                mr = int(concrete.table.multiplication[middle, right])
                lhs = phases[left, middle] * phases[lm, right]
                rhs = phases[left, mr] * phases[middle, right]
                cocycle_residual = max(cocycle_residual, float(abs(lhs - rhs)))
    if cocycle_residual > max(100.0 * tolerance, 1.0e-8):
        raise ValueError(
            f"Computed factor system violates the cocycle condition (residual={cocycle_residual:.6g})."
        )
    cochain = _trivializing_cochain(concrete.table, phases, tolerance)
    shifts.setflags(write=False)
    phases.setflags(write=False)
    if cochain is not None:
        cochain.setflags(write=False)
    return FactorSystem(shifts, phases, cocycle_residual, cochain)


def build_group_irrep(
    table: FiniteGroupTable,
    name: str,
    dimension: int,
    *,
    characters: Mapping[str, complex],
    generators: Mapping[str, np.ndarray] | None = None,
    matrices: Mapping[str, np.ndarray] | None = None,
) -> GroupIrrep:
    if dimension <= 0:
        raise ValueError(f"Irrep {name!r} dimension must be positive.")
    _require_exact_names(characters, table.element_names, f"Irrep {name!r} characters")
    expected_characters = tuple(complex(characters[value]) for value in table.element_names)
    if dimension == 1 and generators is None and matrices is None:
        representation = tuple(
            np.asarray([[value]], dtype=np.complex128) for value in expected_characters
        )
    elif (generators is None) == (matrices is None):
        raise ValueError(
            f"Irrep {name!r} must define exactly one of generators or matrices for dimension {dimension}."
        )
    elif matrices is not None:
        _require_exact_names(matrices, table.element_names, f"Irrep {name!r} matrices")
        representation = tuple(
            _validated_matrix(matrices[element], dimension, f"Irrep {name!r} element {element!r}")
            for element in table.element_names
        )
    else:
        assert generators is not None
        representation = _generate_representation(table, name, dimension, generators)
    _validate_representation(table, name, dimension, representation)
    generated_characters = tuple(complex(np.trace(matrix)) for matrix in representation)
    if not np.allclose(generated_characters, expected_characters, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"Irrep {name!r} matrices do not reproduce its declared characters.")
    output = tuple(np.asarray(matrix, dtype=np.complex128).copy() for matrix in representation)
    for matrix in output:
        matrix.setflags(write=False)
    irrep = GroupIrrep(str(name), int(dimension), table, output, expected_characters)
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
            inner = np.vdot(left.characters, right.characters) / table.order
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


def _group_isomorphisms(actual: FiniteGroupTable, canonical: FiniteGroupTable):
    if actual.order != canonical.order:
        return ()
    identity_a = actual.identity_index
    identity_c = canonical.identity_index
    remaining_a = [index for index in range(actual.order) if index != identity_a]
    by_order = {}
    for index in range(canonical.order):
        if index != identity_c:
            by_order.setdefault(canonical.element_orders[index], []).append(index)
    choices = [by_order.get(actual.element_orders[index], ()) for index in remaining_a]
    if any(not value for value in choices):
        return ()
    output = []
    for values in permutations([index for group in by_order.values() for index in group]):
        mapping = [None] * actual.order
        mapping[identity_a] = identity_c
        valid_orders = True
        for actual_index, canonical_index in zip(remaining_a, values):
            if actual.element_orders[actual_index] != canonical.element_orders[canonical_index]:
                valid_orders = False
                break
            mapping[actual_index] = canonical_index
        if not valid_orders:
            continue
        if all(
            mapping[int(actual.multiplication[left, right])]
            == int(canonical.multiplication[mapping[left], mapping[right]])
            for left in range(actual.order)
            for right in range(actual.order)
        ):
            output.append(tuple(int(value) for value in mapping))
    return tuple(output)


def _point_actions_match(
    concrete: ConcreteFiniteGroup,
    definition: FiniteGroupDefinition,
    mapping,
) -> bool:
    if definition.point_actions is None:
        return False
    return all(
        np.array_equal(concrete.rotations[actual], definition.point_actions[canonical])
        for actual, canonical in enumerate(mapping)
    )


def _point_action_invariants_match(
    concrete: ConcreteFiniteGroup,
    definition: FiniteGroupDefinition,
    mapping,
) -> bool:
    if definition.point_actions is None:
        return False
    return all(
        _linear_action_signature(concrete.rotations[actual])
        == _linear_action_signature(definition.point_actions[canonical])
        for actual, canonical in enumerate(mapping)
    )


def _linear_action_signature(matrix: np.ndarray) -> tuple[int, ...]:
    """Conjugacy invariants distinguish rotations from differently embedded mirrors."""
    value = np.asarray(matrix, dtype=np.int64)
    powers = []
    current = np.eye(value.shape[0], dtype=np.int64)
    for _ in range(value.shape[0]):
        current = current @ value
        powers.append(int(np.trace(current)))
    return (int(round(np.linalg.det(value))), *powers)


def _mapping_sort_key(concrete: ConcreteFiniteGroup, mapping) -> tuple[int, ...]:
    inverse = [0] * len(mapping)
    for actual, canonical in enumerate(mapping):
        inverse[canonical] = actual
    values = []
    for actual in inverse:
        values.extend(int(value) for value in concrete.rotations[actual].reshape(-1))
    return tuple(values)


def _trivializing_cochain(
    table: FiniteGroupTable,
    phases: np.ndarray,
    tolerance: float,
) -> np.ndarray | None:
    if np.max(np.abs(phases - 1.0), initial=0.0) <= tolerance:
        return np.ones(table.order, dtype=np.complex128)
    generators = _minimal_generators(table)
    root_choices = []
    for generator in generators:
        order = table.element_orders[generator]
        current = table.identity_index
        accumulated = 1.0 + 0.0j
        for _ in range(order):
            accumulated *= phases[current, generator]
            current = int(table.multiplication[current, generator])
        base_angle = -np.angle(accumulated) / order
        root_choices.append(
            tuple(np.exp(1j * (base_angle + 2.0 * np.pi * branch / order)) for branch in range(order))
        )
    for selected in product(*root_choices):
        cochain = np.full(table.order, np.nan + 1j * np.nan, dtype=np.complex128)
        cochain[table.identity_index] = 1.0
        for generator, value in zip(generators, selected):
            cochain[generator] = value
        changed = True
        valid = True
        while changed and valid:
            changed = False
            known = np.flatnonzero(np.isfinite(cochain.real))
            for left in known:
                for generator in generators:
                    target = int(table.multiplication[left, generator])
                    candidate = cochain[left] * cochain[generator] * phases[left, generator]
                    if not np.isfinite(cochain[target].real):
                        cochain[target] = candidate / abs(candidate)
                        changed = True
                    elif abs(cochain[target] - candidate) > 100.0 * tolerance:
                        valid = False
                        break
        if not valid or not np.all(np.isfinite(cochain)):
            continue
        residual = max(
            abs(
                cochain[int(table.multiplication[left, right])]
                - cochain[left] * cochain[right] * phases[left, right]
            )
            for left in range(table.order)
            for right in range(table.order)
        )
        if residual <= max(100.0 * tolerance, 1.0e-8):
            return cochain
    return None


def _minimal_generators(table: FiniteGroupTable) -> tuple[int, ...]:
    generated = {table.identity_index}
    generators = []
    for candidate in range(table.order):
        if candidate in generated:
            continue
        generators.append(candidate)
        changed = True
        while changed:
            changed = False
            for left in tuple(generated):
                for generator in generators:
                    for right in (generator, int(table.inverse[generator])):
                        value = int(table.multiplication[left, right])
                        if value not in generated:
                            generated.add(value)
                            changed = True
        if len(generated) == table.order:
            break
    return tuple(generators)


def _generate_representation(
    table: FiniteGroupTable,
    name: str,
    dimension: int,
    generators: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, ...]:
    if not generators:
        raise ValueError(f"Irrep {name!r} generators must not be empty.")
    unknown = sorted(set(generators) - set(table.element_names))
    if unknown:
        raise ValueError(f"Irrep {name!r} references unknown generator elements {unknown}.")
    known: dict[int, np.ndarray] = {
        table.identity_index: np.eye(dimension, dtype=np.complex128)
    }
    generator_items = []
    for element_name, matrix in generators.items():
        element_index = table.element_index(element_name)
        value = _validated_matrix(matrix, dimension, f"Irrep {name!r} generator {element_name!r}")
        existing = known.get(element_index)
        if existing is not None and not np.allclose(existing, value, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"Irrep {name!r} supplies a conflicting identity/generator matrix.")
        known[element_index] = value
        generator_items.append((element_index, value))
    queue = list(known)
    while queue:
        current = queue.pop(0)
        for generator_index, generator_matrix in generator_items:
            result = int(table.multiplication[current, generator_index])
            result_matrix = known[current] @ generator_matrix
            existing = known.get(result)
            if existing is None:
                known[result] = result_matrix
                queue.append(result)
            elif not np.allclose(existing, result_matrix, rtol=0.0, atol=1.0e-8):
                raise ValueError(f"Irrep {name!r} generators violate a group relation.")
    if len(known) != table.order:
        missing = [table.element_names[index] for index in range(table.order) if index not in known]
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
            raise ValueError(f"Irrep {name!r} matrix for {table.element_names[index]!r} is not unitary.")
    for left in range(table.order):
        for right in range(table.order):
            result = int(table.multiplication[left, right])
            if not np.allclose(matrices[result], matrices[left] @ matrices[right], rtol=0.0, atol=1.0e-8):
                raise ValueError(
                    f"Irrep {name!r} violates multiplication for {table.element_names[left]!r} "
                    f"and {table.element_names[right]!r}."
                )


def _validate_class_characters(irrep: GroupIrrep) -> None:
    for conjugacy_class in irrep.table.conjugacy_classes:
        values = [irrep.characters[index] for index in conjugacy_class.element_indices]
        if max((abs(value - values[0]) for value in values), default=0.0) > 1.0e-8:
            raise ValueError(f"Irrep {irrep.name!r} characters are not constant on a conjugacy class.")


def _validated_matrix(value, dimension: int, description: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{description} must be a finite {dimension} x {dimension} matrix.")
    return matrix.copy()


def _require_exact_names(mapping: Mapping[str, object], names, description: str) -> None:
    if set(mapping) != set(names):
        missing = sorted(set(names) - set(mapping))
        extra = sorted(set(mapping) - set(names))
        raise ValueError(f"{description} must cover every element exactly once; missing={missing}, extra={extra}.")
