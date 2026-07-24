from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Mapping

import numpy as np

from .group import SpaceGroup, little_group, reduce_fractional
from ..conventions import BlochConvention
from .crystallography import (
    PointGroupIdentification,
    enumerate_little_group_representations,
    identify_point_group,
    validate_catalog_characters,
)
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
    point_group_symbol: str
    table: FiniteGroupTable
    point_actions: tuple[np.ndarray, ...]
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
    point_group_symbol: str
    point_group_number: int

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


def _extend_one_dimensional_magnetic_irrep(
    full_identification: FiniteGroupIdentification,
    unitary_irrep: ResolvedIrrep,
    tolerance: float,
) -> ResolvedIrrep:
    """Extend a unitary-subgroup irrep to an index-two magnetic corepresentation."""

    if unitary_irrep.dimension != 1:
        raise NotImplementedError(
            "Automatic magnetic site-corepresentation extension currently supports "
            "one-dimensional unitary-subgroup irreps only."
        )
    concrete = full_identification.concrete
    flags = concrete.antiunitary_flags
    antiunitary = [index for index, flag in enumerate(flags) if flag]
    if not antiunitary or len(antiunitary) * 2 != concrete.order:
        raise ValueError("Magnetic site group must have an index-two unitary subgroup.")

    unitary_matrices = {
        operation_index: unitary_irrep.matrix_for_global_index(operation_index)
        for operation_index in unitary_irrep.identification.concrete.operation_indices
    }
    representative = antiunitary[0]
    square = int(concrete.table.multiplication[representative, representative])
    square_matrix = unitary_matrices[concrete.global_index(square)]
    if not np.allclose(square_matrix, np.eye(1), rtol=0.0, atol=tolerance):
        raise NotImplementedError(
            f"Magnetic corepresentation {unitary_irrep.name!r} has a nontrivial "
            "antiunitary square and requires a higher-dimensional extension."
        )

    representative_inverse = int(concrete.table.inverse[representative])
    actual_matrices: list[np.ndarray] = []
    for actual_index, operation_index in enumerate(concrete.operation_indices):
        if not flags[actual_index]:
            matrix = unitary_matrices[operation_index]
        else:
            unitary_index = int(
                concrete.table.multiplication[actual_index, representative_inverse]
            )
            if flags[unitary_index]:
                raise RuntimeError("Magnetic coset decomposition produced an antiunitary element.")
            matrix = unitary_matrices[concrete.global_index(unitary_index)]
        stored = np.asarray(matrix, dtype=np.complex128).copy()
        stored.setflags(write=False)
        actual_matrices.append(stored)

    for left in range(concrete.order):
        for right in range(concrete.order):
            product_index = int(concrete.table.multiplication[left, right])
            right_matrix = actual_matrices[right].conj() if flags[left] else actual_matrices[right]
            product = actual_matrices[left] @ right_matrix
            if not np.allclose(
                product,
                actual_matrices[product_index],
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError(
                    f"Unitary-subgroup irrep {unitary_irrep.name!r} cannot be extended "
                    "to the requested magnetic site group."
                )

    canonical_matrices: list[np.ndarray | None] = [None] * concrete.order
    for actual_index, canonical_index in enumerate(full_identification.actual_to_canonical):
        canonical_matrices[canonical_index] = actual_matrices[actual_index]
    if any(matrix is None for matrix in canonical_matrices):
        raise RuntimeError("Magnetic corepresentation mapping is incomplete.")
    matrices = tuple(matrix for matrix in canonical_matrices if matrix is not None)
    generated = GroupIrrep(
        unitary_irrep.name,
        1,
        full_identification.canonical.table,
        matrices,
        tuple(complex(np.trace(matrix)) for matrix in matrices),
    )
    return ResolvedIrrep(generated, full_identification)


class FiniteGroupLibrary:
    def __init__(self, definitions):
        items = tuple(definitions)
        names = [definition.name for definition in items]
        if not items or len(names) != len(set(names)):
            raise ValueError("Finite-group library names must be unique and non-empty.")
        symbols = [definition.point_group_symbol for definition in items]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Finite-group point_group_symbol values must be unique.")
        self.definitions = items

    def identify(self, concrete: ConcreteFiniteGroup) -> FiniteGroupIdentification:
        point_group = identify_point_group(
            concrete.rotations,
            concrete.group.dimension,
        )
        definitions = tuple(
            definition
            for definition in self.definitions
            if definition.point_group_symbol == point_group.symbol
        )
        if not definitions:
            raise ValueError(
                f"spglib identified point group {point_group.symbol!r}, but no matching "
                "finite-group label catalog is installed."
            )
        matches = []
        for definition in definitions:
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
            if not geometric:
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
                    point_group.symbol,
                    point_group.number,
                )
            )
        if not matches:
            signature = (
                concrete.table.order,
                tuple(sorted(concrete.table.element_orders)),
                tuple(sorted(len(value.element_indices) for value in concrete.table.conjugacy_classes)),
            )
            raise ValueError(
                f"No canonical finite group matches spglib point group "
                f"{point_group.symbol!r} with actual signature {signature}."
            )
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
    bloch_sign: int = 1
    tolerance: float = 1.0e-8
    antiunitary_flags: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Factor-system tolerance must be finite and positive.")
        raw_shifts = np.asarray(self.lattice_shifts)
        if raw_shifts.ndim != 3 or raw_shifts.shape[0] != raw_shifts.shape[1]:
            raise ValueError("Factor-system lattice shifts must have shape (order, order, dimension).")
        shifts = np.rint(raw_shifts).astype(np.int64)
        if not np.allclose(raw_shifts, shifts, rtol=0.0, atol=tolerance):
            raise ValueError("Factor-system lattice shifts must contain integers.")
        phases = np.asarray(self.phases, dtype=np.complex128)
        if phases.shape != shifts.shape[:2] or not np.all(np.isfinite(phases)):
            raise ValueError("Factor-system phases must be a finite order x order matrix.")
        if not np.allclose(np.abs(phases), 1.0, rtol=0.0, atol=max(tolerance, 1.0e-12)):
            raise ValueError("Factor-system phases must have unit modulus.")
        if self.bloch_sign not in {-1, 1}:
            raise ValueError("Factor-system Bloch sign must be +1 or -1.")
        flags = self.antiunitary_flags or (False,) * phases.shape[0]
        flags = tuple(bool(value) for value in flags)
        if len(flags) != phases.shape[0]:
            raise ValueError("Factor-system antiunitary flags must match the group order.")
        if not np.isfinite(self.cocycle_residual) or self.cocycle_residual < 0.0:
            raise ValueError("Factor-system cocycle residual must be finite and non-negative.")
        cochain = self.trivializing_cochain
        if cochain is not None:
            cochain = np.asarray(cochain, dtype=np.complex128)
            if cochain.shape != (phases.shape[0],) or not np.all(np.isfinite(cochain)):
                raise ValueError("Factor-system trivializing cochain has an invalid shape or value.")
            if not np.allclose(
                np.abs(cochain), 1.0, rtol=0.0, atol=max(tolerance, 1.0e-12)
            ):
                raise ValueError("Factor-system trivializing cochain must have unit modulus.")
            cochain = cochain.copy()
            cochain.setflags(write=False)
        shifts = shifts.copy()
        phases = phases.copy()
        shifts.setflags(write=False)
        phases.setflags(write=False)
        object.__setattr__(self, "lattice_shifts", shifts)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "trivializing_cochain", cochain)
        object.__setattr__(self, "cocycle_residual", float(self.cocycle_residual))
        object.__setattr__(self, "bloch_sign", int(self.bloch_sign))
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "antiunitary_flags", flags)

    @property
    def raw_trivial(self) -> bool:
        return self.phase_residual <= self.tolerance

    @property
    def cohomologically_trivial(self) -> bool:
        return self.trivializing_cochain is not None

    @property
    def phase_residual(self) -> float:
        return float(np.max(np.abs(self.phases - 1.0), initial=0.0))

    def cocycle_residual_for(self, product_table) -> float:
        product = np.asarray(product_table, dtype=np.int64)
        order = self.phases.shape[0]
        if product.shape != (order, order):
            raise ValueError("Factor-system product table has an incompatible shape.")
        if np.any(product < 0) or np.any(product >= order):
            raise ValueError("Factor-system product table contains an invalid element index.")
        residual = 0.0
        for left in range(order):
            for middle in range(order):
                for right in range(order):
                    lm = int(product[left, middle])
                    mr = int(product[middle, right])
                    lhs = self.phases[left, middle] * self.phases[lm, right]
                    right_phase = self.phases[middle, right]
                    if self.antiunitary_flags[left]:
                        right_phase = np.conj(right_phase)
                    rhs = self.phases[left, mr] * right_phase
                    residual = max(residual, float(abs(lhs - rhs)))
        return residual

    def assert_compatible(self, other: "FactorSystem", *, tolerance: float | None = None) -> None:
        if not isinstance(other, FactorSystem):
            raise TypeError("Expected another FactorSystem.")
        requested = 0.0 if tolerance is None else float(tolerance)
        if not np.isfinite(requested) or requested < 0.0:
            raise ValueError("Factor-system comparison tolerance must be finite and non-negative.")
        threshold = max(
            self.tolerance,
            other.tolerance,
            requested,
        )
        if self.bloch_sign != other.bloch_sign:
            raise ValueError("Physical and target representations use different Bloch signs.")
        if self.antiunitary_flags != other.antiunitary_flags:
            raise ValueError("Physical and target representations use different antiunitary flags.")
        if self.lattice_shifts.shape != other.lattice_shifts.shape or not np.array_equal(
            self.lattice_shifts, other.lattice_shifts
        ):
            raise ValueError("Physical and target representations use different Seitz lattice shifts.")
        if not np.allclose(self.phases, other.phases, rtol=0.0, atol=threshold):
            raise ValueError("Physical and target representations use different factor-system phases.")


@dataclass(frozen=True)
class ResolvedSmallRepresentation:
    name: str
    operation_indices: tuple[int, ...]
    matrices: tuple[np.ndarray, ...]
    factor_system: FactorSystem
    label_source: str

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Small-representation name must not be empty.")
        indices = tuple(int(value) for value in self.operation_indices)
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("Small-representation operation indices must be unique.")
        matrices = tuple(np.asarray(value, dtype=np.complex128) for value in self.matrices)
        if len(matrices) != len(indices):
            raise ValueError("Small-representation matrices must match operation indices.")
        dimension = matrices[0].shape[0]
        if dimension <= 0 or any(value.shape != (dimension, dimension) for value in matrices):
            raise ValueError("Small-representation matrices must be non-empty and square.")
        if any(not np.all(np.isfinite(value)) for value in matrices):
            raise ValueError("Small-representation matrices must be finite.")
        stored = []
        for matrix in matrices:
            value = matrix.copy()
            value.setflags(write=False)
            stored.append(value)
        source = str(self.label_source).strip()
        if source not in {"catalog", "projective"}:
            raise ValueError("Small-representation label_source must be catalog or projective.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "operation_indices", indices)
        object.__setattr__(self, "matrices", tuple(stored))
        object.__setattr__(self, "label_source", source)

    @property
    def dimension(self) -> int:
        return int(self.matrices[0].shape[0])

    @property
    def characters(self) -> tuple[complex, ...]:
        return tuple(complex(np.trace(matrix)) for matrix in self.matrices)

    def matrix_for_global_index(self, operation_index: int) -> np.ndarray:
        try:
            local = self.operation_indices.index(int(operation_index))
        except ValueError as exc:
            raise ValueError("Operation does not belong to this small representation.") from exc
        return self.matrices[local]


@dataclass(frozen=True)
class ResolvedLittleGroup:
    name: str
    concrete: ConcreteFiniteGroup
    identification: FiniteGroupIdentification
    factor_system: FactorSystem
    irreps: tuple[ResolvedSmallRepresentation, ...]
    k_fractional: np.ndarray | None = None
    reciprocal_lattice_shifts: tuple[tuple[int, ...], ...] = ()

    @property
    def table(self) -> FiniteGroupTable:
        return self.concrete.table

    def require_irreps(self) -> tuple[ResolvedSmallRepresentation, ...]:
        if any(self.factor_system.antiunitary_flags):
            raise NotImplementedError(
                f"Little group {self.name!r} contains antiunitary operations; ordinary character "
                "tables do not describe magnetic corepresentations."
            )
        if not self.irreps:
            raise RuntimeError(f"Little group {self.name!r} has no enumerated small representations.")
        return self.irreps


@dataclass(frozen=True)
class SpaceGroupDefinition:
    name: str
    dimension: int
    tolerance: float
    group: SpaceGroup
    finite_groups: FiniteGroupLibrary
    algebra_tolerance: float = 1.0e-10
    point_group: PointGroupIdentification = field(init=False)
    _identification_cache: dict[tuple[int, ...], FiniteGroupIdentification] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _little_group_cache: dict[tuple[tuple[int, ...], bytes, int], ResolvedLittleGroup] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.dimension != self.group.dimension:
            raise ValueError("Space-group definition dimension does not match its operations.")
        if not np.isfinite(self.algebra_tolerance) or self.algebra_tolerance <= 0.0:
            raise ValueError("Symmetry algebra tolerance must be positive and finite.")
        object.__setattr__(
            self,
            "point_group",
            identify_point_group(
                (operation.rotation for operation in self.group.operations),
                self.dimension,
            ),
        )

    def identify_operations(self, operation_indices) -> FiniteGroupIdentification:
        indices = tuple(int(value) for value in operation_indices)
        cached = self._identification_cache.get(indices)
        if cached is not None:
            return cached
        concrete = ConcreteFiniteGroup.from_space_group(self.group, indices)
        result = self.finite_groups.identify(concrete)
        self._identification_cache[indices] = result
        return result

    def site_irrep(self, operation_indices, name: str) -> ResolvedIrrep:
        indices = tuple(int(value) for value in operation_indices)
        identification = self.identify_operations(indices)
        try:
            return identification.resolved_irrep(name)
        except KeyError as full_group_error:
            unitary_indices = tuple(
                operation_index
                for operation_index in indices
                if not self.group.operations[operation_index].antiunitary
            )
            if len(unitary_indices) == len(indices):
                raise
            try:
                unitary_irrep = self.identify_operations(unitary_indices).resolved_irrep(name)
            except KeyError:
                raise full_group_error
            return _extend_one_dimensional_magnetic_irrep(
                identification,
                unitary_irrep,
                self.algebra_tolerance,
            )

    def resolve_little_group(
        self,
        operation_indices,
        k_fractional,
        *,
        bloch_convention: BlochConvention | None = None,
    ) -> ResolvedLittleGroup:
        convention = BlochConvention() if bloch_convention is None else bloch_convention
        indices = tuple(int(value) for value in operation_indices)
        kpoint = np.asarray(k_fractional, dtype=float)
        if kpoint.shape != (self.dimension,):
            raise ValueError(f"k_fractional must have shape {(self.dimension,)}.")
        reduced_k = reduce_fractional(kpoint, self.tolerance).reduced
        cache_key = (indices, np.ascontiguousarray(reduced_k).tobytes(), convention.sign)
        cached = self._little_group_cache.get(cache_key)
        if cached is not None:
            return cached
        identification = self.identify_operations(indices)
        concrete = identification.concrete
        reciprocal_shifts = []
        for operation_index in indices:
            displacement = self.group.operations[operation_index].act_reciprocal(reduced_k) - reduced_k
            rounded = np.rint(displacement).astype(np.int64)
            if not np.allclose(displacement, rounded, rtol=0.0, atol=self.tolerance):
                operation = self.group.operations[operation_index]
                raise ValueError(
                    f"Operation {operation.name or operation_index!r} is not in the little group "
                    f"at k={reduced_k.tolist()}."
                )
            reciprocal_shifts.append(tuple(int(value) for value in rounded))
        factor = build_factor_system(
            concrete,
            reduced_k,
            self.tolerance,
            algebra_tolerance=self.algebra_tolerance,
            bloch_convention=convention,
        )
        irreps = (
            ()
            if any(factor.antiunitary_flags)
            else _enumerate_resolved_small_representations(
                concrete,
                identification,
                factor,
                self.algebra_tolerance,
            )
        )
        stored_k = reduced_k.copy()
        stored_k.setflags(write=False)
        result = ResolvedLittleGroup(
            identification.canonical.name,
            concrete,
            identification,
            factor,
            irreps,
            stored_k,
            tuple(reciprocal_shifts),
        )
        self._little_group_cache[cache_key] = result
        return result


def identify_finite_group(
    concrete_group: ConcreteFiniteGroup,
    library: FiniteGroupLibrary,
) -> FiniteGroupIdentification:
    return library.identify(concrete_group)


def resolve_little_group(
    definition: SpaceGroupDefinition,
    k_fractional,
    operation_indices=None,
    *,
    bloch_convention: BlochConvention | None = None,
) -> ResolvedLittleGroup:
    if operation_indices is None:
        operation_indices = tuple(
            element.operation_index
            for element in little_group(definition.group, k_fractional)
        )
    return definition.resolve_little_group(
        operation_indices,
        k_fractional,
        bloch_convention=bloch_convention,
    )


def build_factor_system(
    concrete: ConcreteFiniteGroup,
    k_fractional,
    coordinate_tolerance: float,
    *,
    algebra_tolerance: float = 1.0e-10,
    bloch_convention: BlochConvention | None = None,
) -> FactorSystem:
    convention = BlochConvention() if bloch_convention is None else bloch_convention
    if not np.isfinite(coordinate_tolerance) or coordinate_tolerance <= 0.0:
        raise ValueError("Coordinate tolerance must be positive and finite.")
    if not np.isfinite(algebra_tolerance) or algebra_tolerance <= 0.0:
        raise ValueError("Algebra tolerance must be positive and finite.")
    kpoint = np.asarray(k_fractional, dtype=float)
    if kpoint.shape != (concrete.group.dimension,):
        raise ValueError(f"k_fractional must have shape {(concrete.group.dimension,)}.")
    shifts = concrete.lattice_shifts
    phases = np.exp(
        -convention.sign
        * 2j
        * np.pi
        * np.einsum("abd,d->ab", shifts, kpoint)
    )
    provisional = FactorSystem(
        shifts,
        phases,
        0.0,
        None,
        convention.sign,
        algebra_tolerance,
        concrete.antiunitary_flags,
    )
    cocycle_residual = provisional.cocycle_residual_for(concrete.table.multiplication)
    if cocycle_residual > max(100.0 * algebra_tolerance, 1.0e-10):
        raise ValueError(
            f"Computed factor system violates the cocycle condition (residual={cocycle_residual:.6g})."
        )
    cochain = _trivializing_cochain(
        concrete.table,
        phases,
        algebra_tolerance,
        concrete.antiunitary_flags,
    )
    return FactorSystem(
        shifts,
        phases,
        cocycle_residual,
        cochain,
        convention.sign,
        algebra_tolerance,
        concrete.antiunitary_flags,
    )


def _enumerate_resolved_small_representations(
    concrete: ConcreteFiniteGroup,
    identification: FiniteGroupIdentification,
    factor_system: FactorSystem,
    tolerance: float,
) -> tuple[ResolvedSmallRepresentation, ...]:
    generated = enumerate_little_group_representations(
        concrete,
        factor_system,
        tolerance,
    )
    cochain = factor_system.trivializing_cochain
    output = []
    matched_names: set[str] = set()
    for generated_index, matrices in enumerate(generated, start=1):
        characters = np.asarray(
            [np.trace(matrix) for matrix in matrices],
            dtype=np.complex128,
        )
        if cochain is None:
            name = f"P{generated_index}"
            label_source = "projective"
        else:
            ordinary_characters = cochain * characters
            candidates = []
            for catalog_irrep in identification.canonical.irreps:
                expected = np.asarray(
                    [
                        catalog_irrep.characters[canonical]
                        for canonical in identification.actual_to_canonical
                    ],
                    dtype=np.complex128,
                )
                if np.allclose(
                    ordinary_characters,
                    expected,
                    rtol=0.0,
                    atol=max(10.0 * tolerance, 1.0e-7),
                ):
                    candidates.append(catalog_irrep.name)
            if len(candidates) != 1:
                raise ValueError(
                    "Could not uniquely match a spgrep small representation to the "
                    f"{identification.canonical.name} label catalog; matches={candidates}."
                )
            name = candidates[0]
            if name in matched_names:
                raise ValueError(f"spgrep generated duplicate ordinary irrep label {name!r}.")
            matched_names.add(name)
            label_source = "catalog"
        output.append(
            ResolvedSmallRepresentation(
                name,
                concrete.operation_indices,
                matrices,
                factor_system,
                label_source,
            )
        )
    if cochain is not None:
        expected_names = {irrep.name for irrep in identification.canonical.irreps}
        if matched_names != expected_names:
            missing = sorted(expected_names - matched_names)
            raise ValueError(f"spgrep did not generate catalog irreps {missing}.")
        output.sort(
            key=lambda value: next(
                index
                for index, irrep in enumerate(identification.canonical.irreps)
                if irrep.name == value.name
            )
        )
    return tuple(output)


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


def validate_irrep_table(
    table: FiniteGroupTable,
    irreps: tuple[GroupIrrep, ...],
    point_actions: tuple[np.ndarray, ...],
    *,
    dimension: int,
) -> None:
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
    validate_catalog_characters(
        point_actions,
        table,
        (irrep.characters for irrep in irreps),
        dimension=dimension,
    )


def _group_isomorphisms(actual: FiniteGroupTable, canonical: FiniteGroupTable):
    if actual.order != canonical.order:
        return ()
    identity_a = actual.identity_index
    identity_c = canonical.identity_index
    actual_class_sizes = _element_conjugacy_class_sizes(actual)
    canonical_class_sizes = _element_conjugacy_class_sizes(canonical)
    actual_buckets: dict[tuple[int, int], list[int]] = {}
    canonical_buckets: dict[tuple[int, int], list[int]] = {}
    for index in range(actual.order):
        if index != identity_a:
            key = (actual.element_orders[index], actual_class_sizes[index])
            actual_buckets.setdefault(key, []).append(index)
    for index in range(canonical.order):
        if index != identity_c:
            key = (canonical.element_orders[index], canonical_class_sizes[index])
            canonical_buckets.setdefault(key, []).append(index)
    if set(actual_buckets) != set(canonical_buckets) or any(
        len(actual_buckets[key]) != len(canonical_buckets[key]) for key in actual_buckets
    ):
        return ()
    candidates = {
        actual_index: tuple(canonical_buckets[key])
        for key, actual_indices in actual_buckets.items()
        for actual_index in actual_indices
    }
    assignment_order = tuple(
        sorted(
            candidates,
            key=lambda index: (
                len(candidates[index]),
                actual.element_orders[index],
                actual_class_sizes[index],
                index,
            ),
        )
    )
    mapping = [-1] * actual.order
    mapping[identity_a] = identity_c
    used = {identity_c}
    output: list[tuple[int, ...]] = []

    def compatible() -> bool:
        assigned = tuple(index for index, value in enumerate(mapping) if value >= 0)
        for left in assigned:
            canonical_left = mapping[left]
            for right in assigned:
                result = int(actual.multiplication[left, right])
                if mapping[result] < 0:
                    continue
                expected = int(
                    canonical.multiplication[canonical_left, mapping[right]]
                )
                if mapping[result] != expected:
                    return False
        return True

    def search(position: int) -> None:
        if position == len(assignment_order):
            output.append(tuple(mapping))
            return
        actual_index = assignment_order[position]
        for canonical_index in candidates[actual_index]:
            if canonical_index in used:
                continue
            mapping[actual_index] = canonical_index
            used.add(canonical_index)
            if compatible():
                search(position + 1)
            used.remove(canonical_index)
            mapping[actual_index] = -1

    search(0)
    return tuple(output)


def _element_conjugacy_class_sizes(table: FiniteGroupTable) -> tuple[int, ...]:
    sizes = [0] * table.order
    for conjugacy_class in table.conjugacy_classes:
        size = len(conjugacy_class.element_indices)
        for element in conjugacy_class.element_indices:
            sizes[element] = size
    return tuple(sizes)


def _point_actions_match(
    concrete: ConcreteFiniteGroup,
    definition: FiniteGroupDefinition,
    mapping,
) -> bool:
    return all(
        np.array_equal(concrete.rotations[actual], definition.point_actions[canonical])
        for actual, canonical in enumerate(mapping)
    )


def _point_action_invariants_match(
    concrete: ConcreteFiniteGroup,
    definition: FiniteGroupDefinition,
    mapping,
) -> bool:
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
    antiunitary_flags: tuple[bool, ...] = (),
) -> np.ndarray | None:
    flags = antiunitary_flags or (False,) * table.order
    if len(flags) != table.order:
        raise ValueError("Antiunitary flags must match the finite-group order.")
    if np.max(np.abs(phases - 1.0), initial=0.0) <= tolerance:
        return np.ones(table.order, dtype=np.complex128)
    generators = _minimal_generators(table)
    root_choices = []
    for generator in generators:
        order = table.element_orders[generator]
        current = table.identity_index
        accumulated = 1.0 + 0.0j
        exponent = 0
        for _ in range(order):
            accumulated *= phases[current, generator]
            exponent += -1 if flags[current] else 1
            current = int(table.multiplication[current, generator])
        if exponent == 0:
            if abs(accumulated - 1.0) > max(100.0 * tolerance, 1.0e-8):
                return None
            root_choices.append((1.0 + 0.0j,))
        else:
            count = abs(exponent)
            base_angle = -np.angle(accumulated) / exponent
            root_choices.append(
                tuple(
                    np.exp(1j * (base_angle + 2.0 * np.pi * branch / count))
                    for branch in range(count)
                )
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
                    right_value = (
                        np.conj(cochain[generator]) if flags[left] else cochain[generator]
                    )
                    candidate = cochain[left] * right_value * phases[left, generator]
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
                - cochain[left]
                * (np.conj(cochain[right]) if flags[left] else cochain[right])
                * phases[left, right]
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
