from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .definition import SymmetryGroupDefinition
    from .specs import RepresentationAnalysisSpec, SymmetryGaugeSpec
    from .tables import GroupIrrep

from .group import (
    CrystallographicOrbit,
    SpaceGroup,
    SpaceGroupOperation,
    SymmetryKMapping,
    build_crystallographic_orbit,
    build_k_mappings,
)


@dataclass(frozen=True)
class SiteIrrepGenerator:
    operation: SpaceGroupOperation
    matrix: np.ndarray
    match_modulo_lattice: bool = False


@dataclass(frozen=True)
class SiteIrrepSpec:
    name: str
    dimension: int
    identity_matrix: np.ndarray
    generators: tuple[SiteIrrepGenerator, ...]


@dataclass(frozen=True)
class SiteIrrep:
    name: str
    dimension: int
    matrices: tuple[np.ndarray, ...]

    def matrix(self, site_element_index: int) -> np.ndarray:
        return self.matrices[site_element_index]


def build_site_irrep(orbit: CrystallographicOrbit, spec: SiteIrrepSpec) -> SiteIrrep:
    dimension = int(spec.dimension)
    if dimension <= 0:
        raise ValueError("Site-irrep dimension must be positive.")
    identity_matrix = _validated_representation_matrix(spec.identity_matrix, dimension, "identity")
    if not np.allclose(identity_matrix, np.eye(dimension), rtol=0.0, atol=orbit.site_symmetry.tolerance):
        raise ValueError("The site-irrep identity matrix must equal the identity.")

    identity_operation = SpaceGroupOperation.identity(orbit.center.size)
    identity_index = orbit.site_symmetry.element_index(identity_operation)
    known: dict[int, np.ndarray] = {identity_index: identity_matrix}
    generators = []
    for generator_index, generator in enumerate(spec.generators):
        if generator.match_modulo_lattice:
            site_index = next(
                (
                    index
                    for index, element in enumerate(orbit.site_symmetry.elements)
                    if element.operation.equivalent_mod_lattice(
                        generator.operation, orbit.site_symmetry.tolerance
                    )
                ),
                None,
            )
            if site_index is None:
                raise ValueError("Named generator operation is not in the target site-symmetry group.")
        else:
            site_index = orbit.site_symmetry.element_index(generator.operation)
        matrix = _validated_representation_matrix(generator.matrix, dimension, f"generator {generator_index}")
        existing = known.get(site_index)
        if existing is not None and not np.allclose(
            existing, matrix, rtol=0.0, atol=orbit.site_symmetry.tolerance
        ):
            raise ValueError(f"Conflicting matrices were supplied for site element {site_index}.")
        known[site_index] = matrix
        generators.append((site_index, orbit.site_symmetry.elements[site_index].operation, matrix))

    queue = list(known)
    while queue:
        current_index = queue.pop(0)
        current_operation = orbit.site_symmetry.elements[current_index].operation
        current_matrix = known[current_index]
        for _, generator_operation, generator_matrix in generators:
            product_operation = current_operation * generator_operation
            product_index = orbit.site_symmetry.element_index(product_operation)
            product_matrix = current_matrix @ generator_matrix
            existing = known.get(product_index)
            if existing is None:
                known[product_index] = product_matrix
                queue.append(product_index)
            elif not np.allclose(
                existing,
                product_matrix,
                rtol=0.0,
                atol=orbit.site_symmetry.tolerance,
            ):
                raise ValueError("Site-irrep generators violate a group relation.")
    if len(known) != len(orbit.site_symmetry.elements):
        missing = sorted(set(range(len(orbit.site_symmetry.elements))) - set(known))
        raise ValueError(f"Site-irrep generators do not generate the full site group; missing elements {missing}.")

    matrices = tuple(np.asarray(known[index], dtype=np.complex128) for index in range(len(known)))
    _validate_site_representation(orbit, matrices, dimension)
    for matrix in matrices:
        matrix.setflags(write=False)
    return SiteIrrep(spec.name, dimension, matrices)


def _validated_representation_matrix(matrix, dimension: int, description: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.shape != (dimension, dimension):
        raise ValueError(f"Site-irrep {description} matrix has shape {array.shape}; expected {(dimension, dimension)}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Site-irrep {description} matrix contains non-finite values.")
    residual = float(np.linalg.norm(array.conj().T @ array - np.eye(dimension), ord="fro"))
    if residual > 1e-8:
        raise ValueError(f"Site-irrep {description} matrix is not unitary (residual={residual:.6g}).")
    return array.copy()


def _validate_site_representation(
    orbit: CrystallographicOrbit,
    matrices: tuple[np.ndarray, ...],
    dimension: int,
) -> None:
    identity = np.eye(dimension)
    for left_index, left in enumerate(orbit.site_symmetry.elements):
        for right_index, right in enumerate(orbit.site_symmetry.elements):
            product_index = orbit.site_symmetry.element_index(left.operation * right.operation)
            expected = matrices[left_index] @ matrices[right_index]
            if not np.allclose(
                matrices[product_index],
                expected,
                rtol=0.0,
                atol=orbit.site_symmetry.tolerance,
            ):
                raise ValueError(
                    f"Site-irrep multiplication failed for elements {left_index} and {right_index}."
                )
    for index, matrix in enumerate(matrices):
        if not np.allclose(matrix.conj().T @ matrix, identity, rtol=0.0, atol=1e-8):
            raise ValueError(f"Site-irrep matrix {index} is not unitary.")


@dataclass(frozen=True)
class WannierTargetRepresentation:
    name: str
    group: SpaceGroup
    orbit: CrystallographicOrbit
    site_irrep: SiteIrrep

    @property
    def multiplicity(self) -> int:
        return self.orbit.multiplicity

    @property
    def wannier_dimension(self) -> int:
        return self.multiplicity * self.site_irrep.dimension

    def wannier_index(self, irrep_index: int, orbit_index: int) -> int:
        if not 0 <= irrep_index < self.site_irrep.dimension:
            raise IndexError("site-irrep index is out of range.")
        if not 0 <= orbit_index < self.multiplicity:
            raise IndexError("orbit index is out of range.")
        return orbit_index * self.site_irrep.dimension + irrep_index

    def matrix(self, operation: int | SpaceGroupOperation, k_fractional) -> np.ndarray:
        operation_index = (
            int(operation) if isinstance(operation, (int, np.integer)) else self.group.operation_index(operation)
        )
        if not 0 <= operation_index < len(self.group.operations):
            raise IndexError("Space-group operation index is out of range.")
        kpoint = np.asarray(k_fractional, dtype=float)
        if kpoint.shape != (self.group.dimension,) or not np.all(np.isfinite(kpoint)):
            raise ValueError(f"k_fractional must have shape {(self.group.dimension,)} and be finite.")
        transformed_k = self.group.operations[operation_index].act_reciprocal(kpoint)
        dimension = self.site_irrep.dimension
        output = np.zeros((self.wannier_dimension, self.wannier_dimension), dtype=np.complex128)
        for orbit_index in range(self.multiplicity):
            action = self.orbit.action(operation_index, orbit_index)
            lattice_shift = np.asarray(action.lattice_shift, dtype=float)
            phase = np.exp(-2j * np.pi * np.dot(transformed_k, lattice_shift))
            row = slice(action.target_index * dimension, (action.target_index + 1) * dimension)
            column = slice(orbit_index * dimension, (orbit_index + 1) * dimension)
            output[row, column] = phase * self.site_irrep.matrix(action.site_element_index)
        residual = float(np.linalg.norm(output.conj().T @ output - np.eye(self.wannier_dimension), ord="fro"))
        if residual > 1e-8:
            raise FloatingPointError(f"Target Wannier representation is not unitary (residual={residual:.6g}).")
        return output


def combined_target_matrix(
    targets,
    operation: int | SpaceGroupOperation,
    k_fractional,
) -> np.ndarray:
    """Return the block-diagonal target representation in YAML target order."""
    items = tuple(targets)
    if not items:
        return np.empty((0, 0), dtype=np.complex128)
    group = items[0].group
    if any(target.group is not group for target in items):
        raise ValueError("Combined Wannier targets must belong to the same space group.")
    blocks = [target.matrix(operation, k_fractional) for target in items]
    total = sum(block.shape[0] for block in blocks)
    output = np.zeros((total, total), dtype=np.complex128)
    offset = 0
    for block in blocks:
        size = block.shape[0]
        output[offset : offset + size, offset : offset + size] = block
        offset += size
    return output


@dataclass(frozen=True)
class SymmetryModel:
    dimension: int
    tolerance: float
    group: SpaceGroup
    targets: tuple[WannierTargetRepresentation, ...]
    representation_analysis: RepresentationAnalysisSpec | None = None
    symmetry_gauge: SymmetryGaugeSpec | None = None
    group_definition: SymmetryGroupDefinition | None = None

    def target(self, name: str) -> WannierTargetRepresentation:
        for target in self.targets:
            if target.name == name:
                return target
        raise KeyError(f"Unknown Wannier target name: {name!r}.")


@dataclass(frozen=True)
class SymmetryContext:
    model: SymmetryModel
    k_points: tuple[np.ndarray, ...]
    k_mappings: tuple[tuple[SymmetryKMapping, ...], ...]


def build_symmetry_context(model: SymmetryModel, k_points) -> SymmetryContext:
    axes = tuple(np.asarray(axis, dtype=float).copy() for axis in k_points)
    if len(axes) != model.dimension:
        raise ValueError(
            f"Symmetry dimension {model.dimension} does not match the {len(axes)}-dimensional k mesh."
        )
    for axis in axes:
        axis.setflags(write=False)
    mappings = build_k_mappings(model.group, axes)
    return SymmetryContext(model, axes, mappings)


def build_wannier_target(
    name: str,
    group: SpaceGroup,
    center,
    irrep_spec: SiteIrrepSpec,
) -> WannierTargetRepresentation:
    orbit = build_crystallographic_orbit(group, center)
    irrep = build_site_irrep(orbit, irrep_spec)
    return WannierTargetRepresentation(name, group, orbit, irrep)


def build_wannier_target_from_group_irrep(
    name: str,
    group: SpaceGroup,
    center,
    group_irrep: GroupIrrep,
) -> WannierTargetRepresentation:
    orbit = build_crystallographic_orbit(group, center)
    source_indices = tuple(
        element.source_operation_index for element in orbit.site_symmetry.elements
    )
    if set(source_indices) != set(group_irrep.table.operation_indices):
        raise ValueError(
            f"Irrep {group_irrep.name!r} is defined for operations "
            f"{group_irrep.table.operation_indices}, but target {name!r} has site group {source_indices}."
        )
    matrices = tuple(
        np.asarray(group_irrep.matrix_for_global_index(operation_index), dtype=np.complex128).copy()
        for operation_index in source_indices
    )
    _validate_site_representation(orbit, matrices, group_irrep.dimension)
    for matrix in matrices:
        matrix.setflags(write=False)
    site_irrep = SiteIrrep(group_irrep.name, group_irrep.dimension, matrices)
    return WannierTargetRepresentation(name, group, orbit, site_irrep)
