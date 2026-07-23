from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .definition import ResolvedIrrep, SpaceGroupDefinition
    from .specs import RepresentationAnalysisSpec, SymmetryGaugeSpec

from .group import (
    CrystallographicOrbit,
    SpaceGroup,
    SpaceGroupOperation,
    SymmetryKMapping,
    build_crystallographic_orbit,
    build_k_mappings,
    apply_magnetic_bias,
)
from ..conventions import BlochConvention


@dataclass(frozen=True)
class SiteIrrep:
    name: str
    dimension: int
    matrices: tuple[np.ndarray, ...]
    finite_group_name: str
    actual_to_canonical: tuple[int, ...]

    def matrix(self, site_element_index: int) -> np.ndarray:
        return self.matrices[site_element_index]


def _validate_site_representation(
    orbit: CrystallographicOrbit,
    matrices: tuple[np.ndarray, ...],
    dimension: int,
) -> None:
    identity = np.eye(dimension)
    for left_index, left in enumerate(orbit.site_symmetry.elements):
        for right_index, right in enumerate(orbit.site_symmetry.elements):
            product_index = orbit.site_symmetry.element_index(left.operation * right.operation)
            right_matrix = (
                matrices[right_index].conj()
                if left.operation.antiunitary
                else matrices[right_index]
            )
            expected = matrices[left_index] @ right_matrix
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
    bloch_convention: BlochConvention = field(default_factory=BlochConvention)
    _matrix_cache: dict[tuple[int, bytes], np.ndarray] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

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
        cache_key = (operation_index, np.ascontiguousarray(kpoint).tobytes())
        cached = self._matrix_cache.get(cache_key)
        if cached is not None:
            return cached
        transformed_k = self.group.operations[operation_index].act_reciprocal(kpoint)
        dimension = self.site_irrep.dimension
        output = np.zeros((self.wannier_dimension, self.wannier_dimension), dtype=np.complex128)
        for orbit_index in range(self.multiplicity):
            action = self.orbit.action(operation_index, orbit_index)
            lattice_shift = np.asarray(action.lattice_shift, dtype=float)
            phase = np.exp(
                -self.bloch_convention.sign
                * 2j
                * np.pi
                * np.dot(transformed_k, lattice_shift)
            )
            row = slice(action.target_index * dimension, (action.target_index + 1) * dimension)
            column = slice(orbit_index * dimension, (orbit_index + 1) * dimension)
            output[row, column] = phase * self.site_irrep.matrix(action.site_element_index)
        residual = float(np.linalg.norm(output.conj().T @ output - np.eye(self.wannier_dimension), ord="fro"))
        if residual > 1e-8:
            raise FloatingPointError(f"Target Wannier representation is not unitary (residual={residual:.6g}).")
        output.setflags(write=False)
        self._matrix_cache[cache_key] = output
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
    convention = items[0].bloch_convention
    if any(target.bloch_convention != convention for target in items):
        raise ValueError("Combined Wannier targets must use the same Bloch convention.")
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
    group_definition: SpaceGroupDefinition | None = None
    bloch_convention: BlochConvention = field(default_factory=BlochConvention)
    boundary_tolerance: float = 1.0e-6
    magnetic_bias_direction: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.boundary_tolerance) or self.boundary_tolerance <= 0.0:
            raise ValueError("Symmetry boundary tolerance must be positive and finite.")
        if any(target.bloch_convention != self.bloch_convention for target in self.targets):
            raise ValueError("Symmetry targets and model must use the same Bloch convention.")
        bias = self.magnetic_bias_direction
        if bias is not None:
            bias = np.asarray(bias, dtype=float)
            if bias.shape != (3,) or not np.all(np.isfinite(bias)):
                raise ValueError("Magnetic bias must be a finite Cartesian three-vector.")
            norm = float(np.linalg.norm(bias))
            if norm <= self.tolerance:
                raise ValueError("Magnetic bias direction must have non-zero length.")
            bias = bias / norm
            bias.setflags(write=False)
            object.__setattr__(self, "magnetic_bias_direction", bias)

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


def apply_magnetic_bias_to_model(
    model: SymmetryModel,
    real_lattice_vectors,
    magnetic_bias_direction,
) -> SymmetryModel:
    """Classify spatial operations and return the preserved magnetic group."""

    if model.targets:
        raise ValueError("Magnetic bias must be applied before Wannier targets are constructed.")
    group = apply_magnetic_bias(
        model.group,
        real_lattice_vectors,
        magnetic_bias_direction,
        tolerance=model.tolerance,
    )
    definition = model.group_definition
    if definition is not None:
        from .definition import SpaceGroupDefinition

        definition = SpaceGroupDefinition(
            definition.name,
            definition.dimension,
            definition.tolerance,
            group,
            definition.finite_groups,
        )
    return SymmetryModel(
        model.dimension,
        model.tolerance,
        group,
        (),
        model.representation_analysis,
        model.symmetry_gauge,
        definition,
        model.bloch_convention,
        model.boundary_tolerance,
        np.asarray(magnetic_bias_direction, dtype=float),
    )


def build_wannier_target_from_group_irrep(
    name: str,
    group: SpaceGroup,
    center,
    group_irrep: ResolvedIrrep,
    bloch_convention: BlochConvention | None = None,
) -> WannierTargetRepresentation:
    orbit = build_crystallographic_orbit(group, center)
    source_indices = tuple(
        element.source_operation_index for element in orbit.site_symmetry.elements
    )
    concrete_indices = group_irrep.identification.concrete.operation_indices
    if set(source_indices) != set(concrete_indices):
        raise ValueError(
            f"Irrep {group_irrep.name!r} is defined for operations "
            f"{concrete_indices}, but target {name!r} has site group {source_indices}."
        )
    matrices = tuple(
        np.asarray(group_irrep.matrix_for_global_index(operation_index), dtype=np.complex128).copy()
        for operation_index in source_indices
    )
    _validate_site_representation(orbit, matrices, group_irrep.dimension)
    for matrix in matrices:
        matrix.setflags(write=False)
    site_irrep = SiteIrrep(
        group_irrep.name,
        group_irrep.dimension,
        matrices,
        group_irrep.identification.canonical.name,
        group_irrep.identification.actual_to_canonical,
    )
    return WannierTargetRepresentation(
        name,
        group,
        orbit,
        site_irrep,
        BlochConvention() if bloch_convention is None else bloch_convention,
    )
