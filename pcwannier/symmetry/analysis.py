from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .group import SpaceGroup, SpaceGroupOperation
from .specs import (
    DegeneracyTolerance,
    FieldKind,
    RepresentationPointSpec,
)

if TYPE_CHECKING:
    from ..compute.state import StateCollection
    from .bloch import StateBlochSymmetryProvider
    from .representation import SymmetryContext, WannierTargetRepresentation
    from .definition import FactorSystem, ResolvedLittleGroup


def cartesian_field_matrix(
    operation: SpaceGroupOperation,
    real_lattice_vectors,
    field_kind: FieldKind,
    tolerance: float = 1e-8,
) -> np.ndarray:
    lattice = np.asarray(real_lattice_vectors, dtype=float)
    if lattice.shape != (operation.dimension, operation.dimension):
        raise ValueError(
            f"real_lattice_vectors must have shape {(operation.dimension, operation.dimension)}."
        )
    cartesian_rotation = lattice.T @ operation.rotation @ np.linalg.inv(lattice.T)
    orthogonal_residual = float(
        np.linalg.norm(cartesian_rotation.T @ cartesian_rotation - np.eye(operation.dimension), ord="fro")
    )
    if orthogonal_residual > tolerance:
        raise ValueError(
            f"Fractional rotation is not an isometry of the supplied lattice (residual={orthogonal_residual:.6g})."
        )
    if field_kind == FieldKind.SCALAR:
        return np.ones((1, 1), dtype=float)
    if field_kind == FieldKind.ELECTRIC_POLAR_VECTOR:
        return cartesian_rotation
    if field_kind == FieldKind.MAGNETIC_AXIAL_VECTOR:
        return float(np.linalg.det(cartesian_rotation)) * cartesian_rotation
    raise ValueError(f"Unsupported field kind: {field_kind!r}.")


@dataclass(frozen=True)
class LittleGroupElement:
    operation_index: int
    reciprocal_lattice_shift: tuple[int, ...]


@dataclass(frozen=True)
class SewingMatrixRequest:
    operation_index: int
    operation: SpaceGroupOperation
    source_k_fractional: np.ndarray
    target_k_fractional: np.ndarray
    reciprocal_lattice_shift: tuple[int, ...]
    band_indices: tuple[int, ...]
    field_kind: FieldKind
    target_band_indices: tuple[int, ...] | None = None


@runtime_checkable
class SewingMatrixProvider(Protocol):
    def sewing_matrix(self, request: SewingMatrixRequest) -> np.ndarray:
        """Return d_tilde_mn(g,k), with target band m as row and source band n as column."""


@dataclass(frozen=True)
class LittleGroupRepresentationEntry:
    element: LittleGroupElement
    matrix: np.ndarray
    character: complex


@dataclass(frozen=True)
class LittleGroupAnalysis:
    k_fractional: np.ndarray
    band_indices: tuple[int, ...]
    entries: tuple[LittleGroupRepresentationEntry, ...]

    @property
    def characters(self) -> tuple[complex, ...]:
        return tuple(entry.character for entry in self.entries)


@dataclass(frozen=True)
class SewingDiagnostics:
    unitarity_error: float
    leakage: float
    max_composition_residual: float


@dataclass(frozen=True)
class IrrepDecomposition:
    raw_multiplicities: dict[str, complex]
    multiplicities: dict[str, int]
    rounding_residuals: dict[str, float]
    class_character_residuals: dict[str, float] = field(default_factory=dict)

    @property
    def max_residual(self) -> float:
        values = tuple(self.rounding_residuals.values()) + tuple(self.class_character_residuals.values())
        return max(values, default=0.0)


@dataclass(frozen=True)
class RepresentationCompatibility:
    compatible: bool
    target_dimension: int
    physical_dimension: int
    target_multiplicities: dict[str, int]
    physical_multiplicities: dict[str, int]
    missing_irreps: dict[str, int]
    extra_irreps: dict[str, int]
    target_residual: float
    physical_residual: float


@dataclass(frozen=True)
class DegenerateBlock:
    band_indices: tuple[int, ...]
    energies: tuple[complex, ...]
    sewing_matrices: dict[str, np.ndarray]
    characters: dict[str, complex]
    leakage: float
    decomposition: IrrepDecomposition | None = None


@dataclass(frozen=True)
class HighSymmetryPointAnalysis:
    name: str
    requested_k_fractional: np.ndarray
    sampled_k_fractional: np.ndarray
    k_index: tuple[int, ...]
    band_indices: tuple[int, ...]
    little_group_operation_indices: tuple[int, ...]
    sewing_matrices: dict[str, np.ndarray]
    characters: dict[str, complex]
    diagnostics: SewingDiagnostics
    degenerate_blocks: tuple[DegenerateBlock, ...]
    physical_decomposition: IrrepDecomposition | None
    target_characters: dict[str, complex]
    target_decomposition: IrrepDecomposition | None
    compatibility: RepresentationCompatibility | None
    little_group_name: str | None = None
    conjugacy_classes: tuple[tuple[str, ...], ...] = ()
    finite_group_mapping: tuple[tuple[str, str], ...] = ()
    factor_system: FactorSystem | None = None


@dataclass(frozen=True)
class SymmetryAnalysisResult:
    points: tuple[HighSymmetryPointAnalysis, ...]


def little_group(group: SpaceGroup, k_fractional) -> tuple[LittleGroupElement, ...]:
    kpoint = np.asarray(k_fractional, dtype=float)
    if kpoint.shape != (group.dimension,) or not np.all(np.isfinite(kpoint)):
        raise ValueError(f"k_fractional must have shape {(group.dimension,)} and be finite.")
    elements = []
    for operation_index, operation in enumerate(group.operations):
        displacement = operation.act_reciprocal(kpoint) - kpoint
        reciprocal_shift = np.rint(displacement).astype(np.int64)
        if np.allclose(displacement, reciprocal_shift, rtol=0.0, atol=group.tolerance):
            elements.append(
                LittleGroupElement(operation_index, tuple(int(value) for value in reciprocal_shift))
            )
    return tuple(elements)


def analyze_little_group(
    group: SpaceGroup,
    k_fractional,
    band_indices,
    provider: SewingMatrixProvider,
    *,
    field_kind: FieldKind = FieldKind.SCALAR,
) -> LittleGroupAnalysis:
    kpoint = np.asarray(k_fractional, dtype=float)
    bands = _validated_bands(band_indices)
    entries = []
    for element in little_group(group, kpoint):
        operation = group.operations[element.operation_index]
        sewing_at = getattr(provider, "sewing_matrix_at", None)
        if sewing_at is not None:
            raw_matrix = sewing_at(element.operation_index, kpoint, bands, operation=operation)
        else:
            request = SewingMatrixRequest(
                operation_index=element.operation_index,
                operation=operation,
                source_k_fractional=kpoint.copy(),
                target_k_fractional=kpoint.copy(),
                reciprocal_lattice_shift=element.reciprocal_lattice_shift,
                band_indices=bands,
                field_kind=field_kind,
            )
            raw_matrix = provider.sewing_matrix(request)
        matrix = _validated_sewing(raw_matrix, len(bands), operation)
        entries.append(LittleGroupRepresentationEntry(element, matrix, complex(np.trace(matrix))))
    return LittleGroupAnalysis(kpoint.copy(), bands, tuple(entries))


def group_degenerate_bands(
    band_indices,
    energies,
    tolerance: DegeneracyTolerance,
) -> tuple[tuple[int, ...], ...]:
    bands = _validated_bands(band_indices)
    values = np.asarray(energies).reshape(-1)
    if values.size != len(bands) or not np.all(np.isfinite(values)):
        raise ValueError("Degeneracy energies must be finite and match band_indices.")
    ordered = sorted(zip(bands, values), key=lambda item: (float(np.real(item[1])), float(np.imag(item[1]))))
    blocks: list[list[int]] = [[ordered[0][0]]]
    previous = ordered[0][1]
    for band, energy in ordered[1:]:
        if tolerance.equivalent(previous, energy):
            blocks[-1].append(band)
        else:
            blocks.append([band])
        previous = energy
    return tuple(tuple(block) for block in blocks)


def decompose_little_group_characters(
    little_group_definition: ResolvedLittleGroup,
    physical_characters: dict[str, complex],
) -> IrrepDecomposition:
    table = little_group_definition.table
    names = table.operation_names
    if set(physical_characters) != set(names):
        raise ValueError("Physical characters must contain every little-group operation exactly once.")
    physical_values = np.asarray([physical_characters[name] for name in names], dtype=np.complex128)
    cochain = little_group_definition.factor_system.trivializing_cochain
    if cochain is None:
        little_group_definition.require_irreps()
        raise AssertionError("unreachable")
    physical_values = cochain * physical_values
    raw: dict[str, complex] = {}
    rounded: dict[str, int] = {}
    residuals: dict[str, float] = {}
    for irrep in little_group_definition.require_irreps():
        character = np.asarray(irrep.characters, dtype=np.complex128)
        multiplicity = np.vdot(character, physical_values) / table.order
        nearest = int(np.rint(multiplicity.real))
        raw[irrep.name] = complex(multiplicity)
        rounded[irrep.name] = nearest
        residuals[irrep.name] = float(abs(multiplicity - nearest))
    class_residuals = {}
    for conjugacy_class in table.conjugacy_classes:
        class_names = tuple(
            table.element_names[index] for index in conjugacy_class.element_indices
        )
        values = [physical_values[index] for index in conjugacy_class.element_indices]
        average = sum(values) / len(values)
        label = "{" + ",".join(str(name) for name in class_names) + "}"
        class_residuals[label] = max((abs(value - average) for value in values), default=0.0)
    return IrrepDecomposition(raw, rounded, residuals, class_residuals)


def compare_representations(
    target_dimension: int,
    physical_dimension: int,
    target: IrrepDecomposition,
    physical: IrrepDecomposition,
) -> RepresentationCompatibility:
    names = set(target.multiplicities) | set(physical.multiplicities)
    target_values = {name: target.multiplicities.get(name, 0) for name in names}
    physical_values = {name: physical.multiplicities.get(name, 0) for name in names}
    missing = {
        name: target_values[name] - physical_values[name]
        for name in names
        if target_values[name] > physical_values[name]
    }
    extra = {
        name: physical_values[name] - target_values[name]
        for name in names
        if physical_values[name] > target_values[name]
    }
    residuals_are_small = target.max_residual <= 1.0e-5 and physical.max_residual <= 1.0e-5
    if physical_dimension < target_dimension:
        compatible = False
    elif physical_dimension == target_dimension:
        compatible = not missing and not extra
    else:
        compatible = not missing
    compatible = compatible and residuals_are_small
    return RepresentationCompatibility(
        compatible,
        int(target_dimension),
        int(physical_dimension),
        target_values,
        physical_values,
        missing,
        extra,
        target.max_residual,
        physical.max_residual,
    )


def intertwiner_residual(U_source, U_target, D, d_tilde) -> float:
    source = np.asarray(U_source, dtype=np.complex128)
    target = np.asarray(U_target, dtype=np.complex128)
    target_representation = np.asarray(D, dtype=np.complex128)
    physical_representation = np.asarray(d_tilde, dtype=np.complex128)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("U_source and U_target must be M x N_W matrices.")
    if source.shape != target.shape:
        raise ValueError("U_source and U_target must have the same shape.")
    physical_dimension, wannier_dimension = source.shape
    if target_representation.shape != (wannier_dimension, wannier_dimension):
        raise ValueError("D must have shape N_W x N_W.")
    if physical_representation.shape != (physical_dimension, physical_dimension):
        raise ValueError("d_tilde must have shape M x M.")
    return float(
        np.linalg.norm(
            target @ target_representation - physical_representation @ source,
            ord="fro",
        )
    )


def run_symmetry_analysis(
    state: StateCollection,
    context: SymmetryContext,
    *,
    provider: StateBlochSymmetryProvider | None = None,
) -> SymmetryAnalysisResult:
    from .bloch import StateBlochSymmetryProvider

    spec = context.model.representation_analysis
    if spec is None:
        return SymmetryAnalysisResult(())
    provider = provider or StateBlochSymmetryProvider(state, context, field_kind=spec.field_kind)
    return SymmetryAnalysisResult(
        tuple(_analyze_point(state, context, provider, point) for point in spec.points)
    )


def _analyze_point(
    state: StateCollection,
    context: SymmetryContext,
    provider: StateBlochSymmetryProvider,
    point: RepresentationPointSpec,
) -> HighSymmetryPointAnalysis:
    group = context.model.group
    k_index = provider.find_k_index(point.k_fractional)
    sampled_k = np.asarray([context.k_points[axis][k_index[axis]] for axis in range(group.dimension)])
    state_index = tuple(list(k_index) + [0] * (3 - len(k_index)))
    available = tuple(int(value) for value in np.asarray(state.E_idx[state_index]).reshape(-1))
    bands = available if point.band_indices is None else _validated_bands(point.band_indices)
    missing = sorted(set(bands) - set(available))
    if missing:
        raise ValueError(f"Analysis point {point.name!r} is missing actual bands {missing}.")

    elements = little_group(group, point.k_fractional)
    operation_indices = tuple(element.operation_index for element in elements)
    resolved_little_group = (
        context.model.group_definition.resolve_little_group(
            operation_indices, point.k_fractional
        )
        if context.model.group_definition is not None
        else None
    )
    matrices: dict[str, np.ndarray] = {}
    characters: dict[str, complex] = {}
    for element in elements:
        mapping = provider.mapping(element.operation_index, k_index)
        request = provider.request_for_mapping(
            mapping,
            bands,
            source_k_fractional=point.k_fractional,
        )
        matrix = _validated_sewing(
            provider.sewing_matrix(request), len(bands), group.operations[element.operation_index]
        )
        name = _operation_name(group, element.operation_index)
        matrices[name] = matrix
        characters[name] = complex(np.trace(matrix))

    diagnostics = SewingDiagnostics(
        unitarity_error=max(
            (
                float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(len(bands)), ord="fro"))
                for matrix in matrices.values()
            ),
            default=0.0,
        ),
        leakage=max(
            (_projected_subspace_leakage(matrix, len(bands)) for matrix in matrices.values()),
            default=0.0,
        ),
        max_composition_residual=_composition_residual(
            provider, point.k_fractional, k_index, bands, operation_indices, matrices
        ),
    )

    energy_line = np.asarray(state.energy_matrix[state_index])
    energies = tuple(complex(energy_line[band]) for band in bands)
    blocks = group_degenerate_bands(bands, energies, point.degeneracy_tolerance)
    band_position = {band: index for index, band in enumerate(bands)}
    block_results = []
    for block in blocks:
        positions = [band_position[band] for band in block]
        block_matrices = {
            name: matrix[np.ix_(positions, positions)].copy() for name, matrix in matrices.items()
        }
        block_characters = {name: complex(np.trace(matrix)) for name, matrix in block_matrices.items()}
        outside = [index for index in range(len(bands)) if index not in positions]
        leakage = max(
            (
                float(
                    np.sqrt(
                        np.linalg.norm(matrix[np.ix_(outside, positions)], ord="fro") ** 2
                        + np.linalg.norm(matrix[np.ix_(positions, outside)], ord="fro") ** 2
                    )
                )
                for matrix in matrices.values()
            ),
            default=0.0,
        )
        if resolved_little_group is not None:
            decomposition = decompose_little_group_characters(
                resolved_little_group, block_characters
            )
        else:
            decomposition = None
        block_results.append(
            DegenerateBlock(
                block,
                tuple(complex(energy_line[band]) for band in block),
                block_matrices,
                block_characters,
                leakage,
                decomposition,
            )
        )

    if resolved_little_group is not None:
        physical_decomposition = decompose_little_group_characters(
            resolved_little_group, characters
        )
    else:
        physical_decomposition = None
    targets = _selected_targets(context, point)
    target_characters = _target_characters(targets, operation_indices, point.k_fractional)
    if targets and resolved_little_group is not None:
        target_decomposition = decompose_little_group_characters(
            resolved_little_group, target_characters
        )
    else:
        target_decomposition = None
    compatibility = (
        compare_representations(
            sum(target.wannier_dimension for target in targets),
            len(bands),
            target_decomposition,
            physical_decomposition,
        )
        if target_decomposition is not None and physical_decomposition is not None
        else None
    )
    return HighSymmetryPointAnalysis(
        point.name,
        np.asarray(point.k_fractional).copy(),
        sampled_k.copy(),
        k_index,
        bands,
        operation_indices,
        matrices,
        characters,
        diagnostics,
        tuple(block_results),
        physical_decomposition,
        target_characters,
        target_decomposition,
        compatibility,
        None if resolved_little_group is None else resolved_little_group.name,
        (
            ()
            if resolved_little_group is None
            else tuple(
                tuple(
                    resolved_little_group.table.element_names[index]
                    for index in conjugacy_class.element_indices
                )
                for conjugacy_class in resolved_little_group.table.conjugacy_classes
            )
        ),
        (
            ()
            if resolved_little_group is None
            else tuple(
                (
                    resolved_little_group.table.element_names[actual],
                    resolved_little_group.identification.canonical.table.element_names[canonical],
                )
                for actual, canonical in enumerate(
                    resolved_little_group.identification.actual_to_canonical
                )
            )
        ),
        None if resolved_little_group is None else resolved_little_group.factor_system,
    )


def _composition_residual(
    provider: StateBlochSymmetryProvider,
    kpoint: np.ndarray,
    k_index: tuple[int, ...],
    bands: tuple[int, ...],
    operation_indices: tuple[int, ...],
    matrices: dict[str, np.ndarray],
) -> float:
    group = provider.context.model.group
    residual = 0.0
    for left_index in operation_indices:
        left = group.operations[left_index]
        for right_index in operation_indices:
            right = group.operations[right_index]
            right_k = right.act_reciprocal(kpoint)
            right_target_index = provider.find_k_index(right_k)
            left_mapping = provider.mapping(left_index, right_target_index)
            left_request = provider.request_for_mapping(
                left_mapping,
                bands,
                operation=left,
                source_k_fractional=right_k,
            )
            left_matrix = provider.sewing_matrix(left_request)
            product_operation = left * right
            representative_index = group.operation_index(product_operation)
            product_mapping = provider.mapping(representative_index, k_index)
            product_request = provider.request_for_mapping(
                product_mapping,
                bands,
                operation=product_operation,
                source_k_fractional=kpoint,
            )
            product_matrix = provider.sewing_matrix(product_request)
            right_matrix = matrices[_operation_name(group, right_index)]
            residual = max(
                residual,
                float(np.linalg.norm(left_matrix @ right_matrix - product_matrix, ord="fro")),
            )
    return residual


def _target_characters(
    targets: tuple[WannierTargetRepresentation, ...],
    operation_indices: tuple[int, ...],
    kpoint: np.ndarray,
) -> dict[str, complex]:
    if not targets:
        return {}
    group = targets[0].group
    return {
        _operation_name(group, operation_index): sum(
            complex(np.trace(target.matrix(operation_index, kpoint))) for target in targets
        )
        for operation_index in operation_indices
    }


def _selected_targets(
    context: SymmetryContext,
    point: RepresentationPointSpec,
) -> tuple[WannierTargetRepresentation, ...]:
    if point.target_names is None:
        return ()
    return tuple(context.model.target(name) for name in point.target_names)


def _validated_bands(band_indices) -> tuple[int, ...]:
    bands = tuple(int(index) for index in band_indices)
    if not bands or any(index < 0 for index in bands) or len(set(bands)) != len(bands):
        raise ValueError("band_indices must contain unique non-negative actual Bloch-band indices.")
    return bands


def _projected_subspace_leakage(matrix: np.ndarray, dimension: int) -> float:
    deficit = float(dimension - np.linalg.norm(matrix, ord="fro") ** 2)
    roundoff = 1.0e-12 * max(dimension, 1)
    return 0.0 if deficit <= roundoff else float(np.sqrt(deficit))


def _validated_sewing(matrix, dimension: int, operation: SpaceGroupOperation) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    expected = (dimension, dimension)
    if array.shape != expected:
        raise ValueError(
            f"Sewing matrix for operation {operation.name or '<unnamed>'} has shape "
            f"{array.shape}; expected {expected}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Sewing matrix contains non-finite values.")
    return array.copy()


def _operation_name(group: SpaceGroup, operation_index: int) -> str:
    name = group.operations[operation_index].name
    if name is None:
        raise ValueError("Representation analysis requires names for all little-group operations.")
    return name
