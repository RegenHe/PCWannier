from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .group import SpaceGroup, SpaceGroupOperation
from .specs import (
    DegeneracyTolerance,
    FieldKind,
    RepresentationPointSpec,
)
from .twisted import TwistedRepresentation, build_twisted_representation

LOGGER = logging.getLogger(__name__)

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
    if field_kind in {FieldKind.SCALAR, FieldKind.ELECTRIC_Z}:
        return np.ones((1, 1), dtype=float)
    if field_kind == FieldKind.MAGNETIC_AXIAL_Z:
        return np.asarray([[float(np.linalg.det(cartesian_rotation))]], dtype=float)
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
    max_twisted_composition_residual: float = 0.0


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
    unitary_characters: dict[str, complex] = field(default_factory=dict)
    generator_eigenvalues: dict[str, tuple[complex, ...]] = field(default_factory=dict)
    antiunitary_diagnostics: tuple["AntiunitaryOperationDiagnostic", ...] = ()
    coupled_outer_bands: tuple[int, ...] = ()
    candidate_excluded_bands: tuple[int, ...] = ()
    irrep_unavailable_reason: str | None = None
    unitarity_error: float = 0.0
    twisted_composition_residual: float = 0.0


@dataclass(frozen=True)
class AntiunitaryOperationDiagnostic:
    operation_name: str
    square_operation_name: str
    square_eigenvalues: tuple[complex, ...]
    square_residual: float


@dataclass(frozen=True)
class BlochSymmetryPointAnalysis:
    name: str
    requested_k_fractional: np.ndarray
    sampled_k_fractional: np.ndarray
    k_index: tuple[int, ...]
    outer_band_indices: tuple[int, ...]
    band_indices: tuple[int, ...]
    little_group_operation_indices: tuple[int, ...]
    sewing_matrices: dict[str, np.ndarray]
    unitary_characters: dict[str, complex]
    diagnostics: SewingDiagnostics
    degenerate_blocks: tuple[DegenerateBlock, ...]
    physical_decomposition: IrrepDecomposition | None
    little_group_name: str | None = None
    unitary_subgroup_name: str | None = None
    unitary_operation_names: tuple[str, ...] = ()
    antiunitary_operation_names: tuple[str, ...] = ()
    conjugacy_classes: tuple[tuple[str, ...], ...] = ()
    finite_group_mapping: tuple[tuple[str, str], ...] = ()
    factor_system: FactorSystem | None = None
    physical_twisted_representation: TwistedRepresentation | None = None
    outer_unitarity_error: float = 0.0
    outer_candidate_excluded_bands: tuple[int, ...] = ()

    @property
    def characters(self) -> dict[str, complex]:
        """Compatibility alias containing unitary characters only."""

        return self.unitary_characters


@dataclass(frozen=True)
class BlochSymmetryAnalysisResult:
    points: tuple[BlochSymmetryPointAnalysis, ...]


@dataclass(frozen=True)
class TargetCompatibilityAnalysis:
    point_name: str
    target_names: tuple[str, ...]
    target_characters: dict[str, complex]
    target_decomposition: IrrepDecomposition | None
    compatibility: RepresentationCompatibility | None
    target_twisted_representation: TwistedRepresentation | None
    intertwiner_dimension: int | None


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
    physical_twisted_representation: TwistedRepresentation | None = None
    target_twisted_representation: TwistedRepresentation | None = None
    intertwiner_dimension: int | None = None


@dataclass(frozen=True)
class SymmetryAnalysisResult:
    points: tuple[HighSymmetryPointAnalysis, ...]
    bloch: BlochSymmetryAnalysisResult | None = None
    target_compatibilities: tuple[TargetCompatibilityAnalysis, ...] = ()


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
    block_reference = ordered[0][1]
    for band, energy in ordered[1:]:
        if tolerance.equivalent(block_reference, energy):
            blocks[-1].append(band)
        else:
            blocks.append([band])
            block_reference = energy
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


def intertwiner_residual(
    U_source,
    U_target,
    D,
    d_tilde,
    *,
    antiunitary: bool = False,
) -> float:
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
            target @ target_representation
            - physical_representation
            @ (source.conj() if antiunitary else source),
            ord="fro",
        )
    )


def analyze_bloch_symmetry(
    state: StateCollection,
    context: SymmetryContext,
    k_point,
    band_indices=None,
    *,
    provider: StateBlochSymmetryProvider | None = None,
    name: str | None = None,
    split_degenerate_blocks: bool = True,
    degeneracy_tolerance: DegeneracyTolerance | None = None,
    leakage_tolerance: float | None = None,
) -> BlochSymmetryPointAnalysis:
    """Analyze only the physical Bloch representation at one sampled k point."""

    from .bloch import StateBlochSymmetryProvider

    spec = context.model.representation_analysis
    default_degeneracy = DegeneracyTolerance() if spec is None else spec.degeneracy_tolerance
    degeneracy = degeneracy_tolerance or default_degeneracy
    leakage = (
        context.model.tolerance
        if leakage_tolerance is None and spec is None
        else spec.leakage_tolerance if leakage_tolerance is None else float(leakage_tolerance)
    )
    if not np.isfinite(leakage) or leakage <= 0.0:
        raise ValueError("Bloch-symmetry leakage tolerance must be positive and finite.")
    point = RepresentationPointSpec(
        name or "k=" + np.array2string(np.asarray(k_point, dtype=float)),
        np.asarray(k_point, dtype=float),
        None if band_indices is None else _validated_bands(band_indices),
        None,
        degeneracy,
    )
    physical_provider = provider or StateBlochSymmetryProvider(
        state, context, field_kind=(spec.field_kind if spec is not None else state.maxwell.symmetry_field_kind)
    )
    return _analyze_bloch_point(
        state,
        context,
        physical_provider,
        point,
        split_degenerate_blocks=bool(split_degenerate_blocks),
        leakage_tolerance=leakage,
    )


def run_bloch_symmetry_analysis(
    state: StateCollection,
    context: SymmetryContext,
    *,
    provider: StateBlochSymmetryProvider | None = None,
) -> BlochSymmetryAnalysisResult:
    """Analyze configured physical Bloch representations without target data."""

    from .bloch import StateBlochSymmetryProvider

    spec = context.model.representation_analysis
    if spec is None:
        return BlochSymmetryAnalysisResult(())
    physical_provider = provider or StateBlochSymmetryProvider(
        state, context, field_kind=spec.field_kind
    )
    return BlochSymmetryAnalysisResult(
        tuple(
            _analyze_bloch_point(
                state,
                context,
                physical_provider,
                point,
                split_degenerate_blocks=True,
                leakage_tolerance=spec.leakage_tolerance,
            )
            for point in spec.points
        )
    )


def run_symmetry_analysis(
    state: StateCollection,
    context: SymmetryContext,
    *,
    provider: StateBlochSymmetryProvider | None = None,
) -> SymmetryAnalysisResult:
    """Run physical analysis and optional, explicitly requested target comparisons."""

    bloch = run_bloch_symmetry_analysis(state, context, provider=provider)
    spec = context.model.representation_analysis
    if spec is None:
        return SymmetryAnalysisResult((), bloch)
    point_specs = {point.name: point for point in spec.points}
    combined = []
    comparisons = []
    for physical in bloch.points:
        result, comparison = _attach_target_compatibility(
            context, point_specs[physical.name], physical
        )
        combined.append(result)
        if comparison is not None:
            comparisons.append(comparison)
    return SymmetryAnalysisResult(tuple(combined), bloch, tuple(comparisons))


def _analyze_bloch_point(
    state: StateCollection,
    context: SymmetryContext,
    provider: StateBlochSymmetryProvider,
    point: RepresentationPointSpec,
    *,
    split_degenerate_blocks: bool,
    leakage_tolerance: float,
) -> BlochSymmetryPointAnalysis:
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
            operation_indices,
            point.k_fractional,
            bloch_convention=context.model.bloch_convention,
        )
        if context.model.group_definition is not None
        else None
    )
    full_matrices: dict[str, np.ndarray] = {}
    matrices: dict[str, np.ndarray] = {}
    internal_matrices: dict[str, np.ndarray] = {}
    matrices_by_operation: dict[int, np.ndarray] = {}
    for element in elements:
        mapping = provider.mapping(element.operation_index, k_index)
        operation = group.operations[element.operation_index]
        request = provider.request_for_mapping(mapping, available, source_k_fractional=point.k_fractional)
        full_matrix = _validated_sewing(provider.sewing_matrix(request), len(available), operation)
        full_matrices[_operation_name(group, element.operation_index)] = full_matrix
        positions = [available.index(band) for band in bands]
        internal_matrix = full_matrix[np.ix_(positions, positions)].copy()
        band_basis = getattr(provider, "sewing_matrix_in_band_basis", None)
        matrix = (
            internal_matrix
            if band_basis is None
            else _validated_sewing(
                band_basis(
                    mapping,
                    bands,
                    bands,
                    operation=operation,
                    source_k_fractional=point.k_fractional,
                ),
                len(bands),
                operation,
            )
        )
        name = _operation_name(group, element.operation_index)
        matrices[name] = matrix
        internal_matrices[name] = internal_matrix
        matrices_by_operation[element.operation_index] = matrix

    physical_twisted = (
        None
        if resolved_little_group is None
        else build_twisted_representation(
            resolved_little_group,
            operation_indices,
            tuple(matrices_by_operation[index] for index in operation_indices),
        )
    )
    operation_by_name = {
        _operation_name(group, index): group.operations[index] for index in operation_indices
    }
    unitary_characters = {
        name: complex(np.trace(matrix))
        for name, matrix in matrices.items()
        if not operation_by_name[name].antiunitary
    }
    selected_leakage, _ = _subspace_leakage(
        full_matrices, available, bands, leakage_tolerance
    )
    diagnostics = SewingDiagnostics(
        unitarity_error=max(
            (
                float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(len(bands)), ord="fro"))
                for matrix in matrices.values()
            ),
            default=0.0,
        ),
        leakage=selected_leakage,
        max_composition_residual=_composition_residual(
            provider, point.k_fractional, k_index, bands, operation_indices, internal_matrices
        ),
        max_twisted_composition_residual=(
            0.0 if physical_twisted is None else physical_twisted.product_residual
        ),
    )

    energy_line = np.asarray(state.energy_matrix[state_index])
    energies = tuple(complex(energy_line[band]) for band in bands)
    blocks = (
        group_degenerate_bands(bands, energies, point.degeneracy_tolerance)
        if split_degenerate_blocks
        else (bands,)
    )
    band_position = {band: index for index, band in enumerate(bands)}
    generator_indices = _minimal_unitary_generators(group, operation_indices)
    unitary_indices = tuple(
        index for index in operation_indices if not group.operations[index].antiunitary
    )
    antiunitary_indices = tuple(
        index for index in operation_indices if group.operations[index].antiunitary
    )
    unitary_subgroup_name = _unitary_subgroup_name(
        context, unitary_indices, point.k_fractional
    )
    block_results = []
    for block in blocks:
        positions = [band_position[band] for band in block]
        block_matrices = {}
        for element in elements:
            operation = group.operations[element.operation_index]
            name = _operation_name(group, element.operation_index)
            mapping = provider.mapping(element.operation_index, k_index)
            band_basis = getattr(provider, "sewing_matrix_in_band_basis", None)
            block_matrices[name] = (
                matrices[name][np.ix_(positions, positions)].copy()
                if band_basis is None
                else _validated_sewing(
                    band_basis(
                        mapping,
                        block,
                        block,
                        operation=operation,
                        source_k_fractional=point.k_fractional,
                    ),
                    len(block),
                    operation,
                )
            )
        block_unitary_characters = {
            name: complex(np.trace(matrix))
            for name, matrix in block_matrices.items()
            if not operation_by_name[name].antiunitary
        }
        leakage, coupled = _subspace_leakage(
            full_matrices, available, block, leakage_tolerance
        )
        block_by_operation = {
            index: block_matrices[_operation_name(group, index)] for index in operation_indices
        }
        block_twisted = (
            None
            if resolved_little_group is None
            else build_twisted_representation(
                resolved_little_group,
                operation_indices,
                tuple(block_by_operation[index] for index in operation_indices),
            )
        )
        twisted_residual = 0.0 if block_twisted is None else block_twisted.product_residual
        unavailable_reason = _irrep_unavailable_reason(
            resolved_little_group,
            leakage,
            leakage_tolerance,
            twisted_residual,
        )
        if (
            unavailable_reason is None
            and resolved_little_group is not None
            and resolved_little_group.factor_system.cohomologically_trivial
            and not any(resolved_little_group.factor_system.antiunitary_flags)
        ):
            decomposition = decompose_little_group_characters(
                resolved_little_group, block_unitary_characters
            )
        else:
            decomposition = None
        candidates = _candidate_excluded_bands(
            energy_line, available, block, point.degeneracy_tolerance
        )
        generator_eigenvalues = {
            _operation_name(group, index): _sorted_eigenvalues(
                block_matrices[_operation_name(group, index)]
            )
            for index in generator_indices
        }
        antiunitary_diagnostics = _antiunitary_diagnostics(
            group,
            resolved_little_group,
            block_twisted,
            operation_indices,
        )
        block_results.append(
            DegenerateBlock(
                band_indices=block,
                energies=tuple(complex(energy_line[band]) for band in block),
                sewing_matrices=block_matrices,
                characters=block_unitary_characters,
                leakage=leakage,
                decomposition=decomposition,
                unitary_characters=block_unitary_characters,
                generator_eigenvalues=generator_eigenvalues,
                antiunitary_diagnostics=antiunitary_diagnostics,
                coupled_outer_bands=coupled,
                candidate_excluded_bands=candidates,
                irrep_unavailable_reason=unavailable_reason,
                unitarity_error=max(
                    (
                        float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(len(block)), ord="fro"))
                        for matrix in block_matrices.values()
                    ),
                    default=0.0,
                ),
                twisted_composition_residual=twisted_residual,
            )
        )

    if (
        resolved_little_group is not None
        and resolved_little_group.factor_system.cohomologically_trivial
        and not any(resolved_little_group.factor_system.antiunitary_flags)
        and diagnostics.leakage <= leakage_tolerance
        and physical_twisted is not None
        and physical_twisted.product_residual <= leakage_tolerance
    ):
        physical_decomposition = decompose_little_group_characters(
            resolved_little_group, unitary_characters
        )
    else:
        physical_decomposition = None
    outer_unitarity = max(
        (
            float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(len(available)), ord="fro"))
            for matrix in full_matrices.values()
        ),
        default=0.0,
    )
    outer_candidates = _candidate_excluded_bands(
        energy_line, available, bands, point.degeneracy_tolerance
    )
    if diagnostics.leakage > leakage_tolerance or outer_unitarity > leakage_tolerance:
        LOGGER.warning(
            "Bloch symmetry subspace at %s is not closed: bands(1-based)=%s leakage=%.6g "
            "outer_unitarity=%.6g coupled_outer_bands(1-based)=%s "
            "candidate_excluded_bands(1-based)=%s",
            point.name,
            tuple(band + 1 for band in bands),
            diagnostics.leakage,
            outer_unitarity,
            tuple(band + 1 for block in block_results for band in block.coupled_outer_bands),
            tuple(band + 1 for band in outer_candidates),
        )
    return BlochSymmetryPointAnalysis(
        name=point.name,
        requested_k_fractional=np.asarray(point.k_fractional).copy(),
        sampled_k_fractional=sampled_k.copy(),
        k_index=k_index,
        outer_band_indices=available,
        band_indices=bands,
        little_group_operation_indices=operation_indices,
        sewing_matrices=matrices,
        unitary_characters=unitary_characters,
        diagnostics=diagnostics,
        degenerate_blocks=tuple(block_results),
        physical_decomposition=physical_decomposition,
        little_group_name=None if resolved_little_group is None else resolved_little_group.name,
        unitary_subgroup_name=unitary_subgroup_name,
        unitary_operation_names=tuple(_operation_name(group, index) for index in unitary_indices),
        antiunitary_operation_names=tuple(
            _operation_name(group, index) for index in antiunitary_indices
        ),
        conjugacy_classes=_resolved_conjugacy_classes(resolved_little_group),
        finite_group_mapping=_resolved_finite_mapping(resolved_little_group),
        factor_system=None if resolved_little_group is None else resolved_little_group.factor_system,
        physical_twisted_representation=physical_twisted,
        outer_unitarity_error=outer_unitarity,
        outer_candidate_excluded_bands=outer_candidates,
    )


def _attach_target_compatibility(
    context: SymmetryContext,
    point: RepresentationPointSpec,
    physical: BlochSymmetryPointAnalysis,
) -> tuple[HighSymmetryPointAnalysis, TargetCompatibilityAnalysis | None]:
    group = context.model.group
    operation_indices = physical.little_group_operation_indices
    resolved_little_group = (
        context.model.group_definition.resolve_little_group(
            operation_indices,
            point.k_fractional,
            bloch_convention=context.model.bloch_convention,
        )
        if context.model.group_definition is not None
        else None
    )
    targets = _selected_targets(context, point)
    if targets:
        invalid = [block for block in physical.degenerate_blocks if block.irrep_unavailable_reason and block.leakage > context.model.representation_analysis.leakage_tolerance]
        if invalid:
            block = invalid[0]
            raise ValueError(
                f"Degenerate block {block.band_indices} at representation point {point.name!r} "
                f"is not invariant: sewing leakage={block.leakage:.6g}. Select a closed "
                "physical subspace before target compatibility analysis."
            )
    target_matrices_by_operation = {
        operation_index: _combined_target_matrix(targets, operation_index, point.k_fractional)
        for operation_index in operation_indices
    } if targets else {}
    target_characters = {
        _operation_name(group, operation_index): complex(
            np.trace(target_matrices_by_operation[operation_index])
        )
        for operation_index in operation_indices
    } if targets else {}
    target_twisted = (
        None
        if not targets or resolved_little_group is None
        else build_twisted_representation(
            resolved_little_group,
            operation_indices,
            tuple(target_matrices_by_operation[index] for index in operation_indices),
        )
    )
    physical_twisted = physical.physical_twisted_representation
    if target_twisted is not None:
        target_twisted.require_valid(tolerance=context.model.tolerance)
        if physical_twisted is None:
            raise RuntimeError("Target twisted representation has no physical counterpart.")
        physical_twisted.assert_compatible(target_twisted)

    if (
        targets
        and resolved_little_group is not None
        and resolved_little_group.factor_system.cohomologically_trivial
        and not any(resolved_little_group.factor_system.antiunitary_flags)
    ):
        target_decomposition = decompose_little_group_characters(
            resolved_little_group, target_characters
        )
    else:
        target_decomposition = None
    compatibility = (
        compare_representations(
            sum(target.wannier_dimension for target in targets),
            len(physical.band_indices),
            target_decomposition,
            physical.physical_decomposition,
        )
        if target_decomposition is not None and physical.physical_decomposition is not None
        else None
    )
    intertwiner_dimension = None
    if physical_twisted is not None and target_twisted is not None:
        from .gauge import solve_intertwiner_space

        intertwiner_dimension = solve_intertwiner_space(
            physical_twisted,
            target_twisted,
        ).dimension
    high = HighSymmetryPointAnalysis(
        name=physical.name,
        requested_k_fractional=physical.requested_k_fractional,
        sampled_k_fractional=physical.sampled_k_fractional,
        k_index=physical.k_index,
        band_indices=physical.band_indices,
        little_group_operation_indices=operation_indices,
        sewing_matrices=physical.sewing_matrices,
        characters=physical.unitary_characters,
        diagnostics=physical.diagnostics,
        degenerate_blocks=physical.degenerate_blocks,
        physical_decomposition=physical.physical_decomposition,
        target_characters=target_characters,
        target_decomposition=target_decomposition,
        compatibility=compatibility,
        little_group_name=physical.little_group_name,
        conjugacy_classes=physical.conjugacy_classes,
        finite_group_mapping=physical.finite_group_mapping,
        factor_system=physical.factor_system,
        physical_twisted_representation=physical_twisted,
        target_twisted_representation=target_twisted,
        intertwiner_dimension=intertwiner_dimension,
    )
    comparison = None
    if targets:
        comparison = TargetCompatibilityAnalysis(
            point.name,
            tuple(target.name for target in targets),
            target_characters,
            target_decomposition,
            compatibility,
            target_twisted,
            intertwiner_dimension,
        )
    return high, comparison


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
            composed = (
                left_matrix @ right_matrix.conj()
                if left.antiunitary
                else left_matrix @ right_matrix
            )
            residual = max(
                residual,
                float(np.linalg.norm(composed - product_matrix, ord="fro")),
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


def _combined_target_matrix(
    targets: tuple[WannierTargetRepresentation, ...],
    operation_index: int,
    kpoint,
) -> np.ndarray:
    from .representation import combined_target_matrix

    return combined_target_matrix(targets, operation_index, kpoint)


def _selected_targets(
    context: SymmetryContext,
    point: RepresentationPointSpec,
) -> tuple[WannierTargetRepresentation, ...]:
    if point.target_names is None:
        return ()
    return tuple(context.model.target(name) for name in point.target_names)


def _subspace_leakage(
    matrices: dict[str, np.ndarray],
    outer_bands: tuple[int, ...],
    selected_bands: tuple[int, ...],
    tolerance: float,
) -> tuple[float, tuple[int, ...]]:
    selected = [outer_bands.index(band) for band in selected_bands]
    outside = [index for index in range(len(outer_bands)) if index not in selected]
    if not outside:
        return 0.0, ()
    maximum = 0.0
    coupled: set[int] = set()
    for matrix in matrices.values():
        value = np.asarray(matrix, dtype=np.complex128)
        leakage = float(
            np.sqrt(
                np.linalg.norm(value[np.ix_(outside, selected)], ord="fro") ** 2
                + np.linalg.norm(value[np.ix_(selected, outside)], ord="fro") ** 2
            )
        )
        maximum = max(maximum, leakage)
        for index in outside:
            strength = float(
                np.sqrt(
                    np.linalg.norm(value[index, selected]) ** 2
                    + np.linalg.norm(value[selected, index]) ** 2
                )
            )
            if strength > tolerance:
                coupled.add(outer_bands[index])
    return maximum, tuple(sorted(coupled))


def _candidate_excluded_bands(
    energy_line: np.ndarray,
    available_bands: tuple[int, ...],
    selected_bands: tuple[int, ...],
    tolerance: DegeneracyTolerance,
) -> tuple[int, ...]:
    available = set(available_bands)
    selected_energies = [energy_line[index] for index in selected_bands]
    return tuple(
        index
        for index, energy in enumerate(energy_line)
        if index not in available
        and any(tolerance.equivalent(energy, selected) for selected in selected_energies)
    )


def _minimal_unitary_generators(
    group: SpaceGroup,
    operation_indices: tuple[int, ...],
) -> tuple[int, ...]:
    unitary = tuple(
        index for index in operation_indices if not group.operations[index].antiunitary
    )
    allowed = set(unitary)
    generators: list[int] = []

    def generated_closure() -> set[int]:
        closure = {group.identity_index, *generators}
        changed = True
        while changed:
            changed = False
            for left in tuple(closure):
                for right in tuple(closure):
                    product = group.multiply_mod_lattice(left, right).result_index
                    if product in allowed and product not in closure:
                        closure.add(product)
                        changed = True
        return closure

    closure = generated_closure()
    for index in unitary:
        if index == group.identity_index or index in closure:
            continue
        generators.append(index)
        closure = generated_closure()
    return tuple(generators)


def _unitary_subgroup_name(
    context: SymmetryContext,
    operation_indices: tuple[int, ...],
    kpoint,
) -> str | None:
    if context.model.group_definition is None or not operation_indices:
        return None
    resolved = context.model.group_definition.resolve_little_group(
        operation_indices,
        kpoint,
        bloch_convention=context.model.bloch_convention,
    )
    return resolved.name


def _sorted_eigenvalues(matrix: np.ndarray) -> tuple[complex, ...]:
    values = np.linalg.eigvals(np.asarray(matrix, dtype=np.complex128))
    ordered = sorted(
        (complex(value) for value in values),
        key=lambda value: (float(np.angle(value)), float(value.real), float(value.imag)),
    )
    return tuple(ordered)


def _antiunitary_diagnostics(
    group: SpaceGroup,
    resolved_little_group: ResolvedLittleGroup | None,
    representation: TwistedRepresentation | None,
    operation_indices: tuple[int, ...],
) -> tuple[AntiunitaryOperationDiagnostic, ...]:
    if resolved_little_group is None or representation is None:
        return ()
    local_operations = resolved_little_group.concrete.operation_indices
    output = []
    for operation_index in operation_indices:
        operation = group.operations[operation_index]
        if not operation.antiunitary:
            continue
        local = local_operations.index(operation_index)
        result_local = int(representation.product_table[local, local])
        square = representation.matrices[local] @ representation.matrices[local].conj()
        expected = (
            representation.factor_system.phases[local, local]
            * representation.matrices[result_local]
        )
        result_global = local_operations[result_local]
        output.append(
            AntiunitaryOperationDiagnostic(
                _operation_name(group, operation_index),
                _operation_name(group, result_global),
                _sorted_eigenvalues(square),
                float(np.linalg.norm(square - expected, ord="fro")),
            )
        )
    return tuple(output)


def _irrep_unavailable_reason(
    resolved_little_group: ResolvedLittleGroup | None,
    leakage: float,
    tolerance: float,
    twisted_residual: float,
) -> str | None:
    if resolved_little_group is None:
        return "little co-group could not be identified"
    factor = resolved_little_group.factor_system
    if any(factor.antiunitary_flags):
        return "magnetic corepresentation labels are unavailable"
    if not factor.cohomologically_trivial:
        return "projective irrep labels are unavailable for a non-trivial factor system"
    if leakage > tolerance:
        return f"band subspace leakage {leakage:.6g} exceeds {tolerance:.6g}"
    if twisted_residual > tolerance:
        return f"twisted composition residual {twisted_residual:.6g} exceeds {tolerance:.6g}"
    return None


def _resolved_conjugacy_classes(
    resolved_little_group: ResolvedLittleGroup | None,
) -> tuple[tuple[str, ...], ...]:
    if resolved_little_group is None:
        return ()
    return tuple(
        tuple(
            resolved_little_group.table.element_names[index]
            for index in conjugacy_class.element_indices
        )
        for conjugacy_class in resolved_little_group.table.conjugacy_classes
    )


def _resolved_finite_mapping(
    resolved_little_group: ResolvedLittleGroup | None,
) -> tuple[tuple[str, str], ...]:
    if resolved_little_group is None:
        return ()
    return tuple(
        (
            resolved_little_group.table.element_names[actual],
            resolved_little_group.identification.canonical.table.element_names[canonical],
        )
        for actual, canonical in enumerate(
            resolved_little_group.identification.actual_to_canonical
        )
    )


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
