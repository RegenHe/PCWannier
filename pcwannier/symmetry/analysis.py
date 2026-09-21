from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from typing import TYPE_CHECKING

import numpy as np

from .field_action import cartesian_field_matrix
from .group import (
    LittleGroupElement,
    SpaceGroup,
    SpaceGroupOperation,
    little_group,
    periodic_difference,
)
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


@dataclass(frozen=True)
class SewingDiagnostics:
    unitarity_error: float
    leakage: float
    outer_composition_residual: float
    selected_twisted_composition_residual: float = 0.0


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
    leakage: float
    decomposition: IrrepDecomposition | None = None
    approximate_decomposition: IrrepDecomposition | None = None
    character_fit_error: float | None = None
    unitary_characters: dict[str, complex] = field(default_factory=dict)
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
    resolved_little_group: ResolvedLittleGroup | None = None
    unitary_subgroup_name: str | None = None
    unitary_operation_names: tuple[str, ...] = ()
    antiunitary_operation_names: tuple[str, ...] = ()
    physical_twisted_representation: TwistedRepresentation | None = None
    outer_unitarity_error: float = 0.0
    outer_candidate_excluded_bands: tuple[int, ...] = ()

    @property
    def characters(self) -> dict[str, complex]:
        """Unitary characters; antiunitary traces are intentionally excluded."""

        return self.unitary_characters

    @property
    def little_group_name(self) -> str | None:
        return None if self.resolved_little_group is None else self.resolved_little_group.name

    @property
    def factor_system(self) -> FactorSystem | None:
        return (
            None
            if self.resolved_little_group is None
            else self.resolved_little_group.factor_system
        )

    @property
    def conjugacy_classes(self) -> tuple[tuple[str, ...], ...]:
        return _resolved_conjugacy_classes(self.resolved_little_group)

    @property
    def finite_group_mapping(self) -> tuple[tuple[str, str], ...]:
        return _resolved_finite_mapping(self.resolved_little_group)


@dataclass(frozen=True)
class BlochSymmetryAnalysisResult:
    points: tuple[BlochSymmetryPointAnalysis, ...]

    def point(self, name: str) -> BlochSymmetryPointAnalysis:
        matches = tuple(point for point in self.points if point.name == name)
        if not matches:
            raise KeyError(f"Unknown Bloch symmetry analysis point: {name!r}.")
        if len(matches) != 1:
            raise RuntimeError(f"Bloch symmetry analysis point name {name!r} is ambiguous.")
        return matches[0]


@dataclass(frozen=True)
class TargetCompatibilityAnalysis:
    point_name: str
    target_names: tuple[str, ...]
    target_unitary_characters: dict[str, complex]
    target_decomposition: IrrepDecomposition | None
    compatibility: RepresentationCompatibility | None
    target_twisted_representation: TwistedRepresentation | None
    intertwiner_dimension: int | None


@dataclass(frozen=True)
class SymmetryAnalysisResult:
    physical: BlochSymmetryAnalysisResult
    target_compatibilities: tuple[TargetCompatibilityAnalysis, ...] = ()

    def target_compatibility(self, point_name: str) -> TargetCompatibilityAnalysis:
        matches = tuple(
            result
            for result in self.target_compatibilities
            if result.point_name == point_name
        )
        if not matches:
            raise KeyError(f"No target compatibility was requested at point {point_name!r}.")
        if len(matches) != 1:
            raise RuntimeError(f"Target compatibility point name {point_name!r} is ambiguous.")
        return matches[0]


@dataclass(frozen=True)
class GammaZeroRegularizationAnalysis:
    point_name: str
    transverse_band_indices: tuple[int, ...]
    longitudinal_zero_band_indices: tuple[int, ...]
    sewing_matrices: dict[str, np.ndarray]
    unitary_characters: dict[str, complex]
    decomposition: IrrepDecomposition | None
    unitarity_error: float
    twisted_composition_residual: float
    note: str


def regularize_gamma_zero_modes(
    physical: BlochSymmetryAnalysisResult,
    context: SymmetryContext,
    longitudinal_zero_band_indices,
    real_lattice_vectors,
    *,
    energy_tolerance: float,
) -> tuple[BlochSymmetryAnalysisResult, GammaZeroRegularizationAnalysis]:
    """Replace the singular transverse Gamma label by the constant axial T+L space."""

    if not np.isfinite(energy_tolerance) or energy_tolerance < 0.0:
        raise ValueError("Gamma zero-mode tolerance must be finite and non-negative.")
    gamma_points = tuple(
        point
        for point in physical.points
        if np.max(
            np.abs(periodic_difference(point.requested_k_fractional, np.zeros(context.model.dimension)))
        ) <= context.model.tolerance
    )
    if len(gamma_points) != 1:
        raise ValueError(
            "gamma_zero_regularization requires exactly one Gamma representation-analysis point."
        )
    gamma = gamma_points[0]
    zero_blocks = tuple(
        block
        for block in gamma.degenerate_blocks
        if block.energies
        and max(abs(complex(value)) for value in block.energies) <= energy_tolerance
    )
    transverse_bands = tuple(
        band for block in zero_blocks for band in block.band_indices
    )
    longitudinal_bands = tuple(
        int(value) for value in longitudinal_zero_band_indices
    )
    if len(transverse_bands) != 2:
        raise ValueError(
            "Gamma zero regularization requires exactly two physical transverse zero modes; "
            f"found bands(0-based) {tuple(transverse_bands)}."
        )
    if len(longitudinal_bands) != 1:
        raise ValueError(
            "Gamma zero regularization requires exactly one auxiliary longitudinal zero mode; "
            f"found bands(0-based) {tuple(longitudinal_bands)}."
        )
    resolved = gamma.resolved_little_group
    if resolved is None:
        raise ValueError("Gamma zero regularization requires a resolved little group.")
    operation_indices = gamma.little_group_operation_indices
    group = context.model.group
    lattice = np.asarray(real_lattice_vectors, dtype=float)
    matrices_by_operation = {}
    matrices = {}
    for operation_index in operation_indices:
        operation = group.operations[operation_index]
        matrix = cartesian_field_matrix(
            operation,
            lattice,
            FieldKind.MAGNETIC_AXIAL_VECTOR,
            context.model.tolerance,
        )
        if operation.antiunitary:
            matrix = -matrix
        matrix = np.asarray(matrix, dtype=np.complex128)
        matrices_by_operation[operation_index] = matrix
        matrices[_operation_name(group, operation_index)] = matrix
    twisted = build_twisted_representation(
        resolved,
        operation_indices,
        tuple(matrices_by_operation[index] for index in operation_indices),
    )
    unitary_characters = {
        _operation_name(group, index): complex(np.trace(matrices_by_operation[index]))
        for index in operation_indices
        if not group.operations[index].antiunitary
    }
    decomposition = (
        None
        if any(resolved.factor_system.antiunitary_flags)
        else decompose_little_group_characters(resolved, unitary_characters)
    )
    identity = np.eye(3, dtype=np.complex128)
    unitarity = max(
        float(np.linalg.norm(matrix.conj().T @ matrix - identity, ord="fro"))
        for matrix in matrices.values()
    )
    reason = (
        "Gamma zero-frequency transverse subspace is direction dependent; "
        "the ordinary irrep label is intentionally suppressed."
    )
    zero_band_blocks = {block.band_indices for block in zero_blocks}
    replaced_blocks = tuple(
        replace(block, decomposition=None, irrep_unavailable_reason=reason)
        if block.band_indices in zero_band_blocks
        else block
        for block in gamma.degenerate_blocks
    )
    replaced_gamma = replace(
        gamma,
        degenerate_blocks=replaced_blocks,
        physical_decomposition=None,
    )
    replaced_points = tuple(
        replaced_gamma if point is gamma else point for point in physical.points
    )
    note = (
        "The reported representation belongs to the three-dimensional constant axial "
        "T+L space, not to the two physical transverse bands alone."
    )
    return (
        BlochSymmetryAnalysisResult(replaced_points),
        GammaZeroRegularizationAnalysis(
            gamma.name,
            transverse_bands,
            longitudinal_bands,
            matrices,
            unitary_characters,
            decomposition,
            unitarity,
            twisted.product_residual,
            note,
        ),
    )


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
    cochain = little_group_definition.factor_system.trivializing_cochain
    if cochain is not None:
        ordinary_values = cochain * physical_values
        for conjugacy_class in table.conjugacy_classes:
            class_names = tuple(
                table.element_names[index] for index in conjugacy_class.element_indices
            )
            values = [ordinary_values[index] for index in conjugacy_class.element_indices]
            average = sum(values) / len(values)
            label = "{" + ",".join(str(name) for name in class_names) + "}"
            class_residuals[label] = max(
                (abs(value - average) for value in values),
                default=0.0,
            )
    return IrrepDecomposition(raw, rounded, residuals, class_residuals)


def _character_decomposition_error(
    little_group_definition: ResolvedLittleGroup,
    physical_characters: dict[str, complex],
    decomposition: IrrepDecomposition,
) -> float:
    """Maximum trace mismatch after reconstructing the rounded irrep content."""

    table = little_group_definition.table
    observed = np.asarray(
        [physical_characters[name] for name in table.operation_names],
        dtype=np.complex128,
    )
    reconstructed = np.zeros(table.order, dtype=np.complex128)
    for irrep in little_group_definition.require_irreps():
        multiplicity = decomposition.multiplicities.get(irrep.name, 0)
        if multiplicity:
            reconstructed += multiplicity * np.asarray(
                irrep.characters, dtype=np.complex128
            )
    return float(np.max(np.abs(observed - reconstructed), initial=0.0))


def _credible_approximate_decomposition(
    little_group_definition: ResolvedLittleGroup,
    decomposition: IrrepDecomposition,
    physical_dimension: int,
    character_fit_error: float,
    tolerance: float,
) -> bool:
    """Accept a character-pattern label without promoting it to a valid representation."""

    dimensions = {
        irrep.name: irrep.dimension for irrep in little_group_definition.require_irreps()
    }
    multiplicities = decomposition.multiplicities
    if any(value < 0 for value in multiplicities.values()):
        return False
    reconstructed_dimension = sum(
        dimensions[name] * value for name, value in multiplicities.items()
    )
    return bool(
        reconstructed_dimension == physical_dimension
        and character_fit_error <= tolerance
        and decomposition.max_residual <= tolerance
    )


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
    character_tolerance: float | None = None,
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
    character = (
        1.0e-2
        if character_tolerance is None and spec is None
        else spec.character_tolerance
        if character_tolerance is None
        else float(character_tolerance)
    )
    if not np.isfinite(character) or character <= 0.0:
        raise ValueError("Bloch-symmetry character tolerance must be positive and finite.")
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
        character_tolerance=character,
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
    from ..compute.parallel import parallel_map

    def analyze(point: RepresentationPointSpec) -> BlochSymmetryPointAnalysis:
        return _analyze_bloch_point(
            state,
            context,
            physical_provider,
            point,
            split_degenerate_blocks=True,
            leakage_tolerance=spec.leakage_tolerance,
            character_tolerance=spec.character_tolerance,
        )

    get_block = getattr(state, "get_block", None)
    sample_bytes = (
        max(int(np.asarray(get_block(0, 0, 0)).nbytes) * 4, 1 << 20)
        if callable(get_block)
        else 1 << 20
    )
    threads = max(
        1,
        int(
            getattr(
                state,
                "configured_threads",
                getattr(getattr(state, "config", None), "threads", 1),
            )
        ),
    )
    return BlochSymmetryAnalysisResult(
        tuple(
            parallel_map(
                spec.points,
                analyze,
                threads,
                ordered=True,
                bytes_per_task=sample_bytes,
            )
        )
    )


def run_symmetry_analysis(
    state: StateCollection,
    context: SymmetryContext,
    *,
    provider: StateBlochSymmetryProvider | None = None,
) -> SymmetryAnalysisResult:
    """Run physical analysis and optional, explicitly requested target comparisons."""

    physical = run_bloch_symmetry_analysis(state, context, provider=provider)
    spec = context.model.representation_analysis
    if spec is None:
        return SymmetryAnalysisResult(physical)
    point_specs = {point.name: point for point in spec.points}
    comparisons = []
    for physical_point in physical.points:
        comparison = _analyze_target_compatibility(
            context, point_specs[physical_point.name], physical_point
        )
        if comparison is not None:
            comparisons.append(comparison)
    return SymmetryAnalysisResult(physical, tuple(comparisons))


def _analyze_bloch_point(
    state: StateCollection,
    context: SymmetryContext,
    provider: StateBlochSymmetryProvider,
    point: RepresentationPointSpec,
    *,
    split_degenerate_blocks: bool,
    leakage_tolerance: float,
    character_tolerance: float,
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
    matrices_by_operation: dict[int, np.ndarray] = {}
    for element in elements:
        mapping = provider.mapping(element.operation_index, k_index)
        operation = group.operations[element.operation_index]
        request = provider.request_for_mapping(mapping, available, source_k_fractional=point.k_fractional)
        full_matrix = _validated_sewing(provider.sewing_matrix(request), len(available), operation)
        full_matrices[_operation_name(group, element.operation_index)] = full_matrix
        band_basis = getattr(provider, "sewing_matrix_in_band_basis", None)
        matrix = (
            full_matrix[
                np.ix_(
                    [available.index(band) for band in bands],
                    [available.index(band) for band in bands],
                )
            ].copy()
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
    selected_leakage, _ = _band_basis_subspace_leakage(
        provider,
        elements,
        k_index,
        point.k_fractional,
        full_matrices,
        available,
        bands,
        leakage_tolerance,
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
        outer_composition_residual=_composition_residual(
            provider, point.k_fractional, k_index, available, operation_indices, full_matrices
        ),
        selected_twisted_composition_residual=(
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
        leakage, coupled = _band_basis_subspace_leakage(
            provider,
            elements,
            k_index,
            point.k_fractional,
            full_matrices,
            available,
            block,
            leakage_tolerance,
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
        unitarity_error = max(
            (
                float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(len(block)), ord="fro"))
                for matrix in block_matrices.values()
            ),
            default=0.0,
        )
        unavailable_reason = _irrep_unavailable_reason(
            resolved_little_group,
            leakage,
            leakage_tolerance,
            unitarity_error,
            twisted_residual,
        )
        decomposition = None
        approximate_decomposition = None
        character_fit_error = None
        if (
            resolved_little_group is not None
            and not any(resolved_little_group.factor_system.antiunitary_flags)
        ):
            character_decomposition = decompose_little_group_characters(
                resolved_little_group, block_unitary_characters
            )
            character_fit_error = _character_decomposition_error(
                resolved_little_group,
                block_unitary_characters,
                character_decomposition,
            )
            if unavailable_reason is None:
                decomposition = character_decomposition
            elif _credible_approximate_decomposition(
                resolved_little_group,
                character_decomposition,
                len(block),
                character_fit_error,
                character_tolerance,
            ):
                approximate_decomposition = character_decomposition
        candidates = _candidate_excluded_bands(
            energy_line, available, block, point.degeneracy_tolerance
        )
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
                leakage=leakage,
                decomposition=decomposition,
                approximate_decomposition=approximate_decomposition,
                character_fit_error=character_fit_error,
                unitary_characters=block_unitary_characters,
                antiunitary_diagnostics=antiunitary_diagnostics,
                coupled_outer_bands=coupled,
                candidate_excluded_bands=candidates,
                irrep_unavailable_reason=unavailable_reason,
                unitarity_error=unitarity_error,
                twisted_composition_residual=twisted_residual,
            )
        )

    if (
        resolved_little_group is not None
        and not any(resolved_little_group.factor_system.antiunitary_flags)
        and diagnostics.leakage <= leakage_tolerance
        and diagnostics.unitarity_error <= leakage_tolerance
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
    if diagnostics.leakage > leakage_tolerance:
        LOGGER.debug(
            "Selected Bloch symmetry subspace at %s is not closed: "
            "bands(0-based)=%s leakage=%.6g coupled_outer_bands(0-based)=%s",
            point.name,
            tuple(bands),
            diagnostics.leakage,
            tuple(band for block in block_results for band in block.coupled_outer_bands),
        )
    if outer_unitarity > leakage_tolerance:
        LOGGER.debug(
            "Outer Bloch window at %s is not closed: outer_bands(0-based)=%s "
            "unitarity=%.6g candidate_excluded_bands(0-based)=%s",
            point.name,
            tuple(available),
            outer_unitarity,
            tuple(outer_candidates),
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
        resolved_little_group=resolved_little_group,
        unitary_subgroup_name=unitary_subgroup_name,
        unitary_operation_names=tuple(_operation_name(group, index) for index in unitary_indices),
        antiunitary_operation_names=tuple(
            _operation_name(group, index) for index in antiunitary_indices
        ),
        physical_twisted_representation=physical_twisted,
        outer_unitarity_error=outer_unitarity,
        outer_candidate_excluded_bands=outer_candidates,
    )


def _analyze_target_compatibility(
    context: SymmetryContext,
    point: RepresentationPointSpec,
    physical: BlochSymmetryPointAnalysis,
) -> TargetCompatibilityAnalysis | None:
    group = context.model.group
    operation_indices = physical.little_group_operation_indices
    resolved_little_group = physical.resolved_little_group
    targets = _selected_targets(context, point)
    if not targets:
        return None
    analysis_spec = context.model.representation_analysis
    if analysis_spec is None:
        raise RuntimeError("Target compatibility requires representation-analysis settings.")
    invalid = [
        block
        for block in physical.degenerate_blocks
        if block.leakage > analysis_spec.leakage_tolerance
        or block.unitarity_error > analysis_spec.leakage_tolerance
        or block.twisted_composition_residual > analysis_spec.leakage_tolerance
    ]
    if invalid:
        block = invalid[0]
        LOGGER.debug(
            "Degenerate block %s at representation point %s is not closed: "
            "leakage=%.6g unitarity=%.6g twisted_composition=%.6g. "
            "Its irrep label is unavailable, but compatibility of the complete selected "
            "Bloch space will still be analyzed when that space is valid.",
            block.band_indices,
            point.name,
            block.leakage,
            block.unitarity_error,
            block.twisted_composition_residual,
        )
    target_matrices_by_operation = {
        operation_index: context.target_matrix(
            operation_index,
            point.k_fractional,
            targets=targets,
        )
        for operation_index in operation_indices
    }
    target_unitary_characters = {
        _operation_name(group, operation_index): complex(
            np.trace(target_matrices_by_operation[operation_index])
        )
        for operation_index in operation_indices
        if not group.operations[operation_index].antiunitary
    }
    target_twisted = (
        None
        if resolved_little_group is None
        else build_twisted_representation(
            resolved_little_group,
            operation_indices,
            tuple(target_matrices_by_operation[index] for index in operation_indices),
        )
    )
    physical_twisted = physical.physical_twisted_representation
    if target_twisted is not None:
        target_twisted.require_valid(tolerance=context.model.algebra_tolerance)
        if physical_twisted is None:
            raise RuntimeError("Target twisted representation has no physical counterpart.")
        physical_twisted.assert_compatible(target_twisted)

    if (
        resolved_little_group is not None
        and not any(resolved_little_group.factor_system.antiunitary_flags)
    ):
        target_decomposition = decompose_little_group_characters(
            resolved_little_group, target_unitary_characters
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

        gauge_spec = context.model.symmetry_gauge
        numerical_tolerance = (
            analysis_spec.leakage_tolerance
            if gauge_spec is None
            else gauge_spec.tolerance
        )
        numerical_tolerance = max(
            float(numerical_tolerance),
            float(physical_twisted.unitarity_error),
            float(physical_twisted.product_residual),
            float(physical_twisted.cocycle_residual),
        )
        svd_tolerance = (
            context.model.algebra_tolerance
            if gauge_spec is None
            else gauge_spec.svd_relative_tolerance
        )
        intertwiner_dimension = solve_intertwiner_space(
            physical_twisted,
            target_twisted,
            relative_tolerance=svd_tolerance,
            absolute_tolerance=numerical_tolerance,
            representation_tolerance=numerical_tolerance,
        ).dimension
    return TargetCompatibilityAnalysis(
        point.name,
        tuple(target.name for target in targets),
        target_unitary_characters,
        target_decomposition,
        compatibility,
        target_twisted,
        intertwiner_dimension,
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


def _selected_targets(
    context: SymmetryContext,
    point: RepresentationPointSpec,
) -> tuple[WannierTargetRepresentation, ...]:
    if point.target_names is None:
        return ()
    return tuple(context.model.target(name) for name in point.target_names)


def _band_basis_subspace_leakage(
    provider: StateBlochSymmetryProvider,
    elements: tuple[LittleGroupElement, ...],
    k_index: tuple[int, ...],
    kpoint,
    full_matrices: dict[str, np.ndarray],
    outer_bands: tuple[int, ...],
    selected_bands: tuple[int, ...],
    tolerance: float,
) -> tuple[float, tuple[int, ...]]:
    outside_bands = tuple(band for band in outer_bands if band not in selected_bands)
    if not outside_bands:
        return 0.0, ()
    band_basis = getattr(provider, "sewing_matrix_in_band_basis", None)
    if band_basis is None:
        return _matrix_subspace_leakage(
            full_matrices, outer_bands, selected_bands, tolerance
        )

    group = provider.context.model.group
    maximum = 0.0
    coupled: set[int] = set()
    for element in elements:
        operation = group.operations[element.operation_index]
        mapping = provider.mapping(element.operation_index, k_index)
        outside_from_selected = np.asarray(
            band_basis(
                mapping,
                selected_bands,
                outside_bands,
                operation=operation,
                source_k_fractional=kpoint,
            ),
            dtype=np.complex128,
        )
        selected_from_outside = np.asarray(
            band_basis(
                mapping,
                outside_bands,
                selected_bands,
                operation=operation,
                source_k_fractional=kpoint,
            ),
            dtype=np.complex128,
        )
        leakage = float(
            np.sqrt(
                np.linalg.norm(outside_from_selected, ord="fro") ** 2
                + np.linalg.norm(selected_from_outside, ord="fro") ** 2
            )
        )
        maximum = max(maximum, leakage)
        for position, band in enumerate(outside_bands):
            strength = float(
                np.sqrt(
                    np.linalg.norm(outside_from_selected[position, :]) ** 2
                    + np.linalg.norm(selected_from_outside[:, position]) ** 2
                )
            )
            if strength > tolerance:
                coupled.add(band)
    return maximum, tuple(sorted(coupled))


def _matrix_subspace_leakage(
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
    unitarity_error: float,
    twisted_residual: float,
) -> str | None:
    if resolved_little_group is None:
        return "little co-group could not be identified"
    factor = resolved_little_group.factor_system
    if any(factor.antiunitary_flags):
        return "magnetic corepresentation labels are unavailable"
    if leakage > tolerance:
        return f"band subspace leakage {leakage:.6g} exceeds {tolerance:.6g}"
    if unitarity_error > tolerance:
        return f"unitarity residual {unitarity_error:.6g} exceeds {tolerance:.6g}"
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
