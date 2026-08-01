from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..maxwell import FieldKind
from ..symmetry.analysis import (
    BlochSymmetryAnalysisResult,
    BlochSymmetryPointAnalysis,
    decompose_little_group_characters,
)
from ..symmetry.field_action import cartesian_field_matrix
from ..symmetry.group import build_crystallographic_orbit, little_group, periodic_difference
from ..symmetry.representation import (
    SymmetryContext,
    build_wannier_target_from_group_irrep,
)
from .catalog import infer_builtin_catalog_alias, load_ebr_catalog
from .models import (
    BandSymmetryVector,
    EBRAnalysisResult,
    EBRCatalog,
    EBRMatrix,
    SymmetryVectorKey,
)
from .solver import decompose_ebr, enumerate_tetb_decompositions

if TYPE_CHECKING:
    from ..config import IncarConfig
    from ..data import BlochSymmetryRunResult


def build_band_symmetry_vector(
    analysis: BlochSymmetryAnalysisResult,
    catalog: EBRCatalog,
    *,
    tolerance: float = 1.0e-5,
) -> BandSymmetryVector:
    """Build an exact regular-band symmetry vector in catalog point order."""

    rows: list[SymmetryVectorKey] = []
    values: list[int] = []
    dimensions = []
    for catalog_point in catalog.k_points:
        point = _matching_analysis_point(analysis, catalog_point, tolerance)
        _require_unitary_point(point)
        decomposition = point.physical_decomposition
        if decomposition is None:
            raise ValueError(
                f"Representation point {point.name!r} has no exact physical decomposition; "
                "approximate or leakage-limited irrep labels cannot be used for EBR analysis."
            )
        if decomposition.max_residual > tolerance:
            raise ValueError(
                f"Representation point {point.name!r} decomposition residual "
                f"{decomposition.max_residual:.6g} exceeds {tolerance:.6g}."
            )
        resolved = point.resolved_little_group
        if resolved is None:
            raise ValueError(
                f"Representation point {point.name!r} has no resolved little group."
            )
        for irrep in resolved.require_irreps():
            irrep_name = irrep.name
            multiplicity = decomposition.multiplicities.get(irrep_name, 0)
            if multiplicity < 0:
                raise ValueError(
                    f"Physical representation at {point.name!r} has negative irrep multiplicity."
                )
            rows.append(SymmetryVectorKey(catalog_point.name, irrep_name))
            values.append(int(multiplicity))
        dimensions.append(len(point.band_indices))
    if len(set(dimensions)) != 1:
        raise ValueError(
            "EBR analysis points select different band dimensions: "
            f"{dict(zip((point.name for point in catalog.k_points), dimensions))}."
        )
    return BandSymmetryVector(tuple(rows), np.asarray(values), dimensions[0])


def build_ebr_matrix(
    context: SymmetryContext,
    catalog: EBRCatalog,
    *,
    lattice_vectors=None,
) -> EBRMatrix:
    """Generate EBR columns from existing induced target representations."""

    _validate_catalog_context(context, catalog)
    definition = context.model.group_definition
    if definition is None:
        raise ValueError("Dynamic EBR generation requires a resolved space-group definition.")
    if any(operation.antiunitary for operation in context.model.group.operations):
        raise NotImplementedError(
            "EBR catalogs for magnetic corepresentations are not implemented."
        )
    point_data = []
    row_keys = []
    for point in catalog.k_points:
        elements = little_group(context.model.group, point.k_fractional)
        operation_indices = tuple(element.operation_index for element in elements)
        resolved = definition.resolve_little_group(
            operation_indices,
            point.k_fractional,
            bloch_convention=context.model.bloch_convention,
        )
        irreps = resolved.require_irreps()
        point_data.append((point, operation_indices, resolved, irreps))
        row_keys.extend(SymmetryVectorKey(point.name, irrep.name) for irrep in irreps)

    columns = []
    dimensions = []
    values = np.zeros((len(row_keys), len(catalog.ebrs)), dtype=np.int64)
    row_positions = {key: index for index, key in enumerate(row_keys)}
    for column_index, ebr in enumerate(catalog.ebrs):
        orbit = build_crystallographic_orbit(context.model.group, ebr.center)
        declared_multiplicity = _wyckoff_multiplicity(ebr.wyckoff)
        if declared_multiplicity != orbit.multiplicity:
            raise ValueError(
                f"EBR {ebr.name!r} declares Wyckoff {ebr.wyckoff!r}, but its center "
                f"has orbit multiplicity {orbit.multiplicity}."
            )
        if lattice_vectors is not None and definition.dimension == 3:
            definition.validate_wyckoff(ebr.wyckoff, ebr.center, lattice_vectors)
        site_indices = tuple(
            element.source_operation_index for element in orbit.site_symmetry.elements
        )
        site_irrep = definition.site_irrep(site_indices, ebr.site_irrep)
        target = build_wannier_target_from_group_irrep(
            ebr.name,
            context.model.group,
            ebr.center,
            site_irrep,
            context.model.bloch_convention,
        )
        dimensions.append(target.wannier_dimension)
        columns.append(ebr)
        for point, operation_indices, resolved, irreps in point_data:
            characters = {
                context.model.group.operations[index].name or f"g{index}": complex(
                    np.trace(target.matrix(index, point.k_fractional))
                )
                for index in operation_indices
            }
            decomposition = decompose_little_group_characters(resolved, characters)
            threshold = max(100.0 * context.model.algebra_tolerance, 1.0e-7)
            if decomposition.max_residual > threshold:
                raise ValueError(
                    f"Generated EBR {ebr.name!r} has a non-integral decomposition at "
                    f"{point.name!r} (residual={decomposition.max_residual:.6g})."
                )
            represented_dimension = 0
            for irrep in irreps:
                multiplicity = int(decomposition.multiplicities[irrep.name])
                if multiplicity < 0:
                    raise ValueError(
                        f"Generated EBR {ebr.name!r} has negative multiplicity at {point.name!r}."
                    )
                values[row_positions[SymmetryVectorKey(point.name, irrep.name)], column_index] = multiplicity
                represented_dimension += multiplicity * irrep.dimension
            if represented_dimension != target.wannier_dimension:
                raise ValueError(
                    f"Generated EBR {ebr.name!r} has dimension {represented_dimension} at "
                    f"{point.name!r}, expected {target.wannier_dimension}."
                )
    return EBRMatrix(
        tuple(row_keys),
        tuple(columns),
        values,
        np.asarray(dimensions, dtype=np.int64),
    )


def run_ebr_analysis(
    analysis: BlochSymmetryRunResult | BlochSymmetryAnalysisResult,
    context: SymmetryContext,
    config: IncarConfig,
) -> EBRAnalysisResult:
    """Generate the requested EBR problem and solve it without Wannier calculations."""

    physical_analysis = (
        analysis.primary.analysis if hasattr(analysis, "primary") else analysis
    )
    if not isinstance(physical_analysis, BlochSymmetryAnalysisResult):
        raise TypeError("run_ebr_analysis expects a Bloch symmetry analysis result.")
    catalog_reference = str(config.ebr_catalog).strip()
    if catalog_reference.casefold() == "auto":
        catalog_reference = infer_builtin_catalog_alias(
            _context_space_group_number(context)
        )
    catalog = load_ebr_catalog(catalog_reference, base_dir=config.base_dir)
    matrix = build_ebr_matrix(
        context,
        catalog,
        lattice_vectors=config.real_lattice_vectors,
    )
    mode = str(config.ebr_mode).strip().lower()
    if mode == "auto":
        mode = "regular" if config.wannier_subspace == "T+L" else "transverse"
    diagnostics = []
    if mode == "regular":
        vector = build_band_symmetry_vector(physical_analysis, catalog)
        vector = vector.reordered(matrix.row_keys)
        regular = decompose_ebr(
            vector,
            matrix,
            max_states=config.ebr_max_states,
        )
        if not regular:
            diagnostics.append("No non-negative EBR decomposition was found.")
        return EBRAnalysisResult(
            mode,
            catalog,
            vector,
            matrix,
            regular_decompositions=regular,
            diagnostics=tuple(diagnostics),
        )

    field_kind = (
        analysis.primary.field_kind
        if hasattr(analysis, "primary")
        else config.maxwell_problem.symmetry_field_kind
    )
    vector = _build_transverse_symmetry_vector(
        physical_analysis,
        context,
        catalog,
        zero_tolerance=config.gamma_zero_mode_tolerance,
    ).reordered(matrix.row_keys)
    auxiliary_rows = _auxiliary_gamma_rows(
        context,
        catalog,
        matrix,
        config.real_lattice_vectors,
        field_kind,
    )
    solutions = enumerate_tetb_decompositions(
        vector,
        matrix,
        gamma_point_name=catalog.gamma_point.name,
        max_auxiliary_bands=config.ebr_max_auxiliary_bands,
        max_states=config.ebr_max_states,
        required_auxiliary_gamma_rows=auxiliary_rows,
    )
    physical = tuple(solution for solution in solutions if solution.physical)
    optimal = min((solution.auxiliary_dimension for solution in physical), default=None)
    if optimal is None:
        diagnostics.append(
            "No physical TETB decomposition was found within the auxiliary-band limit."
        )
    return EBRAnalysisResult(
        mode,
        catalog,
        vector,
        matrix,
        tetb_solutions=solutions,
        optimal_auxiliary_dimension=optimal,
        diagnostics=tuple(diagnostics),
    )


def _build_transverse_symmetry_vector(
    analysis: BlochSymmetryAnalysisResult,
    context: SymmetryContext,
    catalog: EBRCatalog,
    *,
    zero_tolerance: float,
) -> BandSymmetryVector:
    regular_rows = []
    regular_values = []
    selected_dimensions = []
    for catalog_point in catalog.k_points:
        point = _matching_analysis_point(
            analysis, catalog_point, context.model.tolerance
        )
        _require_unitary_point(point)
        selected_dimensions.append(len(point.band_indices))
        if catalog_point.name != catalog.gamma_point.name:
            decomposition = point.physical_decomposition
            if decomposition is None:
                raise ValueError(
                    f"Non-Gamma representation point {point.name!r} has no exact physical "
                    "decomposition and cannot be used for TETB enumeration."
                )
            point_values = decomposition.multiplicities
        else:
            point_values = _transverse_gamma_multiplicities(
                point, zero_tolerance=zero_tolerance
            )
        resolved = point.resolved_little_group
        if resolved is None:
            raise ValueError(
                f"Representation point {point.name!r} has no resolved little group."
            )
        for irrep in resolved.require_irreps():
            irrep_name = irrep.name
            multiplicity = point_values.get(irrep_name, 0)
            regular_rows.append(SymmetryVectorKey(catalog_point.name, irrep_name))
            regular_values.append(int(multiplicity))
    if len(set(selected_dimensions)) != 1:
        raise ValueError("Transverse representation-analysis points select different band counts.")
    return BandSymmetryVector(
        tuple(regular_rows),
        np.asarray(regular_values),
        selected_dimensions[0],
    )


def _transverse_gamma_multiplicities(
    point: BlochSymmetryPointAnalysis,
    *,
    zero_tolerance: float,
) -> dict[str, int]:
    resolved = point.resolved_little_group
    if resolved is None:
        raise ValueError("Transverse Gamma construction requires a resolved little group.")
    zero_blocks = tuple(
        block
        for block in point.degenerate_blocks
        if block.energies
        and max(abs(complex(value)) for value in block.energies) <= zero_tolerance
    )
    zero_dimension = sum(len(block.band_indices) for block in zero_blocks)
    if zero_dimension != 2:
        if point.physical_decomposition is not None:
            return dict(point.physical_decomposition.multiplicities)
        raise ValueError(
            "Transverse EBR analysis requires exactly two unresolved zero-frequency Gamma "
            f"modes; found dimension {zero_dimension}."
        )
    # The two transverse zero modes do not carry a direction-independent Gamma
    # representation.  Keep their Gamma rows empty and let n_T+L - n_L produce
    # the formal Gamma surrogate.  Positive-frequency Gamma blocks remain exact.
    output = {irrep.name: 0 for irrep in resolved.require_irreps()}
    zero_band_sets = {block.band_indices for block in zero_blocks}
    for block in point.degenerate_blocks:
        if block.band_indices in zero_band_sets:
            continue
        if block.decomposition is None:
            raise ValueError(
                f"Nonzero Gamma block {block.band_indices} has no exact irrep decomposition."
            )
        for name, multiplicity in block.decomposition.multiplicities.items():
            output[name] = output.get(name, 0) + int(multiplicity)
    return output


def _field_representation_decomposition(
    point: BlochSymmetryPointAnalysis,
    context: SymmetryContext,
    field_kind: FieldKind,
    lattice_vectors,
):
    resolved = point.resolved_little_group
    if resolved is None:
        raise ValueError("Field representation requires a resolved little group.")
    return _field_representation_decomposition_resolved(
        resolved,
        point.little_group_operation_indices,
        context,
        field_kind,
        lattice_vectors,
    )


def _field_representation_decomposition_resolved(
    resolved,
    operation_indices,
    context: SymmetryContext,
    field_kind: FieldKind,
    lattice_vectors,
):
    characters = {}
    for operation_index in operation_indices:
        operation = context.model.group.operations[operation_index]
        name = operation.name or f"g{operation_index}"
        characters[name] = complex(
            np.trace(
                cartesian_field_matrix(
                    operation,
                    lattice_vectors,
                    field_kind,
                    context.model.tolerance,
                )
            )
        )
    return decompose_little_group_characters(resolved, characters)


def _auxiliary_gamma_rows(
    context: SymmetryContext,
    catalog: EBRCatalog,
    matrix: EBRMatrix,
    lattice_vectors,
    field_kind: FieldKind,
) -> tuple[int, ...]:
    elements = little_group(context.model.group, catalog.gamma_point.k_fractional)
    indices = tuple(element.operation_index for element in elements)
    resolved = context.model.group_definition.resolve_little_group(
        indices,
        catalog.gamma_point.k_fractional,
        bloch_convention=context.model.bloch_convention,
    )
    if field_kind == FieldKind.ELECTRIC_POLAR_VECTOR:
        auxiliary_kind = FieldKind.SCALAR
    elif field_kind == FieldKind.MAGNETIC_AXIAL_VECTOR:
        auxiliary_kind = FieldKind.PSEUDOSCALAR
    else:
        raise NotImplementedError(
            "TETB transverse analysis requires a 3D polar-electric or axial-magnetic field."
        )
    decomposition = _field_representation_decomposition_resolved(
        resolved,
        indices,
        context,
        auxiliary_kind,
        lattice_vectors,
    )
    names = {
        name
        for name, multiplicity in decomposition.multiplicities.items()
        if multiplicity > 0
    }
    return tuple(
        index
        for index, key in enumerate(matrix.row_keys)
        if key.point_name == catalog.gamma_point.name and key.irrep_name in names
    )


def _matching_analysis_point(
    analysis: BlochSymmetryAnalysisResult,
    catalog_point,
    tolerance: float,
) -> BlochSymmetryPointAnalysis:
    requested = np.asarray(catalog_point.k_fractional, dtype=float)
    matches = tuple(
        point
        for point in analysis.points
        if point.requested_k_fractional.shape == requested.shape
        and np.max(np.abs(periodic_difference(point.requested_k_fractional, requested)))
        <= tolerance
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Representation-analysis point at {requested.tolist()} is ambiguous."
        )
    named = tuple(
        point
        for point in analysis.points
        if point.name.casefold() == catalog_point.name.casefold()
    )
    if len(named) == 1 and named[0].requested_k_fractional.shape == requested.shape:
        return named[0]
    raise ValueError(
        f"EBR catalog point {catalog_point.name!r} at {requested.tolist()} is not covered "
        "by a unique representation_analysis point with the same coordinate or name."
    )


def _require_unitary_point(point: BlochSymmetryPointAnalysis) -> None:
    if point.antiunitary_operation_names:
        raise NotImplementedError(
            f"Representation point {point.name!r} contains antiunitary operations; magnetic "
            "corepresentation EBR catalogs are not implemented."
        )


def _validate_catalog_context(context: SymmetryContext, catalog: EBRCatalog) -> None:
    definition = context.model.group_definition
    if definition is None or definition.hall_number is None:
        raise ValueError("EBR analysis requires a space group with an unambiguous Hall setting.")
    if int(definition.hall_number) != catalog.hall_number:
        raise ValueError(
            f"EBR catalog Hall {catalog.hall_number} does not match calculation Hall "
            f"{definition.hall_number}."
        )
    if catalog.dimension != context.model.dimension:
        raise ValueError("EBR catalog dimension does not match the symmetry context.")
    actual_number = _context_space_group_number(context)
    if actual_number != catalog.space_group_number:
        raise ValueError(
            f"EBR catalog space group {catalog.space_group_number} does not match "
            f"calculation space group {actual_number}."
        )


def _context_space_group_number(context: SymmetryContext) -> int:
    definition = context.model.group_definition
    if definition is None or definition.hall_number is None:
        raise ValueError("Automatic EBR catalog selection requires a Hall-number space group.")
    import spglib

    group_type = spglib.get_spacegroup_type(int(definition.hall_number))
    if group_type is None:
        raise ValueError(f"spglib could not resolve Hall {definition.hall_number}.")
    return int(group_type.number)


def _wyckoff_multiplicity(label: str) -> int:
    digits = ""
    for character in str(label).strip():
        if not character.isdigit():
            break
        digits += character
    if not digits:
        raise ValueError(f"Wyckoff label {label!r} does not begin with a multiplicity.")
    return int(digits)
