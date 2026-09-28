from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, reduce
from itertools import product
from math import gcd
from pathlib import Path
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
    EBRDefinition,
    EBRMatrix,
    EBRSearchStatistics,
    EBRSubspaceCandidate,
    EBRSubspacePointRepresentation,
    SymmetryVectorKey,
)
from .solver import (
    decompose_ebr,
    enumerate_ebr_subspace_solutions,
    enumerate_tetb_decompositions,
)

if TYPE_CHECKING:
    from ..config import IncarConfig
    from ..data import BlochSymmetryRunResult


@dataclass(frozen=True)
class _EBRDecompositionSelection:
    multiplicities: dict[str, int]
    max_residual: float
    approximate_blocks: tuple[tuple[int, ...], ...] = ()


def _ebr_block_decomposition(block):
    """Return an integer irrep label suitable for discrete EBR counting."""

    exact = getattr(block, "decomposition", None)
    if exact is not None:
        return exact, False
    approximate = getattr(block, "approximate_decomposition", None)
    if approximate is not None:
        return approximate, True
    return None, False


def _ebr_point_decomposition(point) -> _EBRDecompositionSelection | None:
    exact = getattr(point, "physical_decomposition", None)
    if exact is not None:
        return _EBRDecompositionSelection(
            dict(exact.multiplicities), float(exact.max_residual)
        )

    blocks = tuple(getattr(point, "degenerate_blocks", ()))
    if not blocks:
        return None
    multiplicities: dict[str, int] = {}
    residual = 0.0
    approximate_blocks = []
    for block in blocks:
        decomposition, approximate = _ebr_block_decomposition(block)
        if decomposition is None:
            return None
        for name, value in decomposition.multiplicities.items():
            multiplicities[name] = multiplicities.get(name, 0) + int(value)
        residual = max(residual, float(decomposition.max_residual))
        if approximate:
            approximate_blocks.append(tuple(int(value) for value in block.band_indices))
    return _EBRDecompositionSelection(
        multiplicities,
        residual,
        tuple(approximate_blocks),
    )


def _approximate_ebr_diagnostics(
    analysis: BlochSymmetryAnalysisResult,
    catalog: EBRCatalog,
    tolerance: float,
) -> tuple[str, ...]:
    diagnostics = []
    for catalog_point in catalog.k_points:
        point = _matching_analysis_point(analysis, catalog_point, tolerance)
        for block in point.degenerate_blocks:
            if block.decomposition is not None or block.approximate_decomposition is None:
                continue
            error = (
                "unknown"
                if block.character_fit_error is None
                else f"{float(block.character_fit_error):.6g}"
            )
            diagnostics.append(
                "EBR counting uses approximate irrep label at "
                f"{point.name}: bands(0-based)={tuple(block.band_indices)}; "
                f"character_error={error}; leakage={float(block.leakage):.6g}."
            )
    return tuple(diagnostics)


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
        decomposition = _ebr_point_decomposition(point)
        if decomposition is None:
            raise ValueError(
                f"Representation point {point.name!r} has no usable physical "
                "decomposition; every selected block needs an exact or accepted "
                "approximate irrep label for EBR analysis."
            )
        if not decomposition.approximate_blocks and decomposition.max_residual > tolerance:
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

    return EBRMatrixBuilder(
        context,
        catalog,
        lattice_vectors=lattice_vectors,
    ).build()


class EBRMatrixBuilder:
    """Run-scoped cache for dynamic EBR target and little-group construction."""

    def __init__(
        self,
        context: SymmetryContext,
        catalog: EBRCatalog,
        *,
        lattice_vectors=None,
    ) -> None:
        self.context = context
        self.catalog = _catalog_in_context_setting(context, catalog)
        self.lattice_vectors = lattice_vectors
        _validate_catalog_context(context, self.catalog)
        self.definition = context.model.group_definition
        if self.definition is None:
            raise ValueError(
                "Dynamic EBR generation requires a resolved space-group definition."
            )
        if any(operation.antiunitary for operation in context.model.group.operations):
            raise NotImplementedError(
                "EBR catalogs for magnetic corepresentations are not implemented."
            )
        self._orbit_cache = {}
        self._site_irrep_cache = {}
        self._target_cache = {}
        self._validated_wyckoff = set()
        self.point_data, self.row_keys = self._prepare_points()
        self.row_positions = {
            key: index for index, key in enumerate(self.row_keys)
        }

    def _prepare_points(self):
        point_data = []
        row_keys = []
        for point in self.catalog.k_points:
            elements = little_group(self.context.model.group, point.k_fractional)
            operation_indices = tuple(
                element.operation_index for element in elements
            )
            resolved = self.definition.resolve_little_group(
                operation_indices,
                point.k_fractional,
                bloch_convention=self.context.model.bloch_convention,
            )
            irreps = resolved.require_irreps()
            point_data.append((point, operation_indices, resolved, irreps))
            row_keys.extend(
                SymmetryVectorKey(point.name, irrep.name) for irrep in irreps
            )
        return tuple(point_data), tuple(row_keys)

    def _orbit(self, ebr: EBRDefinition):
        key = tuple(float(value) for value in ebr.center)
        orbit = self._orbit_cache.get(key)
        if orbit is None:
            orbit = build_crystallographic_orbit(
                self.context.model.group, ebr.center
            )
            self._orbit_cache[key] = orbit
        declared_multiplicity = _wyckoff_multiplicity(ebr.wyckoff)
        if self.lattice_vectors is not None and self.definition.dimension == 3:
            wyckoff_key = (ebr.wyckoff, key)
            if wyckoff_key not in self._validated_wyckoff:
                self.definition.validate_wyckoff(
                    ebr.wyckoff, ebr.center, self.lattice_vectors
                )
                self._validated_wyckoff.add(wyckoff_key)
        elif declared_multiplicity != orbit.multiplicity:
            raise ValueError(
                f"EBR {ebr.name!r} declares Wyckoff {ebr.wyckoff!r}, but its center "
                f"has orbit multiplicity {orbit.multiplicity}."
            )
        return orbit

    def _target(self, ebr: EBRDefinition):
        orbit = self._orbit(ebr)
        site_indices = tuple(
            element.source_operation_index
            for element in orbit.site_symmetry.elements
        )
        target_key = (
            tuple(float(value) for value in ebr.center),
            site_indices,
            ebr.site_irrep,
        )
        cached_target = self._target_cache.get(target_key)
        if cached_target is not None:
            return cached_target
        irrep_key = (site_indices, ebr.site_irrep)
        site_irrep = self._site_irrep_cache.get(irrep_key)
        if site_irrep is None:
            site_irrep = self.definition.site_irrep(
                site_indices, ebr.site_irrep
            )
            self._site_irrep_cache[irrep_key] = site_irrep
        target = build_wannier_target_from_group_irrep(
            ebr.name,
            self.context.model.group,
            ebr.center,
            site_irrep,
            self.context.model.bloch_convention,
            orbit=orbit,
        )
        self._target_cache[target_key] = target
        return target

    def target_representation(self, ebr: EBRDefinition):
        """Return the induced target representation for one catalog EBR.

        This is the shared construction used by the integer EBR matrix and by
        formal tight-binding models.  Keeping it here ensures both consumers
        use the same Wyckoff validation, site-irrep basis, and Bloch convention.
        """

        matching = next(
            (candidate for candidate in self.catalog.ebrs if candidate.name == ebr.name),
            None,
        )
        if matching is None or (
            matching.wyckoff != ebr.wyckoff
            or matching.site_irrep != ebr.site_irrep
            or not np.array_equal(matching.center, ebr.center)
        ):
            raise ValueError(
                f"EBR {ebr.name!r} does not belong to catalog {self.catalog.name!r}."
            )
        return self._target(ebr)

    def build(self) -> EBRMatrix:
        columns = []
        dimensions = []
        values = np.zeros(
            (len(self.row_keys), len(self.catalog.ebrs)), dtype=np.int64
        )
        for column_index, ebr in enumerate(self.catalog.ebrs):
            target = self.target_representation(ebr)
            dimensions.append(target.wannier_dimension)
            columns.append(ebr)
            for point, operation_indices, resolved, irreps in self.point_data:
                characters = {
                    self.context.model.group.operations[index].name
                    or f"g{index}": complex(
                        np.trace(target.matrix(index, point.k_fractional))
                    )
                    for index in operation_indices
                }
                decomposition = decompose_little_group_characters(
                    resolved, characters
                )
                threshold = max(
                    100.0 * self.context.model.algebra_tolerance, 1.0e-7
                )
                if decomposition.max_residual > threshold:
                    raise ValueError(
                        f"Generated EBR {ebr.name!r} has a non-integral decomposition at "
                        f"{point.name!r} (residual={decomposition.max_residual:.6g})."
                    )
                represented_dimension = 0
                for irrep in irreps:
                    multiplicity = int(
                        decomposition.multiplicities[irrep.name]
                    )
                    if multiplicity < 0:
                        raise ValueError(
                            f"Generated EBR {ebr.name!r} has negative multiplicity "
                            f"at {point.name!r}."
                        )
                    row_key = SymmetryVectorKey(point.name, irrep.name)
                    values[self.row_positions[row_key], column_index] = multiplicity
                    represented_dimension += multiplicity * irrep.dimension
                if represented_dimension != target.wannier_dimension:
                    raise ValueError(
                        f"Generated EBR {ebr.name!r} has dimension "
                        f"{represented_dimension} at {point.name!r}, expected "
                        f"{target.wannier_dimension}."
                    )
        return EBRMatrix(
            self.row_keys,
            tuple(columns),
            values,
            np.asarray(dimensions, dtype=np.int64),
        )


def run_ebr_analysis(
    analysis: BlochSymmetryRunResult | BlochSymmetryAnalysisResult,
    context: SymmetryContext,
    config: IncarConfig,
    *,
    catalog: EBRCatalog | str | Path | None = None,
) -> EBRAnalysisResult:
    """Generate and solve an EBR problem for the context's space group.

    The incar-facing path always selects the built-in catalog from the resolved
    space group.  A custom catalog remains available to programmatic callers.
    """

    physical_analysis = (
        analysis.primary.analysis if hasattr(analysis, "primary") else analysis
    )
    band_channels = dict(getattr(analysis, "band_channels", {}))
    if not isinstance(physical_analysis, BlochSymmetryAnalysisResult):
        raise TypeError("run_ebr_analysis expects a Bloch symmetry analysis result.")
    if catalog is None:
        definition = context.model.group_definition
        hall_number = None if definition is None else definition.hall_number
        catalog_reference = infer_builtin_catalog_alias(
            _context_catalog_identifier(context),
            hall_number=hall_number,
        )
        resolved_catalog = load_ebr_catalog(catalog_reference)
    elif isinstance(catalog, EBRCatalog):
        resolved_catalog = catalog
    else:
        definition = context.model.group_definition
        resolved_catalog = load_ebr_catalog(
            catalog,
            base_dir=config.base_dir,
            hall_number=None if definition is None else definition.hall_number,
        )
    catalog = _catalog_in_context_setting(
        context,
        resolved_catalog,
    )
    matrix = build_ebr_matrix(
        context,
        catalog,
        lattice_vectors=config.real_lattice_vectors,
    )
    mode = _resolve_ebr_mode(config, context)
    diagnostics = list(
        _approximate_ebr_diagnostics(
            physical_analysis,
            catalog,
            context.model.tolerance,
        )
    )
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
    if mode == "subspace":
        singular_gamma_zero_modes = _has_transverse_gamma_singularity(
            context,
            field_kind,
            wannier_subspace=config.wannier_subspace,
        )
        target_dimension = config.ebr_subspace_dimension
        if target_dimension is None:
            raise RuntimeError("Internal EBR mode selection lost the subspace dimension.")
        fixed_bands = tuple(
            int(value)
            for value in (
                ()
                if config.ebr_subspace_fixed_bands is None
                else config.ebr_subspace_fixed_bands
            )
        )
        fixed_include_gamma_zero_modes = _validate_subspace_fixed_bands(
            physical_analysis,
            context,
            catalog,
            fixed_bands=fixed_bands,
            target_dimension=target_dimension,
            zero_tolerance=config.gamma_zero_mode_tolerance,
            singular_gamma_zero_modes=singular_gamma_zero_modes,
        )
        skipped_blocks: list[str] = []
        inventory = _build_subspace_inventory(
            physical_analysis,
            context,
            catalog,
            zero_tolerance=config.gamma_zero_mode_tolerance,
            skipped_blocks=skipped_blocks,
            singular_gamma_zero_modes=singular_gamma_zero_modes,
        ).reordered(matrix.row_keys)
        diagnostics.extend(skipped_blocks)
        if singular_gamma_zero_modes:
            auxiliary_rows = _auxiliary_gamma_rows(
                context,
                catalog,
                matrix,
                config.real_lattice_vectors,
                field_kind,
            )
            zero_dimension = _gamma_zero_mode_dimension(
                physical_analysis,
                context,
                catalog,
                config.gamma_zero_mode_tolerance,
            )
            gamma_sectors = _subspace_gamma_sectors(
                fixed_include_gamma_zero_modes, zero_dimension
            )
            max_auxiliary_bands = config.ebr_max_auxiliary_bands
        else:
            auxiliary_rows = ()
            gamma_sectors = (False,)
            max_auxiliary_bands = 0
        candidates = []
        enumerations = []
        for include_gamma_zero_modes in gamma_sectors:
            surrogate = (
                _transverse_gamma_surrogate(
                    context,
                    catalog,
                    matrix,
                    config.real_lattice_vectors,
                    field_kind,
                )
                if include_gamma_zero_modes
                else np.zeros(len(matrix.row_keys), dtype=np.int64)
            )
            sector_name = (
                "include_gamma_zero_modes"
                if include_gamma_zero_modes
                else (
                    "exclude_gamma_zero_modes"
                    if singular_gamma_zero_modes
                    else "ordinary"
                )
            )
            enumeration = enumerate_ebr_subspace_solutions(
                inventory,
                matrix,
                gamma_point_name=catalog.gamma_point.name,
                target_dimension=target_dimension,
                gamma_surrogate=surrogate,
                max_auxiliary_bands=max_auxiliary_bands,
                max_states=config.ebr_max_states,
                required_auxiliary_gamma_rows=auxiliary_rows,
                gamma_sector=sector_name,
            )
            enumerations.append(enumeration)
            for solution, selected_vector in enumeration.solutions:
                point_representations = _match_subspace_representations(
                    physical_analysis,
                    context,
                    catalog,
                    selected_vector,
                    zero_tolerance=config.gamma_zero_mode_tolerance,
                    fixed_band_indices=fixed_bands,
                    include_gamma_zero_modes=include_gamma_zero_modes,
                    singular_gamma_zero_modes=singular_gamma_zero_modes,
                    band_channels=band_channels,
                )
                if point_representations is not None:
                    candidates.append(
                        EBRSubspaceCandidate(
                            solution,
                            selected_vector,
                            point_representations,
                            include_gamma_zero_modes,
                            sector_name,
                        )
                    )
        if singular_gamma_zero_modes:
            for include_gamma_zero_modes, enumeration in zip(
                gamma_sectors, enumerations, strict=True
            ):
                realizable = sum(
                    candidate.includes_gamma_zero_modes == include_gamma_zero_modes
                    for candidate in candidates
                )
                sector = (
                    "including the two singular Gamma zero modes"
                    if include_gamma_zero_modes
                    else "excluding only the two singular Gamma zero modes"
                )
                diagnostics.append(
                    f"Gamma sector {sector}: "
                    f"{len(enumeration.solutions)} algebraic signed-EBR solution(s), "
                    f"{realizable} symmetry-subspace candidate(s). Positive-frequency "
                    "Gamma irreps are enforced in both sectors."
                )
            include_enumeration = next(
                (
                    enumeration
                    for include, enumeration in zip(
                        gamma_sectors, enumerations, strict=True
                    )
                    if include
                ),
                None,
            )
            if include_enumeration is not None and not include_enumeration.solutions:
                gamma = _matching_analysis_point(
                    physical_analysis, catalog.gamma_point, context.model.tolerance
                )
                resolved = gamma.resolved_little_group
                if resolved is not None:
                    positions = {key: index for index, key in enumerate(inventory.row_keys)}
                    available = []
                    possible_dimensions = {0}
                    for irrep in resolved.require_irreps():
                        count = int(
                            inventory.multiplicities[
                                positions[
                                    SymmetryVectorKey(catalog.gamma_point.name, irrep.name)
                                ]
                            ]
                        )
                        if count <= 0:
                            continue
                        available.append(f"{count} {irrep.name}(dim={irrep.dimension})")
                        for _ in range(count):
                            possible_dimensions.update(
                                value + int(irrep.dimension)
                                for value in tuple(possible_dimensions)
                            )
                    required_positive = target_dimension - zero_dimension
                    if required_positive not in possible_dimensions:
                        diagnostics.append(
                            "A subspace containing the two singular Gamma zero modes "
                            f"needs {required_positive} positive-frequency Gamma state(s), "
                            "but the exact available irreps "
                            f"[{', '.join(available) or 'none'}] can realize dimensions "
                            f"{sorted(value for value in possible_dimensions if value <= target_dimension)}."
                        )
        algebraic_count = sum(
            len(enumeration.solutions) for enumeration in enumerations
        )
        search_statistics = EBRSearchStatistics(
            complete=all(enumeration.statistics.complete for enumeration in enumerations),
            gamma_sectors=tuple(
                sector
                for enumeration in enumerations
                for sector in enumeration.statistics.gamma_sectors
            ),
            auxiliary_dimensions_examined=tuple(
                sorted(
                    {
                        value
                        for enumeration in enumerations
                        for value in enumeration.statistics.auxiliary_dimensions_examined
                    }
                )
            ),
            weighted_vectors_generated=sum(
                enumeration.statistics.weighted_vectors_generated
                for enumeration in enumerations
            ),
            signed_combinations_tested=sum(
                enumeration.statistics.signed_combinations_tested
                for enumeration in enumerations
            ),
            algebraic_solutions=algebraic_count,
            unique_signed_solutions=sum(
                enumeration.statistics.unique_signed_solutions
                for enumeration in enumerations
            ),
            completion_solutions=sum(
                enumeration.statistics.completion_solutions
                for enumeration in enumerations
            ),
            realizable_candidates=len(candidates),
            block_realization_count=sum(
                candidate.block_realization_count for candidate in candidates
            ),
            search_limit=config.ebr_max_states,
        )
        optimal = min(
            (candidate.solution.auxiliary_dimension for candidate in candidates),
            default=None,
        )
        if optimal is None:
            gamma = _matching_analysis_point(
                physical_analysis, catalog.gamma_point, context.model.tolerance
            )
            positive_dimensions = [
                len(block.band_indices)
                for block in gamma.degenerate_blocks
                if not singular_gamma_zero_modes
                or not _is_zero_block(block, config.gamma_zero_mode_tolerance)
            ]
            diagnostics.append(
                f"No {target_dimension}-dimensional EBR subspace satisfies the exact "
                "representation-count and symmetry-invariant-subspace constraints."
            )
            dimension_gcd = reduce(gcd, (int(value) for value in matrix.dimensions))
            if target_dimension % dimension_gcd:
                diagnostics.append(
                    f"The EBR generator dimensions have gcd={dimension_gcd}, so a "
                    f"{target_dimension}-dimensional signed EBR combination is impossible."
                )
            elif algebraic_count == 0:
                diagnostics.append(
                    "No signed EBR combination fits the available high-symmetry irrep "
                    "multiplicities, even before complete band-block connectivity is imposed."
                )
            else:
                diagnostics.append(
                    f"{algebraic_count} algebraic EBR candidate(s) fit the irrep inventory, "
                    "but none can be assembled from invariant irrep subspaces of the "
                    "analyzed numerical blocks at every catalog k point."
                )
            required_positive = target_dimension - (
                2 if singular_gamma_zero_modes else 0
            )
            diagnostics.append(
                "Gamma eligible block dimensions are "
                f"{positive_dimensions}; the requested subspace needs "
                f"{required_positive} represented Gamma states."
            )
        return EBRAnalysisResult(
            mode,
            catalog,
            inventory,
            matrix,
            subspace_candidates=tuple(candidates),
            optimal_auxiliary_dimension=optimal,
            diagnostics=tuple(diagnostics),
            subspace_fixed_band_indices=fixed_bands,
            search_statistics=search_statistics,
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


def _build_subspace_inventory(
    analysis: BlochSymmetryAnalysisResult,
    context: SymmetryContext,
    catalog: EBRCatalog,
    *,
    zero_tolerance: float,
    skipped_blocks: list[str] | None = None,
    singular_gamma_zero_modes: bool = True,
) -> BandSymmetryVector:
    """Collect exact complete-block irreps available at every catalog point."""

    rows = []
    values = []
    dimensions = []
    for catalog_point in catalog.k_points:
        point = _matching_analysis_point(analysis, catalog_point, context.model.tolerance)
        _require_unitary_point(point)
        resolved = point.resolved_little_group
        if resolved is None:
            raise ValueError(f"Representation point {point.name!r} has no little group.")
        names = tuple(irrep.name for irrep in resolved.require_irreps())
        counts = {name: 0 for name in names}
        for block in point.degenerate_blocks:
            if (
                singular_gamma_zero_modes
                and catalog_point.name == catalog.gamma_point.name
                and _is_zero_block(block, zero_tolerance)
            ):
                continue
            decomposition, _ = _ebr_block_decomposition(block)
            if decomposition is None:
                if skipped_blocks is not None:
                    reason = block.irrep_unavailable_reason or "no exact irrep decomposition"
                    skipped_blocks.append(
                        f"Ignored incomplete block at {point.name}: bands(0-based)="
                        f"{tuple(block.band_indices)}; reason={reason}."
                    )
                continue
            for name, multiplicity in decomposition.multiplicities.items():
                counts[name] = counts.get(name, 0) + int(multiplicity)
        for name in names:
            rows.append(SymmetryVectorKey(catalog_point.name, name))
            values.append(counts[name])
        dimensions.append(len(point.band_indices))
    if len(set(dimensions)) != 1:
        raise ValueError("EBR subspace analysis points must use one common outer dimension.")
    return BandSymmetryVector(tuple(rows), np.asarray(values), dimensions[0])


def _match_subspace_representations(
    analysis: BlochSymmetryAnalysisResult,
    context: SymmetryContext,
    catalog: EBRCatalog,
    vector: BandSymmetryVector,
    *,
    zero_tolerance: float,
    fixed_band_indices: tuple[int, ...] = (),
    include_gamma_zero_modes: bool = False,
    singular_gamma_zero_modes: bool = True,
    band_channels=None,
) -> tuple[EBRSubspacePointRepresentation, ...] | None:
    band_channels = {} if band_channels is None else dict(band_channels)
    channel_names = tuple(
        dict.fromkeys(
            str(reference.channel)
            for _, reference in sorted(band_channels.items())
        )
    )

    def block_channel_profile(block, selected_dimension: int):
        if not channel_names:
            return (), True
        zero = tuple(0 for _ in channel_names)
        if selected_dimension == 0:
            return zero, True
        counts = {name: 0 for name in channel_names}
        for band_index in block.band_indices:
            reference = band_channels.get(int(band_index))
            if reference is None or str(reference.channel) not in counts:
                return zero, False
            counts[str(reference.channel)] += 1
        active = tuple(name for name, value in counts.items() if value)
        if selected_dimension == len(block.band_indices):
            return tuple(counts[name] for name in channel_names), True
        if len(active) == 1:
            return tuple(
                selected_dimension if name == active[0] else 0
                for name in channel_names
            ), True
        return zero, False

    positions = {key: index for index, key in enumerate(vector.row_keys)}
    output = []
    for catalog_point in catalog.k_points:
        point = _matching_analysis_point(analysis, catalog_point, context.model.tolerance)
        resolved = point.resolved_little_group
        if resolved is None:
            return None
        irreps = tuple(resolved.require_irreps())
        names = tuple(irrep.name for irrep in irreps)
        irrep_dimensions = np.asarray(
            [int(irrep.dimension) for irrep in irreps], dtype=np.int64
        )
        target = np.asarray(
            [vector.multiplicities[positions[SymmetryVectorKey(catalog_point.name, name)]] for name in names],
            dtype=np.int64,
        )
        zero_blocks = tuple(
            block
            for block in point.degenerate_blocks
            if singular_gamma_zero_modes
            and catalog_point.name == catalog.gamma_point.name
            and _is_zero_block(block, zero_tolerance)
        )
        blocks = tuple(
            block
            for block in point.degenerate_blocks
            if not (
                singular_gamma_zero_modes
                and catalog_point.name == catalog.gamma_point.name
                and _is_zero_block(block, zero_tolerance)
            )
            and _ebr_block_decomposition(block)[0] is not None
        )
        fixed = set(fixed_band_indices)
        options = []
        for block in blocks:
            decomposition, _ = _ebr_block_decomposition(block)
            if decomposition is None:
                raise RuntimeError("Internal EBR block filtering lost its decomposition.")
            full = np.asarray(
                [decomposition.multiplicities.get(name, 0) for name in names],
                dtype=np.int64,
            )
            represented_dimension = int(full @ irrep_dimensions)
            if represented_dimension != len(block.band_indices):
                raise ValueError(
                    f"Block {tuple(block.band_indices)} at {point.name!r} has irrep "
                    f"dimension {represented_dimension}, expected {len(block.band_indices)}."
                )
            if fixed.intersection(block.band_indices):
                profile, profile_known = block_channel_profile(
                    block, represented_dimension
                )
                options.append(
                    ((tuple(int(value) for value in full), represented_dimension, profile, profile_known),)
                )
                continue
            block_options = []
            for values in product(*(range(int(value) + 1) for value in full)):
                contribution = np.asarray(values, dtype=np.int64)
                contribution_dimension = int(contribution @ irrep_dimensions)
                profile, profile_known = block_channel_profile(
                    block, contribution_dimension
                )
                block_options.append(
                    (values, contribution_dimension, profile, profile_known)
                )
            options.append(tuple(block_options))

        zero_selection = zero_blocks if include_gamma_zero_modes else ()
        zero_dimension = sum(len(block.band_indices) for block in zero_selection)
        represented_target_dimension = vector.total_dimension - zero_dimension

        @lru_cache(maxsize=None)
        def select(index: int, current_values: tuple[int, ...], selected_dimension: int):
            current = np.asarray(current_values, dtype=np.int64)
            if np.any(current > target) or selected_dimension > represented_target_dimension:
                return 0, (), True
            if index == len(blocks):
                matches = int(
                    np.array_equal(current, target)
                    and selected_dimension == represented_target_dimension
                )
                profiles = (tuple(0 for _ in channel_names),) if matches and channel_names else ()
                return matches, profiles, True
            count = 0
            profiles = set()
            complete = True
            for (
                contribution_values,
                contribution_dimension,
                contribution_profile,
                profile_known,
            ) in options[index]:
                contribution = np.asarray(contribution_values, dtype=np.int64)
                child_count, child_profiles, child_complete = select(
                    index + 1,
                    tuple(int(value) for value in current + contribution),
                    selected_dimension + contribution_dimension,
                )
                if child_count == 0:
                    continue
                count += child_count
                complete = complete and child_complete and profile_known
                if profile_known:
                    profiles.update(
                        tuple(
                            int(left + right)
                            for left, right in zip(contribution_profile, child)
                        )
                        for child in child_profiles
                    )
            return count, tuple(sorted(profiles)), complete

        match_count, raw_profiles, channel_assignment_complete = select(
            0, tuple(0 for _ in target), 0
        )
        if match_count == 0:
            return None
        channel_realizations = ()
        if channel_names:
            zero_profile = tuple(0 for _ in channel_names)
            zero_profile_known = True
            for block in zero_selection:
                profile, known = block_channel_profile(block, len(block.band_indices))
                zero_profile = tuple(
                    int(left + right) for left, right in zip(zero_profile, profile)
                )
                zero_profile_known = zero_profile_known and known
            channel_assignment_complete = (
                channel_assignment_complete and zero_profile_known
            )
            channel_realizations = tuple(
                tuple(
                    (name, int(value))
                    for name, value in zip(
                        channel_names,
                        (
                            int(left + right)
                            for left, right in zip(profile, zero_profile)
                        ),
                    )
                )
                for profile in raw_profiles
            )
        output.append(
            EBRSubspacePointRepresentation(
                catalog_point.name,
                tuple((name, int(value)) for name, value in zip(names, target) if value),
                zero_dimension,
                match_count,
                channel_realizations,
                channel_assignment_complete,
            )
        )
    return tuple(output)


def _validate_subspace_fixed_bands(
    analysis: BlochSymmetryAnalysisResult,
    context: SymmetryContext,
    catalog: EBRCatalog,
    *,
    fixed_bands: tuple[int, ...],
    target_dimension: int,
    zero_tolerance: float,
    singular_gamma_zero_modes: bool = True,
) -> bool:
    """Validate mandatory bands and decide whether Gamma needs the T-mode surrogate."""

    fixed = set(fixed_bands)
    include_gamma_zero_modes = False
    for catalog_point in catalog.k_points:
        point = _matching_analysis_point(analysis, catalog_point, context.model.tolerance)
        available = set(int(value) for value in point.band_indices)
        missing = sorted(fixed - available)
        if missing:
            raise ValueError(
                f"Fixed EBR subspace bands {tuple(missing)} are "
                f"not present at representation point {point.name!r}."
            )
        required_dimension = 0
        for block in point.degenerate_blocks:
            if not fixed.intersection(block.band_indices):
                continue
            is_gamma_zero = (
                singular_gamma_zero_modes
                and catalog_point.name == catalog.gamma_point.name
                and _is_zero_block(block, zero_tolerance)
            )
            if is_gamma_zero:
                include_gamma_zero_modes = True
            elif _ebr_block_decomposition(block)[0] is None:
                reason = block.irrep_unavailable_reason or "no exact irrep decomposition"
                raise ValueError(
                    f"Fixed EBR subspace band at {point.name!r} belongs to incomplete "
                    f"block {tuple(block.band_indices)}: {reason}."
                )
            required_dimension += len(block.band_indices)
        if required_dimension > target_dimension:
            raise ValueError(
                f"Fixed EBR subspace bands require {required_dimension} states at "
                f"{point.name!r}, exceeding ebr_subspace_dimension={target_dimension}."
            )

    if include_gamma_zero_modes:
        gamma = _matching_analysis_point(
            analysis, catalog.gamma_point, context.model.tolerance
        )
        zero_dimension = sum(
            len(block.band_indices)
            for block in gamma.degenerate_blocks
            if _is_zero_block(block, zero_tolerance)
        )
        if zero_dimension != 2:
            raise ValueError(
                "A fixed band intersects the Gamma zero-frequency subspace, but the "
                "analysis does not contain exactly two transverse zero modes; found "
                f"{zero_dimension}."
            )
    return include_gamma_zero_modes


def _gamma_zero_mode_dimension(
    analysis: BlochSymmetryAnalysisResult,
    context: SymmetryContext,
    catalog: EBRCatalog,
    zero_tolerance: float,
) -> int:
    gamma = _matching_analysis_point(
        analysis, catalog.gamma_point, context.model.tolerance
    )
    return sum(
        len(block.band_indices)
        for block in gamma.degenerate_blocks
        if _is_zero_block(block, zero_tolerance)
    )


def _subspace_gamma_sectors(
    fixed_include_gamma_zero_modes: bool,
    zero_mode_dimension: int,
) -> tuple[bool, ...]:
    if fixed_include_gamma_zero_modes:
        if zero_mode_dimension != 2:
            raise ValueError(
                "Fixed Gamma zero modes require exactly two transverse zero modes."
            )
        return (True,)
    return (False, True) if zero_mode_dimension == 2 else (False,)


def _is_zero_block(block, tolerance: float) -> bool:
    return bool(
        block.energies
        and max(abs(complex(value)) for value in block.energies) <= tolerance
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
            decomposition = _ebr_point_decomposition(point)
            if decomposition is None:
                raise ValueError(
                    f"Non-Gamma representation point {point.name!r} has no usable exact "
                    "or accepted approximate decomposition for TETB enumeration."
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
        decomposition = _ebr_point_decomposition(point)
        if decomposition is not None:
            return dict(decomposition.multiplicities)
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
        decomposition, _ = _ebr_block_decomposition(block)
        if decomposition is None:
            raise ValueError(
                f"Nonzero Gamma block {block.band_indices} has no usable exact or "
                "accepted approximate irrep decomposition."
            )
        for name, multiplicity in decomposition.multiplicities.items():
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


def _transverse_gamma_surrogate(
    context: SymmetryContext,
    catalog: EBRCatalog,
    matrix: EBRMatrix,
    lattice_vectors,
    field_kind: FieldKind,
) -> np.ndarray:
    """Return the formal two-transverse-mode Gamma representation T+L - L."""

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
            "Transverse EBR subspace search requires a 3D polar-electric or "
            "axial-magnetic field."
        )
    full = _field_representation_decomposition_resolved(
        resolved, indices, context, field_kind, lattice_vectors
    )
    longitudinal = _field_representation_decomposition_resolved(
        resolved, indices, context, auxiliary_kind, lattice_vectors
    )
    output = np.zeros(len(matrix.row_keys), dtype=np.int64)
    for index, key in enumerate(matrix.row_keys):
        if key.point_name != catalog.gamma_point.name:
            continue
        output[index] = int(full.multiplicities.get(key.irrep_name, 0)) - int(
            longitudinal.multiplicities.get(key.irrep_name, 0)
        )
    return output


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
    if definition is None:
        raise ValueError("EBR analysis requires a resolved space-group definition.")
    if catalog.dimension != context.model.dimension:
        raise ValueError("EBR catalog dimension does not match the symmetry context.")
    if definition.hall_number is None:
        if catalog.space_group_name is None:
            raise ValueError(
                "This non-Hall symmetry context requires a named EBR catalog."
            )
        if _normalized_group_name(catalog.space_group_name) != _normalized_group_name(
            definition.name
        ):
            raise ValueError(
                f"EBR catalog space group {catalog.space_group_name!r} does not match "
                f"calculation space group {definition.name!r}."
            )
        return
    if catalog.space_group_number is None or catalog.hall_number is None:
        raise ValueError("A 3D Hall-setting calculation requires a Hall-setting EBR catalog.")
    actual_number = _context_space_group_number(context)
    if actual_number != catalog.space_group_number:
        raise ValueError(
            f"EBR catalog space group {catalog.space_group_number} does not match "
            f"calculation space group {actual_number}."
        )
    if int(definition.hall_number) != catalog.hall_number:
        raise ValueError(
            f"EBR catalog Hall {catalog.hall_number} does not match calculation Hall "
            f"{definition.hall_number}."
        )


def _catalog_in_context_setting(
    context: SymmetryContext, catalog: EBRCatalog
) -> EBRCatalog:
    """Require an EBR catalog defined in the calculation's exact Hall setting."""

    _validate_catalog_context(context, catalog)
    return catalog


def _context_space_group_number(context: SymmetryContext) -> int:
    definition = context.model.group_definition
    if definition is None or definition.hall_number is None:
        raise ValueError("Automatic EBR catalog selection requires a Hall-number space group.")
    import spglib

    group_type = spglib.get_spacegroup_type(int(definition.hall_number))
    if group_type is None:
        raise ValueError(f"spglib could not resolve Hall {definition.hall_number}.")
    return int(group_type.number)


def _context_catalog_identifier(context: SymmetryContext) -> int | str:
    definition = context.model.group_definition
    if definition is None:
        raise ValueError("Automatic EBR catalog selection requires a space-group definition.")
    if definition.hall_number is None:
        return definition.name
    return _context_space_group_number(context)


def _resolve_ebr_mode(config, context: SymmetryContext) -> str:
    if config.ebr_subspace_dimension is not None:
        return "subspace"
    if context.model.dimension == 2:
        return "regular"
    return "regular" if config.wannier_subspace == "T+L" else "transverse"


def _has_transverse_gamma_singularity(
    context: SymmetryContext,
    field_kind: FieldKind,
    *,
    wannier_subspace: str = "T",
) -> bool:
    subspace = str(wannier_subspace).replace(" ", "").upper()
    return (
        subspace == "T"
        and context.model.dimension == 3
        and field_kind
        in {
            FieldKind.ELECTRIC_POLAR_VECTOR,
            FieldKind.MAGNETIC_AXIAL_VECTOR,
        }
    )


def _normalized_group_name(value: str) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _wyckoff_multiplicity(label: str) -> int:
    digits = ""
    for character in str(label).strip():
        if not character.isdigit():
            break
        digits += character
    if not digits:
        raise ValueError(f"Wyckoff label {label!r} does not begin with a multiplicity.")
    return int(digits)
