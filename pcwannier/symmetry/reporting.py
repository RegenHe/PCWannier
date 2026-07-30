from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import (
        BlochSymmetryAnalysisResult,
        GammaZeroRegularizationAnalysis,
        SymmetryAnalysisResult,
        TargetCompatibilityAnalysis,
    )

LOGGER = logging.getLogger(__name__)


def log_gamma_zero_regularization(result: GammaZeroRegularizationAnalysis) -> None:
    LOGGER.info(
        "Gamma T+L regularization %s: physical_T_bands(1-based)=%s "
        "longitudinal_zero_bands(1-based)=%s irrep=%s unitarity=%.6g "
        "twisted_composition=%.6g note=%s",
        result.point_name,
        tuple(value + 1 for value in result.transverse_band_indices),
        tuple(value + 1 for value in result.longitudinal_zero_band_indices),
        (
            "unavailable"
            if result.decomposition is None
            else _format_irrep_decomposition(result.decomposition)
        ),
        result.unitarity_error,
        result.twisted_composition_residual,
        result.note,
    )


def log_bloch_symmetry_analysis(result: BlochSymmetryAnalysisResult) -> None:
    for point in result.points:
        blocks = tuple(
            tuple(band + 1 for band in block.band_indices)
            for block in point.degenerate_blocks
        )
        factor = point.factor_system
        LOGGER.info(
            "Bloch symmetry point %s: little_co_group=%s unitary_subgroup=%s "
            "unitary_operations=%s antiunitary_operations=%s classes=%s mapping=%s k=%s "
            "outer_bands(1-based)=%s analyzed_bands(1-based)=%s blocks=%s "
            "unitarity=%.6g outer_unitarity=%.6g leakage=%.6g "
            "outer_exact_composition=%.6g selected_twisted_composition=%.6g "
            "factor_phase=%.6g factor_cocycle=%.6g "
            "factor_raw_trivial=%s factor_coboundary_trivial=%s factor_sign=%s "
            "small_representations=%s unitary_characters=%s",
            point.name,
            point.little_group_name or "unresolved",
            point.unitary_subgroup_name or point.little_group_name or "unresolved",
            point.unitary_operation_names,
            point.antiunitary_operation_names,
            point.conjugacy_classes,
            point.finite_group_mapping,
            point.sampled_k_fractional.tolist(),
            tuple(band + 1 for band in point.outer_band_indices),
            tuple(band + 1 for band in point.band_indices),
            blocks,
            point.diagnostics.unitarity_error,
            point.outer_unitarity_error,
            point.diagnostics.leakage,
            point.diagnostics.outer_composition_residual,
            point.diagnostics.selected_twisted_composition_residual,
            0.0 if factor is None else factor.phase_residual,
            0.0 if factor is None else factor.cocycle_residual,
            True if factor is None else factor.raw_trivial,
            True if factor is None else factor.cohomologically_trivial,
            1 if factor is None else factor.bloch_sign,
            (
                ()
                if point.resolved_little_group is None
                else tuple(
                    (irrep.name, irrep.dimension, irrep.label_source)
                    for irrep in point.resolved_little_group.irreps
                )
            ),
            {name: complex(value) for name, value in point.unitary_characters.items()},
        )
        for block in point.degenerate_blocks:
            label = _format_block_irrep(block, include_unavailable_reason=True)
            LOGGER.info(
                "Bloch symmetry block %s bands(1-based)=%s eigenvalues=%s degeneracy=%s "
                "irrep=%s class_character_summary=%s "
                "unitary_characters=%s coupled_outer_bands(1-based)=%s "
                "candidate_excluded_bands(1-based)=%s unitarity=%.6g leakage=%.6g "
                "twisted_composition=%.6g character_fit_error=%s",
                point.name,
                tuple(band + 1 for band in block.band_indices),
                tuple(complex(value) for value in block.energies),
                len(block.band_indices),
                label,
                _class_character_entries(point, block),
                {name: complex(value) for name, value in block.unitary_characters.items()},
                tuple(band + 1 for band in block.coupled_outer_bands),
                tuple(band + 1 for band in block.candidate_excluded_bands),
                block.unitarity_error,
                block.leakage,
                block.twisted_composition_residual,
                block.character_fit_error,
            )
            for diagnostic in block.antiunitary_diagnostics:
                LOGGER.info(
                    "Bloch antiunitary block %s bands(1-based)=%s operation=%s square=%s "
                    "square_eigenvalues=%s square_residual=%.6g",
                    point.name,
                    tuple(band + 1 for band in block.band_indices),
                    diagnostic.operation_name,
                    diagnostic.square_operation_name,
                    diagnostic.square_eigenvalues,
                    diagnostic.square_residual,
                )
        if factor is not None and any(factor.antiunitary_flags):
            LOGGER.info(
                "Symmetry point %s contains antiunitary operations: ordinary irrep labels "
                "unavailable (magnetic corepresentation database is not implemented)",
                point.name,
            )
        elif factor is not None and not factor.cohomologically_trivial:
            LOGGER.info(
                "Symmetry point %s uses a non-trivial projective factor: "
                "small-representation labels are generated in the fixed PCWannier factor gauge",
                point.name,
            )


def log_target_compatibilities(
    results: tuple[TargetCompatibilityAnalysis, ...],
) -> None:
    for result in results:
        LOGGER.info(
            "Target compatibility %s: targets=%s target_unitary_characters=%s "
            "target_irreps=%s compatible=%s direct_intertwiner_dimension=%s",
            result.point_name,
            result.target_names,
            {
                name: complex(value)
                for name, value in result.target_unitary_characters.items()
            },
            (
                {}
                if result.target_decomposition is None
                else result.target_decomposition.multiplicities
            ),
            None if result.compatibility is None else result.compatibility.compatible,
            result.intertwiner_dimension,
        )


def log_symmetry_analysis(result: SymmetryAnalysisResult) -> None:
    log_bloch_symmetry_analysis(result.physical)
    log_target_compatibilities(result.target_compatibilities)


def format_bloch_symmetry_report(
    result: BlochSymmetryAnalysisResult,
    *,
    title: str = "physical",
) -> str:
    """Return a compact, human-readable summary without matrix-level diagnostics."""

    lines = [f"[{title}]"]
    for point in result.points:
        factor = point.factor_system
        if factor is None:
            factor_kind = "ordinary"
        elif any(factor.antiunitary_flags):
            factor_kind = "antiunitary"
        elif factor.cohomologically_trivial:
            factor_kind = "ordinary"
        else:
            factor_kind = "projective"
        lines.extend(
            (
                f"{point.name}: k={_format_vector(point.sampled_k_fractional)}",
                "  group="
                f"{point.little_group_name or 'unresolved'}; "
                f"unitary_subgroup={point.unitary_subgroup_name or point.little_group_name or 'unresolved'}; "
                f"factor={factor_kind}",
                f"  analyzed_bands={_format_bands(point.band_indices)}; "
                f"outer_bands={_format_bands(point.outer_band_indices)}",
            )
        )
        if point.antiunitary_operation_names:
            lines.append(
                "  antiunitary_operations=" + ", ".join(point.antiunitary_operation_names)
            )
        for block in point.degenerate_blocks:
            irrep = _format_block_irrep(block)
            lines.append(
                f"  bands {_format_bands(block.band_indices)}: "
                f"eigenvalues={_format_values(block.energies)}; "
                f"dimension={len(block.band_indices)}; irrep={irrep}"
            )
            for class_label, operation, character, quantity in _class_character_entries(
                point, block
            ):
                lines.append(
                    f"    {class_label}({operation}) {quantity}={_format_number(character)}"
                )
            if block.irrep_unavailable_reason:
                lines.append(f"    note={block.irrep_unavailable_reason}")
        lines.append(
            "  residuals: "
            f"unitarity={point.diagnostics.unitarity_error:.6g}; "
            f"leakage={point.diagnostics.leakage:.6g}; "
            f"composition={point.diagnostics.selected_twisted_composition_residual:.6g}"
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def format_target_compatibility_report(
    results: tuple[TargetCompatibilityAnalysis, ...],
) -> str:
    if not results:
        return ""
    lines = ["[target compatibility]"]
    for result in results:
        compatibility = result.compatibility
        compatible = "unavailable" if compatibility is None else str(compatibility.compatible).lower()
        target_irreps = (
            "unavailable"
            if result.target_decomposition is None
            else _format_irrep_decomposition(result.target_decomposition)
        )
        lines.append(
            f"{result.point_name}: targets={', '.join(result.target_names)}; "
            f"irreps={target_irreps}; compatible={compatible}; "
            f"intertwiner_dimension={result.intertwiner_dimension}"
        )
    return "\n".join(lines)


def format_symmetry_analysis_report(result: SymmetryAnalysisResult) -> str:
    sections = [format_bloch_symmetry_report(result.physical)]
    target = format_target_compatibility_report(result.target_compatibilities)
    if target:
        sections.append(target)
    return "# PCWannier symmetry analysis\n\n" + "\n\n".join(sections) + "\n"


def format_gamma_zero_regularization_report(
    result: GammaZeroRegularizationAnalysis,
) -> str:
    irrep = (
        "unavailable"
        if result.decomposition is None
        else _format_irrep_decomposition(result.decomposition)
    )
    return "\n".join(
        (
            "[Gamma T+L regularization]",
            f"{result.point_name}: transverse_bands={_format_bands(result.transverse_band_indices)}; "
            f"longitudinal_bands={_format_bands(result.longitudinal_zero_band_indices)}; irrep={irrep}",
            f"note={result.note}",
        )
    )


def _format_irrep_decomposition(decomposition) -> str:
    terms = []
    for name, multiplicity in decomposition.multiplicities.items():
        if multiplicity <= 0:
            continue
        terms.append(name if multiplicity == 1 else f"{multiplicity}{name}")
    return " + ".join(terms) or "none"


def _format_block_irrep(block, *, include_unavailable_reason: bool = False) -> str:
    if block.decomposition is not None:
        return _format_irrep_decomposition(block.decomposition)
    if block.approximate_decomposition is not None:
        label = _format_irrep_decomposition(block.approximate_decomposition)
        error = 0.0 if block.character_fit_error is None else block.character_fit_error
        return f"{label} (approximate; character_error={error:.6g})"
    if include_unavailable_reason:
        return f"unavailable ({block.irrep_unavailable_reason or 'invalid representation'})"
    return "unavailable"


def _class_character_entries(point, block):
    resolved = point.resolved_little_group
    if resolved is None:
        return ()
    projective = not resolved.factor_system.cohomologically_trivial
    output = []
    for class_index, conjugacy_class in enumerate(
        resolved.table.conjugacy_classes, start=1
    ):
        representative_index = conjugacy_class.element_indices[0]
        operation = resolved.table.element_names[representative_index]
        if operation not in block.unitary_characters:
            continue
        if projective:
            value = block.unitary_characters[operation]
            quantity = "representative_trace"
        else:
            class_characters = [
                block.unitary_characters[resolved.table.element_names[index]]
                for index in conjugacy_class.element_indices
                if resolved.table.element_names[index] in block.unitary_characters
            ]
            if not class_characters:
                continue
            value = sum(class_characters) / len(class_characters)
            quantity = "character"
        output.append(
            (f"K{class_index}", operation, value, quantity)
        )
    return tuple(output)


def _format_bands(indices) -> str:
    values = tuple(int(value) + 1 for value in indices)
    return ",".join(str(value) for value in values) or "none"


def _format_vector(values) -> str:
    return "(" + ", ".join(f"{float(value):.8g}" for value in values) + ")"


def _format_values(values) -> str:
    return "(" + ", ".join(_format_number(value) for value in values) + ")"


def _format_number(value) -> str:
    number = complex(value)
    if abs(number.imag) <= 1.0e-12:
        return f"{number.real:.10g}"
    return f"{number.real:.10g}{number.imag:+.10g}j"
