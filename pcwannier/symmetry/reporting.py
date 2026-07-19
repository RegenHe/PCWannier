from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import (
        BlochSymmetryAnalysisResult,
        SymmetryAnalysisResult,
        TargetCompatibilityAnalysis,
    )

LOGGER = logging.getLogger(__name__)


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
            "composition=%.6g twisted_composition=%.6g factor_phase=%.6g factor_cocycle=%.6g "
            "factor_raw_trivial=%s factor_coboundary_trivial=%s factor_sign=%s "
            "unitary_characters=%s",
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
            point.diagnostics.max_composition_residual,
            point.diagnostics.max_twisted_composition_residual,
            0.0 if factor is None else factor.phase_residual,
            0.0 if factor is None else factor.cocycle_residual,
            True if factor is None else factor.raw_trivial,
            True if factor is None else factor.cohomologically_trivial,
            1 if factor is None else factor.bloch_sign,
            {name: complex(value) for name, value in point.unitary_characters.items()},
        )
        for block in point.degenerate_blocks:
            label = (
                _format_irrep_decomposition(block.decomposition)
                if block.decomposition is not None
                else f"unavailable ({block.irrep_unavailable_reason or 'invalid representation'})"
            )
            LOGGER.info(
                "Bloch symmetry block %s bands(1-based)=%s eigenvalues=%s degeneracy=%s "
                "irrep=%s generators=%s unitary_characters=%s coupled_outer_bands(1-based)=%s "
                "candidate_excluded_bands(1-based)=%s unitarity=%.6g leakage=%.6g "
                "twisted_composition=%.6g",
                point.name,
                tuple(band + 1 for band in block.band_indices),
                tuple(complex(value) for value in block.energies),
                len(block.band_indices),
                label,
                block.generator_eigenvalues,
                {name: complex(value) for name, value in block.unitary_characters.items()},
                tuple(band + 1 for band in block.coupled_outer_bands),
                tuple(band + 1 for band in block.candidate_excluded_bands),
                block.unitarity_error,
                block.leakage,
                block.twisted_composition_residual,
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
                "ordinary irrep labels unavailable",
                point.name,
            )


def log_target_compatibilities(
    results: tuple[TargetCompatibilityAnalysis, ...],
) -> None:
    for result in results:
        LOGGER.info(
            "Target compatibility %s: targets=%s target_irreps=%s compatible=%s "
            "direct_intertwiner_dimension=%s",
            result.point_name,
            result.target_names,
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


def _format_irrep_decomposition(decomposition) -> str:
    terms = []
    for name, multiplicity in decomposition.multiplicities.items():
        if multiplicity <= 0:
            continue
        terms.append(name if multiplicity == 1 else f"{multiplicity}{name}")
    return " + ".join(terms) or "none"

