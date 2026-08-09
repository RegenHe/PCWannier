from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import EBRAnalysisResult


def write_ebr_outputs(result: EBRAnalysisResult, config, out_dir=None) -> None:
    report_path = _resolve_output(config.ebr_report_file, config.base_dir, out_dir)
    data_path = _resolve_output(config.ebr_data_file, config.base_dir, out_dir)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(format_ebr_report(result), encoding="utf-8")
    if data_path is not None:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            json.dumps(ebr_result_to_dict(result), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )


def format_ebr_report(result: EBRAnalysisResult) -> str:
    names = tuple(column.name for column in result.ebr_matrix.columns)
    lines = [
        "PCWannier EBR analysis",
        f"catalog: {result.catalog.name} ({result.catalog.source or 'unknown'})",
        _format_catalog_group(result.catalog),
        f"mode: {result.mode}",
        f"band_dimension: {result.symmetry_vector.total_dimension}",
        "",
        "Symmetry vector:",
    ]
    for key, value in zip(
        result.symmetry_vector.row_keys, result.symmetry_vector.multiplicities
    ):
        if value:
            lines.append(f"  {key.label}: {int(value)}")
    lines.extend(("", f"EBR columns: {len(names)}"))
    for column, dimension in zip(result.ebr_matrix.columns, result.ebr_matrix.dimensions):
        lines.append(
            f"  {column.name}: {column.site_irrep}@{column.wyckoff}; dimension={int(dimension)}"
        )
    if result.mode == "regular":
        lines.extend(("", f"Regular decompositions: {len(result.regular_decompositions)}"))
        for index, decomposition in enumerate(result.regular_decompositions, start=1):
            lines.append(
                f"  {index}: {_format_combination(decomposition.multiplicities, names)}"
            )
    elif result.mode == "transverse":
        physical = result.physical_tetb_solutions
        lines.extend(
            (
                "",
                f"TETB solutions examined: {len(result.tetb_solutions)}",
                f"Physical optimal solutions: {len(physical)}",
                f"Minimum auxiliary dimension: {result.optimal_auxiliary_dimension}",
            )
        )
        for index, solution in enumerate(physical, start=1):
            lines.extend(
                (
                    f"  solution {index}:",
                    f"    n_T+L = {_format_combination(solution.n_t_plus_l, names)}",
                    f"    n_L   = {_format_combination(solution.n_l, names)}",
                    f"    n_T   = {_format_combination(solution.n_t, names, signed=True)}",
                    f"    auxiliary_dimension = {solution.auxiliary_dimension}",
                    f"    composite = {str(solution.composite).lower()}",
                    "    Gamma surrogate = "
                    + _format_symmetry_vector(
                        result.symmetry_vector.row_keys, solution.gamma_surrogate
                    ),
                )
            )
    else:
        fixed = (
            ",".join(str(band) for band in result.subspace_fixed_band_indices)
            or "none"
        )
        lines.extend(
            (
                "",
                f"Symmetry-subspace candidates: {len(result.subspace_candidates)}",
                f"Minimum auxiliary dimension: {result.optimal_auxiliary_dimension}",
                f"Fixed bands (0-based): {fixed}",
                "n_L convention: the subtracted auxiliary EBR in n_T = n_T+L - n_L; "
                "it is not the number of selected L-channel numerical bands.",
            )
        )
        statistics = result.search_statistics
        if statistics is not None:
            lines.extend(
                (
                    "",
                    "Search statistics:",
                    f"  complete: {str(statistics.complete).lower()}",
                    f"  gamma_sectors: {', '.join(statistics.gamma_sectors) or 'none'}",
                    "  auxiliary_dimensions_examined: "
                    + (",".join(str(value) for value in statistics.auxiliary_dimensions_examined) or "none"),
                    f"  weighted_vectors_generated: {statistics.weighted_vectors_generated}",
                    f"  signed_combinations_tested: {statistics.signed_combinations_tested}",
                    f"  algebraic_solutions: {statistics.algebraic_solutions}",
                    f"  unique_signed_solutions: {statistics.unique_signed_solutions}",
                    f"  completion_solutions: {statistics.completion_solutions}",
                    f"  realizable_ebr_candidates: {statistics.realizable_candidates}",
                    f"  hsp_invariant_subspace_realizations: {statistics.block_realization_count}",
                    f"  max_states_per_sector: {statistics.search_limit}",
                    "  completeness_scope: EBR coefficients and independent high-symmetry-point "
                    "invariant irrep subspaces; global band connectivity and smooth projectors "
                    "are not tested",
                )
            )
        for index, candidate in enumerate(result.subspace_candidates, start=1):
            solution = candidate.solution
            gamma_rule = {
                "ordinary": (
                    "ordinary T+L Gamma representation; enforce complete zero- and "
                    "positive-frequency irreps"
                ),
                "include_gamma_zero_modes": (
                    "include the two singular Gamma zero modes through the T+L-L "
                    "surrogate; enforce all selected positive-frequency Gamma irreps"
                ),
                "exclude_gamma_zero_modes": (
                    "exclude only the two singular Gamma zero modes; enforce all selected "
                    "positive-frequency Gamma irreps"
                ),
            }[candidate.gamma_sector]
            lines.extend(
                (
                    f"  candidate {index}:",
                    f"    n_T+L = {_format_combination(solution.n_t_plus_l, names)}",
                    f"    n_L   = {_format_combination(solution.n_l, names)}",
                    f"    n_T   = {_format_combination(solution.n_t, names, signed=True)}",
                    f"    auxiliary_dimension = {solution.auxiliary_dimension}",
                    f"    Gamma rule = {gamma_rule}",
                    f"    hsp_invariant_subspace_realizations = {candidate.block_realization_count}",
                    "    Gamma zero-mode surrogate = "
                    + _format_symmetry_vector(
                        result.symmetry_vector.row_keys, solution.gamma_surrogate
                    ),
                )
            )
            formal_vector = result.ebr_matrix.values @ solution.n_t
            for point in candidate.point_representations:
                physical = _format_physical_point_representation(
                    point.irrep_multiplicities,
                    unresolved_dimension=point.unresolved_dimension,
                )
                formal = _format_point_symmetry_vector(
                    result.ebr_matrix.row_keys,
                    formal_vector,
                    point.point_name,
                )
                lines.append(
                    f"    {point.point_name}: physical={physical}; "
                    f"formal_signed={formal}; "
                    f"invariant_subspace_realizations={point.alternative_count}"
                    + _format_channel_realizations(point)
                )
    if result.diagnostics:
        lines.extend(("", "Diagnostics:"))
        lines.extend(f"  {message}" for message in result.diagnostics)
    return "\n".join(lines) + "\n"


def ebr_result_to_dict(result: EBRAnalysisResult) -> dict:
    names = tuple(column.name for column in result.ebr_matrix.columns)
    return {
        "mode": result.mode,
        "catalog": {
            "name": result.catalog.name,
            "source": result.catalog.source,
            "space_group_number": result.catalog.space_group_number,
            "hall_number": result.catalog.hall_number,
            "space_group_name": result.catalog.space_group_name,
            "k_points": [
                {"name": point.name, "k": point.k_fractional.tolist()}
                for point in result.catalog.k_points
            ],
        },
        "symmetry_vector": {
            "total_dimension": result.symmetry_vector.total_dimension,
            "rows": [key.label for key in result.symmetry_vector.row_keys],
            "values": result.symmetry_vector.multiplicities.tolist(),
        },
        "ebr_matrix": {
            "rows": [key.label for key in result.ebr_matrix.row_keys],
            "columns": [
                {
                    "name": column.name,
                    "wyckoff": column.wyckoff,
                    "center": column.center.tolist(),
                    "site_irrep": column.site_irrep,
                    "dimension": int(dimension),
                }
                for column, dimension in zip(
                    result.ebr_matrix.columns, result.ebr_matrix.dimensions
                )
            ],
            "values": result.ebr_matrix.values.tolist(),
        },
        "regular_decompositions": [
            {
                "multiplicities": item.multiplicities.tolist(),
                "expression": _format_combination(item.multiplicities, names),
            }
            for item in result.regular_decompositions
        ],
        "tetb_solutions": [
            {
                "n_t_plus_l": item.n_t_plus_l.tolist(),
                "n_l": item.n_l.tolist(),
                "n_t": item.n_t.tolist(),
                "gamma_surrogate": item.gamma_surrogate.tolist(),
                "auxiliary_dimension": item.auxiliary_dimension,
                "physical": item.physical,
                "physical_reason": item.physical_reason,
                "composite": item.composite,
            }
            for item in result.tetb_solutions
        ],
        "subspace_candidates": [
            {
                "n_t_plus_l": item.solution.n_t_plus_l.tolist(),
                "n_l": item.solution.n_l.tolist(),
                "n_t": item.solution.n_t.tolist(),
                "gamma_surrogate": item.solution.gamma_surrogate.tolist(),
                "auxiliary_dimension": item.solution.auxiliary_dimension,
                "selected_symmetry_vector": item.selected_symmetry_vector.multiplicities.tolist(),
                "point_representations": [
                    {
                        "point": point.point_name,
                        "physical_irreps": dict(point.irrep_multiplicities),
                        "formal_signed_irreps": _point_symmetry_multiplicities(
                            result.ebr_matrix.row_keys,
                            result.ebr_matrix.values @ item.solution.n_t,
                            point.point_name,
                        ),
                        "unresolved_dimension": point.unresolved_dimension,
                        "alternative_count": point.alternative_count,
                        "channel_dimension_realizations": [
                            dict(realization)
                            for realization in point.channel_dimension_realizations
                        ],
                        "channel_assignment_complete": point.channel_assignment_complete,
                    }
                    for point in item.point_representations
                ],
                "includes_gamma_zero_modes": item.includes_gamma_zero_modes,
                "gamma_sector": item.gamma_sector,
                "block_realization_count": item.block_realization_count,
            }
            for item in result.subspace_candidates
        ],
        "subspace_fixed_bands": list(result.subspace_fixed_band_indices),
        "search_statistics": (
            None
            if result.search_statistics is None
            else {
                "complete": result.search_statistics.complete,
                "gamma_sectors": list(result.search_statistics.gamma_sectors),
                "auxiliary_dimensions_examined": list(
                    result.search_statistics.auxiliary_dimensions_examined
                ),
                "weighted_vectors_generated": result.search_statistics.weighted_vectors_generated,
                "signed_combinations_tested": result.search_statistics.signed_combinations_tested,
                "algebraic_solutions": result.search_statistics.algebraic_solutions,
                "unique_signed_solutions": result.search_statistics.unique_signed_solutions,
                "completion_solutions": result.search_statistics.completion_solutions,
                "realizable_candidates": result.search_statistics.realizable_candidates,
                "block_realization_count": result.search_statistics.block_realization_count,
                "search_limit": result.search_statistics.search_limit,
            }
        ),
        "optimal_auxiliary_dimension": result.optimal_auxiliary_dimension,
        "diagnostics": list(result.diagnostics),
    }


def _format_combination(values, names, *, signed: bool = False) -> str:
    terms = []
    for raw, name in zip(np.asarray(values, dtype=np.int64), names):
        value = int(raw)
        if value == 0:
            continue
        magnitude = "" if abs(value) == 1 else str(abs(value)) + " "
        term = magnitude + name
        if not terms:
            terms.append(("-" if value < 0 else "") + term)
        else:
            terms.append((" - " if value < 0 else " + ") + term)
    if not terms:
        return "0"
    if not signed and any(int(value) < 0 for value in values):
        raise ValueError("A non-signed EBR combination contains negative multiplicities.")
    return "".join(terms)


def _format_channel_realizations(point) -> str:
    if not point.channel_dimension_realizations:
        return ""
    realizations = " | ".join(
        ",".join(f"{name}:{value}" for name, value in realization)
        for realization in point.channel_dimension_realizations
    )
    suffix = "" if point.channel_assignment_complete else " (partial diagnostics)"
    return f"; channel_dimensions={realizations}{suffix}"


def _format_catalog_group(catalog) -> str:
    if catalog.hall_number is not None:
        return f"space_group: {catalog.space_group_number}; hall: {catalog.hall_number}"
    return f"space_group: {catalog.space_group_name}; dimension: {catalog.dimension}"


def _format_symmetry_vector(keys, values) -> str:
    entries = [
        f"{key.label}={int(value)}"
        for key, value in zip(keys, values)
        if int(value) != 0
    ]
    return ", ".join(entries) or "0"


def _format_physical_point_representation(
    multiplicities, *, unresolved_dimension: int = 0
) -> str:
    terms = []
    if unresolved_dimension:
        terms.append(f"{int(unresolved_dimension)} unresolved transverse zero modes")
    for name, raw in multiplicities:
        value = int(raw)
        terms.append(str(name) if value == 1 else f"{value} {name}")
    return " + ".join(terms) or "0"


def _point_symmetry_multiplicities(keys, values, point_name: str) -> dict[str, int]:
    return {
        key.irrep_name: int(value)
        for key, value in zip(keys, values)
        if key.point_name == point_name and int(value) != 0
    }


def _format_point_symmetry_vector(keys, values, point_name: str) -> str:
    multiplicities = _point_symmetry_multiplicities(keys, values, point_name)
    return _format_combination(
        tuple(multiplicities.values()),
        tuple(multiplicities),
        signed=True,
    )


def _resolve_output(value, base_dir, out_dir) -> Path | None:
    if value is None or value is False or str(value).strip().lower() == "false":
        return None
    path = Path(str(value))
    if out_dir is not None:
        output = Path(out_dir)
        return output / path.name
    return path if path.is_absolute() else Path(base_dir) / path
