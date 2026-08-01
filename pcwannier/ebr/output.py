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
        f"space_group: {result.catalog.space_group_number}; hall: {result.catalog.hall_number}",
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
    else:
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


def _format_symmetry_vector(keys, values) -> str:
    entries = [
        f"{key.label}={int(value)}"
        for key, value in zip(keys, values)
        if int(value) != 0
    ]
    return ", ".join(entries) or "0"


def _resolve_output(value, base_dir, out_dir) -> Path | None:
    if value is None or value is False or str(value).strip().lower() == "false":
        return None
    path = Path(str(value))
    if out_dir is not None:
        output = Path(out_dir)
        return output / path.name
    return path if path.is_absolute() else Path(base_dir) / path
