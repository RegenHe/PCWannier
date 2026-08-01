from __future__ import annotations

import argparse
from copy import copy
from dataclasses import dataclass
import logging
import os
from pathlib import Path

from ._version import __version__
from .config import load_config
from .compute import run_bloch_symmetry_preanalysis, run_calculation
from .ebr import run_ebr_analysis, write_ebr_outputs
from .logging_utils import configure_logging
from .outputs import (
    write_base_figures,
    write_bloch_symmetry_outputs,
    write_interpolation_outputs,
    write_outputs,
)
from .runtime_info import format_elapsed, format_memory, memory_snapshot, now, start_memory_tracking
from .sources import load_input, load_mesh
from .symmetry.crystallography import symmetry_engine_versions
from .symmetry.io import (
    load_builtin_finite_groups,
    load_finite_group,
    load_point_group_from_spglib,
)
from .timing import timed_step

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOptions:
    backend: str | None
    use_cache: bool
    analysis_only: bool
    out_dir: Path | None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=f"PCWannier v{__version__}")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-i", "--input", help="Input incar file path")
    mode.add_argument(
        "-g",
        "--group",
        metavar="NAME",
        help=(
            "Print a finite-group character table and exit, for example: "
            "--group c4v or --group m-3m"
        ),
    )
    parser.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 1, help="Number of worker threads")
    parser.add_argument(
        "--backend",
        choices=("python", "numba", "auto"),
        default=None,
        help="Compute backend override. Default uses compute_backend from incar.",
    )
    parser.add_argument("-l", "--log", default="log.txt", help="Log file path")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    parser.add_argument("-o", "--out", default=None, help="Output directory override")
    analysis_mode = parser.add_mutually_exclusive_group()
    analysis_mode.add_argument(
        "-s",
        "--analyze-symmetry",
        action="store_true",
        help="Analyze outer-window Bloch symmetry, write S/D caches, and exit",
    )
    analysis_mode.add_argument(
        "-e",
        "--analyze-ebr",
        action="store_true",
        help="Analyze Bloch symmetry, find EBR/TETB decompositions, and exit",
    )
    parser.add_argument("-b", "--base", action="store_true", help="Plot projection base functions and exit")
    parser.add_argument("-c", "--cache", action="store_true", help="Use cached calculation matrices")
    parser.add_argument("-I", "--interp", default=None, help="Interpolation mesh point path")
    parser.add_argument("-W", "--interp-wannier", default=None, help="Interpolated Wannier output path")
    parser.add_argument("-M", "--interp-metric", default=None, help="Interpolated metric-material output path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    started_at = now()
    args = parse_args(argv)
    if args.group is not None:
        if args.analyze_symmetry or args.analyze_ebr:
            raise ValueError("Symmetry/EBR analysis requires -i/--input and cannot be used with --group.")
        print(_format_finite_group_table(_load_cli_finite_group(args.group)))
        return 0
    analysis_only = args.analyze_symmetry or args.analyze_ebr
    if analysis_only and (
        args.base
        or args.interp is not None
        or args.interp_wannier is not None
        or args.interp_metric is not None
    ):
        raise ValueError(
            "Analysis-only modes cannot be combined with --base or interpolation outputs."
        )

    start_memory_tracking()
    out_dir = Path(args.out).expanduser().resolve() if args.out is not None else None
    log_path = Path(args.log)
    if out_dir is not None and not log_path.is_absolute():
        log_path = out_dir / log_path
    configure_logging(log_path, getattr(logging, args.log_level))

    if args.interp is None and (args.interp_wannier is not None or args.interp_metric is not None):
        raise ValueError("--interp is required when --interp-wannier or --interp-metric is provided.")

    LOGGER.info("=========  PCWannier v%s  =========", __version__)
    LOGGER.info("input=%s out_dir=%s log=%s threads=%s backend_override=%s", args.input, out_dir, log_path, args.threads, args.backend)

    with timed_step("load config", LOGGER, input=args.input):
        loaded_config = (
            load_config(args.input, mode="bloch_symmetry")
            if analysis_only
            else load_config(args.input)
        )
    config = _apply_run_options(
        loaded_config,
        RunOptions(args.backend, args.cache, analysis_only, out_dir),
    )

    LOGGER.info(
        "config name=%s dataset_type=%s field=%s primary=%s metric=%s curl=%s "
        "metric_file=%s kdim=%s k_shape=%s band_calc_num=%s backend=%s",
        config.name,
        config.dataset_type,
        config.maxwell_problem.field_components.value,
        config.maxwell_problem.primary_field.value,
        config.maxwell_problem.metric_material.value,
        config.maxwell_problem.curl_material.value,
        config.input_path(config.metric_file),
        config.kdim,
        [len(axis) for axis in config.k_points],
        config.band_calc_num,
        config.compute_backend,
    )
    if config.symmetry_context is not None:
        symmetry_model = config.symmetry_context.model
        spglib_version, spgrep_version = symmetry_engine_versions()
        definition = symmetry_model.group_definition
        target_summary = ", ".join(
            f"{target.name}:{target.wannier_dimension}" for target in symmetry_model.targets
        ) or "none"
        LOGGER.info(
            "symmetry file=%s group=%s point_group=%s spglib=%s spgrep=%s "
            "operations=%s targets=%s target_dimensions=%s "
            "constrained_localization=%s output_basis=%s bloch_convention=%s(sign=%s) "
            "magnetic_bias=%s unitary=%s antiunitary=%s",
            config.symmetry_resolved_path
            or config.symmetry_resolved_reference
            or config.input_path(config.symmetry_file),
            (
                definition.name
                if definition is not None
                else "legacy"
            ),
            None if definition is None else definition.point_group.symbol,
            spglib_version,
            spgrep_version,
            len(symmetry_model.group.operations),
            len(symmetry_model.targets),
            target_summary,
            config.symmetry_constrained,
            config.symmetry_output_basis,
            symmetry_model.bloch_convention.name,
            symmetry_model.bloch_convention.sign,
            (
                None
                if symmetry_model.magnetic_bias_direction is None
                else symmetry_model.magnetic_bias_direction.tolist()
            ),
            sum(not operation.antiunitary for operation in symmetry_model.group.operations),
            sum(operation.antiunitary for operation in symmetry_model.group.operations),
        )

    if args.base:
        mesh_path = config.input_path(config.mesh_file)
        with timed_step("load mesh for base figures", LOGGER, file=mesh_path):
            mesh = load_mesh(config)
        with timed_step("write projection base figures", LOGGER, out_dir=out_dir or config.base_dir):
            write_base_figures(config, mesh, out_dir)
        _log_run_summary(started_at)
        return 0

    with timed_step("load input data", LOGGER, dataset_type=config.dataset_type):
        bundle = load_input(config)
    if analysis_only:
        if config.wannier_targets or config.symmetry_constrained:
            LOGGER.info(
                "Bloch symmetry analysis-only mode ignores Wannier targets and constrained-gauge settings."
            )
        with timed_step(
            "run outer-window Bloch symmetry preanalysis",
            LOGGER,
            threads=max(1, int(args.threads)),
            backend=config.compute_backend,
        ):
            result = run_bloch_symmetry_preanalysis(
                bundle,
                threads=max(1, int(args.threads)),
                backend=args.backend,
            )
        with timed_step("write Bloch symmetry caches", LOGGER, out_dir=out_dir or config.base_dir):
            write_bloch_symmetry_outputs(result, config, out_dir)
        if args.analyze_ebr:
            with timed_step("analyze EBR decomposition", LOGGER):
                ebr_result = run_ebr_analysis(result, result.symmetry, config)
            with timed_step("write EBR analysis", LOGGER, out_dir=out_dir or config.base_dir):
                write_ebr_outputs(ebr_result, config, out_dir)
            LOGGER.info(
                "EBR analysis: catalog=%s mode=%s regular_solutions=%s "
                "physical_tetb_solutions=%s subspace_candidates=%s "
                "optimal_auxiliary_dimension=%s",
                ebr_result.catalog.name,
                ebr_result.mode,
                len(ebr_result.regular_decompositions),
                len(ebr_result.physical_tetb_solutions),
                len(ebr_result.subspace_candidates),
                ebr_result.optimal_auxiliary_dimension,
            )
        _log_run_summary(started_at)
        return 0
    with timed_step("run calculation", LOGGER, threads=max(1, int(args.threads)), backend=config.compute_backend):
        result = run_calculation(bundle, threads=max(1, int(args.threads)), backend=args.backend)
    with timed_step("write outputs", LOGGER, out_dir=out_dir or config.base_dir):
        write_outputs(result, config, out_dir)
    if args.interp is not None:
        with timed_step("write interpolation outputs", LOGGER, interp=args.interp):
            write_interpolation_outputs(
                result,
                args.interp,
                args.interp_wannier,
                args.interp_metric,
                out_dir=out_dir,
            )
    _log_run_summary(started_at)
    return 0


def _log_run_summary(started_at: float) -> None:
    LOGGER.info("total runtime: %s", format_elapsed(started_at))
    LOGGER.info("memory usage: %s", format_memory(memory_snapshot()))
    LOGGER.info("Done")


def _apply_run_options(config, options: RunOptions):
    """Apply CLI-only overrides without mutating the parsed incar object."""

    resolved = copy(config)
    if options.backend is not None:
        resolved.compute_backend = options.backend
    if not options.use_cache:
        resolved.use_cached_data = list(getattr(config, "use_cached_data", ()))
        return resolved
    if options.analysis_only:
        resolved.use_cached_data = ["S", "D"]
        _configure_analysis_cache_paths(resolved, options.out_dir)
        return resolved
    resolved.use_cached_data = ["V", "M", "S", "A", "D"]
    if not resolved.symmetry_constrained:
        resolved.use_cached_data.insert(0, "U")
    if options.out_dir is not None:
        _redirect_cache_paths_to_out_dir(resolved, options.out_dir)
    return resolved


def _redirect_cache_paths_to_out_dir(config, out_dir: Path, *, attrs=None) -> None:
    out_dir = Path(out_dir).expanduser().resolve()
    names = attrs or ("M_file", "A_file", "V_file", "U_file", "S_file", "D_file")
    for attr in names:
        value = getattr(config, attr, None)
        if value is None or value is False or str(value).lower() == "false":
            continue
        path = Path(str(value))
        setattr(config, attr, str(out_dir / path.name))


def _configure_analysis_cache_paths(config, out_dir: Path | None) -> None:
    """Point analysis cache input and output at the same absolute directory."""

    directory = Path(out_dir or config.base_dir).expanduser().resolve()
    for attr, default_name in (
        ("S_file", "S.txt"),
        ("D_file", "D.txt"),
        ("longitudinal_S_file", "S_L.txt"),
        ("longitudinal_D_file", "D_L.txt"),
    ):
        value = getattr(config, attr, None)
        filename = (
            default_name
            if value is None or value is False or str(value).lower() == "false"
            else Path(str(value)).name
        )
        setattr(config, attr, str(directory / filename))


def _load_cli_finite_group(value: str):
    raw_value = str(value).strip()
    requested = Path(raw_value).expanduser()
    candidates = [requested]
    if requested.suffix == "":
        candidates.append(requested.with_suffix(".yaml"))
    for candidate in candidates:
        if candidate.is_file():
            return load_finite_group(candidate.resolve())

    key = requested.stem.casefold()
    library = load_builtin_finite_groups()
    matches = [
        definition
        for definition in library.definitions
        if definition.name.casefold() == key
    ]
    if len(matches) == 1:
        return matches[0]
    try:
        return load_point_group_from_spglib(raw_value)
    except ValueError as exc:
        point_group_error = str(exc)
    available = ", ".join(definition.name for definition in library.definitions)
    raise ValueError(
        f"Unknown finite group {value!r}; built-in groups: {available}. "
        f"3D point-group lookup also failed: {point_group_error}"
    )


def _format_finite_group_table(definition) -> str:
    table = definition.table
    classes = table.conjugacy_classes
    class_names = [f"K{index + 1}" for index in range(len(classes))]
    rows = [["irrep", "dim", *class_names]]
    for irrep in definition.irreps:
        rows.append(
            [
                irrep.name,
                str(irrep.dimension),
                *[
                    _format_character(irrep.characters[value.element_indices[0]])
                    for value in classes
                ],
            ]
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    rendered_rows = [
        "  ".join(value.rjust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    class_lines = [
        f"  {name}: {', '.join(table.element_names[index] for index in value.element_indices)}"
        for name, value in zip(class_names, classes)
    ]
    return "\n".join(
        [
            f"Finite group: {definition.name}",
            f"Point-group symbol: {definition.point_group_symbol}",
            f"Order: {table.order}",
            f"Canonical elements: {', '.join(table.element_names)}",
            "Conjugacy classes:",
            *class_lines,
            "Character table:",
            *rendered_rows,
            "Available site_irrep names: "
            + ", ".join(irrep.name for irrep in definition.irreps),
        ]
    )


def _format_character(value: complex) -> str:
    scalar = complex(value)
    tolerance = 1.0e-10
    real = 0.0 if abs(scalar.real) < tolerance else scalar.real
    imag = 0.0 if abs(scalar.imag) < tolerance else scalar.imag
    if imag == 0.0:
        rounded = round(real)
        return str(int(rounded)) if abs(real - rounded) < tolerance else f"{real:.8g}"
    if real == 0.0:
        if abs(imag - 1.0) < tolerance:
            return "i"
        if abs(imag + 1.0) < tolerance:
            return "-i"
        return f"{imag:.8g}i"
    sign = "+" if imag > 0.0 else "-"
    return f"{real:.8g}{sign}{abs(imag):.8g}i"


if __name__ == "__main__":
    raise SystemExit(main())
