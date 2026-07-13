from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from ._version import __version__
from .config import load_config
from .compute import run_calculation
from .logging_utils import configure_logging
from .outputs import write_base_figures, write_interpolation_outputs, write_outputs
from .runtime_info import format_elapsed, format_memory, memory_snapshot, now, start_memory_tracking
from .sources.comsol import load_comsol_mesh, load_input
from .timing import timed_step

LOGGER = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=f"PCWannier v{__version__}")
    parser.add_argument("-i", "--input", required=True, help="Input incar file path")
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
    parser.add_argument("--out", default=None, help="Output directory override")
    parser.add_argument("-b", "--base", action="store_true", help="Plot projection base functions and exit")
    parser.add_argument("-c", "--cache", action="store_true", help="Use cached M/A/V/U matrices")
    parser.add_argument("--interp", default=None, help="Interpolation mesh point path")
    parser.add_argument("--interp-wannier", default=None, help="Interpolated Wannier output path")
    parser.add_argument("--interp-epsilon", default=None, help="Interpolated epsilon output path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    started_at = now()
    start_memory_tracking()
    args = parse_args(argv)
    out_dir = Path(args.out) if args.out is not None else None
    log_path = Path(args.log)
    if out_dir is not None and not log_path.is_absolute():
        log_path = out_dir / log_path
    configure_logging(log_path, getattr(logging, args.log_level))

    if args.interp is None and (args.interp_wannier is not None or args.interp_epsilon is not None):
        raise ValueError("--interp is required when --interp-wannier or --interp-epsilon is provided.")

    LOGGER.info("=========  PCWannier v%s  =========", __version__)
    LOGGER.info("input=%s out_dir=%s log=%s threads=%s backend_override=%s", args.input, out_dir, log_path, args.threads, args.backend)

    with timed_step("load config", LOGGER, input=args.input):
        config = load_config(args.input)
    if args.backend is not None:
        config.compute_backend = args.backend
    if args.cache:
        config.use_cached_data = ["U", "V", "M", "S", "A"]
        if out_dir is not None:
            _redirect_cache_paths_to_out_dir(config, out_dir)

    LOGGER.info(
        "config name=%s dataset_type=%s kdim=%s k_shape=%s band_calc_num=%s backend=%s",
        config.name,
        config.dataset_type,
        config.kdim,
        [len(axis) for axis in config.k_points],
        config.band_calc_num,
        config.compute_backend,
    )
    if config.symmetry_context is not None:
        symmetry_model = config.symmetry_context.model
        target_summary = ", ".join(
            f"{target.name}:{target.wannier_dimension}" for target in symmetry_model.targets
        ) or "none"
        LOGGER.info(
            "symmetry file=%s operations=%s targets=%s target_dimensions=%s constrained_localization=%s",
            config.input_path(config.symmetry_file),
            len(symmetry_model.group.operations),
            len(symmetry_model.targets),
            target_summary,
            config.symmetry_constrained,
        )

    if args.base:
        mesh_path = config.input_path(config.mesh_file)
        if mesh_path is None:
            raise ValueError("mesh_file is required when --base is used.")
        with timed_step("load mesh for base figures", LOGGER, file=mesh_path):
            mesh = load_comsol_mesh(mesh_path)
        with timed_step("write projection base figures", LOGGER, out_dir=out_dir or config.base_dir):
            write_base_figures(config, mesh, out_dir)
        _log_run_summary(started_at)
        return 0

    with timed_step("load input data", LOGGER, dataset_type=config.dataset_type):
        bundle = load_input(config)
    with timed_step("run calculation", LOGGER, threads=max(1, int(args.threads)), backend=config.compute_backend):
        result = run_calculation(bundle, threads=max(1, int(args.threads)), backend=args.backend)
    with timed_step("write outputs", LOGGER, out_dir=out_dir or config.base_dir):
        write_outputs(result, config, out_dir)
    if args.interp is not None:
        with timed_step("write interpolation outputs", LOGGER, interp=args.interp):
            write_interpolation_outputs(result, args.interp, args.interp_wannier, args.interp_epsilon, out_dir=out_dir)
    _log_run_summary(started_at)
    return 0


def _log_run_summary(started_at: float) -> None:
    LOGGER.info("total runtime: %s", format_elapsed(started_at))
    LOGGER.info("memory usage: %s", format_memory(memory_snapshot()))
    LOGGER.info("Done")


def _redirect_cache_paths_to_out_dir(config, out_dir: Path) -> None:
    for attr in ("M_file", "A_file", "V_file", "U_file", "S_file"):
        value = getattr(config, attr)
        if value is None or value is False or str(value).lower() == "false":
            continue
        path = Path(str(value))
        setattr(config, attr, str(out_dir / path.name))


if __name__ == "__main__":
    raise SystemExit(main())
