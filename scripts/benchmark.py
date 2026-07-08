from __future__ import annotations

import argparse
import logging
import time

from pcwannier import load_config
from pcwannier.compute import run_calculation
from pcwannier.logging_utils import configure_logging
from pcwannier.sources.comsol import load_input


def parse_args():
    parser = argparse.ArgumentParser(description="PCWannier benchmark helper")
    parser.add_argument("-i", "--input", default="data/incar")
    parser.add_argument("-t", "--threads", type=int, default=1)
    parser.add_argument("--backend", choices=("python", "numba", "auto"), default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--disable-topology", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(False)
    cfg = load_config(args.input)
    if args.backend is not None:
        cfg.compute_backend = args.backend
    if args.max_iter is not None:
        cfg.max_iter = args.max_iter
    if args.disable_topology:
        cfg.hybrid_Wilson_loop = False
        cfg.Chern_number = False
        cfg.topo_output = "false"

    started = time.perf_counter()
    bundle = load_input(cfg)
    run_calculation(bundle, threads=args.threads, backend=args.backend)
    elapsed = time.perf_counter() - started
    logging.info("benchmark total runtime: %.3f s", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
