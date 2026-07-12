from __future__ import annotations

import logging
import numpy as np

from ..data import InputBundle, RunResult
from ..timing import timed_step
from .backend import resolve_backend
from .context import CalculationContext
from .gradient import Gradient
from .initializer import StateInitializer
from .integration import numba_parallel_policy
from .matrix import MSet
from .parallel import ParallelExecutor
from .state import StateCollection
from .tba import TBAModel
from .threading import blas_thread_limit, threadpool_summary
from .topology import calculate_topology
from .wannier import generate_wannier

LOGGER = logging.getLogger(__name__)


def run_calculation(bundle: InputBundle, *, threads: int = 1, backend: str | None = None) -> RunResult:
    with blas_thread_limit(threads):
        with numba_parallel_policy(max(1, int(threads)) <= 1), ParallelExecutor(threads):
            return _run_calculation(bundle, threads=threads, backend=backend)


def _run_calculation(bundle: InputBundle, *, threads: int = 1, backend: str | None = None) -> RunResult:
    config = bundle.config
    resolved_backend = resolve_backend(backend or config.compute_backend)
    LOGGER.info(
        "Calculation setup: threads=%s backend=%s integration=%s blas=%s k_shape=%s mesh_vertices=%s mesh_triangles=%s",
        threads,
        resolved_backend,
        config.integration_mode,
        threadpool_summary(),
        bundle.fields.shape,
        bundle.mesh.vertices.shape[0],
        bundle.mesh.elements.shape[0],
    )
    state = StateCollection(bundle, backend=resolved_backend, threads=threads)
    with timed_step("check orthogonality", LOGGER):
        report, need_orth = state.check_orthogonality()
    LOGGER.info(
        "Orthogonality report: need_orth=%s max_diag_err=%.6g max_offdiag=%.6g min_lambda=%.6g",
        need_orth,
        float(np.max(report[..., 1])),
        float(np.max(report[..., 2])),
        float(np.min(report[..., 4])),
    )
    if need_orth:
        with timed_step("orthogonalize states", LOGGER):
            state.orthogonalize()
        with timed_step("recheck orthogonality", LOGGER):
            report, need_orth = state.check_orthogonality()
        if need_orth:
            raise RuntimeError("Orthogonalization failed.")
        if config.disable_orth:
            fem_report, _ = state.check_orthogonality(apply_transform=False)
            LOGGER.info(
                "Orthogonalization mode: mixed (strict internally, FEM-normalized output); "
                "output max_diag_err=%.6g max_offdiag=%.6g",
                float(np.max(fem_report[..., 1])),
                float(np.max(fem_report[..., 2])),
            )
        else:
            LOGGER.info("Orthogonalization mode: strict (correction applied internally and to output)")
    else:
        state.ensure_identity_transform()
        LOGGER.info("Orthogonalization mode: identity (input states already orthonormal)")
    with timed_step("extend mesh", LOGGER, extension=config.extension):
        state.extention(config.extension)
    LOGGER.info(
        "Extended mesh: vertices=%s triangles=%s",
        state.extention_mesh.vertices.shape[0],
        state.extention_mesh.elements.shape[0],
    )

    mset = MSet(state, threads=threads)
    with timed_step("initialize M0", LOGGER):
        mset.init_M0()
    initializer = StateInitializer(state, mset, threads=threads)
    with timed_step("projection initialization", LOGGER, max_iter=config.max_iter, err_diff=config.err_diff):
        initializer.iter(config.err_diff, config.max_iter)
    gradient = Gradient(state, mset, threads=threads)
    with timed_step("gradient optimization", LOGGER, max_iter=config.max_iter, epsilon=config.epsilon):
        gradient.iter(config.err_diff, config.max_iter, config.epsilon)
    LOGGER.info(
        "Gradient result: omega=%s omega_I=%s omega_OD=%s omega_D=%s rn_shape=%s",
        float(np.sum(gradient.omega)),
        float(gradient.omega[0]),
        float(gradient.omega[1]),
        float(gradient.omega[2]),
        gradient.rn.shape,
    )

    ctx = CalculationContext(config, state, mset, initializer, gradient)
    with timed_step("generate Wannier functions", LOGGER):
        r_key, wannier, norms = generate_wannier(ctx)
    LOGGER.info(
        "Wannier generated: r=%s shape=%s norm_real_min=%.6g norm_real_max=%.6g "
        "norm_imag_max=%.6g centers_rn=%s",
        r_key,
        wannier.shape,
        float(np.min(np.real(norms))),
        float(np.max(np.real(norms))),
        float(np.max(np.abs(np.imag(norms)))),
        np.real_if_close(gradient.rn.T).tolist(),
    )
    tba = TBAModel(ctx, threads=threads)
    with timed_step("collect hopping matrices", LOGGER):
        hoppings = tba.collect_hoppings()
    LOGGER.info("Hopping matrices collected: count=%s", len(hoppings))
    with timed_step("calculate high-symmetry bands", LOGGER, enabled=bool(config.k_path)):
        band = tba.gen_hs_bands(hoppings) if config.k_path else None
    if band is not None and (config.Chern_number or config.hybrid_Wilson_loop):
        with timed_step("calculate Brillouin-zone bands", LOGGER, k_num=config.k_num):
            tba.gen_bz_bands(band, hoppings)
    with timed_step("calculate topology", LOGGER, enabled=band is not None):
        topology = calculate_topology(band, config) if band is not None else None

    return RunResult(
        config=config,
        mesh=state.mesh,
        extended_mesh=state.extention_mesh,
        extended_epsilon=state.extention_epsilon,
        orthogonality_report=report,
        S=state.S,
        M0=mset.mM0,
        A=initializer.matA,
        V=initializer.matV,
        U=gradient.U,
        omega=gradient.omega,
        rn=gradient.rn,
        wanniers={r_key: wannier},
        wannier_norms=norms,
        hoppings=hoppings,
        band=band,
        topology=topology,
    )
