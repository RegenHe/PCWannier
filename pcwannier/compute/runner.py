from __future__ import annotations

import logging
import numpy as np
from dataclasses import replace

from ..data import BlochSymmetryRunResult, InputBundle, RunResult
from ..symmetry import (
    StateBlochSymmetryProvider,
    construct_symmetry_gauge,
    disentangle_symmetry_constrained,
    evaluate_symmetry_gauge,
    localize_symmetry_constrained,
    outer_band_grid,
    run_bloch_symmetry_analysis,
    run_symmetry_analysis,
    validate_frozen_window_covariance,
    validate_outer_window_closure,
    validate_wannier_symmetry,
)
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


def run_bloch_symmetry_preanalysis(
    bundle: InputBundle,
    *,
    threads: int = 1,
    backend: str | None = None,
) -> BlochSymmetryRunResult:
    """Prepare physical Bloch states, analyze configured points, and stop before Wannier work."""

    if bundle.symmetry is None or bundle.symmetry.model.representation_analysis is None:
        raise ValueError(
            "Bloch symmetry preanalysis requires symmetry_file and representation_analysis."
        )
    with blas_thread_limit(threads):
        with numba_parallel_policy(max(1, int(threads)) <= 1), ParallelExecutor(threads):
            resolved_backend = resolve_backend(backend or bundle.config.compute_backend)
            state, report = _prepare_state(
                bundle, threads=threads, resolved_backend=resolved_backend
            )
            provider = StateBlochSymmetryProvider(
                state,
                bundle.symmetry,
                field_kind=bundle.maxwell.symmetry_field_kind,
            )
            with timed_step("analyze outer-window Bloch symmetry", LOGGER):
                analysis = run_bloch_symmetry_analysis(
                    state, bundle.symmetry, provider=provider
                )
            _log_symmetry_analysis(analysis)
            if state.S is None:
                raise RuntimeError("Raw S overlap cache was not initialized during preanalysis.")
            return BlochSymmetryRunResult(
                config=bundle.config,
                orthogonality_report=report,
                S=state.S,
                symmetry=bundle.symmetry,
                analysis=analysis,
                sewing_matrices=provider.cached_sewing_matrices,
                sewing_calculation_fingerprint=provider.sewing_cache_fingerprint,
            )


def _prepare_state(
    bundle: InputBundle,
    *,
    threads: int,
    resolved_backend: str,
) -> tuple[StateCollection, np.ndarray]:
    config = bundle.config
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
        if config.symmetry_constrained:
            fem_report, _ = state.check_orthogonality(apply_transform=False)
            LOGGER.info(
                "Orthogonalization mode: strict internally, symmetry output basis=%s; "
                "FEM-normalized basis max_diag_err=%.6g max_offdiag=%.6g",
                config.symmetry_output_basis,
                float(np.max(fem_report[..., 1])),
                float(np.max(fem_report[..., 2])),
            )
        elif config.disable_orth:
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
    state.turn_to_bloch()
    return state, report


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
    state, report = _prepare_state(bundle, threads=threads, resolved_backend=resolved_backend)
    symmetry_analysis = None
    symmetry_provider = None
    if bundle.symmetry is not None and (
        bundle.symmetry.model.representation_analysis is not None
        or config.symmetry_constrained
    ):
        symmetry_provider = StateBlochSymmetryProvider(
            state,
            bundle.symmetry,
            field_kind=bundle.maxwell.symmetry_field_kind,
        )
    if bundle.symmetry is not None and bundle.symmetry.model.representation_analysis is not None:
        with timed_step("analyze Bloch symmetry representations", LOGGER):
            symmetry_analysis = run_symmetry_analysis(
                state, bundle.symmetry, provider=symmetry_provider
            )
        _log_symmetry_analysis(symmetry_analysis)
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
    if config.symmetry_constrained:
        with timed_step("projection initialization", LOGGER):
            initializer.prepare()
            # Frozen selectors define the subspace but not the target-column
            # gauge.  Align that frame to (p_x, p_y, s, ...) before applying
            # little-group projection; otherwise circularly split frozen
            # eigenstates can be assigned directly to real target columns.
            initializer.align_to_projection()
    else:
        with timed_step("projection initialization", LOGGER, max_iter=config.max_iter, err_diff=config.err_diff):
            initializer.iter(config.err_diff, config.max_iter)
    gradient = Gradient(state, mset, threads=threads)
    symmetry_gauge = None
    symmetry_localization = None
    symmetry_disentanglement = None
    gauge_spec = bundle.symmetry.model.symmetry_gauge if bundle.symmetry is not None else None
    if config.symmetry_constrained:
        if gauge_spec is None or not gauge_spec.enabled or bundle.symmetry is None:
            raise ValueError(
                "symmetry_constrained=true requires valid symmetry targets and gauge settings in incar."
            )
        if "U" in config.use_cached_data:
            raise ValueError(
                "Cached gradient U is incompatible with symmetry-constrained localization; "
                "use a cached V as the initial gauge instead."
            )
        _validate_symmetry_gauge_prerequisites(symmetry_analysis, gauge_spec.tolerance)
        band_lengths = [len(state.E_idx[index]) for index in state.k_indices()]
        target_dimension = int(config.band_calc_num)
        if min(band_lengths) < target_dimension:
            raise ValueError(
                f"Outer window contains fewer than N_W={target_dimension} states at some k point."
            )
        entangled = any(length > target_dimension for length in band_lengths)
        closure = None
        bands_by_k = outer_band_grid(state)
        if entangled:
            with timed_step("validate symmetry outer window", LOGGER):
                closure = validate_outer_window_closure(
                    state,
                    bundle.symmetry,
                    symmetry_provider,
                    tolerance=gauge_spec.tolerance,
                )
                validate_frozen_window_covariance(
                    initializer,
                    bundle.symmetry,
                    symmetry_provider,
                    bands_by_k,
                    tolerance=gauge_spec.tolerance,
                )
            LOGGER.info(
                "Outer-window symmetry: matrices=%s unitarity_max=%.6g leakage_max=%.6g composition=%.6g",
                closure.matrix_count,
                closure.max_unitarity_error,
                closure.max_leakage,
                closure.max_composition_residual,
            )
        disentangle_max_iter = (
            config.max_iter
            if config.disentangle_max_iter is None
            else config.disentangle_max_iter
        )
        disentangle_err_diff = (
            config.err_diff
            if config.disentangle_err_diff is None
            else config.disentangle_err_diff
        )
        disentangle_projector_tolerance = (
            config.symmetry_tolerance
            if config.disentangle_projector_tolerance is None
            else config.disentangle_projector_tolerance
        )
        identity_only = len(bundle.symmetry.model.group.operations) == 1
        if entangled and identity_only:
            with timed_step(
                "identity-group disentanglement",
                LOGGER,
                max_iter=disentangle_max_iter,
                err_diff=disentangle_err_diff,
            ):
                initializer.run_unconstrained_disentanglement(
                    disentangle_err_diff, disentangle_max_iter
                )
                initializer.align_to_projection()
        with timed_step("construct symmetry-adapted Bloch gauge", LOGGER):
            symmetry_gauge = construct_symmetry_gauge(
                state,
                bundle.symmetry,
                initializer.matV,
                threads=threads,
                tolerance=gauge_spec.tolerance,
                max_iterations=gauge_spec.max_iterations,
                svd_relative_tolerance=gauge_spec.svd_relative_tolerance,
                provider=symmetry_provider,
            )
        if entangled:
            run_iterations = 0 if identity_only else disentangle_max_iter
            with timed_step(
                "symmetry-constrained disentanglement",
                LOGGER,
                max_iter=run_iterations,
                err_diff=disentangle_err_diff,
                projector_tolerance=disentangle_projector_tolerance,
                mixing=config.disentangle_mixing,
            ):
                symmetry_disentanglement = disentangle_symmetry_constrained(
                    initializer,
                    bundle.symmetry,
                    symmetry_gauge,
                    symmetry_provider,
                    closure,
                    err_diff=disentangle_err_diff,
                    max_iter=run_iterations,
                    mixing=config.disentangle_mixing,
                    tolerance=gauge_spec.tolerance,
                    projection_max_iterations=gauge_spec.max_iterations,
                    svd_relative_tolerance=gauge_spec.svd_relative_tolerance,
                    projector_tolerance=disentangle_projector_tolerance,
                )
            initializer.matV = symmetry_disentanglement.optimal_frame
            gauge_residuals = evaluate_symmetry_gauge(
                state,
                bundle.symmetry,
                symmetry_provider,
                initializer.matV,
                symmetry_gauge.band_indices,
                symmetry_disentanglement.diagnostics.max_path_consistency,
                band_indices_by_k=symmetry_disentanglement.outer_band_indices,
            )
            symmetry_gauge = replace(
                symmetry_gauge,
                gauge=initializer.matV,
                residuals=gauge_residuals,
                band_indices_by_k=symmetry_disentanglement.outer_band_indices,
            )
            _log_symmetry_disentanglement(symmetry_disentanglement)
        else:
            initializer.matV = symmetry_gauge.gauge
        mset.initial(initializer.matV)
        with timed_step(
            "symmetry-constrained gradient optimization",
            LOGGER,
            max_iter=config.max_iter,
            epsilon=config.epsilon,
        ):
            symmetry_localization = localize_symmetry_constrained(
                gradient,
                state,
                bundle.symmetry,
                symmetry_gauge,
                symmetry_provider,
                err_diff=config.err_diff,
                max_iter=config.max_iter,
                epsilon=config.epsilon,
                tolerance=gauge_spec.tolerance,
                projection_max_iterations=gauge_spec.max_iterations,
                svd_relative_tolerance=gauge_spec.svd_relative_tolerance,
            )
        symmetry_gauge = replace(
            symmetry_gauge,
            gauge=symmetry_localization.final_gauge,
            residuals=symmetry_localization.residuals,
        )
        _log_symmetry_gauge(symmetry_gauge)
        _log_symmetry_localization(symmetry_localization)
        LOGGER.info("Symmetry-constrained output basis: %s", config.symmetry_output_basis)
        if (
            config.symmetry_output_basis == "strict"
            and config.disable_orth
            and state.is_orthogonalized
        ):
            LOGGER.warning(
                "disable_orth=true is overridden by symmetry_output_basis=strict; final Wannier/TBA "
                "outputs include the non-unitary orthogonalization correction. Set "
                "symmetry_output_basis=fem to preserve the normalized FEM spectrum."
            )
        elif (
            config.symmetry_output_basis == "fem"
            and not config.disable_orth
            and state.is_orthogonalized
        ):
            LOGGER.warning(
                "symmetry_output_basis=fem overrides disable_orth=false for final Wannier/TBA outputs; "
                "internal symmetry calculations remain strictly orthonormalized."
            )
    else:
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

    ctx = CalculationContext(config, state, mset, initializer, gradient, symmetry_gauge)
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
    if symmetry_gauge is not None and gauge_spec.validate_wannier:
        enforce_wannier_residual = config.symmetry_output_basis == "strict"
        with timed_step("validate real-space Wannier symmetry", LOGGER):
            validation = validate_wannier_symmetry(
                ctx,
                bundle.symmetry.model.targets,
                zero_cell_wanniers=wannier,
                tolerance=gauge_spec.real_space_tolerance,
                minimum_retained_norm=gauge_spec.minimum_retained_norm,
                enforce_residual=enforce_wannier_residual,
            )
        symmetry_gauge = replace(symmetry_gauge, real_space_validation=validation)
        ctx.symmetry_gauge = symmetry_gauge
        log_wannier_symmetry = (
            LOGGER.warning
            if not enforce_wannier_residual and validation.max_residual > gauge_spec.real_space_tolerance
            else LOGGER.info
        )
        log_wannier_symmetry(
            "Wannier symmetry: basis=%s max_residual=%.6g mean_residual=%.6g "
            "minimum_retained_norm=%.6g tolerance=%.6g%s",
            config.symmetry_output_basis,
            validation.max_residual,
            validation.mean_residual,
            validation.minimum_retained_norm,
            gauge_spec.real_space_tolerance,
            " (diagnostic only for FEM output)" if not enforce_wannier_residual else "",
        )
    tba = TBAModel(ctx, threads=threads)
    output_spectrum_diagnostics = tba.output_spectrum_diagnostics(symmetry_analysis)
    _log_output_spectrum_diagnostics(output_spectrum_diagnostics, config)
    with timed_step("collect hopping matrices", LOGGER):
        hoppings = tba.collect_hoppings()
    LOGGER.info("Hopping matrices collected: count=%s", len(hoppings))
    hopping_reconstruction_diagnostics = tba.hopping_reconstruction_diagnostics(
        hoppings, symmetry_analysis
    )
    _log_hopping_reconstruction_diagnostics(hopping_reconstruction_diagnostics)
    with timed_step("calculate high-symmetry bands", LOGGER, enabled=bool(config.k_path)):
        band = tba.gen_hs_bands(hoppings) if config.k_path else None
    if band is not None and (config.Chern_number or config.hybrid_Wilson_loop):
        with timed_step("calculate Brillouin-zone bands", LOGGER, k_num=config.k_num):
            tba.gen_bz_bands(band, hoppings)
    with timed_step("calculate topology", LOGGER, enabled=band is not None):
        topology = calculate_topology(band, config) if band is not None else None

    bloch_gauge = state.gen_matrix_on_kmesh(
        lambda i, j, k: np.asarray(ctx.bloch_gauge_at(i, j, k), dtype=np.complex128).copy()
    )
    return RunResult(
        config=config,
        mesh=state.mesh,
        extended_mesh=state.extention_mesh,
        extended_metric_material=state.extended_metric_material,
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
        bloch_gauge=bloch_gauge,
        symmetry=bundle.symmetry,
        symmetry_analysis=symmetry_analysis,
        symmetry_gauge=symmetry_gauge,
        symmetry_localization=symmetry_localization,
        symmetry_disentanglement=symmetry_disentanglement,
        output_spectrum_diagnostics=output_spectrum_diagnostics,
        hopping_reconstruction_diagnostics=hopping_reconstruction_diagnostics,
        sewing_matrices=(
            None if symmetry_provider is None else symmetry_provider.cached_sewing_matrices
        ),
        sewing_calculation_fingerprint=(
            None if symmetry_provider is None else symmetry_provider.sewing_cache_fingerprint
        ),
    )


def _log_symmetry_analysis(result) -> None:
    bloch_result = getattr(result, "bloch", None)
    points = bloch_result.points if bloch_result is not None else result.points
    for point in points:
        blocks = [tuple(band + 1 for band in block.band_indices) for block in point.degenerate_blocks]
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
            getattr(point, "unitary_subgroup_name", None) or point.little_group_name or "unresolved",
            getattr(point, "unitary_operation_names", ()),
            getattr(point, "antiunitary_operation_names", ()),
            point.conjugacy_classes,
            point.finite_group_mapping,
            point.sampled_k_fractional.tolist(),
            tuple(band + 1 for band in getattr(point, "outer_band_indices", point.band_indices)),
            tuple(band + 1 for band in point.band_indices),
            blocks,
            point.diagnostics.unitarity_error,
            getattr(point, "outer_unitarity_error", point.diagnostics.unitarity_error),
            point.diagnostics.leakage,
            point.diagnostics.max_composition_residual,
            point.diagnostics.max_twisted_composition_residual,
            point.factor_system.phase_residual if point.factor_system is not None else 0.0,
            point.factor_system.cocycle_residual if point.factor_system is not None else 0.0,
            point.factor_system.raw_trivial if point.factor_system is not None else True,
            (
                point.factor_system.cohomologically_trivial
                if point.factor_system is not None
                else True
            ),
            point.factor_system.bloch_sign if point.factor_system is not None else 1,
            {
                name: complex(value)
                for name, value in getattr(point, "unitary_characters", point.characters).items()
            },
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
            for antiunitary in block.antiunitary_diagnostics:
                LOGGER.info(
                    "Bloch antiunitary block %s bands(1-based)=%s operation=%s square=%s "
                    "square_eigenvalues=%s square_residual=%.6g",
                    point.name,
                    tuple(band + 1 for band in block.band_indices),
                    antiunitary.operation_name,
                    antiunitary.square_operation_name,
                    antiunitary.square_eigenvalues,
                    antiunitary.square_residual,
                )
        if point.factor_system is not None and any(point.factor_system.antiunitary_flags):
            LOGGER.info(
                "Symmetry point %s contains antiunitary operations: ordinary irrep labels "
                "unavailable (magnetic corepresentation database is not implemented)",
                point.name,
            )
        elif point.factor_system is not None and not point.factor_system.cohomologically_trivial:
            LOGGER.info(
                "Symmetry point %s uses a non-trivial projective factor: "
                "ordinary irrep labels unavailable",
                point.name,
            )
    for comparison in getattr(result, "target_compatibilities", ()):
        LOGGER.info(
            "Target compatibility %s: targets=%s target_irreps=%s compatible=%s "
            "direct_intertwiner_dimension=%s",
            comparison.point_name,
            comparison.target_names,
            (
                {}
                if comparison.target_decomposition is None
                else comparison.target_decomposition.multiplicities
            ),
            None if comparison.compatibility is None else comparison.compatibility.compatible,
            comparison.intertwiner_dimension,
        )


def _format_irrep_decomposition(decomposition) -> str:
    terms = []
    for name, multiplicity in decomposition.multiplicities.items():
        if multiplicity <= 0:
            continue
        terms.append(name if multiplicity == 1 else f"{multiplicity}{name}")
    return " + ".join(terms) or "none"


def _log_output_spectrum_diagnostics(result, config) -> None:
    if result is None:
        LOGGER.info("Output spectrum diagnostics unavailable for an entangled outer window")
        return
    LOGGER.info(
        "Output spectrum: basis=%s max_eigenvalue_drift=%.6g worst_k_index=%s",
        result.basis,
        result.max_eigenvalue_drift,
        result.worst_k_index,
    )
    drift_tolerance = max(
        float(config.symmetry_tolerance),
        float(config.representation_degeneracy_absolute),
    )
    if config.symmetry_constrained and result.max_eigenvalue_drift > drift_tolerance:
        LOGGER.warning(
            "Symmetry output basis %s changes the sampled FEM spectrum: "
            "max_eigenvalue_drift=%.6g at k_index=%s",
            result.basis,
            result.max_eigenvalue_drift,
            result.worst_k_index,
        )
    for splitting in result.degeneracy_splittings:
        if splitting.broken:
            LOGGER.warning(
                "Output basis %s breaks FEM degeneracy at %s bands(actual,1-based)=%s: "
                "raw_gap=%.6g output_gap=%.6g tolerance=%.6g",
                result.basis,
                splitting.point_name,
                tuple(index + 1 for index in splitting.band_indices),
                splitting.reference_gap,
                splitting.output_gap,
                splitting.tolerance,
            )


def _log_hopping_reconstruction_diagnostics(result) -> None:
    LOGGER.info(
        "Hopping reconstruction: max_matrix_error=%.6g max_eigenvalue_error=%.6g "
        "worst_k_index=%s",
        result.max_matrix_error,
        result.max_eigenvalue_error,
        result.worst_k_index,
    )
    for splitting in result.degeneracy_splittings:
        if splitting.broken:
            LOGGER.warning(
                "Configured hopping set breaks output degeneracy at %s bands(actual,1-based)=%s: "
                "direct_output_gap=%.6g reconstructed_gap=%.6g tolerance=%.6g",
                splitting.point_name,
                tuple(index + 1 for index in splitting.band_indices),
                splitting.reference_gap,
                splitting.output_gap,
                splitting.tolerance,
            )


def _validate_symmetry_gauge_prerequisites(analysis, tolerance: float) -> None:
    if analysis is None:
        return
    for point in analysis.points:
        if point.compatibility is not None and not point.compatibility.compatible:
            raise RuntimeError(f"Target representation is incompatible at symmetry point {point.name}.")
        if point.target_twisted_representation is not None and point.intertwiner_dimension == 0:
            raise RuntimeError(
                f"Target representation has no direct intertwiner at symmetry point {point.name}."
            )
        if point.diagnostics.unitarity_error > tolerance:
            raise RuntimeError(
                f"Physical sewing space is not closed at {point.name}: "
                f"unitarity residual={point.diagnostics.unitarity_error:.6g}."
            )
        if point.diagnostics.max_composition_residual > tolerance:
            raise RuntimeError(
                f"Sewing composition residual at {point.name} is "
                f"{point.diagnostics.max_composition_residual:.6g}."
            )


def _log_symmetry_gauge(result) -> None:
    LOGGER.info(
        "Symmetry gauge: stars=%s max_residual=%.6g mean_residual=%.6g "
        "path_residual=%.6g semiunitarity=%.6g",
        len(result.stars.stars),
        result.residuals.max_residual,
        result.residuals.mean_residual,
        result.residuals.max_path_consistency,
        result.residuals.max_semiunitarity_error,
    )
    for diagnostic in result.representative_diagnostics:
        if diagnostic.hom_dimension > 1 or diagnostic.target_commutant_dimension > 1:
            LOGGER.info(
                "Symmetry gauge representative %s: dim_Hom=%s residual_gauge_dimension=%s "
                "iterations=%s residual=%.6g",
                diagnostic.representative_index,
                diagnostic.hom_dimension,
                diagnostic.target_commutant_dimension,
                diagnostic.iterations,
                diagnostic.residual,
            )


def _log_symmetry_localization(result) -> None:
    final = result.iterations[-1]
    initial = result.iterations[0]
    LOGGER.info(
        "Symmetry localization: converged=%s iterations=%s omega_initial=%.12g omega_final=%.12g "
        "gradient_norm=%.6g symmetry_max=%.6g symmetry_mean=%.6g unitarity=%.6g path=%.6g",
        result.converged,
        final.iteration,
        initial.omega,
        final.omega,
        final.gradient_norm,
        final.max_intertwiner_residual,
        final.mean_intertwiner_residual,
        final.max_unitarity_error,
        final.max_path_consistency,
    )


def _log_symmetry_disentanglement(result) -> None:
    final = result.iterations[-1]
    LOGGER.info(
        "Symmetry disentanglement: converged=%s iterations=%s omega_I=%.12g "
        "projector_change=%.6g projector_symmetry=%.6g intertwiner=%.6g "
        "orthonormality=%.6g frozen=%.6g path=%.6g",
        result.converged,
        final.iteration,
        final.omega_i,
        final.projector_change,
        final.max_projector_symmetry_residual,
        final.max_intertwiner_residual,
        final.orthonormality_error,
        final.frozen_window_residual,
        final.path_consistency_residual,
    )
