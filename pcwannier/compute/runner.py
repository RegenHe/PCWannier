from __future__ import annotations

import logging
import numpy as np
from dataclasses import replace

from ..data import (
    BlochSymmetryChannelResult,
    BlochSymmetryRunResult,
    InputBundle,
    RunResult,
)
from ..etbc import complete_transverse_bundle
from ..symmetry.analysis import (
    regularize_gamma_zero_modes,
    run_bloch_symmetry_analysis,
    run_symmetry_analysis,
)
from ..symmetry.bloch import StateBlochSymmetryProvider
from ..symmetry.disentanglement import (
    disentangle_symmetry_constrained,
    outer_band_grid,
    validate_frozen_window_covariance,
    validate_outer_window_closure,
)
from ..symmetry.gauge import construct_symmetry_gauge, evaluate_symmetry_gauge
from ..symmetry.localization import localize_symmetry_constrained
from ..symmetry.representation import build_symmetry_context
from ..symmetry.wannier_validation import validate_wannier_symmetry
from ..timing import timed_step
from ..symmetry.reporting import (
    log_bloch_symmetry_analysis,
    log_gamma_zero_regularization,
    log_symmetry_analysis,
)
from .backend import resolve_backend
from .context import CalculationContext
from .gradient import Gradient
from .initializer import StateInitializer
from .integration import numba_parallel_policy
from .matrix import MSet
from .parallel import ParallelExecutor
from .state import StateCollection
from .subspace_diagnostics import (
    NeighborSubspaceSmoothness,
    outer_channel_smoothness,
    selected_sector_smoothness,
)
from .tba import TBAModel
from .threading import blas_thread_limit, threadpool_summary
from .topology import calculate_topology
from .vector_diagnostics import diagnose_bundle_vector_fields
from .wannier import generate_wannier
from .vector_trials import (
    build_vector_bloch_trial_grid,
    prepare_vector_trial_targets,
)

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
            physical = _analyze_bundle_channel(
                bundle,
                threads=threads,
                resolved_backend=resolved_backend,
                channel_name="physical",
            )
            auxiliary_channels = {}
            for channel_name, load_auxiliary in bundle.auxiliary_bundle_loaders.items():
                auxiliary = load_auxiliary()
                auxiliary_channels[channel_name] = _analyze_bundle_channel(
                    auxiliary,
                    threads=threads,
                    resolved_backend=resolved_backend,
                    channel_name=channel_name,
                )
                del auxiliary
            gamma_regularization = None
            if bundle.config.gamma_zero_regularization:
                longitudinal_zero_bands = bundle.auxiliary_zero_mode_bands.get(
                    "longitudinal"
                )
                if longitudinal_zero_bands is None:
                    raise ValueError(
                        "Gamma zero regularization requires longitudinal zero-mode metadata."
                    )
                gamma_point = physical.analysis.point("Gamma")
                storage_index = tuple(gamma_point.k_index) + (0,) * (
                    3 - len(gamma_point.k_index)
                )
                longitudinal_bands = np.asarray(
                    longitudinal_zero_bands[storage_index], dtype=int
                )
                physical.analysis, gamma_regularization = regularize_gamma_zero_modes(
                    physical.analysis,
                    bundle.symmetry,
                    tuple(int(value) for value in longitudinal_bands),
                    bundle.config.real_lattice_vectors,
                    energy_tolerance=bundle.config.gamma_zero_mode_tolerance,
                )
                log_gamma_zero_regularization(gamma_regularization)
            return BlochSymmetryRunResult(
                config=bundle.config,
                symmetry=bundle.symmetry,
                primary=physical,
                auxiliary_channels=auxiliary_channels,
                gamma_zero_regularization=gamma_regularization,
                band_channels=dict(bundle.band_channels),
            )


def _analyze_bundle_channel(
    bundle: InputBundle,
    *,
    threads: int,
    resolved_backend: str,
    channel_name: str,
) -> BlochSymmetryChannelResult:
    if bundle.symmetry is None:
        raise ValueError("Bloch symmetry channel is missing its symmetry context.")
    field_kind = bundle.analysis_field_kind or bundle.maxwell.symmetry_field_kind
    analysis_context = bundle.symmetry
    if channel_name == "longitudinal" and bundle.zero_modes is not None:
        specification = analysis_context.model.representation_analysis
        if specification is not None:
            longitudinal_specification = replace(
                specification,
                points=tuple(
                    replace(point, band_indices=None)
                    for point in specification.points
                ),
            )
            analysis_context = build_symmetry_context(
                replace(
                    analysis_context.model,
                    representation_analysis=longitudinal_specification,
                ),
                analysis_context.k_points,
            )
    LOGGER.info("Bloch symmetry channel: name=%s field_kind=%s", channel_name, field_kind.value)
    differential_diagnostics = None
    if field_kind.value in {
        "electric_polar_vector",
        "magnetic_axial_vector",
    } and getattr(bundle.mesh, "dimension", 0) == 3:
        is_electric = field_kind.value == "electric_polar_vector"
        quantity = "curl" if channel_name == "longitudinal" else "longitudinal"
        differential_diagnostics = diagnose_bundle_vector_fields(
            bundle,
            quantity=quantity,
            apply_metric=is_electric and quantity == "longitudinal",
        )
        LOGGER.info(
            "Vector-field diagnostic: channel=%s quantity=%s max=%.6g mean=%.6g "
            "worst_k=%s worst_band(0-based)=%s",
            channel_name,
            differential_diagnostics.quantity,
            differential_diagnostics.max_residual,
            differential_diagnostics.mean_residual,
            differential_diagnostics.worst_k_index,
            differential_diagnostics.worst_band_index,
        )
    state, report = _prepare_state(
        bundle,
        threads=threads,
        resolved_backend=resolved_backend,
    )
    provider = StateBlochSymmetryProvider(
        state,
        analysis_context,
        field_kind=field_kind,
    )
    with timed_step("analyze outer-window Bloch symmetry", LOGGER, channel=channel_name):
        analysis = run_bloch_symmetry_analysis(
            state,
            analysis_context,
            provider=provider,
        )
    log_bloch_symmetry_analysis(analysis)
    stencil_count, stencil_bytes = provider.spatial_cache_info
    LOGGER.debug(
        "Spatial symmetry stencil cache: entries=%s size=%.1f MB",
        stencil_count,
        stencil_bytes / (1024.0 * 1024.0),
    )
    provider.release_spatial_cache()
    if state.S is None:
        raise RuntimeError("Raw S overlap cache was not initialized during preanalysis.")
    return BlochSymmetryChannelResult(
        field_kind=field_kind,
        orthogonality_report=report,
        S=state.S,
        analysis=analysis,
        sewing_matrices=provider.cached_sewing_matrices,
        differential_diagnostics=differential_diagnostics,
    )


def _prepare_state(
    bundle: InputBundle,
    *,
    threads: int,
    resolved_backend: str,
    use_overlap_cache: bool = True,
) -> tuple[StateCollection, np.ndarray]:
    config = bundle.config
    state = StateCollection(
        bundle,
        backend=resolved_backend,
        threads=threads,
        use_overlap_cache=use_overlap_cache,
    )
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
    if int(config.kdim) == 3:
        if config.DOS:
            raise NotImplementedError("Three-dimensional DOS output is not implemented.")
        if config.Chern_number or config.hybrid_Wilson_loop:
            raise NotImplementedError("Three-dimensional topology output is not implemented.")
        if config.wannier_figures is not False and str(config.wannier_figures).lower() != "false":
            raise NotImplementedError(
                "Three-dimensional Wannier volume figures are not implemented; set wannier_figures=false."
            )
    resolved_backend = resolve_backend(backend or config.compute_backend)
    integration_family = getattr(
        bundle.mesh,
        "integration_family",
        "finite_element",
    )
    integration_name = (
        "uniform"
        if integration_family == "uniform_grid"
        else config.integration_mode
    )
    LOGGER.info(
        "Calculation setup: threads=%s backend=%s integration=%s discretization=%s "
        "blas=%s k_shape=%s spatial_points=%s spatial_elements=%s",
        threads,
        resolved_backend,
        integration_name,
        integration_family,
        threadpool_summary(),
        bundle.fields.shape,
        bundle.mesh.vertices.shape[0],
        bundle.mesh.elements.shape[0],
    )
    etbc_enabled = (
        config.wannier_subspace == "T+L"
        and config.longitudinal_source == "etbc"
    )
    state, report = _prepare_state(
        bundle,
        threads=threads,
        resolved_backend=resolved_backend,
        use_overlap_cache=not etbc_enabled,
    )
    trial_covariance_diagnostics = ()
    if config.projection_target_bindings:
        if bundle.symmetry is None:
            raise ValueError("Three-dimensional vector projections require symmetry_file.")
        bundle.symmetry, trial_covariance_diagnostics = prepare_vector_trial_targets(
            state, bundle.symmetry
        )
        state.symmetry = bundle.symmetry
    etbc_result = None
    if etbc_enabled:
        with timed_step(
            "construct ETBC auxiliary modes",
            LOGGER,
            rank_tolerance=config.etbc_rank_tolerance,
            auxiliary_eigenvalue=config.etbc_auxiliary_eigenvalue,
        ):
            trial_grid = build_vector_bloch_trial_grid(
                state, context=bundle.symmetry
            )

            def take_trial(index):
                values = trial_grid[index]
                if values is None:
                    raise RuntimeError(
                        f"ETBC trial frame at k={index} was requested more than once."
                    )
                trial_grid[index] = None
                return values

            try:
                etbc_result = complete_transverse_bundle(
                    state,
                    take_trial,
                    auxiliary_eigenvalue=config.etbc_auxiliary_eigenvalue,
                    rank_tolerance=config.etbc_rank_tolerance,
                )
            finally:
                del trial_grid
        bundle = etbc_result.augmented_bundle
        state, report = _prepare_state(
            bundle,
            threads=threads,
            resolved_backend=resolved_backend,
        )
        state._precomputed_vector_projection = etbc_result.trial_projection_matrices
        finite_singular = etbc_result.minimum_nonzero_singular_value
        LOGGER.info(
            "ETBC completion: N_T=%s N_L=%s N_W=%s min_nonzero_singular=%s "
            "max_TL_overlap=%.6g max_augmented_gram=%.6g gamma_regularized=%s",
            etbc_result.transverse_dimension,
            etbc_result.auxiliary_dimension,
            etbc_result.wannier_dimension,
            finite_singular,
            etbc_result.maximum_transverse_auxiliary_overlap,
            etbc_result.maximum_augmented_orthonormality_error,
            etbc_result.gamma_regularized_indices,
        )
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
        log_symmetry_analysis(symmetry_analysis)
    with timed_step("extend mesh", LOGGER, extension=config.extension):
        state.extend(config.extension)
    LOGGER.info(
        "Extended mesh: vertices=%s triangles=%s",
        state.extended_mesh.vertices.shape[0],
        state.extended_mesh.elements.shape[0],
    )

    mset = MSet(state, threads=threads)
    with timed_step("initialize M0", LOGGER):
        mset.init_M0()
    if config.wannier_subspace == "T+L":
        outer_smoothness = outer_channel_smoothness(state, mset)
        outer_entangled = any(
            len(state.E_idx[index]) > int(config.band_calc_num)
            for index in state.k_indices()
        )
        _log_subspace_smoothness(
            outer_smoothness,
            warn_below=None if outer_entangled else 0.5,
            suggestion=(
                "Enlarge the longitudinal outer window and select a smooth auxiliary "
                "subspace by disentanglement."
            ),
        )
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
        "norm_imag_max=%.6g individual_first_moments=%s",
        r_key,
        wannier.shape,
        float(np.min(np.real(norms))),
        float(np.max(np.real(norms))),
        float(np.max(np.abs(np.imag(norms)))),
        np.real_if_close(gradient.rn.T).tolist(),
    )
    if symmetry_gauge is not None and bundle.symmetry is not None:
        _log_symmetry_target_centers(
            gradient.rn,
            config,
            bundle.symmetry.model.targets,
        )
    if symmetry_gauge is not None and gauge_spec.validate_wannier:
        enforce_wannier_residual = config.symmetry_output_basis == "strict"
        input_limited_tolerance = max(
            float(gauge_spec.real_space_tolerance),
            float(symmetry_gauge.residuals.max_residual),
            float(np.sqrt(max(symmetry_gauge.physical_sewing_defect, 0.0))),
        )
        if input_limited_tolerance > gauge_spec.real_space_tolerance:
            LOGGER.warning(
                "Real-space Wannier symmetry accuracy is limited by the selected physical "
                "sewing space: requested=%.6g effective=%.6g sewing_defect=%.6g. "
                "Residuals below the effective tolerance are diagnostic, not evidence of a "
                "Wannier-gauge failure.",
                gauge_spec.real_space_tolerance,
                input_limited_tolerance,
                symmetry_gauge.physical_sewing_defect,
            )
        with timed_step("validate real-space Wannier symmetry", LOGGER):
            validation = validate_wannier_symmetry(
                ctx,
                bundle.symmetry.model.targets,
                zero_cell_wanniers=wannier,
                tolerance=input_limited_tolerance,
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
            "minimum_retained_norm=%.6g requested_tolerance=%.6g effective_tolerance=%.6g%s",
            config.symmetry_output_basis,
            validation.max_residual,
            validation.mean_residual,
            validation.minimum_retained_norm,
            gauge_spec.real_space_tolerance,
            input_limited_tolerance,
            " (diagnostic only for FEM output)" if not enforce_wannier_residual else "",
        )
    if symmetry_provider is not None:
        stencil_count, stencil_bytes = symmetry_provider.spatial_cache_info
        LOGGER.info(
            "Spatial symmetry stencil cache before release: entries=%s size=%.1f MB",
            stencil_count,
            stencil_bytes / (1024.0 * 1024.0),
        )
        symmetry_provider.release_spatial_cache()
    projector_interpolation_enabled = bool(
        config.projector_preserving_band_interpolation
        and config.wannier_subspace == "T+L"
    )
    tba = TBAModel(
        ctx,
        threads=threads,
        projector_preserving=projector_interpolation_enabled,
        fixed_longitudinal_eigenvalue=(
            None if etbc_result is None else etbc_result.auxiliary_eigenvalue
        ),
    )
    if projector_interpolation_enabled:
        LOGGER.info(
            "T/L band interpolation: projector-preserving mode enabled "
            "(longitudinal_source=%s%s)",
            config.longitudinal_source,
            (
                ""
                if etbc_result is None
                else f" fixed_longitudinal_eigenvalue={etbc_result.auxiliary_eigenvalue:.10g}"
            ),
        )
    if config.invert_longitudinal_energies:
        LOGGER.info(
            "Final TBA spectrum: L-channel Maxwell eigenvalues are multiplied by -1 "
            "before the output gauge transformation"
        )
    physical_analysis = None if symmetry_analysis is None else symmetry_analysis.physical
    output_spectrum_diagnostics = tba.output_spectrum_diagnostics(physical_analysis)
    _log_output_spectrum_diagnostics(output_spectrum_diagnostics, config)
    with timed_step("collect hopping matrices", LOGGER):
        hoppings = tba.collect_hoppings()
    LOGGER.info("Hopping matrices collected: count=%s", len(hoppings))
    hopping_reconstruction_diagnostics = tba.hopping_reconstruction_diagnostics(
        hoppings, physical_analysis
    )
    _log_hopping_reconstruction_diagnostics(hopping_reconstruction_diagnostics)
    with timed_step("calculate high-symmetry bands", LOGGER, enabled=bool(config.k_path)):
        band = tba.gen_hs_bands(hoppings) if config.k_path else None
    if band is not None and (config.Chern_number or config.hybrid_Wilson_loop):
        with timed_step("calculate Brillouin-zone bands", LOGGER, k_num=config.k_num):
            tba.gen_bz_bands(band, hoppings)
    with timed_step("calculate topology", LOGGER, enabled=band is not None):
        topology = calculate_topology(band, config) if band is not None else None
    if tba.projector_interpolation_diagnostics is not None:
        interpolation = tba.projector_interpolation_diagnostics
        log_interpolation = (
            LOGGER.warning
            if interpolation.minimum_projector_gap < 1.0e-6
            else LOGGER.info
        )
        log_interpolation(
            "T/L projector interpolation: points=%d min_projector_gap=%.6g "
            "raw_idempotency=%.6g flattened_idempotency=%.6g "
            "removed_cross_sector=%.6g",
            interpolation.point_count,
            interpolation.minimum_projector_gap,
            interpolation.maximum_raw_projector_idempotency_error,
            interpolation.maximum_flattened_projector_idempotency_error,
            interpolation.maximum_removed_cross_sector_component,
        )
    transverse_projectors = tba.transverse_projectors()
    if transverse_projectors is not None:
        _log_subspace_smoothness(
            selected_sector_smoothness(ctx, transverse_projectors),
            warn_below=0.5,
            suggestion=(
                "The selected sector changes rapidly between neighboring k points; "
                "increase the k mesh or improve the longitudinal completion."
            ),
        )

    bloch_gauge = state.gen_matrix_on_kmesh(
        lambda i, j, k: np.asarray(ctx.bloch_gauge_at(i, j, k), dtype=np.complex128).copy()
    )
    return RunResult(
        config=config,
        mesh=state.mesh,
        extended_mesh=state.extended_mesh,
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
        trial_covariance_diagnostics=trial_covariance_diagnostics,
        etbc=etbc_result,
        transverse_projectors=transverse_projectors,
        projector_interpolation_diagnostics=tba.projector_interpolation_diagnostics,
    )


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
                "Output basis %s breaks FEM degeneracy at %s bands(actual,0-based)=%s: "
                "raw_gap=%.6g output_gap=%.6g tolerance=%.6g",
                result.basis,
                splitting.point_name,
                tuple(splitting.band_indices),
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
                "Configured hopping set breaks output degeneracy at %s bands(actual,0-based)=%s: "
                "direct_output_gap=%.6g reconstructed_gap=%.6g tolerance=%.6g",
                splitting.point_name,
                tuple(splitting.band_indices),
                splitting.reference_gap,
                splitting.output_gap,
                splitting.tolerance,
            )


def _log_subspace_smoothness(
    diagnostics: tuple[NeighborSubspaceSmoothness, ...],
    *,
    warn_below: float | None,
    suggestion: str,
) -> None:
    for item in diagnostics:
        log = (
            LOGGER.warning
            if warn_below is not None
            and item.minimum_regular_singular_value < warn_below
            else LOGGER.info
        )
        log(
            "Neighbor subspace smoothness: sector=%s links=%d min_sigma=%.6g "
            "regular_min_sigma=%.6g median_min_sigma=%.6g "
            "worst_k=%s direction=%d worst_regular_k=%s "
            "worst_regular_direction=%d%s",
            item.label,
            item.link_count,
            item.minimum_singular_value,
            item.minimum_regular_singular_value,
            item.median_minimum_singular_value,
            item.worst_source_index,
            item.worst_direction,
            item.worst_regular_source_index,
            item.worst_regular_direction,
            (
                f". {suggestion}"
                if warn_below is not None
                and item.minimum_regular_singular_value < warn_below
                else ""
            ),
        )


def _log_symmetry_target_centers(rn, config, targets) -> None:
    """Report representation-center centroids separately from basis first moments."""
    centers = np.asarray(np.real_if_close(rn), dtype=np.complex128).T
    imaginary = float(np.max(np.abs(centers.imag), initial=0.0))
    if imaginary > 1.0e-10:
        LOGGER.warning("Wannier first moments contain an imaginary residual of %.6g.", imaginary)
    centers = centers.real
    lattice = (
        np.asarray(config.real_lattice_vectors, dtype=float)
        * float(config.lattice_const)
    )
    fractional = centers @ np.linalg.inv(lattice)
    offset = 0
    for target in targets:
        irrep_dimension = int(target.site_irrep.dimension)
        for orbit_index, orbit_point in enumerate(target.orbit.points):
            indices = tuple(
                offset + target.wannier_index(irrep_index, orbit_index)
                for irrep_index in range(irrep_dimension)
            )
            expected = np.asarray(orbit_point.position, dtype=float)
            displacements = fractional[list(indices)] - expected[None, :]
            displacements -= np.rint(displacements)
            mean_displacement = np.mean(displacements, axis=0)
            multiplet_center = np.mod(expected + mean_displacement, 1.0)
            centroid_error = float(np.linalg.norm(mean_displacement))
            individual_spread = max(
                (float(np.linalg.norm(value - mean_displacement)) for value in displacements),
                default=0.0,
            )
            LOGGER.info(
                "Wannier target center: target=%s orbit=%s indices=%s expected=%s "
                "multiplet_center=%s centroid_error=%.6g individual_first_moment_spread=%.6g",
                target.name,
                orbit_index,
                indices,
                expected.tolist(),
                multiplet_center.tolist(),
                centroid_error,
                individual_spread,
            )
        offset += target.wannier_dimension


def _validate_symmetry_gauge_prerequisites(analysis, tolerance: float) -> None:
    if analysis is None:
        return
    for compatibility in analysis.target_compatibilities:
        if compatibility.compatibility is not None and not compatibility.compatibility.compatible:
            raise RuntimeError(
                "Target representation is incompatible at symmetry point "
                f"{compatibility.point_name}."
            )
        if (
            compatibility.target_twisted_representation is not None
            and compatibility.intertwiner_dimension == 0
        ):
            raise RuntimeError(
                "Target representation has no direct intertwiner at symmetry point "
                f"{compatibility.point_name}."
            )
    for point in analysis.physical.points:
        if point.diagnostics.unitarity_error > tolerance:
            LOGGER.warning(
                "Physical sewing space is not fully closed at %s: unitarity residual=%.6g "
                "exceeds %.6g. Continuing with this outer window; final selected-gauge "
                "residuals will be reported separately.",
                point.name,
                point.diagnostics.unitarity_error,
                tolerance,
            )
        if point.diagnostics.outer_composition_residual > tolerance:
            LOGGER.warning(
                "Physical sewing composition residual at %s is %.6g and exceeds %.6g. "
                "Continuing with the approximate outer-space representation.",
                point.name,
                point.diagnostics.outer_composition_residual,
                tolerance,
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
    free = tuple(
        diagnostic
        for diagnostic in result.representative_diagnostics
        if diagnostic.hom_dimension > 1 or diagnostic.target_commutant_dimension > 1
    )
    if free:
        LOGGER.info(
            "Symmetry gauge freedom: representatives=%s max_dim_Hom=%s max_commutant_dim=%s",
            len(free),
            max(diagnostic.hom_dimension for diagnostic in free),
            max(diagnostic.target_commutant_dimension for diagnostic in free),
        )
        for diagnostic in free:
            LOGGER.debug(
                "Symmetry gauge representative %s: dim_Hom=%s commutant_dim=%s "
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
