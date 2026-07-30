from __future__ import annotations

from dataclasses import replace
import logging

import numpy as np

from ..projections import (
    ProjectionRecord3D,
    TrialCovarianceDiagnostics,
)
from ..symmetry.field_action import cartesian_field_matrix
from ..symmetry.representation import (
    SiteIrrep,
    WannierTargetRepresentation,
    build_symmetry_context,
)
from .initializer import StateBases

LOGGER = logging.getLogger(__name__)


def _cartesian_rotation(operation, lattice_vectors) -> np.ndarray:
    lattice = np.asarray(lattice_vectors, dtype=float)
    return lattice.T @ operation.rotation @ np.linalg.inv(lattice.T)


def _center_cartesian(config, center_fractional) -> np.ndarray:
    lattice = np.asarray(config.real_lattice_vectors, dtype=float)
    center = np.asarray(center_fractional, dtype=float) @ lattice
    center += np.asarray(config.origin, dtype=float)
    return center * float(config.lattice_const)


def _evaluate_transformed_trial(
    state,
    record: ProjectionRecord3D,
    trial,
    points_cartesian: np.ndarray,
    center_fractional,
    operation,
) -> np.ndarray:
    center = _center_cartesian(state.config, center_fractional)
    rotation = _cartesian_rotation(
        operation, state.config.real_lattice_vectors
    )
    preimage_displacement = (points_cartesian - center[None, :]) @ rotation
    values = trial.evaluate(
        preimage_displacement,
        record.frame,
        float(state.config.lattice_const),
        StateBases.Radial,
    )
    if operation.antiunitary:
        values = state.maxwell.apply_time_reversal(values)
    component_matrix = cartesian_field_matrix(
        operation,
        state.config.real_lattice_vectors,
        state.maxwell.symmetry_field_kind,
        state.config.symmetry_tolerance,
    )
    return np.asarray(values @ component_matrix.T, dtype=np.complex128)


def _normalize_sample_columns(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.complex128)
    norms = np.sqrt(np.sum(np.abs(matrix) ** 2, axis=(0, 2), dtype=np.float64))
    if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(float).tiny):
        raise ValueError("A representative 3D trial function has zero or non-finite sample norm.")
    return matrix / norms[None, :, None]


def _trial_representation(
    state,
    record: ProjectionRecord3D,
    target: WannierTargetRepresentation,
) -> tuple[tuple[np.ndarray, ...], TrialCovarianceDiagnostics]:
    points = np.asarray(state.mesh.vertices, dtype=float)
    if points.shape[0] > 16384:
        indices = np.linspace(0, points.shape[0] - 1, 16384, dtype=np.intp)
        points = points[indices]
    identity = target.group.operations[target.group.identity_index]
    basis = np.stack(
        [
            _evaluate_transformed_trial(
                state,
                record,
                trial,
                points,
                record.frac_position,
                identity,
            )
            for trial in record.states
        ],
        axis=1,
    )
    basis = _normalize_sample_columns(basis)
    flat_basis = basis.transpose(0, 2, 1).reshape(-1, len(record.states))
    gram = flat_basis.conj().T @ flat_basis
    gram_error = float(np.linalg.norm(gram - np.eye(len(record.states)), ord="fro"))
    pinv = np.linalg.pinv(flat_basis, rcond=1.0e-12)
    input_matrices = []
    operation_residuals = []
    for site_element in target.orbit.site_symmetry.elements:
        transformed = np.stack(
            [
                _evaluate_transformed_trial(
                    state,
                    record,
                    trial,
                    points,
                    record.frac_position,
                    site_element.operation,
                )
                for trial in record.states
            ],
            axis=1,
        )
        transformed = _normalize_sample_columns(transformed)
        flat_transformed = transformed.transpose(0, 2, 1).reshape(
            -1, len(record.states)
        )
        matrix = pinv @ flat_transformed
        residual = float(
            np.linalg.norm(flat_transformed - flat_basis @ matrix, ord="fro")
            / max(np.linalg.norm(flat_transformed, ord="fro"), np.finfo(float).tiny)
        )
        input_matrices.append(np.asarray(matrix, dtype=np.complex128))
        operation = target.group.operations[site_element.source_operation_index]
        operation_residuals.append((operation.name or f"g{site_element.source_operation_index}", residual))

    standard = target.site_irrep.matrices
    antiunitary = tuple(
        element.operation.antiunitary for element in target.orbit.site_symmetry.elements
    )
    transform, equivalence_residual = _find_semilinear_basis_transform(
        tuple(input_matrices), standard, antiunitary
    )
    changed_matrices = tuple(
        np.asarray(
            transform @ dmat @ (transform.T if is_antiunitary else transform.conj().T),
            dtype=np.complex128,
        )
        for dmat, is_antiunitary in zip(standard, antiunitary)
    )
    max_closure = max((value for _, value in operation_residuals), default=0.0)
    diagnostics = TrialCovarianceDiagnostics(
        max(max_closure, equivalence_residual, gram_error),
        tuple(operation_residuals),
        max(max_closure, gram_error),
        transform,
    )
    return changed_matrices, diagnostics


def _find_semilinear_basis_transform(
    physical: tuple[np.ndarray, ...],
    target: tuple[np.ndarray, ...],
    antiunitary: tuple[bool, ...],
) -> tuple[np.ndarray, float]:
    dimension = physical[0].shape[0]
    rng = np.random.default_rng(41021)
    seeds = []
    if not any(antiunitary):
        from ..symmetry.gauge import solve_intertwiner_space

        space = solve_intertwiner_space(physical, target)
        seeds.extend(np.asarray(value, dtype=np.complex128) for value in space.basis)
        if space.basis:
            for _ in range(8):
                coefficients = rng.normal(size=len(space.basis)) + 1j * rng.normal(
                    size=len(space.basis)
                )
                seeds.append(
                    sum(
                        coefficient * value
                        for coefficient, value in zip(coefficients, space.basis)
                    )
                )
    seeds.extend([np.eye(dimension, dtype=np.complex128)] + [
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension))
        for _ in range(8)
    ])
    best = None
    best_residual = np.inf
    for seed in seeds:
        matrix = np.asarray(seed, dtype=np.complex128)
        full_rank = False
        for _ in range(30):
            matrix = sum(
                pmat
                @ (matrix.conj() if is_antiunitary else matrix)
                @ tmat.conj().T
                for pmat, tmat, is_antiunitary in zip(
                    physical, target, antiunitary
                )
            ) / len(physical)
            left, singular, right = np.linalg.svd(matrix, full_matrices=False)
            if singular.size != dimension or singular[-1] <= 1.0e-12 * max(singular[0], 1.0):
                break
            matrix = left @ right
            full_rank = True
        if not full_rank:
            continue
        residual = max(
            float(
                np.linalg.norm(
                    pmat @ (matrix.conj() if is_antiunitary else matrix)
                    - matrix @ tmat,
                    ord="fro",
                )
            )
            for pmat, tmat, is_antiunitary in zip(physical, target, antiunitary)
        )
        if residual < best_residual:
            best = matrix.copy()
            best_residual = residual
    if best is None:
        return np.eye(dimension, dtype=np.complex128), float("inf")
    return best, best_residual


def prepare_vector_trial_targets(state, context):
    """Validate user trial bases and express target matrices in those bases."""

    bindings = tuple(state.config.projection_target_bindings)
    if not bindings:
        return context, ()
    replacements = {}
    diagnostics = []
    for binding in bindings:
        target = context.model.target(binding.target_name)
        matrices, report = _trial_representation(state, binding.projection, target)
        diagnostics.append(report)
        if report.max_residual > state.config.symmetry_tolerance:
            message = (
                f"3D projection {binding.projection.wyckoff!r} for target {target.name!r} "
                f"does not carry site irrep {target.site_irrep.name!r}: "
                f"covariance residual={report.max_residual:.6g}."
            )
            if state.config.symmetry_constrained:
                raise ValueError(message)
            LOGGER.warning(message)
        new_irrep = SiteIrrep(
            target.site_irrep.name,
            target.site_irrep.dimension,
            matrices,
            target.site_irrep.finite_group_name,
            target.site_irrep.actual_to_canonical,
        )
        replacements[target.name] = replace(target, site_irrep=new_irrep)
        LOGGER.info(
            "3D trial covariance: target=%s wyckoff=%s functions=%s orbit=%s residual=%.6g",
            target.name,
            binding.projection.wyckoff,
            len(binding.projection.states),
            target.multiplicity,
            report.max_residual,
        )
    targets = tuple(replacements.get(target.name, target) for target in context.model.targets)
    model = replace(context.model, targets=targets)
    return build_symmetry_context(model, context.k_points), tuple(diagnostics)


def _tile_translations(extension) -> tuple[tuple[int, ...], ...]:
    axes = []
    for size in extension:
        center = int(np.floor((int(size) - 1) / 2.0))
        axes.append(tuple(int(value - center) for value in range(int(size))))
    return tuple(tuple(int(value) for value in item) for item in np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T)


def build_vector_bloch_trials(state, k_index) -> np.ndarray:
    """Construct orbit-expanded 3D vector Bloch sums on the primitive grid."""

    context = state.config.symmetry_context
    bindings = tuple(state.config.projection_target_bindings)
    if context is None or not bindings:
        raise ValueError("3D vector trials require bound Wannier targets and symmetry context.")
    points = np.asarray(state.mesh.vertices, dtype=float)
    lattice = np.asarray(state.config.real_lattice_vectors, dtype=float) * float(
        state.config.lattice_const
    )
    k_fractional = np.asarray(
        [state.config.k_points[axis][k_index[axis]] for axis in range(3)], dtype=float
    )
    translations = _tile_translations(state.config.extension)
    columns = []
    for binding in bindings:
        target = context.model.target(binding.target_name)
        record = binding.projection
        for orbit_point in target.orbit.points:
            operation = orbit_point.representative_operation
            for trial in record.states:
                total = np.zeros((points.shape[0], 3), dtype=np.complex128)
                for translation in translations:
                    shifted_center = np.asarray(orbit_point.position, dtype=float) + np.asarray(
                        translation, dtype=float
                    )
                    phase = np.exp(
                        state.bloch_sign
                        * 2j
                        * np.pi
                        * np.dot(k_fractional, np.asarray(translation, dtype=float))
                    )
                    total += phase * _evaluate_transformed_trial(
                        state,
                        record,
                        trial,
                        points,
                        shifted_center,
                        operation,
                    )
                columns.append(total)
    values = np.stack(columns, axis=1)
    norms = state.inner_product.norms(values, name="3D projection Bloch-sum norms")
    invalid = np.flatnonzero(~np.isfinite(norms) | (norms <= 0.0))
    if invalid.size:
        raise ValueError(
            "3D projection Bloch sums have zero or non-finite norms at columns "
            f"{invalid.tolist()}."
        )
    return values / np.sqrt(norms)[None, :, None]
