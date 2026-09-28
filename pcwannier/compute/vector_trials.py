from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
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
from .kspace import is_complete_uniform_k_mesh
from .parallel import parallel_map
from .prepared import ProjectionSeed

LOGGER = logging.getLogger(__name__)


def _minimum_image_fractional(
    fractional: np.ndarray,
    lattice: np.ndarray,
    periods: np.ndarray,
) -> np.ndarray:
    """Reduce displacements using the Cartesian metric of a skew lattice."""

    values = np.asarray(fractional, dtype=float)
    basis = np.asarray(lattice, dtype=float)
    period_values = np.asarray(periods, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != period_values.size:
        raise ValueError("Fractional displacements and periods have incompatible shapes.")
    if basis.shape != (period_values.size, period_values.size):
        raise ValueError("Minimum-image lattice has an invalid shape.")
    nearest = np.floor(values / period_values[None, :] + 0.5).astype(np.int64)
    best = None
    best_norm = None
    for offset in product((-1, 0, 1), repeat=period_values.size):
        image = nearest + np.asarray(offset, dtype=np.int64)[None, :]
        candidate = values - image * period_values[None, :]
        cartesian = candidate @ basis
        norm = np.einsum("pi,pi->p", cartesian, cartesian, optimize=True)
        if best is None:
            best = candidate
            best_norm = norm
            continue
        update = norm < best_norm
        best[update] = candidate[update]
        best_norm[update] = norm[update]
    assert best is not None
    return best


@dataclass(frozen=True)
class _PreparedTrialColumn:
    record: ProjectionRecord3D
    trial: object
    operation: object
    base_center_cartesian: np.ndarray
    cartesian_rotation: np.ndarray
    component_matrix: np.ndarray
    lattice: np.ndarray
    inverse_lattice: np.ndarray
    lattice_scale: float


def _prepare_trial_column(state, record, orbit_point, trial) -> _PreparedTrialColumn:
    operation = orbit_point.representative_operation
    return _PreparedTrialColumn(
        record=record,
        trial=trial,
        operation=operation,
        base_center_cartesian=_center_cartesian(
            state.config, orbit_point.position
        ),
        cartesian_rotation=_cartesian_rotation(
            operation, state.config.real_lattice_vectors
        ),
        component_matrix=cartesian_field_matrix(
            operation,
            state.config.real_lattice_vectors,
            state.maxwell.symmetry_field_kind,
            state.config.symmetry_tolerance,
        ),
        lattice=np.asarray(state.config.real_lattice_vectors, dtype=float),
        inverse_lattice=np.linalg.inv(
            np.asarray(state.config.real_lattice_vectors, dtype=float)
        ),
        lattice_scale=float(state.config.lattice_const),
    )


def _evaluate_prepared_trial(
    state,
    prepared: _PreparedTrialColumn,
    points_cartesian: np.ndarray,
    translation: np.ndarray,
    periods: tuple[int, ...],
) -> np.ndarray:
    lattice = prepared.lattice
    scale = prepared.lattice_scale
    center = prepared.base_center_cartesian + (translation @ lattice) * scale
    displacement = points_cartesian - center[None, :]
    fractional = (displacement / scale) @ prepared.inverse_lattice
    period_values = np.asarray(periods, dtype=float)
    fractional = _minimum_image_fractional(
        fractional,
        lattice,
        period_values,
    )
    displacement = (fractional @ lattice) * scale
    values = prepared.trial.evaluate(
        displacement @ prepared.cartesian_rotation,
        prepared.record.frame,
        scale,
        StateBases.Radial,
    )
    if prepared.operation.antiunitary:
        values = state.maxwell.apply_time_reversal(values)
    return np.asarray(
        values @ prepared.component_matrix.T, dtype=np.complex128
    )


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
    *,
    supercell_periods=None,
) -> np.ndarray:
    center = _center_cartesian(state.config, center_fractional)
    displacement = points_cartesian - center[None, :]
    if supercell_periods is not None:
        periods = np.asarray(supercell_periods, dtype=float).reshape(-1)
        lattice = np.asarray(state.config.real_lattice_vectors, dtype=float)
        if periods.shape != (lattice.shape[0],) or np.any(periods <= 0.0):
            raise ValueError("Born-von Karman periods must be positive and match the lattice dimension.")
        scale = float(state.config.lattice_const)
        fractional = (displacement / scale) @ np.linalg.inv(lattice)
        # The trial lives on the finite Born-von Karman torus selected by the
        # k mesh. Use the Cartesian metric: component-wise wrapping is not a
        # symmetry-covariant nearest-image rule for skew primitive lattices.
        fractional = _minimum_image_fractional(
            fractional,
            lattice,
            periods,
        )
        displacement = (fractional @ lattice) * scale
    rotation = _cartesian_rotation(
        operation, state.config.real_lattice_vectors
    )
    preimage_displacement = displacement @ rotation
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
    lattice = (
        np.asarray(state.config.real_lattice_vectors, dtype=float)
        * float(state.config.lattice_const)
    )
    radius = 0.35 * float(np.min(np.linalg.norm(lattice, axis=1)))
    axis = np.linspace(-radius, radius, 9, dtype=float)
    offsets = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    points = _center_cartesian(state.config, record.frac_position)[None, :] + offsets
    identity = target.group.operations[target.group.identity_index]
    # Site-irrep covariance is a local statement.  Sampling a primitive cell
    # clips angular orbitals whenever q lies on its boundary and can report a
    # false closure failure.  A deterministic Cartesian cloud around q tests
    # the local angular/component action without introducing periodic-image
    # ties; full Bloch-sum covariance is validated separately.
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


def _born_von_karman_translations(k_points) -> tuple[tuple[int, ...], ...]:
    axes = []
    for axis in k_points:
        size = len(axis)
        center = int(np.floor((int(size) - 1) / 2.0))
        axes.append(tuple(int(value - center) for value in range(int(size))))
    return tuple(tuple(int(value) for value in item) for item in np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T)


def _born_von_karman_translation_axes(k_points) -> tuple[np.ndarray, ...]:
    axes = []
    for axis in k_points:
        size = int(len(axis))
        center = int(np.floor((size - 1) / 2.0))
        axes.append(np.arange(size, dtype=int) - center)
    return tuple(axes)


def _translation_bloch_fft(
    images: np.ndarray,
    k_points,
    bloch_sign: int,
) -> np.ndarray:
    """Transform translation-resolved trial images to the configured k mesh."""

    k_shape = tuple(int(len(axis)) for axis in k_points)
    values = np.asarray(images, dtype=np.complex128)
    if values.shape[: len(k_shape)] != k_shape:
        raise ValueError(
            f"Translation images start with shape {values.shape[:len(k_shape)]}; "
            f"expected {k_shape}."
        )
    sign = int(bloch_sign)
    if sign not in {-1, 1}:
        raise ValueError(f"Bloch sign must be +1 or -1; got {bloch_sign}.")
    if not is_complete_uniform_k_mesh(k_points):
        raise ValueError(
            "FFT Bloch sums require each sampled k axis to be a complete uniform "
            "periodic mesh with spacing 1/N."
        )

    translation_axes = _born_von_karman_translation_axes(k_points)
    offset_phase = np.ones(k_shape, dtype=np.complex128)
    index_phase = np.ones(k_shape, dtype=np.complex128)
    dimension = len(k_shape)
    for axis_index, (axis, translations) in enumerate(
        zip(k_points, translation_axes)
    ):
        reshape = [1] * dimension
        reshape[axis_index] = translations.size
        offset_phase *= np.exp(
            sign
            * 2j
            * np.pi
            * float(np.asarray(axis, dtype=float)[0])
            * translations
        ).reshape(reshape)
        center = int(np.floor((translations.size - 1) / 2.0))
        indices = np.arange(translations.size, dtype=float)
        index_phase *= np.exp(
            -sign * 2j * np.pi * indices * center / float(translations.size)
        ).reshape(reshape)

    trailing = (1,) * (values.ndim - dimension)
    weighted = values * offset_phase.reshape(k_shape + trailing)
    axes = tuple(range(dimension))
    if sign > 0:
        transformed = np.fft.ifftn(weighted, axes=axes) * int(np.prod(k_shape))
    else:
        transformed = np.fft.fftn(weighted, axes=axes)
    transformed *= index_phase.reshape(k_shape + trailing)
    return np.asarray(transformed, dtype=np.complex128)


def _vector_trial_columns(state, context):
    columns = []
    for binding in tuple(state.config.projection_target_bindings):
        target = context.model.target(binding.target_name)
        record = binding.projection
        for orbit_point in target.orbit.points:
            for trial in record.states:
                columns.append((record, orbit_point, trial))
    return tuple(columns)


class VectorTrialFrameSource:
    """Re-iterable, chunked FFT source for orbit-expanded vector trials."""

    def __init__(
        self,
        state,
        *,
        context=None,
        workspace_bytes: int = 256 << 20,
    ) -> None:
        self.state = state
        self.context = (
            context
            or getattr(state, "symmetry", None)
            or state.config.symmetry_context
        )
        if self.context is None or not tuple(
            state.config.projection_target_bindings
        ):
            raise ValueError(
                "3D vector trials require bound Wannier targets and symmetry context."
            )
        self.k_points = tuple(state.config.k_points)
        self.k_shape = tuple(int(len(axis)) for axis in self.k_points)
        if not is_complete_uniform_k_mesh(self.k_points):
            raise ValueError(
                "Streaming vector trials require a complete uniform periodic k mesh."
            )
        self.points = np.asarray(state.mesh.vertices, dtype=float)
        raw_columns = _vector_trial_columns(state, self.context)
        self.columns = tuple(
            _prepare_trial_column(state, record, orbit_point, trial)
            for record, orbit_point, trial in raw_columns
        )
        self.trial_count = len(self.columns)
        if self.trial_count != int(state.config.band_calc_num):
            raise ValueError(
                f"Vector trial basis contains {self.trial_count} columns; expected "
                f"band_calc_num={state.config.band_calc_num}."
            )
        self.point_count = int(self.points.shape[0])
        self.workspace_bytes = max(1 << 20, int(workspace_bytes))
        self.periods = tuple(int(len(axis)) for axis in self.k_points)
        translation_axes = _born_von_karman_translation_axes(self.k_points)
        self.translation_vectors = {
            index: np.asarray(
                [translation_axes[axis][index[axis]] for axis in range(len(index))],
                dtype=float,
            )
            for index in np.ndindex(self.k_shape)
        }
        k_count = int(np.prod(self.k_shape))
        workers = max(1, int(state.configured_threads))
        bytes_per_point = (
            k_count
            * 3
            * np.dtype(np.complex128).itemsize
            * (self.trial_count + 2 * workers)
        )
        self.point_chunk = max(
            1,
            min(self.point_count, self.workspace_bytes // max(1, bytes_per_point)),
        )
        self._pass_count = 0

    def iter_raw_chunks(self):
        self._pass_count += 1
        k_count = int(np.prod(self.k_shape))
        bytes_per_task = (
            2
            * k_count
            * self.point_chunk
            * 3
            * np.dtype(np.complex128).itemsize
        )
        LOGGER.info(
            "Vector trial stream pass=%d k_points=%d spatial_points=%d columns=%d "
            "point_chunk=%d workspace=%.1f MB",
            self._pass_count,
            k_count,
            self.point_count,
            self.trial_count,
            self.point_chunk,
            self.workspace_bytes / (1024.0**2),
        )
        for start in range(0, self.point_count, self.point_chunk):
            stop = min(start + self.point_chunk, self.point_count)
            local_points = self.points[start:stop]
            output = np.empty(
                self.k_shape
                + (local_points.shape[0], self.trial_count, 3),
                dtype=np.complex128,
            )

            def calculate_column(item):
                column, prepared = item
                images = np.empty(
                    self.k_shape + (local_points.shape[0], 3),
                    dtype=np.complex128,
                )
                for translation_index, translation in self.translation_vectors.items():
                    images[translation_index] = _evaluate_prepared_trial(
                        self.state,
                        prepared,
                        local_points,
                        translation,
                        self.periods,
                    )
                transformed = _translation_bloch_fft(
                    images, self.k_points, self.state.bloch_sign
                )
                return column, transformed

            for column, transformed in parallel_map(
                enumerate(self.columns),
                calculate_column,
                self.state.configured_threads,
                ordered=False,
                bytes_per_task=bytes_per_task,
                memory_budget_bytes=self.workspace_bytes,
            ):
                output[..., column, :] = transformed
            yield start, stop, output

    def projection_seed(self) -> ProjectionSeed:
        """Accumulate normalized trial overlaps without materializing trial fields."""

        inner = self.state.inner_product
        if getattr(inner, "domain_kind", None) != "points":
            raise TypeError(
                "Streaming vector projection currently requires a uniform grid."
            )
        metric = np.asarray(inner.metric, dtype=np.complex128).reshape(-1)
        if np.max(np.abs(metric.imag), initial=0.0) > 1.0e-12:
            raise ValueError("Vector trial projection requires a real metric.")
        point_weight = float(inner.point_weight)
        indices = tuple(self.state.k_indices())
        norms = {
            index: np.zeros(self.trial_count, dtype=np.float64)
            for index in indices
        }
        overlaps = {
            index: np.zeros(
                (len(self.state.E_idx[index]), self.trial_count),
                dtype=np.complex128,
            )
            for index in indices
        }
        for start, stop, trial_chunk in self.iter_raw_chunks():
            weights = metric.real[start:stop] * point_weight
            for index in indices:
                trials = np.asarray(trial_chunk[index], dtype=np.complex128)
                block = self.state.get_block(*index)[:, start:stop]
                phase = self.state.get_phase(*index)[start:stop]
                full_fields = block * phase[None, :, None]
                norms[index] += np.einsum(
                    "pic,pic,p->i",
                    trials.conj(),
                    trials,
                    weights,
                    optimize=True,
                ).real
                overlaps[index] += np.einsum(
                    "mpc,pic,p->mi",
                    full_fields.conj(),
                    trials,
                    weights,
                    optimize=True,
                )
        matrices = np.empty(self.k_shape, dtype=object)
        for index in indices:
            invalid = np.flatnonzero(
                ~np.isfinite(norms[index]) | (norms[index] <= 0.0)
            )
            if invalid.size:
                raise ValueError(
                    f"3D projection Bloch sums at k={index} have invalid norms "
                    f"in columns {invalid.tolist()}."
                )
            normalized_overlap = overlaps[index] / np.sqrt(norms[index])[None, :]
            matrices[index] = self.state.overlap_to_internal_basis(
                index, normalized_overlap
            )
        return ProjectionSeed(matrices, source="streaming vector trial")


def build_vector_bloch_trial_source(
    state,
    *,
    context=None,
    workspace_bytes: int = 256 << 20,
) -> VectorTrialFrameSource:
    return VectorTrialFrameSource(
        state,
        context=context,
        workspace_bytes=workspace_bytes,
    )


def build_vector_bloch_trial_grid(
    state,
    *,
    context=None,
    workspace_bytes: int = 256 << 20,
) -> np.ndarray:
    """Construct all k-point trial frames with one FFT over BvK translations."""

    context = context or getattr(state, "symmetry", None) or state.config.symmetry_context
    bindings = tuple(state.config.projection_target_bindings)
    if context is None or not bindings:
        raise ValueError("3D vector trials require bound Wannier targets and symmetry context.")
    k_points = tuple(state.config.k_points)
    k_shape = tuple(int(len(axis)) for axis in k_points)
    if not is_complete_uniform_k_mesh(k_points):
        LOGGER.warning(
            "Vector trial k mesh is not a complete uniform periodic grid; "
            "falling back to direct per-k Bloch sums."
        )
        output = np.empty(k_shape, dtype=object)
        for index in np.ndindex(k_shape):
            output[index] = build_vector_bloch_trials(state, index, context=context)
        return output

    source = build_vector_bloch_trial_source(
        state,
        context=context,
        workspace_bytes=workspace_bytes,
    )
    output = np.empty(k_shape, dtype=object)
    for index in np.ndindex(k_shape):
        output[index] = np.empty(
            (source.point_count, source.trial_count, 3), dtype=np.complex128
        )
    for start, stop, chunk in source.iter_raw_chunks():
        for index in np.ndindex(k_shape):
            output[index][start:stop] = chunk[index]

    for index in np.ndindex(k_shape):
        values = output[index]
        norms = state.inner_product.norms(
            np.swapaxes(values, 1, 2),
            name=f"3D projection Bloch-sum norms at k={index}",
        )
        invalid = np.flatnonzero(~np.isfinite(norms) | (norms <= 0.0))
        if invalid.size:
            raise ValueError(
                "3D projection Bloch sums have zero or non-finite norms at "
                f"k={index}, columns={invalid.tolist()}."
            )
        values /= np.sqrt(norms)[None, :, None]
    return output


def build_vector_bloch_trials(state, k_index, *, context=None) -> np.ndarray:
    """Construct orbit-expanded 3D vector Bloch sums on the primitive grid."""

    context = context or getattr(state, "symmetry", None) or state.config.symmetry_context
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
    periods = tuple(int(len(axis)) for axis in state.config.k_points)
    translations = _born_von_karman_translations(state.config.k_points)
    columns = []
    for record, orbit_point, trial in _vector_trial_columns(state, context):
        operation = orbit_point.representative_operation
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
                supercell_periods=periods,
            )
        columns.append(total)
    values = np.stack(columns, axis=1)
    # Trial construction uses (point, trial, component), while the inner-product
    # column convention is (point, component, column).
    norms = state.inner_product.norms(
        np.swapaxes(values, 1, 2),
        name="3D projection Bloch-sum norms",
    )
    invalid = np.flatnonzero(~np.isfinite(norms) | (norms <= 0.0))
    if invalid.size:
        raise ValueError(
            "3D projection Bloch sums have zero or non-finite norms at columns "
            f"{invalid.tolist()}."
        )
    return values / np.sqrt(norms)[None, :, None]
