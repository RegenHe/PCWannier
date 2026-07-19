from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from .cache import SewingMatrixCache, SewingMatrixCacheEntry, load_sewing_matrix_cache
from .group import SpaceGroupOperation, SymmetryKMapping, periodic_difference, reduce_fractional
from .specs import BlochConvention, FieldKind

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..compute.state import StateCollection
    from .analysis import SewingMatrixRequest
    from .representation import SymmetryContext


def fractional_mesh_vertices(mesh, real_lattice_vectors, lattice_const: float) -> np.ndarray:
    """Convert Cartesian mesh vertices to row-vector fractional coordinates."""
    lattice = np.asarray(real_lattice_vectors, dtype=float)
    dimension = lattice.shape[0]
    vertices = np.asarray(mesh.vertices, dtype=float)
    if lattice.shape != (dimension, dimension) or vertices.shape[1] != dimension:
        raise ValueError("Mesh and real-space lattice dimensions do not match.")
    if not np.isfinite(lattice_const) or lattice_const <= 0.0:
        raise ValueError("lattice_const must be positive and finite.")
    return (vertices / float(lattice_const)) @ np.linalg.inv(lattice)


@dataclass(frozen=True)
class BarycentricStencil:
    vertex_indices: np.ndarray
    weights: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 2:
            return np.einsum(
                "nvc,vc->nv",
                array[:, self.vertex_indices],
                self.weights,
                optimize=True,
            )
        if array.ndim == 3:
            return np.einsum(
                "nvcd,vc->nvd",
                array[:, self.vertex_indices, :],
                self.weights,
                optimize=True,
            )
        raise ValueError("Bloch fields must have shape (bands, vertices[, components]).")


class PeriodicTriangleInterpolator:
    """Linear interpolation on periodic copies of the original FEM triangles."""

    def __init__(self, fractional_vertices, elements, *, tolerance: float = 1.0e-8):
        vertices = np.asarray(fractional_vertices, dtype=float)
        triangles = np.asarray(elements, dtype=np.intp)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise NotImplementedError("Periodic FEM symmetry interpolation currently supports 2D meshes.")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("Symmetry interpolation requires triangular elements.")
        self.vertices = vertices
        self.elements = triangles
        self.tolerance = float(tolerance)
        canonical = np.mod(vertices, 1.0)
        canonical[np.abs(canonical - 1.0) <= tolerance] = 0.0
        self._node_tree = cKDTree(canonical, boxsize=1.0)
        self._node_representative, self.periodic_node_classes = self._periodic_node_groups(
            canonical
        )
        self._build_periodic_triangles()

    def _periodic_node_groups(
        self, canonical: np.ndarray
    ) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
        parent = np.arange(len(canonical), dtype=np.intp)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            root = min(left_root, right_root)
            parent[max(left_root, right_root)] = root

        for left, right in sorted(self._node_tree.query_pairs(r=self.tolerance, p=np.inf)):
            left = int(left)
            right = int(right)
            difference = self.vertices[left] - self.vertices[right]
            lattice_shift = np.rint(difference).astype(np.int64)
            if not np.allclose(
                difference, lattice_shift, rtol=0.0, atol=self.tolerance
            ):
                raise ValueError(
                    "Distinct FEM nodes are closer than the symmetry tolerance after periodic "
                    f"reduction: nodes=({left}, {right}). Reduce symmetry_tolerance."
                )
            if not np.any(lattice_shift):
                raise ValueError(
                    "The FEM mesh contains coincident duplicate nodes, so symmetry interpolation "
                    f"is ambiguous: nodes=({left}, {right})."
                )
            union(left, right)
        groups: dict[int, list[int]] = {}
        for index in range(len(canonical)):
            groups.setdefault(find(index), []).append(index)
        classes = tuple(
            tuple(values) for _, values in sorted(groups.items()) if len(values) > 1
        )
        representative = np.arange(len(canonical), dtype=np.intp)
        for values in classes:
            representative[np.asarray(values, dtype=np.intp)] = min(values)
        representative.setflags(write=False)
        return representative, classes

    def _build_periodic_triangles(self) -> None:
        shifts = np.asarray(tuple(product((-1, 0, 1), repeat=2)), dtype=float)
        base = self.vertices[self.elements]
        tiled = (base[None, :, :, :] + shifts[:, None, None, :]).reshape(-1, 3, 2)
        self._triangles = tiled
        self._triangle_sources = np.tile(self.elements, (len(shifts), 1))
        lower = np.min(tiled, axis=1)
        upper = np.max(tiled, axis=1)
        self._lower = lower
        self._upper = upper

        count = max(8, int(np.ceil(np.sqrt(tiled.shape[0]))))
        self._grid_count = count
        self._buckets: dict[tuple[int, int], list[int]] = {}
        for triangle_index, (lo, hi) in enumerate(zip(lower, upper)):
            cell_lo = np.floor(np.clip(lo, 0.0, 1.0 - np.finfo(float).eps) * count).astype(int)
            cell_hi = np.floor(np.clip(hi, 0.0, 1.0 - np.finfo(float).eps) * count).astype(int)
            if np.any(hi < -self.tolerance) or np.any(lo > 1.0 + self.tolerance):
                continue
            for ix in range(max(0, cell_lo[0]), min(count - 1, cell_hi[0]) + 1):
                for iy in range(max(0, cell_lo[1]), min(count - 1, cell_hi[1]) + 1):
                    self._buckets.setdefault((ix, iy), []).append(triangle_index)

    def stencil(self, points) -> BarycentricStencil:
        query = np.asarray(points, dtype=float)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("Interpolation points must have shape (count, 2).")
        query = np.asarray([reduce_fractional(point, self.tolerance).reduced for point in query])
        indices = np.empty((len(query), 3), dtype=np.intp)
        weights = np.empty((len(query), 3), dtype=float)

        distances, nearest = self._node_tree.query(query, k=1, p=np.inf)
        for point_index, point in enumerate(query):
            if distances[point_index] <= self.tolerance:
                node = int(self._node_representative[int(nearest[point_index])])
                indices[point_index] = (node, node, node)
                weights[point_index] = (1.0, 0.0, 0.0)
                continue
            cell = np.floor(np.clip(point, 0.0, 1.0 - np.finfo(float).eps) * self._grid_count).astype(int)
            candidates = self._buckets.get((int(cell[0]), int(cell[1])), ())
            found = self._find_triangle(point, candidates)
            if found is None:
                raise ValueError(
                    "A symmetry-transformed point lies outside the periodic FEM mesh: "
                    f"fractional coordinate {point.tolist()}."
                )
            triangle_index, barycentric = found
            indices[point_index] = self._triangle_sources[triangle_index]
            weights[point_index] = barycentric
        return BarycentricStencil(indices, weights)

    def _find_triangle(self, point: np.ndarray, candidates) -> tuple[int, np.ndarray] | None:
        best = None
        best_violation = np.inf
        for triangle_index in candidates:
            if np.any(point < self._lower[triangle_index] - self.tolerance) or np.any(
                point > self._upper[triangle_index] + self.tolerance
            ):
                continue
            triangle = self._triangles[triangle_index]
            matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
            determinant = float(np.linalg.det(matrix))
            if abs(determinant) <= np.finfo(float).eps:
                continue
            uv = np.linalg.solve(matrix, point - triangle[0])
            barycentric = np.array([1.0 - uv[0] - uv[1], uv[0], uv[1]])
            violation = float(max(0.0, -np.min(barycentric), np.max(barycentric) - 1.0))
            if violation <= self.tolerance and violation < best_violation:
                best = (int(triangle_index), barycentric)
                best_violation = violation
        return best


class BlochSymmetryAction:
    def __init__(
        self,
        fractional_vertices,
        elements,
        real_lattice_vectors,
        *,
        bloch_sign: int = -1,
        tolerance: float = 1.0e-8,
    ):
        if bloch_sign not in {-1, 1}:
            raise ValueError("Bloch sign must be -1 or 1.")
        self.fractional_vertices = np.asarray(fractional_vertices, dtype=float)
        self.real_lattice_vectors = np.asarray(real_lattice_vectors, dtype=float)
        self.bloch_sign = int(bloch_sign)
        self.tolerance = float(tolerance)
        self.interpolator = PeriodicTriangleInterpolator(
            self.fractional_vertices, elements, tolerance=tolerance
        )
        self._stencils: dict[tuple[bytes, bytes], BarycentricStencil] = {}

    def apply(
        self,
        values: np.ndarray,
        operation: SpaceGroupOperation,
        source_k_fractional,
        field_kind: FieldKind,
        *,
        time_reversal=None,
    ) -> np.ndarray:
        source_k = np.asarray(source_k_fractional, dtype=float)
        transformed_k = operation.act_reciprocal(source_k)
        stencil = self._stencil(operation)
        sampled = stencil.apply(values)
        component_matrix = _cartesian_field_matrix(
            operation, self.real_lattice_vectors, field_kind, self.tolerance
        )
        if field_kind in {
            FieldKind.SCALAR,
            FieldKind.ELECTRIC_Z,
            FieldKind.MAGNETIC_AXIAL_Z,
        }:
            transformed = component_matrix[0, 0] * sampled
        else:
            if sampled.ndim != 3 or sampled.shape[2] != operation.dimension:
                raise ValueError(
                    f"{field_kind.value} fields require {operation.dimension} Cartesian components."
                )
            transformed = np.einsum("ab,nvb->nva", component_matrix, sampled, optimize=True)
        if operation.antiunitary:
            if time_reversal is None:
                raise ValueError("Antiunitary Bloch actions require a field time-reversal callback.")
            transformed = np.asarray(time_reversal(transformed), dtype=np.complex128)
        phase = np.exp(
            -self.bloch_sign * 2j * np.pi * np.dot(transformed_k, operation.translation)
        )
        return phase * transformed

    def _stencil(self, operation: SpaceGroupOperation) -> BarycentricStencil:
        reduced_tau = reduce_fractional(operation.translation, self.tolerance).reduced
        key = (operation.rotation.tobytes(), np.round(reduced_tau / self.tolerance).astype(np.int64).tobytes())
        cached = self._stencils.get(key)
        if cached is not None:
            return cached
        inverse_rotation = np.linalg.inv(operation.rotation)
        preimages = (self.fractional_vertices - operation.translation) @ inverse_rotation.T
        stencil = self.interpolator.stencil(preimages)
        self._stencils[key] = stencil
        return stencil


def _cartesian_field_matrix(
    operation: SpaceGroupOperation,
    real_lattice_vectors,
    field_kind: FieldKind,
    tolerance: float,
) -> np.ndarray:
    # Kept local to avoid a bloch.py <-> analysis.py import cycle.
    lattice = np.asarray(real_lattice_vectors, dtype=float)
    rotation = lattice.T @ operation.rotation @ np.linalg.inv(lattice.T)
    residual = float(np.linalg.norm(rotation.T @ rotation - np.eye(operation.dimension), ord="fro"))
    if residual > tolerance:
        raise ValueError(f"Fractional rotation is not a lattice isometry (residual={residual:.6g}).")
    if field_kind in {FieldKind.SCALAR, FieldKind.ELECTRIC_Z}:
        return np.ones((1, 1), dtype=float)
    if field_kind == FieldKind.MAGNETIC_AXIAL_Z:
        return np.asarray([[float(np.linalg.det(rotation))]], dtype=float)
    if field_kind == FieldKind.ELECTRIC_POLAR_VECTOR:
        return rotation
    if field_kind == FieldKind.MAGNETIC_AXIAL_VECTOR:
        return float(np.linalg.det(rotation)) * rotation
    raise ValueError(f"Unsupported field kind: {field_kind!r}.")


def coefficient_metric_overlap(left, right, metric) -> np.ndarray:
    """Return C_left^dagger S C_right for coefficient-space states."""
    left_array = np.asarray(left, dtype=np.complex128)
    right_array = np.asarray(right, dtype=np.complex128)
    metric_array = np.asarray(metric, dtype=np.complex128)
    if left_array.ndim != 2 or right_array.ndim != 2 or metric_array.ndim != 2:
        raise ValueError("Coefficient states and metric must be matrices.")
    if metric_array.shape[0] != metric_array.shape[1]:
        raise ValueError("Coefficient-space metric must be square.")
    if left_array.shape[0] != metric_array.shape[0] or right_array.shape[0] != metric_array.shape[1]:
        raise ValueError("Coefficient state dimensions do not match the metric.")
    return left_array.conj().T @ metric_array @ right_array


class StateBlochSymmetryProvider:
    """Metric-weighted sewing matrices for periodic scalar COMSOL states."""

    def __init__(
        self,
        state: StateCollection,
        context: SymmetryContext,
        *,
        field_kind: FieldKind | None = None,
    ):
        if not state.is_bloch:
            raise ValueError("StateCollection must store periodic Bloch parts before symmetry analysis.")
        if field_kind is None:
            maxwell = getattr(state, "maxwell", None)
            field_kind = (
                FieldKind.SCALAR
                if maxwell is None
                else maxwell.symmetry_field_kind
            )
        if field_kind not in {
            FieldKind.SCALAR,
            FieldKind.ELECTRIC_Z,
            FieldKind.MAGNETIC_AXIAL_Z,
        }:
            raise NotImplementedError(
                "Automatic StateCollection sewing currently supports scalar COMSOL fields only."
            )
        self.state = state
        self.context = context
        self.maxwell = getattr(state, "maxwell", None)
        self._time_reversal = (
            np.conj
            if self.maxwell is None and field_kind == FieldKind.SCALAR
            else None if self.maxwell is None else self.maxwell.apply_time_reversal
        )
        self.field_kind = field_kind
        self.dimension = context.model.dimension
        dataset_convention = BlochConvention.for_dataset(state.config.dataset_type)
        if dataset_convention.sign != context.model.bloch_convention.sign:
            raise ValueError(
                f"Symmetry model uses Bloch sign {context.model.bloch_convention.sign}, but "
                f"dataset {state.config.dataset_type!r} requires {dataset_convention.sign}."
            )
        self.bloch_sign = context.model.bloch_convention.sign
        shape = tuple(len(axis) for axis in context.k_points)
        self._k_indices = tuple(np.ndindex(shape))
        self._k_fractional_points = np.asarray(
            [self._fractional_at(index) for index in self._k_indices], dtype=float
        )
        fractional = fractional_mesh_vertices(
            state.mesh, state.config.real_lattice_vectors, state.config.lattice_const
        )
        self.fractional_vertices = fractional
        self.action = BlochSymmetryAction(
            fractional,
            state.mesh.elements,
            state.config.real_lattice_vectors,
            bloch_sign=self.bloch_sign,
            tolerance=context.model.tolerance,
        )
        self._validate_periodic_node_data(context.model.boundary_tolerance)
        self._sewing_cache: dict[tuple[object, ...], SewingMatrixCacheEntry] = {}
        self._cache_fingerprint: str | None = None
        cached_names = {
            str(value).upper() for value in getattr(state.config, "use_cached_data", ())
        }
        self._cache_required = "D" in cached_names
        if self._cache_required:
            value = getattr(state.config, "D_file", None)
            input_path = getattr(state.config, "input_path", None)
            path = input_path(value) if callable(input_path) else None
            if path is None:
                raise ValueError("D cache requested, but D_file is disabled.")
            self._load_disk_cache(load_sewing_matrix_cache(path))
            LOGGER.info("Loaded D symmetry matrix cache: file=%s matrices=%s", path, len(self._sewing_cache))

    def sewing_matrix(self, request: SewingMatrixRequest) -> np.ndarray:
        if request.field_kind != self.field_kind:
            raise ValueError(
                f"Sewing request field kind {request.field_kind.value!r} does not match "
                f"the configured Maxwell field kind {self.field_kind.value!r}."
            )
        source_index, source_representative, source_shift = self._locate_k(request.source_k_fractional)
        transformed_k = request.operation.act_reciprocal(request.source_k_fractional)
        target_index, target_representative, target_shift = self._locate_k(transformed_k)
        expected_target = np.asarray(request.target_k_fractional, dtype=float)
        if np.max(np.abs(periodic_difference(target_representative, expected_target))) > self.context.model.tolerance:
            raise ValueError("Sewing request target k does not match the transformed source k.")
        requested_shift = np.asarray(request.reciprocal_lattice_shift, dtype=np.int64)
        direct_shift = np.rint(transformed_k - target_representative).astype(np.int64)
        if not np.array_equal(requested_shift, direct_shift):
            raise ValueError(
                f"Sewing reciprocal shift {requested_shift.tolist()} does not match "
                f"Rk-k'={direct_shift.tolist()}."
            )

        source_bands = tuple(int(value) for value in request.band_indices)
        target_bands = (
            source_bands
            if request.target_band_indices is None
            else tuple(int(value) for value in request.target_band_indices)
        )
        cache_key = self._sewing_cache_key(request)
        cached = self._sewing_cache.get(cache_key)
        if cached is not None:
            return self._select_cached_bands(cached, source_bands, target_bands)
        if self._cache_required:
            raise ValueError(
                "Sewing matrix cache does not contain the requested exact Seitz action: "
                f"operation={request.operation.name or request.operation_index}, "
                f"source_k={np.asarray(request.source_k_fractional).tolist()}, "
                f"source_bands={source_bands}, target_bands={target_bands}."
            )

        full_source_bands = self._actual_bands(source_index)
        full_target_bands = self._actual_bands(target_index)
        source = self._orthonormal_block(source_index, full_source_bands)
        if np.any(source_shift):
            source = source * self._fiber_phase(source_shift)[None, :]
        transformed = self.action.apply(
            source,
            request.operation,
            request.source_k_fractional,
            request.field_kind,
            time_reversal=self._time_reversal,
        )
        target = self._orthonormal_block(target_index, full_target_bands)
        if np.any(target_shift):
            target = target * self._fiber_phase(target_shift)[None, :]
        matrix = self.state.metric_overlap(
            target,
            transformed,
            chunk_size=64,
        )
        entry = SewingMatrixCacheEntry(
            operation_rotation=request.operation.rotation.copy(),
            operation_translation=request.operation.translation.copy(),
            source_k_fractional=np.asarray(request.source_k_fractional, dtype=float).copy(),
            target_k_fractional=target_representative.copy(),
            reciprocal_lattice_shift=tuple(int(value) for value in direct_shift),
            source_band_indices=full_source_bands,
            target_band_indices=full_target_bands,
            field_kind=request.field_kind.value,
            matrix=np.asarray(matrix, dtype=np.complex128).copy(),
            antiunitary=request.operation.antiunitary,
        )
        self._sewing_cache[cache_key] = entry
        return self._select_cached_bands(entry, source_bands, target_bands)

    @property
    def cached_sewing_matrices(self) -> tuple[SewingMatrixCacheEntry, ...]:
        """Return deterministic, output-ready copies of all integrated sewing matrices."""
        return tuple(self._sewing_cache[key] for key in sorted(self._sewing_cache))

    @property
    def sewing_cache_fingerprint(self) -> str:
        if self._cache_fingerprint is None:
            self._cache_fingerprint = self._calculation_fingerprint()
        return self._cache_fingerprint

    def sewing_matrix_at(
        self,
        operation_index: int,
        source_k_fractional,
        band_indices,
        *,
        operation: SpaceGroupOperation | None = None,
    ) -> np.ndarray:
        """Canonicalize a source k point and evaluate its sewing matrix."""
        source_index = self.find_k_index(source_k_fractional)
        mapping = self.mapping(operation_index, source_index)
        request = self.request_for_mapping(
            mapping,
            band_indices,
            operation=operation,
            source_k_fractional=source_k_fractional,
        )
        return self.sewing_matrix(request)

    def sewing_matrix_for_mapping(self, mapping: SymmetryKMapping, band_indices) -> np.ndarray:
        """Convenience API for a precomputed symmetry k mapping."""
        return self.sewing_matrix(self.request_for_mapping(mapping, band_indices))

    def sewing_matrix_between_mapping(
        self,
        mapping: SymmetryKMapping,
        source_band_indices,
        target_band_indices,
    ) -> np.ndarray:
        """Return the target-outer by source-outer sewing matrix."""
        return self.sewing_matrix(
            self.request_for_mapping(
                mapping,
                source_band_indices,
                target_band_indices=target_band_indices,
            )
        )

    def sewing_matrix_in_band_basis(
        self,
        mapping: SymmetryKMapping,
        source_band_indices,
        target_band_indices=None,
        *,
        operation: SpaceGroupOperation | None = None,
        source_k_fractional=None,
    ) -> np.ndarray:
        """Return sewing in a band-local Lowdin basis without another field integral.

        The cached sewing matrix uses the full outer-window orthogonalization.  That
        transform may mix non-degenerate eigenstates.  Representation analysis
        instead recovers the normalized FEM-band overlap and orthogonalizes only
        the requested source and target subspaces.
        """

        operation = operation or self.context.model.group.operations[mapping.operation_index]
        source_index = tuple(mapping.source_k_index)
        target_index = tuple(mapping.target_k_index)
        full_source = self._actual_bands(source_index)
        full_target = self._actual_bands(target_index)
        requested_source = tuple(int(value) for value in source_band_indices)
        requested_target = (
            requested_source
            if target_band_indices is None
            else tuple(int(value) for value in target_band_indices)
        )
        missing_source = sorted(set(requested_source) - set(full_source))
        missing_target = sorted(set(requested_target) - set(full_target))
        if missing_source or missing_target:
            raise ValueError(
                "Band-local sewing requested absent outer-window bands: "
                f"source_missing={missing_source}, target_missing={missing_target}."
            )

        request = self.request_for_mapping(
            mapping,
            full_source,
            operation=operation,
            source_k_fractional=source_k_fractional,
            target_band_indices=full_target,
        )
        internal = np.asarray(self.sewing_matrix(request), dtype=np.complex128)
        transforms = self.state.get_transform()
        source_transform = np.asarray(
            transforms[self._state_index(source_index)], dtype=np.complex128
        )
        target_transform = np.asarray(
            transforms[self._state_index(target_index)], dtype=np.complex128
        )
        if source_transform.shape != (len(full_source), len(full_source)) or target_transform.shape != (
            len(full_target), len(full_target)
        ):
            raise ValueError("Outer-window orthogonalization transform has an invalid shape.")

        try:
            source_inverse = np.linalg.inv(source_transform)
            target_inverse = np.linalg.inv(target_transform)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Outer-window orthogonalization transform is singular.") from exc
        source_right_inverse = (
            source_inverse.conj() if operation.antiunitary else source_inverse
        )
        fem_overlap = target_inverse.conj().T @ internal @ source_right_inverse
        source_metric = source_inverse.conj().T @ source_inverse
        target_metric = target_inverse.conj().T @ target_inverse

        source_positions = [full_source.index(band) for band in requested_source]
        target_positions = [full_target.index(band) for band in requested_target]
        source_lowdin = _inverse_sqrt_hermitian(
            source_metric[np.ix_(source_positions, source_positions)],
            description="source band block overlap",
        )
        target_lowdin = _inverse_sqrt_hermitian(
            target_metric[np.ix_(target_positions, target_positions)],
            description="target band block overlap",
        )
        block = fem_overlap[np.ix_(target_positions, source_positions)]
        source_factor = source_lowdin.conj() if operation.antiunitary else source_lowdin
        return target_lowdin.conj().T @ block @ source_factor

    def request_for_mapping(
        self,
        mapping: SymmetryKMapping,
        band_indices,
        *,
        operation: SpaceGroupOperation | None = None,
        source_k_fractional=None,
        target_band_indices=None,
    ) -> SewingMatrixRequest:
        from .analysis import SewingMatrixRequest

        operation = operation or self.context.model.group.operations[mapping.operation_index]
        source_index = tuple(mapping.source_k_index)
        target_index = tuple(mapping.target_k_index)
        source_k = self._fractional_at(source_index) if source_k_fractional is None else np.asarray(source_k_fractional)
        target_k = self._fractional_at(target_index)
        transformed = operation.act_reciprocal(source_k)
        shift = np.rint(transformed - target_k).astype(np.int64)
        return SewingMatrixRequest(
            operation_index=mapping.operation_index,
            operation=operation,
            source_k_fractional=np.asarray(source_k, dtype=float),
            target_k_fractional=target_k,
            reciprocal_lattice_shift=tuple(int(value) for value in shift),
            band_indices=tuple(int(value) for value in band_indices),
            field_kind=self.field_kind,
            target_band_indices=(
                None
                if target_band_indices is None
                else tuple(int(value) for value in target_band_indices)
            ),
        )

    def mapping(self, operation_index: int, source_index) -> SymmetryKMapping:
        source = tuple(int(value) for value in source_index)
        shape = tuple(len(axis) for axis in self.context.k_points)
        flat = int(np.ravel_multi_index(source, shape))
        mapping = self.context.k_mappings[operation_index][flat]
        if mapping.source_k_index != source:
            raise RuntimeError("Symmetry k-mapping order is inconsistent with the k mesh.")
        return mapping

    def find_k_index(self, k_fractional) -> tuple[int, ...]:
        return self._locate_k(k_fractional)[0]

    def _locate_k(self, k_fractional) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
        point = np.asarray(k_fractional, dtype=float)
        distances = np.max(np.abs(periodic_difference(self._k_fractional_points, point)), axis=1)
        flat = int(np.argmin(distances))
        if distances[flat] > self.context.model.tolerance:
            raise ValueError(f"k point {point.tolist()} is not present in the configured symmetry k mesh.")
        representative = self._k_fractional_points[flat]
        shift_float = point - representative
        shift = np.rint(shift_float).astype(np.int64)
        if not np.allclose(shift_float, shift, rtol=0.0, atol=self.context.model.tolerance):
            raise ValueError("Periodic k-point matching did not yield an integer reciprocal shift.")
        return self._k_indices[flat], representative, shift

    def _fractional_at(self, index) -> np.ndarray:
        return np.asarray(
            [self.context.k_points[axis][index[axis]] for axis in range(self.dimension)],
            dtype=float,
        )

    def _state_index(self, index) -> tuple[int, int, int]:
        values = list(index) + [0, 0, 0]
        return int(values[0]), int(values[1]), int(values[2])

    def _orthonormal_block(self, index, band_indices) -> np.ndarray:
        state_index = self._state_index(index)
        actual = tuple(int(value) for value in np.asarray(self.state.E_idx[state_index]).reshape(-1))
        requested = tuple(int(value) for value in band_indices)
        missing = sorted(set(requested) - set(actual))
        if missing:
            raise ValueError(f"Actual Bloch bands {missing} are missing at k index {tuple(index)}.")
        local = [actual.index(band) for band in requested]
        block = self.state.get_block(*state_index)
        correction = np.asarray(self.state.get_transform()[state_index], dtype=np.complex128)
        orthonormal = (block.T @ correction).T
        return orthonormal[local]

    def _actual_bands(self, index) -> tuple[int, ...]:
        state_index = self._state_index(index)
        return tuple(int(value) for value in np.asarray(self.state.E_idx[state_index]).reshape(-1))

    def _validate_periodic_node_data(self, tolerance: float) -> None:
        classes = self.action.interpolator.periodic_node_classes
        if not classes:
            return
        metric = np.asarray(self.state.metric_material).reshape(-1)
        if metric.size != self.fractional_vertices.shape[0]:
            raise ValueError("Metric material size does not match the symmetry FEM mesh.")
        metric_scale = max(float(np.max(np.abs(metric), initial=0.0)), np.finfo(float).tiny)
        for nodes in classes:
            values = metric[np.asarray(nodes, dtype=np.intp)]
            residual = float(np.max(np.abs(values - values[0]), initial=0.0) / metric_scale)
            if residual > tolerance:
                raise ValueError(
                    "Periodic-equivalent FEM nodes have inconsistent metric material values: "
                    f"nodes={nodes}, relative_residual={residual:.6g}, tolerance={tolerance:.6g}."
                )

        for k_index in self._k_indices:
            state_index = self._state_index(k_index)
            block = self.state.get_block(*state_index)
            scales = np.maximum(
                np.max(np.abs(block), axis=1),
                np.finfo(float).tiny,
            )
            actual_bands = tuple(
                int(value) for value in np.asarray(self.state.E_idx[state_index]).reshape(-1)
            )
            for nodes in classes:
                node_indices = np.asarray(nodes, dtype=np.intp)
                values = block[:, node_indices]
                residuals = np.max(np.abs(values - values[:, :1]), axis=1) / scales
                worst = int(np.argmax(residuals))
                if residuals[worst] > tolerance:
                    band = actual_bands[worst] if worst < len(actual_bands) else worst
                    raise ValueError(
                        "Periodic-equivalent FEM nodes have inconsistent periodic Bloch fields: "
                        f"k={state_index}, band={band}, nodes={nodes}, "
                        f"relative_residual={residuals[worst]:.6g}, tolerance={tolerance:.6g}."
                    )

    def _fiber_phase(self, reciprocal_shift) -> np.ndarray:
        shift = np.asarray(reciprocal_shift, dtype=float)
        return np.exp(
            -self.bloch_sign * 2j * np.pi * (self.fractional_vertices @ shift)
        )

    def _sewing_cache_key(self, request: SewingMatrixRequest) -> tuple[object, ...]:
        return self._cache_key(
            request.operation.rotation,
            request.operation.translation,
            request.source_k_fractional,
            request.field_kind.value,
            request.operation.antiunitary,
        )

    def _cache_key(
        self,
        rotation,
        translation,
        source_k,
        field_kind: str,
        antiunitary: bool = False,
    ) -> tuple[object, ...]:
        tolerance = self.context.model.tolerance
        source = tuple(np.rint(np.asarray(source_k) / tolerance).astype(np.int64))
        translation_key = tuple(np.rint(np.asarray(translation) / tolerance).astype(np.int64))
        return (
            np.ascontiguousarray(rotation, dtype=np.int64).tobytes(),
            translation_key,
            source,
            str(field_kind),
            bool(antiunitary),
        )

    def _select_cached_bands(
        self,
        entry: SewingMatrixCacheEntry,
        source_bands: tuple[int, ...],
        target_bands: tuple[int, ...],
    ) -> np.ndarray:
        missing_source = sorted(set(source_bands) - set(entry.source_band_indices))
        missing_target = sorted(set(target_bands) - set(entry.target_band_indices))
        if missing_source or missing_target:
            raise ValueError(
                "Requested bands are absent from the sewing cache entry: "
                f"source_missing={missing_source}, target_missing={missing_target}."
            )
        columns = [entry.source_band_indices.index(band) for band in source_bands]
        rows = [entry.target_band_indices.index(band) for band in target_bands]
        return np.asarray(entry.matrix[np.ix_(rows, columns)], dtype=np.complex128).copy()

    def _load_disk_cache(self, cache: SewingMatrixCache) -> None:
        expected_shape = tuple(len(axis) for axis in self.context.k_points)
        if cache.dimension != self.dimension:
            raise ValueError(
                f"Sewing cache dimension {cache.dimension} does not match symmetry dimension {self.dimension}."
            )
        if cache.bloch_sign != self.bloch_sign:
            raise ValueError(
                f"Sewing cache Bloch sign {cache.bloch_sign} does not match current sign {self.bloch_sign}."
            )
        if cache.k_shape != expected_shape:
            raise ValueError(
                f"Sewing cache k shape {cache.k_shape} does not match current shape {expected_shape}."
            )
        if cache.calculation_fingerprint != self.sewing_cache_fingerprint:
            raise ValueError(
                "Sewing cache calculation fingerprint does not match the current mesh, fields, "
                "metric material, field components, orthogonalization, lattice, or integration settings."
            )
        for entry in cache.entries:
            try:
                field_kind = FieldKind(entry.field_kind)
            except ValueError as exc:
                raise ValueError(f"Sewing cache has unknown field kind {entry.field_kind!r}.") from exc
            source_index, _, _ = self._locate_k(entry.source_k_fractional)
            operation = SpaceGroupOperation(
                entry.operation_rotation,
                entry.operation_translation,
                antiunitary=entry.antiunitary,
            )
            transformed = operation.act_reciprocal(entry.source_k_fractional)
            target_index, target_representative, _ = self._locate_k(transformed)
            if np.max(np.abs(periodic_difference(target_representative, entry.target_k_fractional))) > self.context.model.tolerance:
                raise ValueError("Sewing cache target k is inconsistent with its exact Seitz action.")
            shift = tuple(int(value) for value in np.rint(transformed - target_representative))
            if shift != entry.reciprocal_lattice_shift:
                raise ValueError(
                    f"Sewing cache reciprocal shift {entry.reciprocal_lattice_shift} does not match {shift}."
                )
            source_bands = self._actual_bands(source_index)
            target_bands = self._actual_bands(target_index)
            if entry.source_band_indices != source_bands or entry.target_band_indices != target_bands:
                raise ValueError(
                    "Sewing cache outer-window band ids do not match the current calculation: "
                    f"cached=({entry.source_band_indices}, {entry.target_band_indices}), "
                    f"current=({source_bands}, {target_bands})."
                )
            expected_matrix_shape = (len(target_bands), len(source_bands))
            if entry.matrix.shape != expected_matrix_shape or not np.all(np.isfinite(entry.matrix)):
                raise ValueError(
                    f"Sewing cache matrix has invalid shape or values; expected {expected_matrix_shape}."
                )
            key = self._cache_key(
                entry.operation_rotation,
                entry.operation_translation,
                entry.source_k_fractional,
                field_kind.value,
                entry.antiunitary,
            )
            previous = self._sewing_cache.get(key)
            if previous is not None and not np.allclose(
                previous.matrix, entry.matrix, rtol=0.0, atol=self.context.model.tolerance
            ):
                raise ValueError("Sewing cache contains conflicting duplicate entries.")
            self._sewing_cache[key] = entry

    def _calculation_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"PCWannier sewing input v2\0")
        for label, value in (
            ("real_lattice_vectors", self.state.config.real_lattice_vectors),
            ("lattice_const", [self.state.config.lattice_const]),
            ("bloch_sign", [self.bloch_sign]),
            ("symmetry_tolerance", [self.context.model.tolerance]),
            ("mesh_vertices", self.state.mesh.vertices),
            ("mesh_elements", self.state.mesh.elements),
            ("metric_material", self.state.metric_material),
        ):
            _update_array_digest(digest, label, value)
        digest.update(self.field_kind.value.encode("utf-8"))
        maxwell = getattr(self.state, "maxwell", None)
        if maxwell is not None:
            digest.update(maxwell.field_components.value.encode("utf-8"))
            digest.update(maxwell.metric_material.value.encode("utf-8"))
        digest.update(str(self.state.integration_mode).encode("utf-8"))
        bias = self.context.model.magnetic_bias_direction
        if bias is not None:
            _update_array_digest(digest, "magnetic_bias_direction", bias)
        for operation in self.context.model.group.operations:
            digest.update(bytes((int(operation.antiunitary),)))
        transforms = self.state.get_transform()
        for k_index in self._k_indices:
            state_index = self._state_index(k_index)
            _update_array_digest(digest, f"bands:{state_index}", self._actual_bands(k_index))
            _update_array_digest(digest, f"field:{state_index}", self.state.get_block(*state_index))
            _update_array_digest(digest, f"transform:{state_index}", transforms[state_index])
        return digest.hexdigest()


def _inverse_sqrt_hermitian(matrix, *, description: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    value = 0.5 * (value + value.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    threshold = 1.0e-12 * scale
    if np.min(eigenvalues, initial=np.inf) <= threshold:
        raise ValueError(
            f"{description} is not positive definite; minimum eigenvalue="
            f"{float(np.min(eigenvalues)):.6g}."
        )
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.conj().T


def _update_array_digest(digest, label: str, value) -> None:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"Cannot fingerprint object array {label!r}.")
    digest.update(label.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    if array.flags.c_contiguous:
        digest.update(memoryview(array).cast("B"))
        return
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        order="C",
        buffersize=1 << 17,
    )
    for chunk in iterator:
        contiguous = np.ascontiguousarray(chunk)
        digest.update(memoryview(contiguous).cast("B"))
