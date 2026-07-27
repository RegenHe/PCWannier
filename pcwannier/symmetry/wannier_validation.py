from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from ..compute.wannier import generate_wannier
from .field_action import cartesian_field_matrix
from .bloch import fractional_mesh_vertices


@dataclass(frozen=True)
class WannierSymmetryEntry:
    operation_index: int
    operation_name: str
    target_name: str
    source_wannier_index: int
    target_wannier_indices: tuple[int, ...]
    lattice_shift: tuple[int, ...]
    residual: float
    retained_norm: float


@dataclass(frozen=True)
class WannierSymmetryValidation:
    entries: tuple[WannierSymmetryEntry, ...]
    max_residual: float
    mean_residual: float
    minimum_retained_norm: float


@dataclass(frozen=True)
class _PartialStencil:
    vertex_indices: np.ndarray
    weights: np.ndarray
    valid_vertices: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim < 1 or array.shape[0] <= np.max(self.vertex_indices, initial=-1):
            raise ValueError("Localized Wannier values must use the mesh-point axis first.")
        output = np.zeros(
            (self.vertex_indices.shape[0],) + array.shape[1:], dtype=array.dtype
        )
        valid = self.valid_vertices
        if array.ndim == 1:
            output[valid] = np.einsum(
                "vc,vc->v",
                array[self.vertex_indices[valid]],
                self.weights[valid],
                optimize=True,
            )
            return output
        output[valid] = np.einsum(
            "vc,vc...->v...",
            self.weights[valid],
            array[self.vertex_indices[valid]],
            optimize=True,
        )
        return output


class _TriangleInterpolator:
    def __init__(self, vertices, elements, tolerance: float):
        self.vertices = np.asarray(vertices, dtype=float)
        self.elements = np.asarray(elements, dtype=np.intp)
        self.tolerance = float(tolerance)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise NotImplementedError("Wannier symmetry validation currently supports 2D meshes.")
        self.triangles = self.vertices[self.elements]
        self.lower = np.min(self.triangles, axis=1)
        self.upper = np.max(self.triangles, axis=1)
        self.domain_lower = np.min(self.vertices, axis=0)
        self.domain_upper = np.max(self.vertices, axis=0)
        self.extent = np.maximum(self.domain_upper - self.domain_lower, self.tolerance)
        self.grid_count = max(8, int(np.ceil(np.sqrt(len(self.elements)))))
        self.tree = cKDTree(self.vertices)
        self.buckets: dict[tuple[int, int], list[int]] = {}
        for triangle_index, (lower, upper) in enumerate(zip(self.lower, self.upper)):
            cell_lower = self._cell(lower)
            cell_upper = self._cell(upper)
            for ix in range(cell_lower[0], cell_upper[0] + 1):
                for iy in range(cell_lower[1], cell_upper[1] + 1):
                    self.buckets.setdefault((ix, iy), []).append(triangle_index)

    def stencil(self, points) -> _PartialStencil:
        query = np.asarray(points, dtype=float)
        indices = np.zeros((len(query), 3), dtype=np.intp)
        weights = np.zeros((len(query), 3), dtype=float)
        valid = np.zeros(len(query), dtype=bool)
        distances, nearest = self.tree.query(query, k=1, p=np.inf)
        for point_index, point in enumerate(query):
            if np.any(point < self.domain_lower - self.tolerance) or np.any(
                point > self.domain_upper + self.tolerance
            ):
                continue
            if distances[point_index] <= self.tolerance:
                node = int(nearest[point_index])
                indices[point_index] = (node, node, node)
                weights[point_index] = (1.0, 0.0, 0.0)
                valid[point_index] = True
                continue
            found = self._find(point, self.buckets.get(tuple(self._cell(point)), ()))
            if found is None:
                continue
            triangle_index, barycentric = found
            indices[point_index] = self.elements[triangle_index]
            weights[point_index] = barycentric
            valid[point_index] = True
        return _PartialStencil(indices, weights, valid)

    def _cell(self, point: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(point) - self.domain_lower) / self.extent
        return np.floor(np.clip(scaled, 0.0, 1.0 - np.finfo(float).eps) * self.grid_count).astype(int)

    def _find(self, point: np.ndarray, candidates) -> tuple[int, np.ndarray] | None:
        best = None
        best_violation = np.inf
        for triangle_index in candidates:
            if np.any(point < self.lower[triangle_index] - self.tolerance) or np.any(
                point > self.upper[triangle_index] + self.tolerance
            ):
                continue
            triangle = self.triangles[triangle_index]
            matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
            if abs(float(np.linalg.det(matrix))) <= np.finfo(float).eps:
                continue
            uv = np.linalg.solve(matrix, point - triangle[0])
            barycentric = np.array([1.0 - uv[0] - uv[1], uv[0], uv[1]])
            violation = float(max(0.0, -np.min(barycentric), np.max(barycentric) - 1.0))
            if violation <= self.tolerance and violation < best_violation:
                best = (int(triangle_index), barycentric)
                best_violation = violation
        return best


class _RegularGridInterpolator:
    """Partial multilinear interpolation on an extended non-periodic grid."""

    def __init__(self, grid, tolerance: float):
        self.shape = tuple(int(value) for value in grid.shape)
        self.dimension = len(self.shape)
        self.tolerance = float(tolerance)
        fractional = np.asarray(grid.fractional_vertices, dtype=float).reshape(
            self.shape + (self.dimension,)
        )
        self.start = fractional[(0,) * self.dimension]
        self.step = np.asarray(
            [
                1.0 / int(grid.base_shape[axis])
                for axis in range(self.dimension)
            ],
            dtype=float,
        )

    def stencil(self, points) -> _PartialStencil:
        query = np.asarray(points, dtype=float)
        if query.ndim != 2 or query.shape[1] != self.dimension:
            raise ValueError(
                f"Interpolation points must have shape (N, {self.dimension})."
            )
        scaled = (query - self.start) / self.step
        rounded = np.rint(scaled)
        scaled = np.where(
            np.abs(scaled - rounded) <= self.tolerance / self.step,
            rounded,
            scaled,
        )
        upper_bound = np.asarray(self.shape, dtype=float) - 1.0
        valid = np.all(
            (scaled >= -self.tolerance / self.step)
            & (scaled <= upper_bound + self.tolerance / self.step),
            axis=1,
        )
        scaled = np.clip(scaled, 0.0, upper_bound)
        lower = np.floor(scaled).astype(np.int64)
        fraction = scaled - lower
        for axis, size in enumerate(self.shape):
            if size == 1:
                lower[:, axis] = 0
                fraction[:, axis] = 0.0
            else:
                at_upper = lower[:, axis] >= size - 1
                lower[at_upper, axis] = size - 2
                fraction[at_upper, axis] = 1.0

        corners = tuple(product((0, 1), repeat=self.dimension))
        indices = np.zeros((len(query), len(corners)), dtype=np.intp)
        weights = np.zeros((len(query), len(corners)), dtype=float)
        for corner_index, corner in enumerate(corners):
            corner_array = np.asarray(corner, dtype=np.int64)
            multi = lower + corner_array
            for axis, size in enumerate(self.shape):
                if size == 1:
                    multi[:, axis] = 0
            indices[:, corner_index] = np.ravel_multi_index(
                tuple(multi[:, axis] for axis in range(self.dimension)),
                self.shape,
            )
            component_weights = np.where(
                corner_array[None, :] == 0,
                1.0 - fraction,
                fraction,
            )
            weights[:, corner_index] = np.prod(component_weights, axis=1)
        return _PartialStencil(indices, weights, valid)


def validate_wannier_symmetry(
    ctx,
    targets,
    *,
    zero_cell_wanniers: np.ndarray | None = None,
    tolerance: float = 1.0e-6,
    minimum_retained_norm: float = 0.99,
    enforce_residual: bool = True,
) -> WannierSymmetryValidation:
    """Validate induced real-space Wannier transformations on supported meshes."""
    if ctx.symmetry_gauge is None:
        raise ValueError("Real-space symmetry validation requires a symmetry-adapted Bloch gauge.")
    state = ctx.state
    if state.extention_mesh is None or state.extended_metric_material is None:
        raise ValueError("Wannier symmetry validation requires the extended mesh.")
    target_items = tuple(targets)
    if not target_items:
        raise ValueError("Wannier symmetry validation requires at least one target.")
    group = target_items[0].group
    mesh = state.extention_mesh
    if zero_cell_wanniers is None:
        _, zero_cell_wanniers, _ = generate_wannier(ctx)
    zero_cell = np.asarray(zero_cell_wanniers, dtype=np.complex128)
    expected_dimension = sum(target.wannier_dimension for target in target_items)
    expected_shape = (mesh.vertices.shape[0], expected_dimension)
    if state.get_extention_block(0, 0, 0).ndim == 3:
        expected_shape += (state.get_extention_block(0, 0, 0).shape[-1],)
    if zero_cell.shape != expected_shape:
        raise ValueError(
            f"Zero-cell Wannier array has shape {zero_cell.shape}; "
            f"expected {expected_shape}."
        )

    needed_shifts = {
        tuple(int(value) for value in action.lattice_shift)
        for target in target_items
        for operation_actions in target.orbit.actions
        for action in operation_actions
    }
    cell_fields: dict[tuple[int, ...], np.ndarray] = {(0,) * group.dimension: zero_cell}
    for shift in sorted(needed_shifts):
        if shift not in cell_fields:
            _, fields, _ = generate_wannier(ctx, list(shift))
            cell_fields[shift] = fields

    fractional = fractional_mesh_vertices(
        mesh, ctx.config.real_lattice_vectors, ctx.config.lattice_const
    )
    lattice = np.asarray(ctx.config.real_lattice_vectors, dtype=float)
    physical_tolerance = max(
        group.tolerance * float(ctx.config.lattice_const),
        np.finfo(float).eps * max(float(np.max(np.abs(mesh.vertices))), 1.0) * 128.0,
    )
    integration_family = getattr(mesh, "integration_family", "finite_element")
    if integration_family == "uniform_grid":
        interpolator = _RegularGridInterpolator(mesh, group.tolerance)
    elif integration_family == "finite_element":
        interpolator = _TriangleInterpolator(
            mesh.vertices,
            mesh.elements,
            physical_tolerance,
        )
    else:
        raise ValueError(f"Unknown spatial integration family {integration_family!r}.")
    if state.extended_inner_product is None:
        raise RuntimeError("Extended metric inner product has not been initialized.")
    operation_data = {}
    for operation_index, operation in enumerate(group.operations):
        preimage_fractional = (fractional - operation.translation) @ np.linalg.inv(operation.rotation).T
        if integration_family == "uniform_grid":
            stencil = interpolator.stencil(preimage_fractional)
            if not np.any(stencil.valid_vertices):
                raise RuntimeError(
                    f"No common interior grid points remain for operation {operation.name}."
                )
            inner_product = state.extended_inner_product.restrict_points(
                stencil.valid_vertices
            )
        else:
            preimage_cartesian = (
                preimage_fractional
                @ lattice
                * float(ctx.config.lattice_const)
            )
            stencil = interpolator.stencil(preimage_cartesian)
            valid_elements = np.all(
                stencil.valid_vertices[mesh.elements],
                axis=1,
            )
            if not np.any(valid_elements):
                raise RuntimeError(
                    f"No common interior triangles remain for operation {operation.name}."
                )
            inner_product = state.extended_inner_product.restrict_elements(
                valid_elements
            )
        field_matrix = cartesian_field_matrix(
            operation,
            lattice,
            state.maxwell.symmetry_field_kind,
            group.tolerance,
        )
        operation_data[operation_index] = (
            stencil,
            inner_product,
            np.asarray(field_matrix, dtype=np.complex128),
        )

    full_inner_product = state.extended_inner_product
    entries = []
    offset = 0
    for target in target_items:
        irrep_dimension = target.site_irrep.dimension
        for operation_index, operation in enumerate(group.operations):
            stencil, valid_inner_product, component_matrix = operation_data[operation_index]
            for orbit_index in range(target.multiplicity):
                action = target.orbit.action(operation_index, orbit_index)
                shift = tuple(int(value) for value in action.lattice_shift)
                site_matrix = target.site_irrep.matrix(action.site_element_index)
                for irrep_index in range(irrep_dimension):
                    source_index = offset + target.wannier_index(irrep_index, orbit_index)
                    transformed = stencil.apply(zero_cell[:, source_index])
                    if operation.antiunitary:
                        transformed = state.maxwell.apply_time_reversal(transformed)
                    if transformed.ndim == 1:
                        transformed = complex(component_matrix[0, 0]) * transformed
                    else:
                        transformed = transformed @ component_matrix.T
                    target_indices = tuple(
                        offset + target.wannier_index(row, action.target_index)
                        for row in range(irrep_dimension)
                    )
                    expected = sum(
                        site_matrix[row, irrep_index] * cell_fields[shift][:, target_indices[row]]
                        for row in range(irrep_dimension)
                    )
                    residual_field = transformed - expected
                    residual_norm = _field_norm(valid_inner_product, residual_field)
                    transformed_norm = _field_norm(valid_inner_product, transformed)
                    expected_norm = _field_norm(valid_inner_product, expected)
                    full_source_norm = _field_norm(
                        full_inner_product, zero_cell[:, source_index]
                    )
                    full_expected_norm = _field_norm(full_inner_product, expected)
                    denominator = max(np.sqrt(transformed_norm), np.sqrt(expected_norm), 1.0e-15)
                    residual = float(np.sqrt(residual_norm) / denominator)
                    retained = float(
                        min(
                            transformed_norm / max(full_source_norm, 1.0e-30),
                            expected_norm / max(full_expected_norm, 1.0e-30),
                        )
                    )
                    entries.append(
                        WannierSymmetryEntry(
                            operation_index,
                            operation.name or f"g{operation_index}",
                            target.name,
                            source_index,
                            target_indices,
                            shift,
                            residual,
                            retained,
                        )
                    )
        offset += target.wannier_dimension

    max_residual = max((entry.residual for entry in entries), default=0.0)
    mean_residual = float(np.mean([entry.residual for entry in entries])) if entries else 0.0
    retained = min((entry.retained_norm for entry in entries), default=1.0)
    result = WannierSymmetryValidation(tuple(entries), max_residual, mean_residual, retained)
    if retained < minimum_retained_norm:
        raise RuntimeError(
            f"Wannier symmetry validation retained only {retained:.6g} of the norm; "
            f"required {minimum_retained_norm:.6g}. Increase extension."
        )
    if enforce_residual and max_residual > tolerance:
        raise RuntimeError(
            f"Real-space Wannier symmetry residual {max_residual:.6g} exceeds {tolerance:.6g}."
        )
    return result


def _field_norm(inner_product, field) -> float:
    return inner_product.norm(field, name="Wannier symmetry norm")
