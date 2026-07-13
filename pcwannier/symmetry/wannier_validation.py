from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..compute.integration import (
    MeshIntegralView,
    integrate_weighted_abs2_columns,
    validated_real,
)
from ..compute.wannier import generate_wannier
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
        output = np.zeros(array.shape[:-1] + (self.vertex_indices.shape[0],), dtype=array.dtype)
        valid = self.valid_vertices
        if array.ndim == 1:
            output[valid] = np.einsum(
                "vc,vc->v",
                array[self.vertex_indices[valid]],
                self.weights[valid],
                optimize=True,
            )
            return output
        raise ValueError("Localized Wannier interpolation currently expects one scalar field.")


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


def validate_wannier_symmetry(
    ctx,
    targets,
    *,
    zero_cell_wanniers: np.ndarray | None = None,
    tolerance: float = 1.0e-6,
    minimum_retained_norm: float = 0.99,
) -> WannierSymmetryValidation:
    """Validate the induced real-space Wannier transformation on a 2D scalar FEM mesh."""
    if ctx.symmetry_gauge is None:
        raise ValueError("Real-space symmetry validation requires a symmetry-adapted Bloch gauge.")
    state = ctx.state
    if state.extention_mesh is None or state.extention_epsilon is None:
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
    if zero_cell.shape != (mesh.vertices.shape[0], expected_dimension):
        raise ValueError(
            f"Zero-cell Wannier array has shape {zero_cell.shape}; "
            f"expected {(mesh.vertices.shape[0], expected_dimension)}."
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
    interpolator = _TriangleInterpolator(mesh.vertices, mesh.elements, physical_tolerance)
    operation_data = {}
    for operation_index, operation in enumerate(group.operations):
        preimage_fractional = (fractional - operation.translation) @ np.linalg.inv(operation.rotation).T
        preimage_cartesian = preimage_fractional @ lattice * float(ctx.config.lattice_const)
        stencil = interpolator.stencil(preimage_cartesian)
        valid_elements = np.all(stencil.valid_vertices[mesh.elements], axis=1)
        if not np.any(valid_elements):
            raise RuntimeError(f"No common interior triangles remain for operation {operation.name}.")
        view = MeshIntegralView(
            np.asarray(mesh.elements[valid_elements], dtype=np.intp),
            np.asarray(mesh.tri_weights[valid_elements], dtype=float),
            mesh.vertices.shape[0],
        )
        operation_data[operation_index] = (stencil, view)

    full_view = state.extention_integral_view
    epsilon = state.extention_epsilon
    entries = []
    offset = 0
    for target in target_items:
        irrep_dimension = target.site_irrep.dimension
        for operation_index, operation in enumerate(group.operations):
            stencil, valid_view = operation_data[operation_index]
            for orbit_index in range(target.multiplicity):
                action = target.orbit.action(operation_index, orbit_index)
                shift = tuple(int(value) for value in action.lattice_shift)
                site_matrix = target.site_irrep.matrix(action.site_element_index)
                for irrep_index in range(irrep_dimension):
                    source_index = offset + target.wannier_index(irrep_index, orbit_index)
                    transformed = stencil.apply(zero_cell[:, source_index])
                    target_indices = tuple(
                        offset + target.wannier_index(row, action.target_index)
                        for row in range(irrep_dimension)
                    )
                    expected = sum(
                        site_matrix[row, irrep_index] * cell_fields[shift][:, target_indices[row]]
                        for row in range(irrep_dimension)
                    )
                    residual_field = transformed - expected
                    residual_norm = _field_norm(valid_view, epsilon, residual_field, state)
                    transformed_norm = _field_norm(valid_view, epsilon, transformed, state)
                    expected_norm = _field_norm(valid_view, epsilon, expected, state)
                    full_source_norm = _field_norm(full_view, epsilon, zero_cell[:, source_index], state)
                    full_expected_norm = _field_norm(full_view, epsilon, expected, state)
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
    if max_residual > tolerance:
        raise RuntimeError(
            f"Real-space Wannier symmetry residual {max_residual:.6g} exceeds {tolerance:.6g}."
        )
    return result


def _field_norm(view, epsilon, field, state) -> float:
    value = integrate_weighted_abs2_columns(
        view,
        epsilon,
        np.asarray(field).reshape(-1, 1),
        chunk_size=1,
        backend=state.compute_backend,
        mode=state.integration_mode,
    )
    return float(validated_real(np.atleast_1d(value), "Wannier symmetry norm")[0])

