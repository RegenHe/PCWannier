from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable
import copy
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from .config import IncarConfig
from .conventions import BlochConvention, BlochFieldRepresentation
from .maxwell import FieldKind, MaxwellProblem

if TYPE_CHECKING:
    from .etbc import ETBCCompletionResult
    from .symmetry.analysis import (
        BlochSymmetryAnalysisResult,
        GammaZeroRegularizationAnalysis,
        SymmetryAnalysisResult,
    )
    from .symmetry.cache import SewingMatrixCacheEntry
    from .symmetry.disentanglement import SymmetryDisentanglementResult
    from .symmetry.gauge import SymmetryGaugeResult
    from .symmetry.localization import SymmetryLocalizationResult
    from .symmetry.representation import SymmetryContext


def periodic_axis_coordinates(
    size: int,
    *,
    sample_offset: float = 0.0,
) -> np.ndarray:
    """Return one half-open periodic fractional grid axis.

    A single-sample direction is geometrically degenerate and is represented
    at the cell center rather than at the otherwise equivalent ``-1/2`` edge.
    """

    count = int(size)
    offset = float(sample_offset)
    if count <= 0:
        raise ValueError("Periodic grid axis size must be positive.")
    if not np.isfinite(offset):
        raise ValueError("Periodic grid sample offset must be finite.")
    if count == 1:
        return np.zeros(1, dtype=float)
    return (np.arange(count, dtype=float) + offset) / count - 0.5


class Mesh:
    def __init__(self, vertices: np.ndarray, elements: np.ndarray, edge: np.ndarray | None = None) -> None:
        self.vertices = np.asarray(vertices, dtype=float)
        self.elements = np.asarray(elements, dtype=np.intp)
        self.edge = None if edge is None else np.asarray(edge, dtype=np.intp)
        tree = cKDTree(self.vertices)
        dists, _ = tree.query(self.vertices, k=2)
        self.mindist = float(np.min(dists[:, 1])) if len(self.vertices) > 1 else 0.0
        self.tri_weights: np.ndarray | None = None
        self._boundary_mask_cache: np.ndarray | None = None
        self._precompute_tri_weights()

    def _precompute_tri_weights(self) -> None:
        elems = self.elements
        verts = self.vertices
        v0 = verts[elems[:, 0]]
        v1 = verts[elems[:, 1]]
        v2 = verts[elems[:, 2]]
        self.tri_weights = np.abs(
            (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
            - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
        ) / 6.0

    def func(self, fn, offset=(0.0, 0.0)) -> np.ndarray:
        dx = self.vertices[:, 0] - offset[0]
        dy = self.vertices[:, 1] - offset[1]
        return np.asarray(fn(dx, dy), dtype=np.complex128)

    def rfunc(self, fn, offset=(0.0, 0.0), ang=0.0) -> np.ndarray:
        dx = self.vertices[:, 0] - offset[0]
        dy = self.vertices[:, 1] - offset[1]
        return np.asarray(fn(np.hypot(dx, dy), np.arctan2(dy, dx) + np.deg2rad(ang)), dtype=np.complex128)

    def extension(
        self,
        n: list[int],
        real_lattice_vectors: list[list[float]],
        lattice_const: float,
    ) -> np.ndarray:
        if len(n) < 2 or n[0] < 1 or n[1] < 1:
            raise ValueError("extension must contain two positive integers.")

        original_vertices = self.vertices.copy()
        original_elements = self.elements.copy()
        nv = original_vertices.shape[0]
        ne = original_elements.shape[0]
        tile_offsets = self._extension_offsets(n, real_lattice_vectors, lattice_const)

        tiled_vertices = (original_vertices[None, :, :] + tile_offsets[:, None, :]).reshape(-1, original_vertices.shape[1])
        tiled_elements = (
            original_elements[None, :, :] + (np.arange(tile_offsets.shape[0], dtype=np.intp) * nv)[:, None, None]
        ).reshape(tile_offsets.shape[0] * ne, original_elements.shape[1])
        tiled_mapping = np.tile(np.arange(nv, dtype=np.intp), tile_offsets.shape[0])

        raw_to_unique, unique_raw = self._merge_extension_vertices(tiled_vertices, nv)

        offset_x = (
            real_lattice_vectors[0][0] * np.floor((n[0] - 1) / 2)
            + real_lattice_vectors[1][0] * np.floor((n[1] - 1) / 2)
        ) * lattice_const
        offset_y = (
            real_lattice_vectors[0][1] * np.floor((n[0] - 1) / 2)
            + real_lattice_vectors[1][1] * np.floor((n[1] - 1) / 2)
        ) * lattice_const
        self.vertices = tiled_vertices[unique_raw] - np.array([offset_x, offset_y])
        self.elements = raw_to_unique[tiled_elements]
        self.edge = None
        self._boundary_mask_cache = None
        self._precompute_tri_weights()
        return tiled_mapping[unique_raw]

    def _extension_offsets(
        self,
        n: list[int],
        real_lattice_vectors: list[list[float]],
        lattice_const: float,
    ) -> np.ndarray:
        a1 = np.asarray(real_lattice_vectors[0], dtype=float) * lattice_const
        a2 = np.asarray(real_lattice_vectors[1], dtype=float) * lattice_const
        offsets = np.empty((int(n[0]) * int(n[1]), 2), dtype=float)
        pos = 0
        for i in range(int(n[0])):
            for j in range(int(n[1])):
                offsets[pos] = i * a1 + j * a2
                pos += 1
        return offsets

    def _merge_extension_vertices(self, vertices: np.ndarray, original_vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
        raw_count = vertices.shape[0]
        parent = np.arange(raw_count, dtype=np.intp)
        boundary_mask = self._boundary_vertex_mask(original_vertex_count)
        candidates = np.flatnonzero(np.tile(boundary_mask, raw_count // original_vertex_count))

        coordinate_scale = max(float(np.max(np.abs(vertices))) if vertices.size else 0.0, 1.0)
        threshold = max(np.finfo(float).eps * coordinate_scale * 128.0, coordinate_scale * 1e-12)
        if candidates.size > 1:
            self._merge_by_coordinate_hash(parent, vertices, candidates, threshold)

        reps = np.fromiter((self._find(parent, idx) for idx in range(raw_count)), dtype=np.intp, count=raw_count)
        unique_raw, raw_to_unique = np.unique(reps, return_inverse=True)
        return raw_to_unique.astype(np.intp, copy=False), unique_raw.astype(np.intp, copy=False)

    def _boundary_vertex_mask(self, original_vertex_count: int) -> np.ndarray:
        if self._boundary_mask_cache is not None and self._boundary_mask_cache.size == original_vertex_count:
            return self._boundary_mask_cache.copy()

        mask = np.zeros(original_vertex_count, dtype=bool)
        # Imported edge blocks may also contain material interfaces. Triangle
        # edges used only once form the exterior boundary of the integration mesh.
        elems = np.asarray(self.elements, dtype=np.intp)
        edges = np.vstack((elems[:, [0, 1]], elems[:, [1, 2]], elems[:, [2, 0]]))
        edges.sort(axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        if boundary_edges.size:
            mask[np.unique(boundary_edges.reshape(-1))] = True
        self._boundary_mask_cache = mask.copy()
        return mask

    @classmethod
    def _merge_by_coordinate_hash(
        cls,
        parent: np.ndarray,
        vertices: np.ndarray,
        candidates: np.ndarray,
        threshold: float,
    ) -> None:
        inv_tol = 1.0 / threshold
        buckets: dict[tuple[int, ...], list[int]] = {}
        dim = vertices.shape[1]
        neighbor_offsets = list(product((-1, 0, 1), repeat=dim))
        for raw_idx in candidates:
            raw_idx = int(raw_idx)
            key_arr = np.rint(vertices[raw_idx] * inv_tol).astype(np.int64)
            key = tuple(int(x) for x in key_arr)
            for delta in neighbor_offsets:
                near_key = tuple(key[axis] + int(delta[axis]) for axis in range(dim))
                for other_idx in buckets.get(near_key, ()):
                    if np.linalg.norm(vertices[raw_idx] - vertices[other_idx]) < threshold:
                        cls._union_min(parent, raw_idx, int(other_idx))
            buckets.setdefault(key, []).append(raw_idx)

    @staticmethod
    def _find(parent: np.ndarray, idx: int) -> int:
        root = idx
        while parent[root] != root:
            root = int(parent[root])
        while parent[idx] != idx:
            nxt = int(parent[idx])
            parent[idx] = root
            idx = nxt
        return root

    @classmethod
    def _union_min(cls, parent: np.ndarray, a: int, b: int) -> None:
        root_a = cls._find(parent, a)
        root_b = cls._find(parent, b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b

    def __deepcopy__(self, memo=None):
        return Mesh(copy.deepcopy(self.vertices, memo), copy.deepcopy(self.elements, memo), copy.deepcopy(self.edge, memo))


class PeriodicGrid:
    """Uniform periodic sampling grid in one crystallographic cell.

    The lattice vectors are stored as rows. ``sample_offset=0`` gives the
    half-open MPB grid u_i = i / N - 1/2.  The class exposes 2D plot
    triangles for existing output code, but numerical integration never uses
    those triangles.
    """

    integration_family = "uniform_grid"

    def __init__(
        self,
        shape,
        lattice_vectors,
        *,
        sample_offset=0.0,
    ) -> None:
        grid_shape = tuple(int(value) for value in shape)
        if not grid_shape or len(grid_shape) > 3 or any(value <= 0 for value in grid_shape):
            raise ValueError("Periodic grid shape must contain one to three positive integers.")
        lattice = np.asarray(lattice_vectors, dtype=float)
        dimension = len(grid_shape)
        if lattice.shape != (dimension, dimension) or not np.all(np.isfinite(lattice)):
            raise ValueError(
                f"Periodic grid lattice must have finite shape {(dimension, dimension)}."
            )
        determinant = float(np.linalg.det(lattice))
        if not np.isfinite(determinant) or abs(determinant) <= np.finfo(float).tiny:
            raise ValueError("Periodic grid lattice must have positive non-zero volume.")
        offset = np.broadcast_to(np.asarray(sample_offset, dtype=float), (dimension,)).copy()
        if not np.all(np.isfinite(offset)):
            raise ValueError("Periodic grid sample offsets must be finite.")

        self.base_shape = grid_shape
        self.shape = grid_shape
        self.lattice_vectors = lattice
        self.sample_offset = offset
        self.base_cell_volume = abs(determinant)
        self.cell_volume = self.base_cell_volume
        self.tile_shape = (1,) * dimension
        self.edge = None
        self._set_fractional_vertices(self._base_fractional_vertices())

    @property
    def dimension(self) -> int:
        return len(self.shape)

    @property
    def point_count(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def _base_fractional_vertices(self) -> np.ndarray:
        axes = [
            periodic_axis_coordinates(
                size,
                sample_offset=self.sample_offset[axis],
            )
            for axis, size in enumerate(self.base_shape)
        ]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, self.dimension)

    def _set_fractional_vertices(self, fractional: np.ndarray) -> None:
        self.fractional_vertices = np.asarray(fractional, dtype=float)
        self.vertices = self.fractional_vertices @ self.lattice_vectors
        self.elements = self._plot_elements()
        self.tri_weights = self._plot_triangle_weights()
        if self.point_count > 1:
            steps = [
                np.linalg.norm(self.lattice_vectors[axis]) / self.shape[axis]
                for axis in range(self.dimension)
            ]
            self.mindist = float(min(steps))
        else:
            self.mindist = 0.0

    def _plot_elements(self) -> np.ndarray:
        if self.dimension != 2:
            return np.empty((0, 3), dtype=np.intp)
        nx, ny = self.shape
        if nx < 2 or ny < 2:
            return np.empty((0, 3), dtype=np.intp)
        cells_i, cells_j = np.meshgrid(
            np.arange(nx - 1, dtype=np.intp),
            np.arange(ny - 1, dtype=np.intp),
            indexing="ij",
        )
        lower = (cells_i * ny + cells_j).reshape(-1)
        right = lower + ny
        upper = lower + 1
        diagonal = right + 1
        first = np.column_stack((lower, right, diagonal))
        second = np.column_stack((lower, diagonal, upper))
        return (
            np.stack((first, second), axis=1)
            .reshape(-1, 3)
            .astype(np.intp, copy=False)
        )

    def _plot_triangle_weights(self) -> np.ndarray:
        if self.elements.size == 0:
            return np.empty(0, dtype=float)
        triangles = self.vertices[self.elements]
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        determinant = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
        return np.abs(determinant) / 6.0

    def func(self, fn, offset=(0.0, 0.0)) -> np.ndarray:
        if self.dimension != 2:
            raise NotImplementedError("Projection functions currently support 2D grids.")
        dx = self.vertices[:, 0] - offset[0]
        dy = self.vertices[:, 1] - offset[1]
        return np.asarray(fn(dx, dy), dtype=np.complex128)

    def rfunc(self, fn, offset=(0.0, 0.0), ang=0.0) -> np.ndarray:
        if self.dimension != 2:
            raise NotImplementedError("Projection functions currently support 2D grids.")
        dx = self.vertices[:, 0] - offset[0]
        dy = self.vertices[:, 1] - offset[1]
        return np.asarray(
            fn(np.hypot(dx, dy), np.arctan2(dy, dx) + np.deg2rad(ang)),
            dtype=np.complex128,
        )

    def extension(
        self,
        n: list[int],
        real_lattice_vectors: list[list[float]],
        lattice_const: float,
    ) -> np.ndarray:
        extension = tuple(int(value) for value in n)
        if len(extension) != self.dimension or any(value <= 0 for value in extension):
            raise ValueError(
                f"extension must contain {self.dimension} positive integers."
            )
        expected_lattice = np.asarray(real_lattice_vectors, dtype=float) * float(lattice_const)
        if expected_lattice.shape != self.lattice_vectors.shape or not np.allclose(
            expected_lattice,
            self.lattice_vectors,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError("Periodic grid lattice does not match the calculation lattice.")

        base_shape = self.base_shape
        new_shape = tuple(base_shape[axis] * extension[axis] for axis in range(self.dimension))
        center_tiles = np.floor((np.asarray(extension, dtype=float) - 1.0) / 2.0)
        axes = []
        for axis, size in enumerate(new_shape):
            base_size = base_shape[axis]
            tile = np.arange(size, dtype=np.int64) // base_size
            local = np.arange(size, dtype=np.int64) % base_size
            base_axis = periodic_axis_coordinates(
                base_size,
                sample_offset=self.sample_offset[axis],
            )
            axes.append(
                tile
                - center_tiles[axis]
                + base_axis[local]
            )
        fractional = np.stack(
            np.meshgrid(*axes, indexing="ij"), axis=-1
        ).reshape(-1, self.dimension)
        global_indices = np.indices(new_shape, dtype=np.int64)
        mapping_multi = tuple(
            (global_indices[axis] % base_shape[axis]).reshape(-1)
            for axis in range(self.dimension)
        )
        mapping = np.ravel_multi_index(mapping_multi, base_shape).astype(np.intp)

        self.shape = new_shape
        self.tile_shape = extension
        self.cell_volume = self.base_cell_volume * int(np.prod(extension))
        self._set_fractional_vertices(fractional)
        return mapping

    def __deepcopy__(self, memo=None):
        result = PeriodicGrid(
            self.base_shape,
            copy.deepcopy(self.lattice_vectors, memo),
            sample_offset=copy.deepcopy(self.sample_offset, memo),
        )
        if self.tile_shape != (1,) * self.dimension:
            result.extension(
                list(self.tile_shape),
                (self.lattice_vectors / 1.0).tolist(),
                1.0,
            )
        return result


@dataclass
class RawData:
    point_matrix: np.ndarray
    value_matrix: np.ndarray
    column_parameters: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class BandChannelReference:
    channel: str
    source_band_index: int

    @property
    def label(self) -> str:
        return f"{self.channel}:{self.source_band_index}"


@dataclass(frozen=True)
class InputBundle:
    config: IncarConfig
    maxwell: MaxwellProblem
    bloch_convention: BlochConvention
    mesh: Mesh | PeriodicGrid
    fields: np.ndarray
    metric_material: np.ndarray
    energies: np.ndarray
    band_indices: np.ndarray
    inner_band_indices: np.ndarray
    energy_matrix: np.ndarray
    field_representation: BlochFieldRepresentation = BlochFieldRepresentation.FULL_BLOCH
    symmetry: SymmetryContext | None = None
    analysis_field_kind: FieldKind | None = None
    zero_modes: np.ndarray | None = None
    band_channels: dict[int, BandChannelReference] = field(default_factory=dict)
    auxiliary_zero_mode_bands: dict[str, np.ndarray] = field(default_factory=dict)
    auxiliary_bundle_loaders: dict[str, Callable[[], "InputBundle"]] = field(
        default_factory=dict
    )
    base_hamiltonians: np.ndarray | None = None


@dataclass
class BandResult:
    k_path: np.ndarray
    k_axis: np.ndarray
    high_sym_points: list[list[Any]]
    energies: np.ndarray
    dos_energy: np.ndarray | None = None
    dos_components: np.ndarray | None = None
    bz_eigvals: np.ndarray | None = None
    bz_eigvecs: np.ndarray | None = None
    groups: list[list[int]] = field(default_factory=list)


@dataclass
class TopologyResult:
    wilson: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int | None]] = field(default_factory=dict)
    chern: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)
    chern_bands: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class DegeneracySplittingDiagnostic:
    point_name: str
    band_indices: tuple[int, ...]
    reference_gap: float
    output_gap: float
    tolerance: float

    @property
    def broken(self) -> bool:
        return self.reference_gap <= self.tolerance and self.output_gap > self.tolerance


@dataclass(frozen=True)
class OutputSpectrumDiagnostics:
    basis: str
    max_eigenvalue_drift: float
    worst_k_index: tuple[int, int, int]
    degeneracy_splittings: tuple[DegeneracySplittingDiagnostic, ...] = ()


@dataclass(frozen=True)
class HoppingReconstructionDiagnostics:
    max_matrix_error: float
    max_eigenvalue_error: float
    worst_k_index: tuple[int, int, int]
    degeneracy_splittings: tuple[DegeneracySplittingDiagnostic, ...] = ()


@dataclass
class RunResult:
    config: IncarConfig
    mesh: Mesh | PeriodicGrid
    extended_mesh: Mesh | PeriodicGrid
    extended_metric_material: np.ndarray
    orthogonality_report: np.ndarray
    S: np.ndarray | None
    M0: np.ndarray
    A: np.ndarray
    V: np.ndarray
    U: np.ndarray
    omega: np.ndarray
    rn: np.ndarray
    wanniers: dict[tuple[int, ...], np.ndarray]
    wannier_norms: np.ndarray
    hoppings: dict[tuple[int, int, int], np.ndarray]
    band: BandResult | None
    topology: TopologyResult | None
    bloch_gauge: np.ndarray | None = None
    symmetry: SymmetryContext | None = None
    symmetry_analysis: SymmetryAnalysisResult | None = None
    symmetry_gauge: SymmetryGaugeResult | None = None
    symmetry_localization: SymmetryLocalizationResult | None = None
    symmetry_disentanglement: SymmetryDisentanglementResult | None = None
    output_spectrum_diagnostics: OutputSpectrumDiagnostics | None = None
    hopping_reconstruction_diagnostics: HoppingReconstructionDiagnostics | None = None
    sewing_matrices: tuple[SewingMatrixCacheEntry, ...] | None = None
    trial_covariance_diagnostics: tuple[Any, ...] = ()
    etbc: ETBCCompletionResult | None = None
    transverse_projectors: np.ndarray | None = None


@dataclass
class BlochSymmetryChannelResult:
    field_kind: FieldKind
    orthogonality_report: np.ndarray
    S: np.ndarray
    analysis: BlochSymmetryAnalysisResult
    sewing_matrices: tuple[SewingMatrixCacheEntry, ...]
    differential_diagnostics: "VectorFieldDifferentialDiagnostics | None" = None


@dataclass
class BlochSymmetryRunResult:
    config: IncarConfig
    symmetry: SymmetryContext
    primary: BlochSymmetryChannelResult
    auxiliary_channels: dict[str, BlochSymmetryChannelResult] = field(default_factory=dict)
    gamma_zero_regularization: "GammaZeroRegularizationAnalysis | None" = None
    band_channels: dict[int, BandChannelReference] = field(default_factory=dict)
