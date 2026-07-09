from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

from .config import IncarConfig


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

        threshold = max(float(self.mindist) * 0.5, 1e-12)
        if candidates.size > 1:
            self._merge_by_coordinate_hash(parent, vertices, candidates, threshold)

        reps = np.fromiter((self._find(parent, idx) for idx in range(raw_count)), dtype=np.intp, count=raw_count)
        unique_raw, raw_to_unique = np.unique(reps, return_inverse=True)
        return raw_to_unique.astype(np.intp, copy=False), unique_raw.astype(np.intp, copy=False)

    def _boundary_vertex_mask(self, original_vertex_count: int) -> np.ndarray:
        if self._boundary_mask_cache is not None and self._boundary_mask_cache.size == original_vertex_count:
            return self._boundary_mask_cache.copy()

        mask = np.zeros(original_vertex_count, dtype=bool)
        if self.edge is not None and self.edge.size:
            mask[np.unique(np.asarray(self.edge, dtype=np.intp).reshape(-1))] = True
            self._boundary_mask_cache = mask.copy()
            return mask

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

    def match(self, new_vertices: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tree = cKDTree(new_vertices)
        dists, idxs = tree.query(vertices, k=1)
        idx_existing = np.where(dists < self.mindist * 0.5)[0]
        idx_new = [idxs[i] for i in idx_existing]
        return np.asarray(idx_new, dtype=np.intp), np.asarray(idx_existing, dtype=np.intp)

    def rebuild_index(self, space_to_original_mapping=None) -> tuple[dict[int, int], np.ndarray]:
        used_indices = sorted(set(int(x) for x in self.elements.flatten()))
        new_vertices = np.asarray([self.vertices[i] for i in used_indices])
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
        if space_to_original_mapping is None:
            new_mapping = np.arange(len(self.vertices), dtype=np.intp)[used_indices]
        else:
            mapping = np.asarray(space_to_original_mapping, dtype=np.intp)
            new_mapping = mapping[used_indices]

        self.elements = np.asarray([[old_to_new[int(idx)] for idx in element] for element in self.elements], dtype=np.intp)
        self.vertices = new_vertices
        return old_to_new, np.asarray(new_mapping, dtype=np.intp)

    def __deepcopy__(self, memo=None):
        return Mesh(copy.deepcopy(self.vertices, memo), copy.deepcopy(self.elements, memo), copy.deepcopy(self.edge, memo))


@dataclass
class RawData:
    point_matrix: np.ndarray
    value_matrix: np.ndarray


@dataclass
class FieldData:
    name: str
    mesh: Mesh
    field: np.ndarray


@dataclass
class InputBundle:
    config: IncarConfig
    mesh: Mesh
    fields: np.ndarray
    epsilon: np.ndarray
    energies: np.ndarray
    band_indices: np.ndarray
    inner_band_indices: np.ndarray
    energy_matrix: np.ndarray


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
    wilson: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = field(default_factory=dict)
    chern: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)


@dataclass
class RunResult:
    config: IncarConfig
    mesh: Mesh
    extended_mesh: Mesh
    extended_epsilon: np.ndarray
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
