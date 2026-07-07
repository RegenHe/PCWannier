from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy

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
        space_to_original_mapping: np.ndarray = np.arange(len(original_vertices), dtype=np.intp)

        for i in range(n[0]):
            for j in range(n[1]):
                if i == 0 and j == 0:
                    continue
                offset_x = (real_lattice_vectors[0][0] * i + real_lattice_vectors[1][0] * j) * lattice_const
                offset_y = (real_lattice_vectors[0][1] * i + real_lattice_vectors[1][1] * j) * lattice_const
                new_elements = original_elements + int(np.max(self.elements)) + 1
                new_vertices = original_vertices + np.array([offset_x, offset_y])

                idx_new, idx_existing = self.match(new_vertices, self.vertices)
                for new_idx, old_idx in zip(idx_new, idx_existing):
                    new_elements[new_elements == (new_idx + int(np.max(self.elements)) + 1)] = old_idx

                self.elements = np.vstack((self.elements, new_elements))
                self.vertices = np.vstack((self.vertices, new_vertices))
                space_to_original_mapping = np.hstack(
                    (space_to_original_mapping, np.arange(len(original_vertices), dtype=np.intp))
                )
                _, space_to_original_mapping = self.rebuild_index(space_to_original_mapping)

        offset_x = (
            real_lattice_vectors[0][0] * np.floor((n[0] - 1) / 2)
            + real_lattice_vectors[1][0] * np.floor((n[1] - 1) / 2)
        ) * lattice_const
        offset_y = (
            real_lattice_vectors[0][1] * np.floor((n[0] - 1) / 2)
            + real_lattice_vectors[1][1] * np.floor((n[1] - 1) / 2)
        ) * lattice_const
        self.vertices = self.vertices - np.array([offset_x, offset_y])
        self._precompute_tri_weights()
        return np.asarray(space_to_original_mapping, dtype=np.intp)

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
