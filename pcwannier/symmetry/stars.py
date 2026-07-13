from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .group import SymmetryKMapping
from .representation import SymmetryContext


@dataclass(frozen=True)
class SymmetryStarMember:
    k_index: tuple[int, ...]
    flat_index: int
    paths: tuple[SymmetryKMapping, ...]

    @property
    def canonical_path(self) -> SymmetryKMapping:
        if not self.paths:
            raise RuntimeError("A symmetry-star member has no path from its representative.")
        return self.paths[0]


@dataclass(frozen=True)
class SymmetryKStar:
    index: int
    representative_index: tuple[int, ...]
    representative_flat_index: int
    members: tuple[SymmetryStarMember, ...]


@dataclass(frozen=True)
class SymmetryStarPartition:
    k_shape: tuple[int, ...]
    stars: tuple[SymmetryKStar, ...]
    k_to_star: np.ndarray

    def star_for(self, k_index) -> SymmetryKStar:
        index = tuple(int(value) for value in k_index)
        return self.stars[int(self.k_to_star[index])]


def build_symmetry_stars(context: SymmetryContext) -> SymmetryStarPartition:
    """Partition a closed uniform k mesh into deterministic point-group orbits."""
    shape = tuple(len(axis) for axis in context.k_points)
    count = int(np.prod(shape))
    all_indices = tuple(tuple(int(value) for value in index) for index in np.ndindex(shape))
    unassigned = set(range(count))
    stars: list[SymmetryKStar] = []
    k_to_star = np.full(shape, -1, dtype=np.intp)

    while unassigned:
        representative_flat = min(unassigned)
        representative = all_indices[representative_flat]
        paths_by_target: dict[int, list[SymmetryKMapping]] = {}
        for operation_mappings in context.k_mappings:
            mapping = operation_mappings[representative_flat]
            if mapping.source_k_index != representative:
                raise RuntimeError("Symmetry k mappings are not stored in flat-index order.")
            target_flat = int(np.ravel_multi_index(mapping.target_k_index, shape))
            paths_by_target.setdefault(target_flat, []).append(mapping)

        member_flats = set(paths_by_target)
        if not member_flats <= unassigned:
            overlap = sorted(member_flats - unassigned)
            raise RuntimeError(f"Symmetry stars overlap at flat k indices {overlap[:8]}.")

        # A star must remain closed when any operation acts on any member.
        for member_flat in member_flats:
            for operation_mappings in context.k_mappings:
                target = operation_mappings[member_flat].target_k_index
                target_flat = int(np.ravel_multi_index(target, shape))
                if target_flat not in member_flats:
                    raise RuntimeError(
                        f"Symmetry star represented by {representative} is not group-closed."
                    )

        members = []
        for flat_index in sorted(member_flats):
            paths = tuple(sorted(paths_by_target[flat_index], key=lambda item: item.operation_index))
            members.append(SymmetryStarMember(all_indices[flat_index], flat_index, paths))
        star_index = len(stars)
        star = SymmetryKStar(star_index, representative, representative_flat, tuple(members))
        stars.append(star)
        for member in members:
            k_to_star[member.k_index] = star_index
        unassigned.difference_update(member_flats)

    if np.any(k_to_star < 0):
        raise RuntimeError("Symmetry-star construction did not cover the complete k mesh.")
    k_to_star.setflags(write=False)
    return SymmetryStarPartition(shape, tuple(stars), k_to_star)

