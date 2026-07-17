from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


MIN_FRACTIONAL_TOLERANCE = 64.0 * np.finfo(float).eps


@dataclass(frozen=True)
class PeriodicImage:
    reduced: np.ndarray
    lattice_shift: np.ndarray


@dataclass(frozen=True)
class SeitzProduct:
    """A product reduced to the stored Seitz representative set.

    ``g_left g_right = t_lattice_shift g_result`` uses a lattice
    translation on the left. ``result_index`` is an index into the owning
    :class:`SpaceGroup` operation list.
    """

    result_index: int
    lattice_shift: tuple[int, ...]


def reduce_fractional(vector, tolerance: float = 1e-8) -> PeriodicImage:
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Fractional coordinates must be a finite one-dimensional vector.")
    shift = np.floor(values).astype(np.int64)
    reduced = values - shift
    near_zero = np.abs(reduced) <= tolerance
    near_one = np.abs(reduced - 1.0) <= tolerance
    reduced[near_zero] = 0.0
    reduced[near_one] = 0.0
    shift[near_one] += 1
    if np.any(reduced < -tolerance) or np.any(reduced >= 1.0 + tolerance):
        raise FloatingPointError("Periodic coordinate reduction failed.")
    return PeriodicImage(reduced, shift)


def periodic_difference(left, right) -> np.ndarray:
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return (difference + 0.5) % 1.0 - 0.5


def periodic_equivalent(left, right, tolerance: float = 1e-8) -> bool:
    return bool(np.max(np.abs(periodic_difference(left, right)), initial=0.0) <= tolerance)


def _integer_shift(values, tolerance: float, *, description: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    rounded = np.rint(array).astype(np.int64)
    residual = float(np.max(np.abs(array - rounded), initial=0.0))
    if residual > tolerance:
        raise ValueError(f"{description} is not a lattice vector (residual={residual:.6g}).")
    return rounded


@dataclass(frozen=True, eq=False)
class SpaceGroupOperation:
    rotation: np.ndarray
    translation: np.ndarray
    name: str | None = None

    def __post_init__(self) -> None:
        raw_rotation = np.asarray(self.rotation)
        if raw_rotation.ndim != 2 or raw_rotation.shape[0] != raw_rotation.shape[1]:
            raise ValueError("Space-group rotation must be a square matrix.")
        rotation = np.rint(raw_rotation).astype(np.int64)
        if not np.allclose(raw_rotation, rotation, rtol=0.0, atol=1e-12):
            raise ValueError("Space-group rotation must contain integers.")
        determinant = int(round(float(np.linalg.det(rotation))))
        if abs(determinant) != 1:
            raise ValueError(f"Space-group rotation must be unimodular; det(R)={determinant}.")
        translation = np.asarray(self.translation, dtype=float)
        if translation.shape != (rotation.shape[0],) or not np.all(np.isfinite(translation)):
            raise ValueError(f"Space-group translation must have shape {(rotation.shape[0],)} and be finite.")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @property
    def dimension(self) -> int:
        return int(self.rotation.shape[0])

    @property
    def reciprocal_rotation(self) -> np.ndarray:
        reciprocal = np.rint(np.linalg.inv(self.rotation).T).astype(np.int64)
        return reciprocal

    @classmethod
    def identity(cls, dimension: int, name: str | None = "E") -> SpaceGroupOperation:
        return cls(np.eye(dimension, dtype=np.int64), np.zeros(dimension), name)

    @classmethod
    def lattice_translation(cls, shift, name: str | None = None) -> SpaceGroupOperation:
        shift = np.asarray(shift, dtype=float)
        return cls(np.eye(shift.size, dtype=np.int64), shift, name)

    def __mul__(self, other: SpaceGroupOperation) -> SpaceGroupOperation:
        if not isinstance(other, SpaceGroupOperation) or self.dimension != other.dimension:
            return NotImplemented
        rotation = self.rotation @ other.rotation
        translation = self.rotation @ other.translation + self.translation
        return SpaceGroupOperation(rotation, translation)

    def inverse(self) -> SpaceGroupOperation:
        rotation = np.rint(np.linalg.inv(self.rotation)).astype(np.int64)
        return SpaceGroupOperation(rotation, -(rotation @ self.translation))

    def act_real(self, fractional_position) -> np.ndarray:
        position = np.asarray(fractional_position, dtype=float)
        if position.shape != (self.dimension,):
            raise ValueError(f"Real-space point must have shape {(self.dimension,)}.")
        return self.rotation @ position + self.translation

    def act_reciprocal(self, fractional_k) -> np.ndarray:
        kpoint = np.asarray(fractional_k, dtype=float)
        if kpoint.shape != (self.dimension,):
            raise ValueError(f"Reciprocal-space point must have shape {(self.dimension,)}.")
        return self.reciprocal_rotation @ kpoint

    def act_real_reduced(self, fractional_position, tolerance: float = 1e-8) -> PeriodicImage:
        return reduce_fractional(self.act_real(fractional_position), tolerance)

    def act_reciprocal_reduced(self, fractional_k, tolerance: float = 1e-8) -> PeriodicImage:
        return reduce_fractional(self.act_reciprocal(fractional_k), tolerance)

    def strictly_equal(self, other: SpaceGroupOperation, tolerance: float = 1e-8) -> bool:
        return bool(
            isinstance(other, SpaceGroupOperation)
            and np.array_equal(self.rotation, other.rotation)
            and np.allclose(self.translation, other.translation, rtol=0.0, atol=tolerance)
        )

    def equivalent_mod_lattice(self, other: SpaceGroupOperation, tolerance: float = 1e-8) -> bool:
        if not isinstance(other, SpaceGroupOperation) or not np.array_equal(self.rotation, other.rotation):
            return False
        difference = self.translation - other.translation
        return bool(np.allclose(difference, np.rint(difference), rtol=0.0, atol=tolerance))


@dataclass(frozen=True)
class SpaceGroup:
    operations: tuple[SpaceGroupOperation, ...]
    tolerance: float = 1e-8
    identity_index: int = 0

    def __init__(self, operations: Iterable[SpaceGroupOperation], tolerance: float = 1e-8):
        items = tuple(operations)
        if not items:
            raise ValueError("A space group must contain at least one operation.")
        if (
            not np.isfinite(tolerance)
            or tolerance < MIN_FRACTIONAL_TOLERANCE
            or tolerance >= 0.25
        ):
            raise ValueError(
                "Symmetry tolerance must be finite, at least "
                f"{MIN_FRACTIONAL_TOLERANCE:.6g}, and smaller than 0.25."
            )
        dimension = items[0].dimension
        if any(operation.dimension != dimension for operation in items):
            raise ValueError("All space-group operations must have the same dimension.")
        names = [operation.name for operation in items if operation.name is not None]
        if len(names) != len(set(names)):
            raise ValueError("Space-group operation names must be unique.")
        for left_index, left in enumerate(items):
            for right_index in range(left_index):
                if left.equivalent_mod_lattice(items[right_index], tolerance):
                    raise ValueError(
                        f"Space-group operations {right_index} and {left_index} are equivalent modulo a lattice translation."
                    )
        identity = SpaceGroupOperation.identity(dimension)
        identity_matches = [idx for idx, operation in enumerate(items) if operation.equivalent_mod_lattice(identity, tolerance)]
        if len(identity_matches) != 1:
            raise ValueError("The space-group operation list must contain exactly one identity modulo lattice translations.")
        object.__setattr__(self, "operations", items)
        object.__setattr__(self, "tolerance", float(tolerance))
        object.__setattr__(self, "identity_index", identity_matches[0])
        self._validate_group_laws()

    @property
    def dimension(self) -> int:
        return self.operations[0].dimension

    def operation_index(self, operation: SpaceGroupOperation, *, modulo_lattice: bool = True) -> int:
        for index, candidate in enumerate(self.operations):
            matches = (
                candidate.equivalent_mod_lattice(operation, self.tolerance)
                if modulo_lattice
                else candidate.strictly_equal(operation, self.tolerance)
            )
            if matches:
                return index
        raise ValueError("Operation is not an element of the supplied space group.")

    def operation_by_name(self, name: str) -> SpaceGroupOperation:
        for operation in self.operations:
            if operation.name == name:
                return operation
        raise KeyError(f"Unknown space-group operation name: {name!r}.")

    def multiply_mod_lattice(self, left_index: int, right_index: int) -> SeitzProduct:
        """Multiply representatives and expose the discarded lattice shift."""

        left = int(left_index)
        right = int(right_index)
        if not 0 <= left < len(self.operations) or not 0 <= right < len(self.operations):
            raise IndexError("Space-group operation index is out of range.")
        exact_product = self.operations[left] * self.operations[right]
        result_index = self.operation_index(exact_product)
        representative = self.operations[result_index]
        lattice_shift = _integer_shift(
            exact_product.translation - representative.translation,
            self.tolerance,
            description="Seitz representative product translation",
        )
        return SeitzProduct(
            result_index,
            tuple(int(value) for value in lattice_shift),
        )

    def _validate_group_laws(self) -> None:
        for operation in self.operations:
            self.operation_index(operation.inverse())
        for left in range(len(self.operations)):
            for right in range(len(self.operations)):
                self.multiply_mod_lattice(left, right)


@dataclass(frozen=True)
class SymmetryKMapping:
    operation_index: int
    source_k_index: tuple[int, ...]
    target_k_index: tuple[int, ...]
    reciprocal_lattice_shift: tuple[int, ...]


def build_k_mappings(
    group: SpaceGroup,
    k_points: Iterable[np.ndarray],
) -> tuple[tuple[SymmetryKMapping, ...], ...]:
    axes = tuple(np.asarray(axis, dtype=float) for axis in k_points)
    if len(axes) != group.dimension or any(axis.ndim != 1 or axis.size == 0 for axis in axes):
        raise ValueError(f"k_points must contain {group.dimension} non-empty one-dimensional axes.")
    shape = tuple(int(axis.size) for axis in axes)
    multi_indices = tuple(np.ndindex(shape))
    raw_points = np.asarray([[axes[axis][idx[axis]] for axis in range(group.dimension)] for idx in multi_indices])
    canonical = np.asarray([reduce_fractional(point, group.tolerance).reduced for point in raw_points])
    tree = cKDTree(canonical, boxsize=1.0)
    duplicate_pairs = tree.query_pairs(r=group.tolerance, p=np.inf)
    if duplicate_pairs:
        first = min(duplicate_pairs)
        raise ValueError(f"k mesh contains periodically duplicate samples at flat indices {first}.")

    all_mappings = []
    for operation_index, operation in enumerate(group.operations):
        operation_mappings = []
        for flat_index, (source_index, source_k) in enumerate(zip(multi_indices, raw_points)):
            transformed = operation.act_reciprocal(source_k)
            transformed_reduced = reduce_fractional(transformed, group.tolerance).reduced
            neighbor_count = min(2, len(canonical))
            distances, targets = tree.query(
                transformed_reduced, k=neighbor_count, p=np.inf
            )
            distances = np.atleast_1d(distances)
            targets = np.atleast_1d(targets)
            distance = float(distances[0])
            target_flat = int(targets[0])
            if not np.isfinite(distance) or distance > group.tolerance:
                raise ValueError(
                    f"k mesh is not closed under operation {operation.name or operation_index}: "
                    f"source index {source_index} maps to {transformed.tolist()}."
                )
            if neighbor_count > 1 and distances[1] <= group.tolerance:
                raise ValueError(
                    f"Ambiguous symmetry k mapping under operation "
                    f"{operation.name or operation_index}: source index {source_index} is within "
                    f"tolerance of flat targets {target_flat} and {int(targets[1])}."
                )
            target_k = raw_points[target_flat]
            reciprocal_shift = _integer_shift(
                transformed - target_k,
                group.tolerance,
                description="Reciprocal-space symmetry shift",
            )
            operation_mappings.append(
                SymmetryKMapping(
                    operation_index,
                    tuple(int(value) for value in source_index),
                    tuple(int(value) for value in multi_indices[target_flat]),
                    tuple(int(value) for value in reciprocal_shift),
                )
            )
        all_mappings.append(tuple(operation_mappings))
    return tuple(all_mappings)


@dataclass(frozen=True)
class SiteSymmetryElement:
    source_operation_index: int
    lattice_shift: tuple[int, ...]
    operation: SpaceGroupOperation
    strictly_fixed: bool


@dataclass(frozen=True)
class SiteSymmetryGroup:
    center: np.ndarray
    strict_elements: tuple[SiteSymmetryElement, ...]
    elements: tuple[SiteSymmetryElement, ...]
    tolerance: float

    def element_index(self, operation: SpaceGroupOperation) -> int:
        for index, element in enumerate(self.elements):
            if element.operation.strictly_equal(operation, self.tolerance):
                return index
        raise ValueError("Operation is not an element of the site-symmetry group.")


@dataclass(frozen=True)
class OrbitPoint:
    index: int
    position: np.ndarray
    representative_operation_index: int
    representative_operation: SpaceGroupOperation


@dataclass(frozen=True)
class OrbitAction:
    operation_index: int
    source_index: int
    target_index: int
    lattice_shift: tuple[int, ...]
    site_element_index: int
    site_element: SpaceGroupOperation


@dataclass(frozen=True)
class CrystallographicOrbit:
    center: np.ndarray
    points: tuple[OrbitPoint, ...]
    site_symmetry: SiteSymmetryGroup
    actions: tuple[tuple[OrbitAction, ...], ...]

    @property
    def multiplicity(self) -> int:
        return len(self.points)

    def action(self, operation_index: int, source_index: int) -> OrbitAction:
        return self.actions[operation_index][source_index]


def build_crystallographic_orbit(group: SpaceGroup, center) -> CrystallographicOrbit:
    center = np.asarray(center, dtype=float)
    if center.shape != (group.dimension,) or not np.all(np.isfinite(center)):
        raise ValueError(f"Wannier center must have shape {(group.dimension,)} and be finite.")
    reduced_center = reduce_fractional(center, group.tolerance)
    if np.any(reduced_center.lattice_shift != 0) or not np.allclose(
        reduced_center.reduced, center, rtol=0.0, atol=group.tolerance
    ):
        raise ValueError("Wannier target center must lie in the canonical unit cell [0, 1).")

    site_elements = []
    strict_elements = []
    for operation_index, operation in enumerate(group.operations):
        displacement = operation.act_real(center) - center
        rounded = np.rint(displacement).astype(np.int64)
        if not np.allclose(displacement, rounded, rtol=0.0, atol=group.tolerance):
            continue
        corrected = SpaceGroupOperation.lattice_translation(-rounded) * operation
        if not np.allclose(corrected.act_real(center), center, rtol=0.0, atol=group.tolerance):
            raise FloatingPointError("Failed to construct a strict site-symmetry operation.")
        entry = SiteSymmetryElement(
            operation_index,
            tuple(int(value) for value in rounded),
            corrected,
            bool(np.all(rounded == 0)),
        )
        site_elements.append(entry)
        if entry.strictly_fixed:
            strict_elements.append(entry)
    site_group = SiteSymmetryGroup(center.copy(), tuple(strict_elements), tuple(site_elements), group.tolerance)

    order = (group.identity_index,) + tuple(idx for idx in range(len(group.operations)) if idx != group.identity_index)
    orbit_points: list[OrbitPoint] = []
    for operation_index in order:
        operation = group.operations[operation_index]
        periodic_image = operation.act_real_reduced(center, group.tolerance)
        image = periodic_image.reduced
        if any(periodic_equivalent(image, point.position, group.tolerance) for point in orbit_points):
            continue
        representative = (
            SpaceGroupOperation.lattice_translation(-periodic_image.lattice_shift) * operation
        )
        if not np.allclose(
            representative.act_real(center), image, rtol=0.0, atol=group.tolerance
        ):
            raise FloatingPointError("Failed to construct an exact orbit representative.")
        image.setflags(write=False)
        orbit_points.append(
            OrbitPoint(len(orbit_points), image, operation_index, representative)
        )
    if len(group.operations) != len(orbit_points) * len(site_elements):
        raise ValueError("Orbit-stabilizer relation failed for the supplied operation list and target center.")

    all_actions = []
    for operation_index, operation in enumerate(group.operations):
        operation_actions = []
        for source in orbit_points:
            raw_target = operation.act_real(source.position)
            target_index = next(
                (
                    point.index
                    for point in orbit_points
                    if periodic_equivalent(raw_target, point.position, group.tolerance)
                ),
                None,
            )
            if target_index is None:
                raise ValueError("Space-group operation maps an orbit point outside the constructed orbit.")
            target = orbit_points[target_index]
            lattice_shift = _integer_shift(
                raw_target - target.position,
                group.tolerance,
                description="Orbit lattice shift",
            )
            g_source = source.representative_operation
            g_target = target.representative_operation
            translation = SpaceGroupOperation.lattice_translation(-lattice_shift)
            site_operation = g_target.inverse() * translation * operation * g_source
            site_index = site_group.element_index(site_operation)
            operation_actions.append(
                OrbitAction(
                    operation_index,
                    source.index,
                    target.index,
                    tuple(int(value) for value in lattice_shift),
                    site_index,
                    site_operation,
                )
            )
        all_actions.append(tuple(operation_actions))
    return CrystallographicOrbit(center.copy(), tuple(orbit_points), site_group, tuple(all_actions))
