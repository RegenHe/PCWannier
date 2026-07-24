from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np
import spglib
import spgrep
from spgrep.symmetry.enumerate import enumerate_unitary_irreps

if TYPE_CHECKING:
    from .definition import FactorSystem
    from .tables import ConcreteFiniteGroup, FiniteGroupTable


@dataclass(frozen=True)
class CrystallographicEmbedding:
    """Embed fractional 2D crystallographic data in spglib's 3D convention."""

    dimension: int

    def __post_init__(self) -> None:
        if self.dimension not in {2, 3}:
            raise ValueError("Crystallographic symmetry currently supports dimension 2 or 3.")

    @property
    def ambient_dimension(self) -> int:
        return 3

    def rotation(self, value) -> np.ndarray:
        raw = np.asarray(value)
        expected = (self.dimension, self.dimension)
        if raw.shape != expected or not np.all(np.isfinite(raw)):
            raise ValueError(f"Fractional rotation must be a finite matrix with shape {expected}.")
        rounded = np.rint(raw).astype(np.int64)
        if not np.array_equal(raw, rounded):
            raise ValueError("Fractional crystallographic rotations must contain integers.")
        if abs(round(float(np.linalg.det(rounded)))) != 1:
            raise ValueError("Fractional crystallographic rotations must be unimodular.")
        if self.dimension == 3:
            output = rounded.copy()
        else:
            output = np.eye(3, dtype=np.int64)
            output[:2, :2] = rounded
        output.setflags(write=False)
        return output

    def vector(self, value) -> np.ndarray:
        raw = np.asarray(value, dtype=float)
        if raw.shape != (self.dimension,) or not np.all(np.isfinite(raw)):
            raise ValueError(
                f"Fractional crystallographic vector must have shape {(self.dimension,)}."
            )
        output = np.zeros(3, dtype=float)
        output[: self.dimension] = raw
        output.setflags(write=False)
        return output

    def rotations(self, values: Iterable[np.ndarray]) -> np.ndarray:
        output = np.asarray([self.rotation(value) for value in values], dtype=np.int64)
        if output.ndim != 3 or output.shape[1:] != (3, 3):
            raise ValueError("At least one crystallographic rotation is required.")
        return output


@dataclass(frozen=True)
class PointGroupIdentification:
    symbol: str
    number: int
    transformation_matrix: np.ndarray

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip()
        if not symbol:
            raise ValueError("spglib returned an empty point-group symbol.")
        number = int(self.number)
        if number <= 0:
            raise ValueError("spglib returned an invalid point-group number.")
        transform = np.asarray(self.transformation_matrix, dtype=np.int64)
        if transform.shape != (3, 3):
            raise ValueError("spglib point-group transformation must have shape (3, 3).")
        transform = transform.copy()
        transform.setflags(write=False)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "number", number)
        object.__setattr__(self, "transformation_matrix", transform)


def identify_point_group(
    rotations: Iterable[np.ndarray],
    dimension: int,
) -> PointGroupIdentification:
    embedded = CrystallographicEmbedding(dimension).rotations(rotations)
    result = spglib.get_pointgroup(embedded)
    if result is None or len(result) != 3:
        raise ValueError("spglib could not identify the crystallographic point group.")
    symbol, number, transformation = result
    return PointGroupIdentification(symbol, number, transformation)


def enumerate_projective_representations(
    rotations: Iterable[np.ndarray],
    factor_phases,
    product_table,
    *,
    dimension: int,
    tolerance: float,
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Enumerate irreps for the supplied, already convention-adjusted factor system."""

    threshold = max(float(tolerance), 1.0e-10)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("Representation tolerance must be finite and positive.")
    embedded = CrystallographicEmbedding(dimension).rotations(rotations)
    phases = np.asarray(factor_phases, dtype=np.complex128)
    product = np.asarray(product_table, dtype=np.int64)
    order = embedded.shape[0]
    if phases.shape != (order, order) or product.shape != (order, order):
        raise ValueError("Factor phases and multiplication table must match the group order.")

    raw_irreps, _ = enumerate_unitary_irreps(
        embedded,
        factor_system=phases,
        real=False,
        method="Neto",
        rtol=threshold,
        atol=threshold,
    )
    output = []
    for raw in raw_irreps:
        matrices = np.asarray(raw, dtype=np.complex128)
        if matrices.ndim != 3 or matrices.shape[0] != order:
            raise ValueError("spgrep returned an irrep with an invalid shape.")
        irrep_dimension = matrices.shape[1]
        if matrices.shape[2] != irrep_dimension or irrep_dimension <= 0:
            raise ValueError("spgrep returned a non-square or empty irrep.")
        if not np.all(np.isfinite(matrices)):
            raise ValueError("spgrep returned a non-finite irrep.")
        identity = np.eye(irrep_dimension)
        unitary_residual = max(
            float(np.linalg.norm(matrix.conj().T @ matrix - identity, ord="fro"))
            for matrix in matrices
        )
        product_residual = max(
            float(
                np.linalg.norm(
                    matrices[left] @ matrices[right]
                    - phases[left, right] * matrices[int(product[left, right])],
                    ord="fro",
                )
            )
            for left in range(order)
            for right in range(order)
        )
        allowed = max(100.0 * threshold, 1.0e-8)
        if unitary_residual > allowed or product_residual > allowed:
            raise ValueError(
                "spgrep returned an invalid projective representation: "
                f"unitarity={unitary_residual:.6g}, product={product_residual:.6g}."
            )
        stored = []
        for matrix in matrices:
            value = matrix.copy()
            value.setflags(write=False)
            stored.append(value)
        output.append(tuple(stored))

    if sum(values[0].shape[0] ** 2 for values in output) != order:
        raise ValueError("spgrep irreps do not satisfy sum(dim^2) = group order.")
    output.sort(key=_representation_sort_key)
    return tuple(output)


def enumerate_little_group_representations(
    concrete: ConcreteFiniteGroup,
    factor_system: FactorSystem,
    tolerance: float,
) -> tuple[tuple[np.ndarray, ...], ...]:
    if any(concrete.antiunitary_flags):
        raise ValueError("Unitary spgrep enumeration cannot include antiunitary operations.")
    return enumerate_projective_representations(
        concrete.rotations,
        factor_system.phases,
        concrete.table.multiplication,
        dimension=concrete.group.dimension,
        tolerance=tolerance,
    )


def validate_catalog_characters(
    rotations: Iterable[np.ndarray],
    table: FiniteGroupTable,
    declared_characters: Iterable[Iterable[complex]],
    *,
    dimension: int,
    tolerance: float = 1.0e-8,
) -> None:
    generated = enumerate_projective_representations(
        rotations,
        np.ones((table.order, table.order), dtype=np.complex128),
        table.multiplication,
        dimension=dimension,
        tolerance=tolerance,
    )
    available = [
        np.asarray([np.trace(matrix) for matrix in representation], dtype=np.complex128)
        for representation in generated
    ]
    for declared in declared_characters:
        value = np.asarray(tuple(declared), dtype=np.complex128)
        match = next(
            (
                index
                for index, candidate in enumerate(available)
                if candidate.shape == value.shape
                and np.allclose(candidate, value, rtol=0.0, atol=max(10.0 * tolerance, 1.0e-7))
            ),
            None,
        )
        if match is None:
            raise ValueError("Declared character table does not match spgrep irreps.")
        available.pop(match)
    if available:
        raise ValueError("Declared character table is incomplete relative to spgrep irreps.")


def symmetry_engine_versions() -> tuple[str, str]:
    return str(spglib.__version__), str(spgrep.__version__)


def _representation_sort_key(matrices: tuple[np.ndarray, ...]) -> tuple:
    characters = [complex(np.trace(matrix)) for matrix in matrices]
    flattened = []
    for value in characters:
        flattened.extend((round(value.real, 12), round(value.imag, 12)))
    return matrices[0].shape[0], tuple(flattened)
