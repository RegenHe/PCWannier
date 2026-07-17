from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from .definition import FactorSystem, ResolvedLittleGroup

if TYPE_CHECKING:
    from .representation import SymmetryContext


@dataclass(frozen=True)
class TwistedRepresentation:
    """A unitary little-co-group representation with a shared factor system."""

    matrices: tuple[np.ndarray, ...]
    product_table: np.ndarray
    factor_system: FactorSystem
    antiunitary_flags: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        matrices = tuple(np.asarray(matrix, dtype=np.complex128) for matrix in self.matrices)
        if not matrices:
            raise ValueError("A twisted representation requires at least one matrix.")
        dimension = matrices[0].shape[0]
        if dimension <= 0 or any(matrix.shape != (dimension, dimension) for matrix in matrices):
            raise ValueError("Twisted-representation matrices must have one common square shape.")
        if any(not np.all(np.isfinite(matrix)) for matrix in matrices):
            raise ValueError("Twisted-representation matrices contain non-finite values.")

        order = len(matrices)
        product = np.asarray(self.product_table, dtype=np.int64)
        if product.shape != (order, order):
            raise ValueError("Twisted-representation product table has an incompatible shape.")
        if np.any(product < 0) or np.any(product >= order):
            raise ValueError("Twisted-representation product table contains an invalid index.")
        if self.factor_system.phases.shape != (order, order):
            raise ValueError("Twisted representation and factor system have different orders.")

        flags = self.antiunitary_flags or (False,) * order
        flags = tuple(bool(value) for value in flags)
        if len(flags) != order:
            raise ValueError("antiunitary_flags must contain one value per group element.")

        frozen_matrices = []
        for matrix in matrices:
            value = matrix.copy()
            value.setflags(write=False)
            frozen_matrices.append(value)
        product = product.copy()
        product.setflags(write=False)
        object.__setattr__(self, "matrices", tuple(frozen_matrices))
        object.__setattr__(self, "product_table", product)
        object.__setattr__(self, "antiunitary_flags", flags)

    @property
    def order(self) -> int:
        return len(self.matrices)

    @property
    def dimension(self) -> int:
        return int(self.matrices[0].shape[0])

    @property
    def cocycle_residual(self) -> float:
        return self.factor_system.cocycle_residual_for(self.product_table)

    @property
    def product_residual(self) -> float:
        if any(self.antiunitary_flags):
            raise NotImplementedError(
                "Antiunitary twisted-representation products are not implemented."
            )
        residual = 0.0
        for left in range(self.order):
            for right in range(self.order):
                result = int(self.product_table[left, right])
                expected = self.factor_system.phases[left, right] * self.matrices[result]
                residual = max(
                    residual,
                    float(np.linalg.norm(self.matrices[left] @ self.matrices[right] - expected, ord="fro")),
                )
        return residual

    def require_valid(self, *, tolerance: float | None = None) -> None:
        threshold = _validation_tolerance(self.factor_system.tolerance, tolerance)
        cocycle = self.cocycle_residual
        if cocycle > threshold:
            raise ValueError(
                f"Factor system violates the cocycle condition: residual={cocycle:.6g}, "
                f"tolerance={threshold:.6g}."
            )
        product = self.product_residual
        if product > threshold:
            raise ValueError(
                f"Matrices violate the twisted representation product: residual={product:.6g}, "
                f"tolerance={threshold:.6g}."
            )

    def assert_compatible(
        self,
        other: "TwistedRepresentation",
        *,
        tolerance: float | None = None,
    ) -> None:
        if not isinstance(other, TwistedRepresentation):
            raise TypeError("Expected another TwistedRepresentation.")
        if not np.array_equal(self.product_table, other.product_table):
            raise ValueError("Physical and target representations use different product tables.")
        if self.antiunitary_flags != other.antiunitary_flags:
            raise ValueError("Physical and target representations use different antiunitary flags.")
        self.factor_system.assert_compatible(other.factor_system, tolerance=tolerance)

    def trivialized_matrices(self, *, tolerance: float | None = None) -> tuple[np.ndarray, ...]:
        """Return ordinary matrices rho'_a = c_a rho_a when omega is a coboundary."""

        cochain = self.factor_system.trivializing_cochain
        if cochain is None:
            raise NotImplementedError(
                "This factor system is not cohomologically trivial; ordinary irreps cannot be used."
            )
        matrices = tuple(cochain[index] * matrix for index, matrix in enumerate(self.matrices))
        threshold = _validation_tolerance(self.factor_system.tolerance, tolerance)
        residual = _ordinary_product_residual(matrices, self.product_table)
        if residual > threshold:
            raise ValueError(
                f"Trivialized matrices do not form an ordinary representation: "
                f"residual={residual:.6g}, tolerance={threshold:.6g}."
            )
        output = []
        for matrix in matrices:
            value = np.asarray(matrix, dtype=np.complex128).copy()
            value.setflags(write=False)
            output.append(value)
        return tuple(output)


def build_twisted_representation(
    little_group: ResolvedLittleGroup,
    operation_indices: Sequence[int],
    matrices: Sequence[np.ndarray],
    *,
    antiunitary_flags: Sequence[bool] | None = None,
) -> TwistedRepresentation:
    """Order full-space-group matrices by the resolved little co-group."""

    indices = tuple(int(value) for value in operation_indices)
    values = tuple(matrices)
    if len(indices) != len(values) or len(indices) != len(set(indices)):
        raise ValueError("Little-group operation indices and matrices must be unique and aligned.")
    expected = little_group.concrete.operation_indices
    if set(indices) != set(expected):
        raise ValueError("Matrices do not cover the resolved little group exactly once.")
    matrix_by_operation = dict(zip(indices, values))

    if antiunitary_flags is None:
        flags_by_operation: Mapping[int, bool] = {index: False for index in indices}
    else:
        flags = tuple(bool(value) for value in antiunitary_flags)
        if len(flags) != len(indices):
            raise ValueError("antiunitary_flags must align with operation_indices.")
        flags_by_operation = dict(zip(indices, flags))

    return TwistedRepresentation(
        tuple(matrix_by_operation[index] for index in expected),
        little_group.table.multiplication,
        little_group.factor_system,
        tuple(flags_by_operation[index] for index in expected),
    )


def build_little_group_twisted_pair(
    context: "SymmetryContext",
    k_fractional,
    operation_indices: Sequence[int],
    physical_matrices: Sequence[np.ndarray],
    target_matrices: Sequence[np.ndarray],
) -> tuple[TwistedRepresentation, TwistedRepresentation]:
    definition = context.model.group_definition
    if definition is None:
        raise ValueError("Twisted little-group representations require a space-group definition.")
    indices = tuple(int(value) for value in operation_indices)
    resolved = definition.resolve_little_group(
        indices,
        k_fractional,
        bloch_convention=context.model.bloch_convention,
    )
    physical = build_twisted_representation(resolved, indices, physical_matrices)
    target = build_twisted_representation(resolved, indices, target_matrices)
    physical.assert_compatible(target)
    return physical, target


def _ordinary_product_residual(matrices, product_table) -> float:
    residual = 0.0
    for left in range(len(matrices)):
        for right in range(len(matrices)):
            result = int(product_table[left, right])
            residual = max(
                residual,
                float(np.linalg.norm(matrices[left] @ matrices[right] - matrices[result], ord="fro")),
            )
    return residual


def _validation_tolerance(base: float, requested: float | None) -> float:
    value = 0.0 if requested is None else float(requested)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("Twisted-representation tolerance must be finite and non-negative.")
    return max(float(base), 1.0e-12, value)
