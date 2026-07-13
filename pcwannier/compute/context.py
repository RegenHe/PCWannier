from __future__ import annotations

from dataclasses import dataclass

from .gradient import Gradient
from .initializer import StateInitializer
from .matrix import MSet
from .state import StateCollection
from ..config import IncarConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..symmetry import SymmetryGaugeResult


@dataclass
class CalculationContext:
    config: IncarConfig
    state: StateCollection
    mset: MSet
    initializer: StateInitializer
    gradient: Gradient
    symmetry_gauge: SymmetryGaugeResult | None = None

    def bloch_gauge_at(self, i: int, j: int, k: int):
        return self.initializer.matV[i, j, k] @ self.gradient.U[i, j, k]

    def state_coefficients_at(self, i: int, j: int, k: int):
        strict_symmetry = bool(self.config.symmetry_constrained) and self.symmetry_gauge is not None
        zero_transform = bool(self.config.disable_orth) and not strict_symmetry
        transform = self.state.get_transform(zero_transform)
        return transform[i, j, k] @ self.bloch_gauge_at(i, j, k)
