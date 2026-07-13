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

    def internal_state_coefficients_at(self, i: int, j: int, k: int):
        """Coefficients in the internally orthonormalized Bloch basis."""
        transform = self.state.get_transform(False)
        return transform[i, j, k] @ self.bloch_gauge_at(i, j, k)

    def output_state_coefficients_at(self, i: int, j: int, k: int):
        """Coefficients used for final Wannier and tight-binding outputs."""
        if bool(self.config.symmetry_constrained) and self.symmetry_gauge is not None:
            fem_output = self.config.symmetry_output_basis == "fem"
        else:
            fem_output = bool(self.config.disable_orth)
        transform = self.state.get_transform(fem_output)
        return transform[i, j, k] @ self.bloch_gauge_at(i, j, k)

    def state_coefficients_at(self, i: int, j: int, k: int):
        """Backward-compatible alias for final output coefficients."""
        return self.output_state_coefficients_at(i, j, k)
