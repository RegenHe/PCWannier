from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..data import InputBundle


@dataclass(frozen=True)
class NLCSettings:
    """Controls for Neighbor-based Longitudinal Completion (NLC).

    The cutoff uses reciprocal Cartesian coordinates without the 2*pi/a factor.
    Auxiliary energies are negative constant-eta Laplacian Ritz values.
    """

    cutoff: float = 4.0
    eta: float = 10.0
    max_iterations: int = 800
    projector_tolerance: float = 1.0e-4
    cost_tolerance: float = 1.0e-7
    mixing: float = 0.5
    rank_tolerance: float = 1.0e-6
    filter_candidates: bool = True
    pin_gamma: bool = True

    def __post_init__(self) -> None:
        for name in ("cutoff", "eta", "projector_tolerance", "rank_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"NLC {name} must be positive and finite.")
        if self.rank_tolerance >= 1:
            raise ValueError("NLC rank_tolerance must lie in (0, 1).")
        if not np.isfinite(self.cost_tolerance) or self.cost_tolerance < 0:
            raise ValueError("NLC cost_tolerance must be non-negative and finite.")
        if not np.isfinite(self.mixing) or not 0 < self.mixing <= 1:
            raise ValueError("NLC mixing must lie in (0, 1].")
        if isinstance(self.max_iterations, (bool, np.bool_)) or not isinstance(
            self.max_iterations, (int, np.integer)
        ) or self.max_iterations < 0:
            raise ValueError("NLC max_iterations must be a non-negative integer.")
        if not isinstance(self.filter_candidates, (bool, np.bool_)) or not isinstance(
            self.pin_gamma, (bool, np.bool_)
        ):
            raise ValueError("NLC filter_candidates and pin_gamma must be booleans.")

    @classmethod
    def from_config(cls, config) -> NLCSettings:
        return cls(
            cutoff=config.nlc_cutoff,
            eta=config.nlc_eta,
            max_iterations=config.nlc_max_iter,
            projector_tolerance=config.nlc_projector_tolerance,
            cost_tolerance=config.nlc_err_diff,
            mixing=config.nlc_mixing,
            rank_tolerance=config.nlc_rank_tolerance,
            filter_candidates=config.nlc_filter_candidates,
            pin_gamma=config.nlc_pin_gamma,
        )


@dataclass(frozen=True)
class NLCCompletionResult:
    transverse_dimension: int
    auxiliary_dimension: int
    wannier_dimension: int
    settings: NLCSettings
    converged: bool
    iterations: int
    initial_cost: float
    final_cost: float
    history: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]

    @property
    def omega_i(self) -> float:
        """Gauge-invariant spread for the configured, directed neighbor stencil."""
        return self.final_cost / 2

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["method"] = "Neighbor-based Longitudinal Completion (NLC)"
        result["omega_i"] = self.omega_i
        result["auxiliary_energy_model"] = "negative constant-eta Laplacian Ritz spectrum"
        return result


@dataclass(frozen=True)
class NLCCompletionArtifacts:
    result: NLCCompletionResult
    augmented_bundle: InputBundle = field(repr=False)
