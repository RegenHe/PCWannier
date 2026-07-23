from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlochConvention:
    """Sign convention in psi_k = exp(sign * i k.r) u_k."""

    sign: int = 1
    name: str = "standard"

    def __post_init__(self) -> None:
        if self.sign not in {-1, 1}:
            raise ValueError("Bloch convention sign must be -1 or 1.")
        if not str(self.name).strip():
            raise ValueError("Bloch convention name must not be empty.")
