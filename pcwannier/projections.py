from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.special import sph_harm_y


def _frozen_vector(values, *, name: str, dimension: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (dimension,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with {dimension} components.")
    output = array.copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class LocalFrame3D:
    """Right-handed Cartesian frame stored as column vectors (ex, ey, ez)."""

    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray

    @classmethod
    def from_xz(cls, x, z, *, tolerance: float = 1.0e-12) -> "LocalFrame3D":
        xvec = _frozen_vector(x, name="local frame x", dimension=3)
        zvec = _frozen_vector(z, name="local frame z", dimension=3)
        xnorm = float(np.linalg.norm(xvec))
        znorm = float(np.linalg.norm(zvec))
        if xnorm <= tolerance or znorm <= tolerance:
            raise ValueError("Local-frame x and z vectors must be non-zero.")
        ez = np.asarray(zvec / znorm, dtype=float)
        x_perp = np.asarray(xvec - np.dot(xvec, ez) * ez, dtype=float)
        x_perp_norm = float(np.linalg.norm(x_perp))
        if x_perp_norm <= tolerance * max(xnorm, 1.0):
            raise ValueError("Local-frame x and z vectors must not be parallel.")
        ex = x_perp / x_perp_norm
        ey = np.cross(ez, ex)
        ey /= np.linalg.norm(ey)
        for vector in (ex, ey, ez):
            vector.setflags(write=False)
        return cls(ex, ey, ez)

    @property
    def matrix(self) -> np.ndarray:
        output = np.column_stack((self.ex, self.ey, self.ez))
        output.setflags(write=False)
        return output

    def local_coordinates(self, cartesian_vectors) -> np.ndarray:
        vectors = np.asarray(cartesian_vectors, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError("Cartesian coordinates must have shape (points, 3).")
        return vectors @ self.matrix

    def global_direction(self, local_direction) -> np.ndarray:
        direction = _frozen_vector(
            local_direction, name="orbital vector direction", dimension=3
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-14:
            raise ValueError("Orbital vector direction must be non-zero.")
        output = self.matrix @ (direction / norm)
        output.setflags(write=False)
        return output


def real_spherical_harmonic(l: int, m: int, theta, phi) -> np.ndarray:
    """Condon-Shortley real spherical harmonic with p_x at (l,m)=(1,1)."""

    if l < 0 or abs(m) > l:
        raise ValueError(f"Real spherical harmonic requires l >= 0 and -l <= m <= l; got {l}, {m}.")
    theta_array = np.asarray(theta, dtype=float)
    phi_array = np.asarray(phi, dtype=float)
    value = sph_harm_y(l, abs(m), theta_array, phi_array)
    if m < 0:
        result = np.sqrt(2.0) * ((-1) ** m) * value.imag
    elif m > 0:
        result = np.sqrt(2.0) * ((-1) ** m) * value.real
    else:
        result = value.real
    return np.asarray(result, dtype=float)


@dataclass(frozen=True)
class VectorHydrogenicOrbital:
    n: int
    l: int
    m: int
    zeta: float
    direction: np.ndarray

    def __post_init__(self) -> None:
        if self.n <= 0 or self.l < 0 or self.l >= self.n:
            raise ValueError("Hydrogenic orbital requires n > 0 and 0 <= l < n.")
        if abs(self.m) > self.l:
            raise ValueError(
                f"Hydrogenic orbital requires -l <= m <= l; got l={self.l}, m={self.m}."
            )
        if not np.isfinite(self.zeta) or self.zeta <= 0.0:
            raise ValueError("Hydrogenic orbital zeta must be positive and finite.")
        direction = _frozen_vector(
            self.direction, name="orbital vector direction", dimension=3
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-14:
            raise ValueError("Orbital vector direction must be non-zero.")
        direction = direction / norm
        direction.setflags(write=False)
        object.__setattr__(self, "direction", direction)

    def evaluate(
        self,
        displacement_cartesian,
        frame: LocalFrame3D,
        lattice_const: float,
        radial_factory: Callable[[int, int], Callable],
    ) -> np.ndarray:
        displacement = np.asarray(displacement_cartesian, dtype=float)
        local = frame.local_coordinates(displacement)
        radius_cartesian = np.linalg.norm(local, axis=1)
        radius = radius_cartesian / float(lattice_const)
        theta = np.zeros_like(radius)
        nonzero = radius_cartesian > np.finfo(float).eps
        theta[nonzero] = np.arccos(
            np.clip(local[nonzero, 2] / radius_cartesian[nonzero], -1.0, 1.0)
        )
        phi = np.arctan2(local[:, 1], local[:, 0])
        scalar = radial_factory(self.n, self.l)(radius, self.zeta) * real_spherical_harmonic(
            self.l, self.m, theta, phi
        )
        return np.asarray(scalar[:, None] * frame.global_direction(self.direction)[None, :])


@dataclass(frozen=True)
class TrialLinearCombination:
    orbitals: tuple[VectorHydrogenicOrbital, ...]
    coefficients: tuple[complex, ...]

    def __post_init__(self) -> None:
        if not self.orbitals or len(self.orbitals) != len(self.coefficients):
            raise ValueError("Trial linear combination requires matching non-empty orbital and coefficient lists.")
        if not np.all(np.isfinite(np.asarray(self.coefficients, dtype=np.complex128))):
            raise ValueError("Trial linear-combination coefficients must be finite.")

    @classmethod
    def primitive(cls, orbital: VectorHydrogenicOrbital) -> "TrialLinearCombination":
        return cls((orbital,), (1.0 + 0.0j,))

    def evaluate(
        self,
        displacement_cartesian,
        frame: LocalFrame3D,
        lattice_const: float,
        radial_factory: Callable[[int, int], Callable],
    ) -> np.ndarray:
        point_count = np.asarray(displacement_cartesian).shape[0]
        result = np.zeros((point_count, 3), dtype=np.complex128)
        for orbital, coefficient in zip(self.orbitals, self.coefficients):
            result += coefficient * orbital.evaluate(
                displacement_cartesian, frame, lattice_const, radial_factory
            )
        return result


@dataclass(frozen=True)
class ProjectionRecord3D:
    wyckoff: str
    frac_position: np.ndarray
    frame: LocalFrame3D
    states: tuple[TrialLinearCombination, ...]

    def __post_init__(self) -> None:
        label = str(self.wyckoff).strip()
        if not label:
            raise ValueError("Projection Wyckoff label must not be empty.")
        center = _frozen_vector(
            self.frac_position, name="3D projection center", dimension=3
        )
        if not self.states:
            raise ValueError("A 3D projection record must define at least one trial function.")
        object.__setattr__(self, "wyckoff", label)
        object.__setattr__(self, "frac_position", center)


@dataclass(frozen=True)
class ProjectionTargetBinding:
    projection: ProjectionRecord3D
    target_name: str
    site_irrep: str


@dataclass(frozen=True)
class TrialCovarianceDiagnostics:
    max_residual: float
    residual_per_operation: tuple[tuple[str, float], ...]
    closure_residual: float
    basis_transform: np.ndarray | None = None


@dataclass(frozen=True)
class TrialBasisSet:
    values: np.ndarray
    centers_fractional: np.ndarray
    target_slices: tuple[slice, ...]
    diagnostics: tuple[TrialCovarianceDiagnostics, ...] = ()


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    output: list[str] = []
    start = 0
    square = round_depth = curly = 0
    for index, character in enumerate(text):
        if character == "[":
            square += 1
        elif character == "]":
            square -= 1
        elif character == "(":
            round_depth += 1
        elif character == ")":
            round_depth -= 1
        elif character == "{":
            curly += 1
        elif character == "}":
            curly -= 1
        elif character == delimiter and square == round_depth == curly == 0:
            output.append(text[start:index].strip())
            start = index + 1
        if min(square, round_depth, curly) < 0:
            raise ValueError(f"Unbalanced delimiters in {text!r}.")
    if square or round_depth or curly:
        raise ValueError(f"Unbalanced delimiters in {text!r}.")
    output.append(text[start:].strip())
    return [value for value in output if value]

