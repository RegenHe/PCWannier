from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _hermitian(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    return 0.5 * (array + np.conjugate(np.swapaxes(array, -2, -1)))


@dataclass(frozen=True)
class ProjectorBandInterpolationDiagnostics:
    point_count: int
    minimum_projector_gap: float
    maximum_raw_projector_idempotency_error: float
    maximum_flattened_projector_idempotency_error: float
    maximum_removed_cross_sector_component: float

    def merged(
        self, other: "ProjectorBandInterpolationDiagnostics"
    ) -> "ProjectorBandInterpolationDiagnostics":
        return ProjectorBandInterpolationDiagnostics(
            self.point_count + other.point_count,
            min(self.minimum_projector_gap, other.minimum_projector_gap),
            max(
                self.maximum_raw_projector_idempotency_error,
                other.maximum_raw_projector_idempotency_error,
            ),
            max(
                self.maximum_flattened_projector_idempotency_error,
                other.maximum_flattened_projector_idempotency_error,
            ),
            max(
                self.maximum_removed_cross_sector_component,
                other.maximum_removed_cross_sector_component,
            ),
        )


class ProjectorPreservingBandInterpolator:
    """Preserve transverse and longitudinal sectors between sampled k points."""

    def __init__(
        self,
        transverse_dimension: int,
        wannier_dimension: int,
        *,
        fixed_longitudinal_eigenvalue: float | None = None,
    ) -> None:
        self.transverse_dimension = int(transverse_dimension)
        self.wannier_dimension = int(wannier_dimension)
        self.longitudinal_dimension = (
            self.wannier_dimension - self.transverse_dimension
        )
        self.fixed_longitudinal_eigenvalue = (
            None
            if fixed_longitudinal_eigenvalue is None
            else float(fixed_longitudinal_eigenvalue)
        )
        if not 0 < self.transverse_dimension < self.wannier_dimension:
            raise ValueError(
                "Projector-preserving interpolation requires 0 < N_T < N_W; "
                f"got N_T={self.transverse_dimension}, N_W={self.wannier_dimension}."
            )
        if self.fixed_longitudinal_eigenvalue is not None and not np.isfinite(
            self.fixed_longitudinal_eigenvalue
        ):
            raise ValueError("Fixed longitudinal eigenvalue must be finite.")

    def projector_in_wannier_basis(
        self,
        basis_overlap: np.ndarray,
        output_coefficients: np.ndarray,
        transverse_source_indices: np.ndarray,
    ) -> np.ndarray:
        """Return the selected T projector in the orthonormal output basis."""

        overlap = np.asarray(basis_overlap, dtype=np.complex128)
        coefficients = np.asarray(output_coefficients, dtype=np.complex128)
        source_indices = np.asarray(transverse_source_indices, dtype=int).reshape(-1)
        source_dimension = overlap.shape[0] if overlap.ndim == 2 else -1
        if overlap.shape != (source_dimension, source_dimension):
            raise ValueError(f"Source overlap must be square; got {overlap.shape}.")
        if coefficients.shape != (source_dimension, self.wannier_dimension):
            raise ValueError(
                "Projector output coefficients must have shape "
                f"{(source_dimension, self.wannier_dimension)}; got "
                f"{coefficients.shape}."
            )
        if source_indices.size != self.transverse_dimension:
            raise ValueError(
                "Transverse source selector has the wrong dimension: "
                f"{source_indices.size} != {self.transverse_dimension}."
            )
        if (
            np.unique(source_indices).size != source_indices.size
            or np.any(source_indices < 0)
            or np.any(source_indices >= source_dimension)
        ):
            raise ValueError("Transverse source indices are invalid or duplicated.")
        if not np.all(np.isfinite(overlap)) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Projector interpolation inputs contain non-finite values.")

        overlap = _hermitian(overlap)
        transverse_gram = overlap[np.ix_(source_indices, source_indices)]
        eigenvalues = np.linalg.eigvalsh(transverse_gram)
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(np.min(eigenvalues)) <= np.finfo(float).eps * scale:
            raise ValueError(
                "The transverse source Gram matrix is singular: "
                f"eigenvalues={eigenvalues.tolist()}."
            )
        transverse_overlap = overlap[source_indices, :] @ coefficients
        projector = transverse_overlap.conj().T @ np.linalg.solve(
            transverse_gram, transverse_overlap
        )
        flattened, _ = self.flatten_projectors(projector[None, ...])
        return flattened[0]

    def flatten_projectors(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, ProjectorBandInterpolationDiagnostics]:
        projectors = _hermitian(values)
        expected = (self.wannier_dimension, self.wannier_dimension)
        if projectors.ndim != 3 or projectors.shape[1:] != expected:
            raise ValueError(
                f"Projector batch must have shape (Nk, {expected[0]}, {expected[1]}); "
                f"got {projectors.shape}."
            )
        if not np.all(np.isfinite(projectors)):
            raise ValueError("Interpolated projectors contain non-finite values.")

        eigenvalues, eigenvectors = np.linalg.eigh(projectors)
        selected = eigenvectors[..., -self.transverse_dimension :]
        flattened = selected @ np.conjugate(np.swapaxes(selected, -2, -1))
        raw_errors = np.linalg.norm(
            projectors @ projectors - projectors, axis=(-2, -1)
        )
        flattened_errors = np.linalg.norm(
            flattened @ flattened - flattened, axis=(-2, -1)
        )
        boundary = self.longitudinal_dimension - 1
        gaps = eigenvalues[:, boundary + 1] - eigenvalues[:, boundary]
        diagnostics = ProjectorBandInterpolationDiagnostics(
            int(projectors.shape[0]),
            float(np.min(gaps)),
            float(np.max(raw_errors)),
            float(np.max(flattened_errors)),
            0.0,
        )
        return _hermitian(flattened), diagnostics

    def constrain_hamiltonians(
        self,
        transverse_hamiltonians: np.ndarray,
        longitudinal_hamiltonians: np.ndarray,
        approximate_projectors: np.ndarray,
    ) -> tuple[np.ndarray, ProjectorBandInterpolationDiagnostics]:
        h_transverse = _hermitian(transverse_hamiltonians)
        h_longitudinal = _hermitian(longitudinal_hamiltonians)
        expected = (
            h_transverse.shape[0],
            self.wannier_dimension,
            self.wannier_dimension,
        )
        if h_transverse.ndim != 3 or h_transverse.shape != expected:
            raise ValueError(
                f"Transverse Hamiltonian batch has an invalid shape: {h_transverse.shape}."
            )
        if h_longitudinal.shape != expected:
            raise ValueError(
                f"Longitudinal Hamiltonian batch has an invalid shape: {h_longitudinal.shape}."
            )
        if not np.all(np.isfinite(h_transverse)) or not np.all(
            np.isfinite(h_longitudinal)
        ):
            raise ValueError("Interpolated Hamiltonians contain non-finite values.")

        projector, diagnostics = self.flatten_projectors(approximate_projectors)
        if projector.shape[0] != h_transverse.shape[0]:
            raise ValueError("Hamiltonian and projector batches differ in size.")
        identity = np.eye(self.wannier_dimension, dtype=np.complex128)[None, ...]
        complement = identity - projector
        raw = h_transverse + h_longitudinal
        physical = projector @ h_transverse @ projector
        if self.fixed_longitudinal_eigenvalue is None:
            longitudinal = complement @ h_longitudinal @ complement
        else:
            longitudinal = self.fixed_longitudinal_eigenvalue * complement
        output = physical + longitudinal
        diagnostics = ProjectorBandInterpolationDiagnostics(
            diagnostics.point_count,
            diagnostics.minimum_projector_gap,
            diagnostics.maximum_raw_projector_idempotency_error,
            diagnostics.maximum_flattened_projector_idempotency_error,
            float(np.max(np.linalg.norm(raw - output, axis=(-2, -1)))),
        )
        return _hermitian(output), diagnostics
