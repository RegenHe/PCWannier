from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import ceil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import scipy.linalg
from scipy.optimize import least_squares

from .compute.band_path import sample_fractional_band_path
from .data import BandResult
from .symmetry.representation import (
    CombinedTargetRepresentation,
    SymmetryContext,
)
from .ebr.analysis import EBRMatrixBuilder
from .ebr.catalog import load_ebr_catalog
from .ebr.models import EBRCatalog

__all__ = [
    "FormalTBFitResult",
    "FormalTightBindingModel",
    "build_formal_tight_binding",
    "build_formal_tight_binding_from_ebr",
    "fit_formal_tight_binding",
]


@dataclass(frozen=True)
class FormalTBSeed:
    """One Hermitian real-space hopping seed before symmetry projection."""

    translation: tuple[int, ...]
    row: int
    column: int
    component: str
    distance: float

    def __post_init__(self) -> None:
        if self.component not in {"diagonal", "real", "imaginary"}:
            raise ValueError(f"Unknown formal-TB seed component {self.component!r}.")
        if self.component == "diagonal" and (
            self.row != self.column or any(self.translation)
        ):
            raise ValueError("A diagonal formal-TB seed must be an on-site matrix element.")

    @property
    def label(self) -> str:
        vector = ",".join(str(value) for value in self.translation)
        return f"{self.component}[{self.row},{self.column};R=({vector})]"


@dataclass(frozen=True)
class FormalTBFitResult:
    parameters: np.ndarray
    fitted_eigenvalues: np.ndarray
    rms_error: float
    max_error: float
    auxiliary_maximum: float | None
    success: bool
    message: str
    function_evaluations: int

    def __post_init__(self) -> None:
        parameters = _readonly_array(self.parameters, ndim=1, name="Fit parameters")
        eigenvalues = _readonly_array(
            self.fitted_eigenvalues, ndim=2, name="Fitted eigenvalues"
        )
        if not np.isfinite(self.rms_error) or self.rms_error < 0.0:
            raise ValueError("Fit RMS error must be finite and non-negative.")
        if not np.isfinite(self.max_error) or self.max_error < 0.0:
            raise ValueError("Fit maximum error must be finite and non-negative.")
        if self.auxiliary_maximum is not None and not np.isfinite(
            self.auxiliary_maximum
        ):
            raise ValueError("Auxiliary-band maximum must be finite when present.")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "fitted_eigenvalues", eigenvalues)
        object.__setattr__(self, "function_evaluations", int(self.function_evaluations))


@dataclass(frozen=True)
class FormalTightBindingModel:
    """Finite-range Hamiltonian basis projected onto an induced EBR.

    The free parameters are real.  Every basis matrix is Hermitian and obeys
    the same space-group covariance as ``representation``. ``hopping_order=n``
    includes real-space orbital bonds with Cartesian length ``|d| <= n a``,
    where the supplied lattice vectors and orbital centers are expressed in
    units of the lattice scale ``a``.
    """

    representation: CombinedTargetRepresentation
    lattice_vectors: np.ndarray
    hopping_order: int
    hopping_cutoff: float
    bond_distances: tuple[float, ...]
    seeds: tuple[FormalTBSeed, ...]
    enforce_time_reversal: bool
    ebr_multiplicities: tuple[tuple[str, int], ...] = ()
    _basis_cache: dict[bytes, np.ndarray] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        lattice = _readonly_array(
            self.lattice_vectors, ndim=2, name="Real lattice vectors", real=True
        )
        dimension = self.representation.group.dimension
        if lattice.shape != (dimension, dimension):
            raise ValueError(
                f"Real lattice vectors must have shape {(dimension, dimension)}."
            )
        if abs(float(np.linalg.det(lattice))) <= np.finfo(float).eps:
            raise ValueError("Real lattice vectors must be invertible.")
        if int(self.hopping_order) < 0:
            raise ValueError("Formal-TB hopping order must be non-negative.")
        if not np.isfinite(self.hopping_cutoff) or self.hopping_cutoff < 0.0:
            raise ValueError("Formal-TB hopping cutoff must be finite and non-negative.")
        if not self.seeds:
            raise ValueError("A formal tight-binding model requires at least one parameter.")
        object.__setattr__(self, "lattice_vectors", lattice)
        object.__setattr__(self, "hopping_order", int(self.hopping_order))
        object.__setattr__(self, "hopping_cutoff", float(self.hopping_cutoff))
        object.__setattr__(self, "_basis_cache", {})

    @property
    def dimension(self) -> int:
        return self.representation.dimension

    @property
    def parameter_count(self) -> int:
        return len(self.seeds)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(seed.label for seed in self.seeds)

    @property
    def orbital_centers(self) -> np.ndarray:
        return _target_orbital_centers(self.representation)

    def basis(self, k_fractional) -> np.ndarray:
        """Return symmetry-allowed Hermitian basis matrices at one k point."""

        kpoint = _validate_kpoint(k_fractional, self.representation.group.dimension)
        key = np.ascontiguousarray(kpoint).tobytes()
        cached = self._basis_cache.get(key)
        if cached is not None:
            return cached
        values = _project_seed_basis(
            self.representation,
            self.seeds,
            kpoint,
            enforce_time_reversal=self.enforce_time_reversal,
        )
        values.setflags(write=False)
        self._basis_cache[key] = values
        return values

    def hamiltonian(self, k_fractional, parameters) -> np.ndarray:
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.parameter_count,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Formal-TB parameters must be finite with shape {(self.parameter_count,)}."
            )
        matrix = np.einsum("p,pij->ij", values, self.basis(k_fractional), optimize=True)
        return _hermitian(matrix)

    def eigenvalues(self, k_points, parameters) -> np.ndarray:
        points = _validate_kpoints(k_points, self.representation.group.dimension)
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.parameter_count,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Formal-TB parameters must be finite with shape {(self.parameter_count,)}."
            )
        basis = np.asarray([self.basis(point) for point in points])
        hamiltonians = np.einsum("p,kpij->kij", values, basis, optimize=True)
        return np.linalg.eigvalsh(_hermitian(hamiltonians))

    def transfer_parameters_from(
        self,
        source: "FormalTightBindingModel",
        parameters,
        *,
        k_points=None,
    ) -> np.ndarray:
        """Express a compatible formal Hamiltonian in this parameter basis.

        Independently generated shell orders use QR-selected coordinates, so
        their parameter indices need not be nested even though their physical
        Hamiltonian spaces are.  This least-squares coordinate transfer is the
        appropriate warm start when increasing the shell order.
        """

        if not isinstance(source, FormalTightBindingModel):
            raise TypeError("source must be a FormalTightBindingModel.")
        if source.dimension != self.dimension:
            raise ValueError("Formal models must have the same dimension.")
        if not np.allclose(
            source.lattice_vectors, self.lattice_vectors, rtol=1.0e-12, atol=1.0e-12
        ) or not np.allclose(
            source.orbital_centers, self.orbital_centers, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError("Formal models must use the same lattice and orbital centers.")
        source_parameters = np.asarray(parameters, dtype=float)
        if source_parameters.shape != (source.parameter_count,) or not np.all(
            np.isfinite(source_parameters)
        ):
            raise ValueError(
                "Source parameters must be finite with shape "
                f"{(source.parameter_count,)}."
            )
        points = (
            _default_sample_points(
                self.representation.group.dimension,
                max(16, self.parameter_count + source.parameter_count),
            )
            if k_points is None
            else _validate_kpoints(k_points, self.representation.group.dimension)
        )
        design = np.vstack(
            [_hermitian_coordinates(self.basis(point)).T for point in points]
        )
        target = np.concatenate(
            [
                _hermitian_coordinates(
                    source.hamiltonian(point, source_parameters)[None, :, :]
                )[0]
                for point in points
            ]
        )
        transferred, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        return np.asarray(transferred, dtype=float)

    def covariance_residual(self, parameters, k_points=None) -> float:
        points = (
            _default_sample_points(self.representation.group.dimension, 8)
            if k_points is None
            else _validate_kpoints(k_points, self.representation.group.dimension)
        )
        maximum = 0.0
        for point in points:
            source = self.hamiltonian(point, parameters)
            for operation_index, operation in enumerate(
                self.representation.group.operations
            ):
                target_k = operation.act_reciprocal(point)
                target = self.hamiltonian(target_k, parameters)
                matrix = self.representation.matrix(operation_index, point)
                transformed = np.conj(source) if operation.antiunitary else source
                expected = matrix @ transformed @ matrix.conj().T
                maximum = max(maximum, float(np.linalg.norm(target - expected, ord="fro")))
        return maximum

    def time_reversal_residual(self, parameters, k_points=None) -> float:
        if not self.enforce_time_reversal:
            raise ValueError("This formal model does not impose time reversal.")
        points = (
            _default_sample_points(self.representation.group.dimension, 8)
            if k_points is None
            else _validate_kpoints(k_points, self.representation.group.dimension)
        )
        return max(
            float(
                np.linalg.norm(
                    self.hamiltonian(-point, parameters)
                    - np.conj(self.hamiltonian(point, parameters)),
                    ord="fro",
                )
            )
            for point in points
        )

    def band_structure(self, parameters, k_path) -> BandResult:
        """Evaluate the formal model on an incar-style high-symmetry path."""

        points, axis, high_symmetry_points = sample_fractional_band_path(
            k_path, self.representation.group.dimension
        )
        return BandResult(
            points,
            axis,
            high_symmetry_points,
            self.eigenvalues(points, parameters),
        )

    def plot_bands(self, filename, parameters, k_path) -> BandResult:
        """Evaluate and write a band figure using PCWannier's standard plotter."""

        from .outputs import plot_band

        result = self.band_structure(parameters, k_path)
        plot_band(filename, result, None)
        return result


def build_formal_tight_binding(
    representation: CombinedTargetRepresentation,
    lattice_vectors,
    *,
    hopping_order: int,
    rank_tolerance: float = 1.0e-10,
    enforce_time_reversal: bool | None = None,
) -> FormalTightBindingModel:
    """Enumerate and project all orbital bonds with ``|d| <= hopping_order*a``."""

    if not isinstance(representation, CombinedTargetRepresentation):
        raise TypeError("representation must be a CombinedTargetRepresentation.")
    order = int(hopping_order)
    if order != hopping_order or order < 0:
        raise ValueError("hopping_order must be a non-negative integer.")
    if not np.isfinite(rank_tolerance) or not 0.0 < rank_tolerance < 1.0:
        raise ValueError("rank_tolerance must lie in (0, 1).")
    lattice = np.asarray(lattice_vectors, dtype=float)
    dimension = representation.group.dimension
    if lattice.shape != (dimension, dimension) or not np.all(np.isfinite(lattice)):
        raise ValueError(f"lattice_vectors must have finite shape {(dimension, dimension)}.")
    if abs(float(np.linalg.det(lattice))) <= np.finfo(float).eps:
        raise ValueError("lattice_vectors must be invertible.")

    if enforce_time_reversal is None:
        enforce_time_reversal = not any(
            operation.antiunitary for operation in representation.group.operations
        )
    if enforce_time_reversal and any(
        operation.antiunitary for operation in representation.group.operations
    ):
        raise ValueError(
            "Bare time reversal cannot be imposed automatically on a magnetic group; "
            "set enforce_time_reversal=False."
        )

    centers = _target_orbital_centers(representation)
    candidates, bond_distances = _enumerate_hopping_seeds(
        centers, lattice, order
    )
    samples = _default_sample_points(
        dimension,
        max(16, ceil(len(candidates) / max(representation.dimension**2, 1)) + 8),
    )
    design = np.vstack(
        [
            _hermitian_coordinates(
                _project_seed_basis(
                    representation,
                    candidates,
                    point,
                    enforce_time_reversal=bool(enforce_time_reversal),
                )
            ).T
            for point in samples
        ]
    )
    singular_values = np.linalg.svd(design, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("Symmetry projection removed every formal hopping seed.")
    rank = int(
        np.count_nonzero(singular_values > float(rank_tolerance) * singular_values[0])
    )
    if rank == 0:
        raise ValueError("Symmetry projection produced no independent Hamiltonian parameter.")
    _, _, pivots = scipy.linalg.qr(design, mode="economic", pivoting=True)
    selected = tuple(candidates[int(index)] for index in pivots[:rank])
    model = FormalTightBindingModel(
        representation,
        lattice,
        order,
        float(order),
        bond_distances,
        selected,
        bool(enforce_time_reversal),
    )
    scale = max(1.0, np.sqrt(float(model.dimension)))
    residual = model.covariance_residual(
        np.linspace(0.25, 1.0, model.parameter_count), samples[:8]
    )
    if residual > max(1.0e-8, 100.0 * float(rank_tolerance)) * scale:
        raise FloatingPointError(
            "Generated formal Hamiltonian basis violates target covariance "
            f"(residual={residual:.6g})."
        )
    return model


def build_formal_tight_binding_from_ebr(
    context: SymmetryContext,
    catalog: EBRCatalog | str | Path,
    multiplicities: Mapping[str, int],
    lattice_vectors,
    *,
    hopping_order: int,
    rank_tolerance: float = 1.0e-10,
    enforce_time_reversal: bool | None = None,
) -> FormalTightBindingModel:
    """Build a formal model directly from a positive EBR combination."""

    if not isinstance(context, SymmetryContext):
        raise TypeError("context must be a SymmetryContext.")
    if isinstance(catalog, EBRCatalog):
        resolved_catalog = catalog
    else:
        definition = context.model.group_definition
        resolved_catalog = load_ebr_catalog(
            catalog,
            hall_number=None if definition is None else definition.hall_number,
        )
    builder = EBRMatrixBuilder(
        context, resolved_catalog, lattice_vectors=lattice_vectors
    )
    definitions = {ebr.name: ebr for ebr in builder.catalog.ebrs}
    unknown = sorted(set(multiplicities) - set(definitions))
    if unknown:
        raise ValueError(f"Unknown EBR names for {builder.catalog.name!r}: {unknown}.")
    targets = []
    normalized = []
    for name, raw_count in multiplicities.items():
        count = int(raw_count)
        if count != raw_count or count < 0:
            raise ValueError(f"EBR multiplicity for {name!r} must be a non-negative integer.")
        if count == 0:
            continue
        target = builder.target_representation(definitions[name])
        targets.extend(target for _ in range(count))
        normalized.append((name, count))
    if not targets:
        raise ValueError("At least one positive EBR multiplicity is required.")
    model = build_formal_tight_binding(
        CombinedTargetRepresentation(tuple(targets)),
        lattice_vectors,
        hopping_order=hopping_order,
        rank_tolerance=rank_tolerance,
        enforce_time_reversal=enforce_time_reversal,
    )
    return FormalTightBindingModel(
        model.representation,
        model.lattice_vectors,
        model.hopping_order,
        model.hopping_cutoff,
        model.bond_distances,
        model.seeds,
        model.enforce_time_reversal,
        tuple(normalized),
    )


def fit_formal_tight_binding(
    model: FormalTightBindingModel,
    k_points,
    target_eigenvalues,
    *,
    eigenvalue_indices: Sequence[int] | None = None,
    initial_parameters=None,
    auxiliary_band_count: int = 0,
    auxiliary_ceiling: float | None = None,
    auxiliary_weight: float = 1.0,
    max_nfev: int = 10000,
    starts: int = 4,
    random_seed: int = 0,
) -> FormalTBFitResult:
    """Fit real formal-TB parameters to sorted target eigenvalues.

    ``eigenvalue_indices`` selects model eigenvalues after ascending sorting.
    This makes it possible to fit only the transverse sector of a T+L model.
    ``auxiliary_ceiling`` optionally penalizes the lowest auxiliary bands when
    they rise above a chosen value, commonly zero in photonic TETB fits.
    """

    points = _validate_kpoints(k_points, model.representation.group.dimension)
    target = np.asarray(target_eigenvalues, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
    if target.shape[0] != points.shape[0] or not np.all(np.isfinite(target)):
        raise ValueError(
            "target_eigenvalues must be finite with one row per k point."
        )
    if eigenvalue_indices is None:
        if target.shape[1] != model.dimension:
            raise ValueError(
                "eigenvalue_indices is required when fitting fewer than all model bands."
            )
        indices = np.arange(model.dimension, dtype=int)
    else:
        indices = np.asarray(tuple(eigenvalue_indices), dtype=int)
        if indices.shape != (target.shape[1],) or len(set(indices.tolist())) != len(indices):
            raise ValueError(
                "eigenvalue_indices must contain one unique index per target band."
            )
        if np.any(indices < 0) or np.any(indices >= model.dimension):
            raise ValueError("eigenvalue_indices contains an out-of-range model band.")
    auxiliary = int(auxiliary_band_count)
    if auxiliary < 0 or auxiliary > model.dimension:
        raise ValueError("auxiliary_band_count is out of range.")
    if auxiliary_ceiling is not None and not np.isfinite(auxiliary_ceiling):
        raise ValueError("auxiliary_ceiling must be finite when provided.")
    if not np.isfinite(auxiliary_weight) or auxiliary_weight <= 0.0:
        raise ValueError("auxiliary_weight must be positive and finite.")
    if int(max_nfev) <= 0:
        raise ValueError("max_nfev must be positive.")
    if int(starts) <= 0:
        raise ValueError("starts must be positive.")
    if initial_parameters is None:
        initial = np.zeros(model.parameter_count, dtype=float)
        initial[0] = float(np.mean(target))
    else:
        initial = np.asarray(initial_parameters, dtype=float)
        if initial.shape != (model.parameter_count,) or not np.all(np.isfinite(initial)):
            raise ValueError(
                f"initial_parameters must have finite shape {(model.parameter_count,)}."
            )

    sampled_basis = np.asarray([model.basis(point) for point in points])

    def spectrum(parameters):
        hamiltonians = np.einsum(
            "p,kpij->kij", parameters, sampled_basis, optimize=True
        )
        return np.linalg.eigvalsh(_hermitian(hamiltonians))

    def spectrum_and_jacobian(parameters):
        hamiltonians = np.einsum(
            "p,kpij->kij", parameters, sampled_basis, optimize=True
        )
        eigenvalues, eigenvectors = np.linalg.eigh(_hermitian(hamiltonians))
        derivatives = np.einsum(
            "kin,kpij,kjn->knp",
            eigenvectors.conj(),
            sampled_basis,
            eigenvectors,
            optimize=True,
        ).real
        return eigenvalues, derivatives

    def residual(parameters):
        all_eigenvalues = spectrum(parameters)
        pieces = [(all_eigenvalues[:, indices] - target).reshape(-1)]
        if auxiliary and auxiliary_ceiling is not None:
            violation = np.maximum(
                all_eigenvalues[:, :auxiliary] - float(auxiliary_ceiling), 0.0
            )
            pieces.append(np.sqrt(float(auxiliary_weight)) * violation.reshape(-1))
        return np.concatenate(pieces)

    def jacobian(parameters):
        all_eigenvalues, derivatives = spectrum_and_jacobian(parameters)
        pieces = [derivatives[:, indices, :].reshape(-1, model.parameter_count)]
        if auxiliary and auxiliary_ceiling is not None:
            active = (
                all_eigenvalues[:, :auxiliary] > float(auxiliary_ceiling)
            )[..., None]
            pieces.append(
                np.sqrt(float(auxiliary_weight))
                * (active * derivatives[:, :auxiliary, :]).reshape(
                    -1, model.parameter_count
                )
            )
        return np.vstack(pieces)

    rng = np.random.default_rng(int(random_seed))
    energy_scale = max(float(np.ptp(target)), float(np.std(target)), 1.0)
    starting_points = [initial]
    for _ in range(1, int(starts)):
        starting_points.append(
            initial
            + rng.normal(
                scale=energy_scale / max(np.sqrt(model.parameter_count), 1.0),
                size=model.parameter_count,
            )
        )
    attempts = [
        least_squares(
            residual,
            candidate,
            jac=jacobian,
            max_nfev=int(max_nfev),
            ftol=1.0e-13,
            xtol=1.0e-13,
            gtol=1.0e-13,
        )
        for candidate in starting_points
    ]
    optimized = min(attempts, key=lambda item: float(np.dot(item.fun, item.fun)))
    fitted_all = spectrum(optimized.x)
    fitted = fitted_all[:, indices]
    errors = fitted - target
    auxiliary_maximum = (
        None
        if auxiliary == 0
        else float(np.max(fitted_all[:, :auxiliary]))
    )
    return FormalTBFitResult(
        optimized.x,
        fitted,
        float(np.sqrt(np.mean(np.square(errors)))),
        float(np.max(np.abs(errors))),
        auxiliary_maximum,
        bool(optimized.success),
        str(optimized.message),
        int(optimized.nfev),
    )


def _project_seed_basis(
    representation: CombinedTargetRepresentation,
    seeds: Sequence[FormalTBSeed],
    kpoint: np.ndarray,
    *,
    enforce_time_reversal: bool,
) -> np.ndarray:
    values = _space_group_projected_seed_basis(representation, seeds, kpoint)
    if enforce_time_reversal:
        reverse = _space_group_projected_seed_basis(representation, seeds, -kpoint)
        values = 0.5 * (values + np.conj(reverse))
    return _hermitian(values)


def _space_group_projected_seed_basis(
    representation: CombinedTargetRepresentation,
    seeds: Sequence[FormalTBSeed],
    kpoint: np.ndarray,
) -> np.ndarray:
    group = representation.group
    output = np.zeros(
        (len(seeds), representation.dimension, representation.dimension),
        dtype=np.complex128,
    )
    for operation_index, operation in enumerate(group.operations):
        source_k = operation.inverse().act_reciprocal(kpoint)
        raw = _raw_seed_basis(
            seeds,
            source_k,
            representation.dimension,
            representation.bloch_convention.sign,
        )
        if operation.antiunitary:
            raw = np.conj(raw)
        matrix = representation.matrix(operation_index, source_k)
        output += np.einsum(
            "ab,pbc,dc->pad", matrix, raw, matrix.conj(), optimize=True
        )
    return output / float(len(group.operations))


def _raw_seed_basis(
    seeds: Sequence[FormalTBSeed],
    kpoint: np.ndarray,
    dimension: int,
    bloch_sign: int,
) -> np.ndarray:
    output = np.zeros((len(seeds), dimension, dimension), dtype=np.complex128)
    if not seeds:
        return output
    translations = np.asarray([seed.translation for seed in seeds], dtype=float)
    phases = np.exp(
        int(bloch_sign) * 2j * np.pi * (translations @ np.asarray(kpoint, dtype=float))
    )
    for index, (seed, phase) in enumerate(zip(seeds, phases)):
        if seed.component == "diagonal":
            output[index, seed.row, seed.column] = 1.0
            continue
        factor = 1.0 if seed.component == "real" else 1.0j
        output[index, seed.row, seed.column] = factor * phase
        output[index, seed.column, seed.row] = np.conj(factor * phase)
    return output


def _enumerate_hopping_seeds(
    centers: np.ndarray,
    lattice: np.ndarray,
    hopping_order: int,
) -> tuple[tuple[FormalTBSeed, ...], tuple[float, ...]]:
    """Enumerate Hermitian seeds within a Cartesian orbital-bond cutoff."""

    dimension = lattice.shape[0]
    singular_minimum = float(np.min(np.linalg.svd(lattice, compute_uv=False)))
    tolerance = 1.0e-9 * max(float(np.linalg.norm(lattice, ord=2)), 1.0)
    cutoff = float(hopping_order)
    center_offsets = centers[None, :, :] - centers[:, None, :]
    max_center_offset = float(
        np.max(np.linalg.norm(center_offsets @ lattice, axis=-1), initial=0.0)
    )
    radius = max(
        1,
        int(ceil((cutoff + max_center_offset + tolerance) / singular_minimum)) + 1,
    )
    records = []
    for translation in product(range(-radius, radius + 1), repeat=dimension):
        vector = np.asarray(translation, dtype=int)
        for row in range(len(centers)):
            for column in range(len(centers)):
                key = tuple(translation) + (row, column)
                reverse = tuple((-vector).tolist()) + (column, row)
                if key > reverse:
                    continue
                displacement = vector + centers[column] - centers[row]
                distance = float(np.linalg.norm(displacement @ lattice))
                if distance <= cutoff + tolerance:
                    records.append(
                        (tuple(int(value) for value in vector), row, column, distance)
                    )

    selected_distances = _unique_positive_distances(
        (record[3] for record in records), tolerance
    )
    seeds = []
    for translation, row, column, distance in records:
        self_adjoint = row == column and not any(translation)
        if self_adjoint:
            seeds.append(FormalTBSeed(translation, row, column, "diagonal", distance))
        else:
            seeds.append(FormalTBSeed(translation, row, column, "real", distance))
            seeds.append(FormalTBSeed(translation, row, column, "imaginary", distance))
    return tuple(seeds), tuple(float(value) for value in selected_distances)


def _unique_positive_distances(values, tolerance: float) -> tuple[float, ...]:
    output = []
    for value in sorted(float(item) for item in values if float(item) > tolerance):
        if not output or abs(value - output[-1]) > tolerance:
            output.append(value)
    return tuple(output)


def _target_orbital_centers(
    representation: CombinedTargetRepresentation,
) -> np.ndarray:
    centers = []
    for target in representation.targets:
        for point in target.orbit.points:
            centers.extend(
                np.asarray(point.position, dtype=float)
                for _ in range(target.site_irrep.dimension)
            )
    output = np.asarray(centers, dtype=float)
    expected = (representation.dimension, representation.group.dimension)
    if output.shape != expected:
        raise RuntimeError(
            f"Target orbital centers have shape {output.shape}; expected {expected}."
        )
    output.setflags(write=False)
    return output


def _hermitian_coordinates(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.complex128)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("Hermitian-coordinate input must have shape (count, N, N).")
    dimension = values.shape[1]
    pieces = [np.real(np.diagonal(values, axis1=1, axis2=2))]
    scale = np.sqrt(2.0)
    for row in range(dimension):
        for column in range(row + 1, dimension):
            pieces.append((scale * np.real(values[:, row, column]))[:, None])
            pieces.append((scale * np.imag(values[:, row, column]))[:, None])
    return np.concatenate(pieces, axis=1)


def _default_sample_points(dimension: int, count: int) -> np.ndarray:
    roots = np.sqrt(np.asarray((2.0, 3.0, 5.0, 7.0, 11.0), dtype=float)[:dimension])
    indices = np.arange(1, int(count) + 1, dtype=float)[:, None]
    points = np.mod(indices * roots[None, :], 1.0) - 0.5
    points[0] = 0.0
    return points


def _validate_kpoint(values, dimension: int) -> np.ndarray:
    point = np.asarray(values, dtype=float)
    if point.shape != (dimension,) or not np.all(np.isfinite(point)):
        raise ValueError(f"k point must have finite shape {(dimension,)}.")
    return point


def _validate_kpoints(values, dimension: int) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != dimension or not np.all(np.isfinite(points)):
        raise ValueError(f"k_points must have finite shape (count, {dimension}).")
    return points


def _readonly_array(values, *, ndim: int, name: str, real: bool = False) -> np.ndarray:
    dtype = float if real else None
    output = np.asarray(values, dtype=dtype)
    if output.ndim != ndim or not np.all(np.isfinite(output)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array.")
    output = output.copy()
    output.setflags(write=False)
    return output


def _hermitian(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    return 0.5 * (array + np.conjugate(np.swapaxes(array, -2, -1)))
