from __future__ import annotations

from itertools import product

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

from ..data import (
    BandResult,
    DegeneracySplittingDiagnostic,
    HoppingReconstructionDiagnostics,
    OutputSpectrumDiagnostics,
)
from .context import CalculationContext
from .band_path import sample_fractional_band_path
from .kspace import get_kxyz
from .parallel import parallel_map


class TBAModel:
    def __init__(
        self,
        ctx: CalculationContext,
        threads: int = 1,
    ):
        self.ctx = ctx
        self.config = ctx.config
        self.state = ctx.state
        self.threads = max(1, int(threads))
        self.hoppings: list[np.ndarray] | None = None
        self._projected_hamiltonians: np.ndarray | None = None
        self._projected_k_cart: np.ndarray | None = None
        self._projector_source_positions = self._prepare_projector_source_positions()
        self._transverse_projector_grid: np.ndarray | None = None

    def transverse_projectors(self) -> np.ndarray | None:
        """Return sampled P_T(k) matrices in the final Wannier basis."""

        if self._projector_source_positions is None:
            return None
        if self._transverse_projector_grid is not None:
            return self._transverse_projector_grid
        output = np.empty(self.state.k_shape, dtype=object)
        for index in self.state.k_indices():
            output[index] = self._transverse_projector_at(index)
        self._transverse_projector_grid = output
        return output

    def gen_hopping(self, r: list[int] | tuple[int, ...] | None = None) -> np.ndarray:
        if r is None:
            r = [0, 0, 0]
        k_cart, projected = self._projected_k_hamiltonians()
        return self._fourier_coefficient(projected, k_cart, r)

    def _fourier_coefficient(
        self,
        matrices: np.ndarray,
        k_cart: np.ndarray,
        r: list[int] | tuple[int, ...],
    ) -> np.ndarray:
        config = self.config
        kdim = int(config.kdim)
        avec = np.asarray(config.real_lattice_vectors, dtype=float)
        dim = avec.shape[1]
        r_use = (list(r) + [0, 0, 0])[:kdim]
        r_cart = np.zeros(dim, dtype=float)
        for axis in range(kdim):
            r_cart += r_use[axis] * avec[axis, :]
        r_cart *= float(config.lattice_const)
        phase = np.exp(1j * (-self.state.bloch_sign) * (k_cart @ r_cart))
        return (
            np.sum(
                np.asarray(matrices, dtype=np.complex128)
                * phase[:, None, None],
                axis=0,
            )
            / self.state.get_k_num()
        )

    def _projected_k_hamiltonians(self) -> tuple[np.ndarray, np.ndarray]:
        if self._projected_hamiltonians is not None and self._projected_k_cart is not None:
            return self._projected_k_cart, self._projected_hamiltonians

        config = self.config
        state = self.state
        dim = np.asarray(config.real_lattice_vectors).shape[1]
        band_count = int(config.band_calc_num)
        k_count = state.get_k_num()
        k_cart = np.empty((k_count, dim), dtype=float)
        projected = np.empty((k_count, band_count, band_count), dtype=np.complex128)
        for pos, (i, j, k) in enumerate(state.k_indices()):
            umat = self.ctx.output_state_coefficients_at(i, j, k)
            base_hamiltonian = self._output_base_hamiltonian_at((i, j, k))
            projected[pos] = np.conj(umat).T @ base_hamiltonian @ umat
            k_cart[pos] = get_kxyz(config, [i, j, k])[:dim]
        self._projected_k_cart = k_cart
        self._projected_hamiltonians = projected
        return k_cart, projected

    def _prepare_projector_source_positions(
        self,
    ) -> dict[tuple[int, int, int], np.ndarray] | None:
        channels = getattr(self.state, "band_channels", {})
        has_longitudinal = any(
            str(reference.channel).strip().upper() == "L"
            for reference in channels.values()
        )
        if not has_longitudinal:
            return None

        positions: dict[tuple[int, int, int], np.ndarray] = {}
        counts = set()
        for index in self.state.k_indices():
            actual_bands = np.asarray(self.state.E_idx[index], dtype=int).reshape(-1)
            transverse = []
            longitudinal = []
            for position, actual_band in enumerate(actual_bands):
                reference = channels.get(int(actual_band))
                if reference is None:
                    raise ValueError(
                        "Writing P_T requires T/L channel metadata for actual band "
                        f"{actual_band} at k={index}."
                    )
                channel = str(reference.channel).strip().upper()
                if channel == "L":
                    longitudinal.append(position)
                elif channel in {"H", "T"}:
                    transverse.append(position)
                else:
                    raise ValueError(
                        "Unsupported band channel while constructing P_T: "
                        f"{reference.channel!r}."
                    )
            if not transverse or not longitudinal:
                raise ValueError(
                    "Writing P_T requires both T and L states at every k point; "
                    f"k={index} has T={len(transverse)}, L={len(longitudinal)}."
                )
            positions[tuple(index)] = np.asarray(transverse, dtype=int)
            counts.add(len(transverse))
        if len(counts) != 1:
            raise ValueError(
                "Writing P_T requires a fixed transverse dimension; found "
                f"{sorted(counts)}."
            )
        return positions

    def _transverse_projector_at(
        self, index: tuple[int, int, int]
    ) -> np.ndarray:
        if self._projector_source_positions is None:
            raise RuntimeError("The current state has no mixed T/L channel metadata.")
        if self.state.S is None:
            raise ValueError("Writing P_T requires the raw S(k) matrices.")

        transverse_positions = self._projector_source_positions[index]
        coefficients = np.asarray(
            self.ctx.output_state_coefficients_at(*index), dtype=np.complex128
        )
        source_dimension, wannier_dimension = coefficients.shape
        normalization = np.eye(source_dimension, dtype=np.complex128)
        normalization_grid = getattr(self.state, "normalization_transform", None)
        if bool(getattr(self.state, "is_orthogonalized", False)) and (
            normalization_grid is not None
        ):
            normalization = np.asarray(normalization_grid[index], dtype=np.complex128)
        raw_overlap = np.asarray(self.state.S[index], dtype=np.complex128)
        if raw_overlap.shape != (source_dimension, source_dimension):
            raise ValueError(
                f"Raw overlap at k={index} has shape {raw_overlap.shape}; expected "
                f"{(source_dimension, source_dimension)}."
            )
        overlap = normalization.conj().T @ raw_overlap @ normalization
        transverse_gram = overlap[np.ix_(transverse_positions, transverse_positions)]
        eigenvalues = np.linalg.eigvalsh(self._hermitian_batch(transverse_gram))
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(np.min(eigenvalues)) <= np.finfo(float).eps * scale:
            raise ValueError(
                f"The transverse source Gram matrix at k={index} is singular: "
                f"eigenvalues={eigenvalues.tolist()}."
            )
        transverse_overlap = overlap[transverse_positions, :] @ coefficients
        matrix_elements = transverse_overlap.conj().T @ np.linalg.solve(
            transverse_gram, transverse_overlap
        )
        matrix_elements = self._hermitian_batch(matrix_elements)
        rank = min(int(transverse_positions.size), wannier_dimension)
        _, eigenvectors = np.linalg.eigh(matrix_elements)
        selected = eigenvectors[:, -rank:]
        return self._hermitian_batch(selected @ selected.conj().T)

    def _band_hamiltonian_factory(
        self,
        hoppings: dict[tuple[int, int, int], np.ndarray],
    ):
        h0 = np.asarray(hoppings[(0, 0, 0)], dtype=np.complex128)
        neighbors = self._band_neighbors(hoppings)
        hop_array = self._hoppings_for_neighbors(hoppings, neighbors)
        return self._h_of_k_factory(h0, neighbors, hop_array)

    def output_spectrum_diagnostics(self, symmetry_analysis=None) -> OutputSpectrumDiagnostics | None:
        """Compare the output-basis Hamiltonian with isolated FEM eigenvalues."""
        _, projected = self._projected_k_hamiltonians()
        band_count = int(self.config.band_calc_num)
        indices = tuple(self.state.k_indices())
        if any(len(np.asarray(self.state.E[index]).reshape(-1)) != band_count for index in indices):
            return None

        raw = np.asarray([
            np.linalg.eigvalsh(self._hermitian_batch(self._output_base_hamiltonian_at(index)))
            for index in indices
        ])
        output = np.linalg.eigvalsh(self._hermitian_batch(projected))
        errors = np.max(np.abs(output - raw), axis=1)
        worst = int(np.argmax(errors))
        basis = self.config.output_basis
        return OutputSpectrumDiagnostics(
            basis,
            float(errors[worst]),
            tuple(int(value) for value in indices[worst]),
            self._degeneracy_splittings(projected, symmetry_analysis),
        )

    def hopping_reconstruction_diagnostics(
        self,
        hoppings: dict[tuple[int, int, int], np.ndarray],
        symmetry_analysis=None,
    ) -> HoppingReconstructionDiagnostics:
        """Measure truncation error on the original sampled k mesh."""
        k_cart, projected = self._projected_k_hamiltonians()
        h0 = np.asarray(hoppings[(0, 0, 0)], dtype=np.complex128)
        neighbors = self._band_neighbors(hoppings)
        hop_array = self._hoppings_for_neighbors(hoppings, neighbors)
        reconstructed = self._h_of_k_factory(h0, neighbors, hop_array)(k_cart)
        matrix_errors = np.linalg.norm(reconstructed - projected, axis=(1, 2))
        direct_eigenvalues = np.linalg.eigvalsh(self._hermitian_batch(projected))
        reconstructed_eigenvalues = np.linalg.eigvalsh(self._hermitian_batch(reconstructed))
        eigenvalue_errors = np.max(np.abs(reconstructed_eigenvalues - direct_eigenvalues), axis=1)
        worst = int(np.argmax(eigenvalue_errors))
        indices = tuple(self.state.k_indices())
        return HoppingReconstructionDiagnostics(
            float(np.max(matrix_errors)),
            float(eigenvalue_errors[worst]),
            tuple(int(value) for value in indices[worst]),
            self._degeneracy_splittings(
                reconstructed,
                symmetry_analysis,
                reference_hamiltonians=projected,
            ),
        )

    def _degeneracy_splittings(
        self,
        hamiltonians: np.ndarray,
        symmetry_analysis,
        *,
        reference_hamiltonians: np.ndarray | None = None,
    ) -> tuple[DegeneracySplittingDiagnostic, ...]:
        if symmetry_analysis is None:
            return ()
        band_count = int(self.config.band_calc_num)
        output = []
        for point in symmetry_analysis.points:
            state_index = tuple((list(point.k_index) + [0, 0, 0])[:3])
            actual_bands = tuple(int(value) for value in np.asarray(self.state.E_idx[state_index]).reshape(-1))
            if len(actual_bands) != band_count:
                continue
            raw_energies = np.real(self._output_energies_at(state_index))
            sorted_local = np.argsort(raw_energies)
            rank_by_band = {actual_bands[local]: rank for rank, local in enumerate(sorted_local)}
            flat = int(np.ravel_multi_index(state_index, self.state.k_shape))
            output_energies = np.linalg.eigvalsh(self._hermitian_batch(hamiltonians[flat]))
            for block in point.degenerate_blocks:
                if len(block.band_indices) < 2 or any(band not in rank_by_band for band in block.band_indices):
                    continue
                ranks = [rank_by_band[band] for band in block.band_indices]
                raw_values = np.asarray([raw_energies[actual_bands.index(band)] for band in block.band_indices])
                output_values = output_energies[ranks]
                if reference_hamiltonians is None:
                    reference_values = raw_values
                else:
                    reference_eigenvalues = np.linalg.eigvalsh(
                        self._hermitian_batch(reference_hamiltonians[flat])
                    )
                    reference_values = reference_eigenvalues[ranks]
                scale = max(float(np.max(np.abs(raw_values))), 1.0)
                tolerance = float(self.config.representation_degeneracy_absolute) + float(
                    self.config.representation_degeneracy_relative
                ) * scale
                output.append(
                    DegeneracySplittingDiagnostic(
                        point.name,
                        tuple(int(value) for value in block.band_indices),
                        float(np.max(reference_values) - np.min(reference_values)),
                        float(np.max(output_values) - np.min(output_values)),
                        tolerance,
                    )
                )
        return tuple(output)

    def _output_energies_at(self, index: tuple[int, int, int]) -> np.ndarray:
        """Return energies used only by final hopping and band construction."""

        energies = np.asarray(self.state.E[index], dtype=np.complex128).reshape(-1)
        if not bool(getattr(self.config, "invert_longitudinal_energies", False)):
            return energies
        actual_bands = np.asarray(self.state.E_idx[index], dtype=int).reshape(-1)
        if actual_bands.size != energies.size:
            raise ValueError(
                f"Band channel metadata at k={index} does not match the energy window."
            )
        channels = getattr(self.state, "band_channels", {})
        signs = np.ones(energies.size, dtype=float)
        longitudinal = 0
        for position, actual_band in enumerate(actual_bands):
            reference = channels.get(int(actual_band))
            if reference is not None and str(reference.channel).upper() == "L":
                signs[position] = -1.0
                longitudinal += 1
        if longitudinal == 0:
            raise ValueError(
                "invert_longitudinal_energies=true was requested, but the output window "
                f"contains no L-channel bands at k={index}."
            )
        return energies * signs

    def _output_base_hamiltonian_at(
        self, index: tuple[int, int, int]
    ) -> np.ndarray:
        getter = getattr(self.state, "base_hamiltonian_at", None)
        matrix = (
            np.asarray(getter(index), dtype=np.complex128)
            if callable(getter)
            else np.diag(np.asarray(self.state.E[index], dtype=np.complex128))
        )
        if not bool(getattr(self.config, "invert_longitudinal_energies", False)):
            return matrix
        # The legacy file-based L path supplies Maxwell eigenstates, so its
        # source Hamiltonian is diagonal before the Wannier gauge transform.
        diagonal = np.diag(np.diag(matrix))
        scale = max(float(np.linalg.norm(matrix, ord="fro")), 1.0)
        if np.linalg.norm(matrix - diagonal, ord="fro") > 1.0e-10 * scale:
            raise ValueError(
                "invert_longitudinal_energies requires a diagonal source Hamiltonian."
            )
        return np.diag(self._output_energies_at(index))

    @staticmethod
    def _hermitian_batch(matrices: np.ndarray) -> np.ndarray:
        array = np.asarray(matrices, dtype=np.complex128)
        return 0.5 * (array + np.conjugate(np.swapaxes(array, -2, -1)))

    def collect_hoppings(self) -> dict[tuple[int, int, int], np.ndarray]:
        complete_neighbors = self.R_half_rect(self.state.k_shape)
        self._projected_k_hamiltonians()

        def calc_residue(row):
            representatives = self._wigner_seitz_representatives(row)
            if self.is_nyquist(row, self.state.k_shape):
                representatives = tuple(
                    representative
                    for representative in representatives
                    if representative < tuple(-value for value in representative)
                )
            if not representatives:
                raise RuntimeError(
                    f"No independent Wigner-Seitz representative found for {row}."
                )
            coefficient = self.gen_hopping(row) / float(len(representatives))
            return tuple((representative, coefficient) for representative in representatives)

        out = {(0, 0, 0): self.gen_hopping((0, 0, 0))}
        rows = [tuple(int(value) for value in row) for row in complete_neighbors]
        for entries in parallel_map(rows, calc_residue, self.threads):
            for key, hopping in entries:
                if key in out:
                    raise RuntimeError(
                        f"Duplicate Wigner-Seitz hopping representative {key}."
                    )
                out[key] = hopping
        return out

    def _wigner_seitz_representatives(
        self, residue: tuple[int, ...]
    ) -> tuple[tuple[int, int, int], ...]:
        """Return every shortest real-space representative of a mesh residue."""

        shape = np.asarray(self.state.k_shape, dtype=int)
        kdim = int(self.config.kdim)
        base = np.asarray(residue, dtype=int)[:kdim] % shape[:kdim]
        lattice = (
            np.asarray(self.config.real_lattice_vectors, dtype=float)[:kdim]
            * float(self.config.lattice_const)
        )
        winners: list[np.ndarray] = []
        for radius in range(1, 5):
            translations = np.asarray(
                list(product(range(-radius, radius + 1), repeat=kdim)),
                dtype=int,
            )
            candidates = base[None, :] + translations * shape[None, :kdim]
            cartesian = candidates @ lattice
            distance_squared = np.einsum(
                "ij,ij->i", cartesian, cartesian, optimize=True
            )
            minimum = float(np.min(distance_squared))
            tolerance = 1.0e-12 * max(minimum, 1.0)
            winner_mask = np.abs(distance_squared - minimum) <= tolerance
            winners = [row.copy() for row in candidates[winner_mask]]
            winner_translations = translations[winner_mask]
            if not np.any(np.abs(winner_translations) == radius):
                break
        else:
            raise RuntimeError(
                "Could not bound the Wigner-Seitz representative search for "
                f"residue={tuple(int(value) for value in base)}."
            )

        padded = {
            tuple((row.tolist() + [0, 0, 0])[:3]) for row in winners
        }
        return tuple(sorted(padded))

    def gen_hs_bands(self, hoppings: dict[tuple[int, int, int], np.ndarray]) -> BandResult:
        config = self.config
        kdim = int(config.kdim)
        k_path, k_axis, high_sym_points = sample_fractional_band_path(
            config.k_path, kdim
        )

        h_of_k = self._band_hamiltonian_factory(hoppings)
        hks = h_of_k(self._kfrac_to_kcart(k_path))
        energies = np.linalg.eigvalsh(hks) if config.hermitian else np.sort(np.linalg.eigvals(hks))

        dos_energy = None
        dos_components = None
        if config.DOS in (1, 2, 3):
            dos_energy, dos_components = self._calculate_dos(h_of_k, kdim)

        return BandResult(k_path, k_axis, high_sym_points, energies, dos_energy, dos_components)

    def _calculate_dos(self, h_of_k, kdim: int) -> tuple[np.ndarray, np.ndarray]:
        config = self.config
        mesh = np.asarray(config.DOS_Brillouin_mesh, dtype=int)[:kdim]
        if mesh.size != kdim or np.any(mesh <= 0):
            raise ValueError(f"DOS_Brillouin_mesh must contain {kdim} positive integers.")
        axes = [np.arange(int(count), dtype=float) / int(count) for count in mesh]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, kdim)
        hgrid = h_of_k(self._kfrac_to_kcart(grid))
        if not config.hermitian:
            raise NotImplementedError("DOS currently supports Hermitian TBA models only.")
        eigvals, eigvecs = np.linalg.eigh(hgrid)

        eta = float(config.DOS_eps)
        if not np.isfinite(eta) or eta <= 0.0:
            raise ValueError("DOS_eps must be a positive finite number.")
        energy_count = int(config.DOS_num)
        if energy_count < 2:
            raise ValueError("DOS_num must be at least 2.")
        padding = 10.0 * eta
        energy_axis = np.linspace(float(eigvals.min()) - padding, float(eigvals.max()) + padding, energy_count)
        component_count = int(config.band_calc_num) if config.DOS == 2 else 1
        components = np.empty((component_count, energy_count), dtype=np.float64)
        orbital_weights = np.abs(eigvecs) ** 2
        nk = eigvals.shape[0]
        for eidx, energy in enumerate(energy_axis):
            lorentz = eta / (np.pi * ((energy - eigvals) ** 2 + eta**2))
            if config.DOS == 2:
                components[:, eidx] = np.einsum("kob,kb->o", orbital_weights, lorentz, optimize=True) / nk
            else:
                components[0, eidx] = np.sum(lorentz) / nk
        if config.DOS == 2:
            total = np.sum(components, axis=0)
            direct = np.array(
                [np.sum(eta / (np.pi * ((energy - eigvals) ** 2 + eta**2))) / nk for energy in energy_axis]
            )
            residual = float(np.max(np.abs(total - direct)))
            if residual > 1e-10 * max(float(np.max(np.abs(direct))), 1.0):
                raise FloatingPointError(f"PDOS components do not sum to total DOS (residual={residual:.6g}).")
        return energy_axis, components

    def gen_bz_bands(self, result: BandResult, hoppings: dict[tuple[int, int, int], np.ndarray]) -> None:
        config = self.config
        kdim = int(config.kdim)
        k_num = np.asarray(config.k_num, dtype=int)[:kdim]
        axes = [np.linspace(-0.5, 0.5, n, endpoint=False) for n in k_num]
        kfrac = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        nk_shape = kfrac.shape[:-1]
        k_flat = kfrac.reshape(int(np.prod(nk_shape)), kdim)
        hks = self._band_hamiltonian_factory(hoppings)(
            self._kfrac_to_kcart(k_flat)
        )
        eigvals, eigvecs = np.linalg.eigh(hks)
        result.bz_eigvals = eigvals.reshape(*nk_shape, int(config.band_calc_num))
        result.bz_eigvecs = eigvecs.reshape(*nk_shape, int(config.band_calc_num), int(config.band_calc_num))
        result.groups = self.group_bands(result.bz_eigvals, delta_rel=1e-2)

    def _kfrac_to_kcart(self, kfrac: np.ndarray) -> np.ndarray:
        reciprocal = np.asarray(self.config.reciprocal_lattice_vectors, dtype=float)[: self.config.kdim, :]
        return (kfrac @ reciprocal) * (2.0 * np.pi / float(self.config.lattice_const))

    def _hoppings_for_neighbors(
        self,
        hoppings: dict[tuple[int, int, int], np.ndarray],
        neighbors: np.ndarray,
    ) -> np.ndarray:
        rows = np.asarray(neighbors, dtype=int)
        if rows.size == 0:
            band_count = int(self.config.band_calc_num)
            return np.empty((0, band_count, band_count), dtype=np.complex128)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        kdim = int(self.config.kdim)
        if rows.ndim != 2 or rows.shape[1] < kdim:
            raise ValueError(f"neighbor vectors must have at least {kdim} components.")
        rows = rows[:, :kdim]
        selected = []
        for row in rows:
            key = tuple((row.tolist() + [0, 0, 0])[:3])
            value = hoppings.get(key)
            if value is None:
                value = self.gen_hopping(key)
            selected.append(np.asarray(value, dtype=np.complex128))
        return np.asarray(selected, dtype=np.complex128)

    def _band_neighbors(
        self,
        hoppings: dict[tuple[int, int, int], np.ndarray],
    ) -> np.ndarray:
        """Return the configured interpolation range, or every available R vector."""

        configured = np.asarray(self.config.neighbor, dtype=int)
        if configured.size:
            if configured.ndim == 1:
                configured = configured.reshape(1, -1)
            return configured
        complete = [key for key in hoppings if key != (0, 0, 0)]
        return np.asarray(complete, dtype=int).reshape(-1, 3)

    def _h_of_k_factory(self, h0: np.ndarray, neigh: np.ndarray, hops: np.ndarray):
        config = self.config
        band_count = int(config.band_calc_num)
        avec = np.asarray(config.real_lattice_vectors, dtype=float)
        dim = avec.shape[1]
        delta_r = (neigh[:, :dim] @ avec) * float(config.lattice_const) if neigh.size else np.zeros((0, dim))
        nyq_mask = np.array([self.is_nyquist(r, self.state.k_shape) for r in neigh], dtype=bool) if neigh.size else np.array([], dtype=bool)
        sign = self.state.bloch_sign
        h0_hermitian = 0.5 * (h0 + h0.conj().T)

        def h_of_k(k_cart: np.ndarray) -> np.ndarray:
            if neigh.size == 0:
                return np.broadcast_to(h0_hermitian, (k_cart.shape[0], band_count, band_count)).copy()
            phase = np.exp(1j * sign * (k_cart @ delta_r.T))
            if np.any(~nyq_mask):
                hi = np.einsum("mn,nab->mab", phase[:, ~nyq_mask], hops[~nyq_mask])
            else:
                hi = np.zeros((k_cart.shape[0], band_count, band_count), dtype=np.complex128)
            if np.any(nyq_mask):
                nyquist = np.einsum("mn,nab->mab", phase[:, nyq_mask], hops[nyq_mask])
                hq = 0.5 * (nyquist + np.conjugate(np.swapaxes(nyquist, -2, -1)))
            else:
                hq = 0.0
            return h0_hermitian + hi + np.conjugate(np.swapaxes(hi, -2, -1)) + hq

        return h_of_k

    @staticmethod
    def is_nyquist(r, kshape) -> bool:
        shape = tuple(int(value) for value in kshape)
        coords = (list(map(int, r)) + [0] * len(shape))[: len(shape)]
        all_zero = True
        for axis, n_axis in enumerate(shape):
            val = coords[axis] % n_axis
            if val != 0:
                all_zero = False
            if n_axis % 2 == 0:
                if (2 * val) % n_axis != 0:
                    return False
            elif val != 0:
                return False
        return not all_zero

    @staticmethod
    def R_half_rect(kshape) -> np.ndarray:
        shape = tuple(int(n_axis) for n_axis in kshape)
        out = []
        for residues in product(*(range(n_axis) for n_axis in shape)):
            if all(value == 0 for value in residues):
                continue
            negative = tuple((-value) % n_axis for value, n_axis in zip(residues, shape))
            if residues != negative and residues > negative:
                continue
            signed = tuple(
                value if value <= n_axis // 2 else value - n_axis
                for value, n_axis in zip(residues, shape)
            )
            out.append((signed + (0, 0, 0))[:3])
        return np.asarray(out, dtype=int).reshape(-1, 3)

    @staticmethod
    def group_bands(energies: np.ndarray, delta_rel=1e-3, delta_abs=None):
        energies = np.asarray(energies)
        nk = int(np.prod(energies.shape[:-1]))
        nb = int(energies.shape[-1])
        flat = energies.reshape(nk, nb)
        span = float(flat.max() - flat.min())
        delta = float(delta_rel) * span
        if delta_abs is not None:
            delta = max(delta, float(delta_abs))
        mindiff = np.full((nb, nb), np.inf, dtype=float)
        denom = max(nb * nb, 1)
        block = max(1, int(64e6 // (8 * denom)))
        for start in range(0, nk, block):
            local = np.min(np.abs(flat[start : start + block, :, None] - flat[start : start + block, None, :]), axis=0)
            np.minimum(mindiff, local, out=mindiff)
        adjacency = mindiff <= delta
        np.fill_diagonal(adjacency, True)
        graph = scipy.sparse.csr_matrix(adjacency | adjacency.T)
        comp_count, labels = scipy.sparse.csgraph.connected_components(graph, directed=False)
        groups = [[] for _ in range(comp_count)]
        for band, label in enumerate(labels):
            groups[label].append(band)
        groups = [sorted(group) for group in groups if group]
        groups.sort(key=lambda group: group[0])
        return groups
