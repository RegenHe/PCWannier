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
from .kspace import get_kxyz
from .parallel import parallel_map


class TBAModel:
    def __init__(self, ctx: CalculationContext, threads: int = 1):
        self.ctx = ctx
        self.config = ctx.config
        self.state = ctx.state
        self.threads = max(1, int(threads))
        self.hoppings: list[np.ndarray] | None = None
        self._projected_hamiltonians: np.ndarray | None = None
        self._projected_k_cart: np.ndarray | None = None

    def gen_hopping(self, r: list[int] | tuple[int, ...] | None = None) -> np.ndarray:
        if r is None:
            r = [0, 0, 0]
        config = self.config
        state = self.state
        kdim = int(config.kdim)
        avec = np.asarray(config.real_lattice_vectors)
        dim = avec.shape[1]
        r_use = (list(r) + [0, 0, 0])[:kdim]
        r_cart = np.zeros(dim, dtype=float)
        for axis in range(kdim):
            r_cart += r_use[axis] * avec[axis, :]
        r_cart *= float(config.lattice_const)

        band_count = int(config.band_calc_num)
        sign = state.bloch_sign
        k_cart, projected = self._projected_k_hamiltonians()
        phase = np.exp(1j * (-sign) * (k_cart @ r_cart))
        return np.sum(projected * phase[:, None, None], axis=0) / state.get_k_num()

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
            energy = np.asarray(state.E[i, j, k], dtype=np.complex128)
            projected[pos] = np.conj(umat).T @ (energy[:, None] * umat)
            k_cart[pos] = get_kxyz(config, [i, j, k])[:dim]
        self._projected_k_cart = k_cart
        self._projected_hamiltonians = projected
        return k_cart, projected

    def output_spectrum_diagnostics(self, symmetry_analysis=None) -> OutputSpectrumDiagnostics | None:
        """Compare the output-basis Hamiltonian with isolated FEM eigenvalues."""
        _, projected = self._projected_k_hamiltonians()
        band_count = int(self.config.band_calc_num)
        indices = tuple(self.state.k_indices())
        if any(len(np.asarray(self.state.E[index]).reshape(-1)) != band_count for index in indices):
            return None

        raw = np.asarray(
            [np.sort(np.real(np.asarray(self.state.E[index], dtype=np.complex128))) for index in indices]
        )
        output = np.linalg.eigvalsh(self._hermitian_batch(projected))
        errors = np.max(np.abs(output - raw), axis=1)
        worst = int(np.argmax(errors))
        basis = (
            self.config.symmetry_output_basis
            if self.config.symmetry_constrained
            else ("fem" if self.config.disable_orth else "strict")
        )
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
        neighbors = np.asarray(self.config.neighbor, dtype=int)
        hop_array = np.asarray(
            [hoppings[tuple((list(row) + [0, 0, 0])[:3])] for row in neighbors],
            dtype=np.complex128,
        )
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
            raw_energies = np.real(np.asarray(self.state.E[state_index], dtype=np.complex128))
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

    @staticmethod
    def _hermitian_batch(matrices: np.ndarray) -> np.ndarray:
        array = np.asarray(matrices, dtype=np.complex128)
        return 0.5 * (array + np.conjugate(np.swapaxes(array, -2, -1)))

    def collect_hoppings(self) -> dict[tuple[int, int, int], np.ndarray]:
        if not self.config.neighbor:
            self.config.neighbor = self.R_half_rect(self.state.k_shape).tolist()
        self._projected_k_hamiltonians()

        r_list = [(0, 0, 0)] + [tuple((list(r) + [0, 0, 0])[:3]) for r in self.config.neighbor]

        def calc_r(r3):
            return tuple(int(x) for x in r3), self.gen_hopping(r3)

        out = {}
        for key, hopping in parallel_map(r_list, calc_r, self.threads):
            out[key] = hopping
        return out

    def gen_hs_bands(self, hoppings: dict[tuple[int, int, int], np.ndarray]) -> BandResult:
        config = self.config
        kdim = int(config.kdim)
        high_sym_points = []
        k_list_parts = [np.array(config.k_path[0]["point"], dtype=float)[:kdim]]
        total = 0
        first_point = np.asarray(config.k_path[0]["point"], dtype=float)[:kdim]
        last_point = np.asarray(config.k_path[-1]["point"], dtype=float)[:kdim]
        endpoints_equivalent = self._periodically_equivalent(first_point, last_point)
        for idx, point in enumerate(config.k_path):
            high_sym_points.append([point["name"], total])
            if idx == len(config.k_path) - 1 and endpoints_equivalent:
                break
            num = int(point["num"])
            start = np.asarray(point["point"], dtype=float)[:kdim]
            stop = np.asarray(config.k_path[(idx + 1) % len(config.k_path)]["point"], dtype=float)[:kdim]
            if num > 0:
                seg = np.stack([np.linspace(start[axis], stop[axis], num + 1)[1:] for axis in range(kdim)], axis=1)
                k_list_parts.append(seg)
            total += num
        if not endpoints_equivalent:
            high_sym_points.append([config.k_path[0]["name"], total])
        k_path = np.vstack(k_list_parts)
        k_axis = np.arange(0, total + 1)

        h0 = np.asarray(hoppings[(0, 0, 0)], dtype=np.complex128)
        neigh = np.asarray(config.neighbor, dtype=int)
        hops = np.asarray([hoppings[tuple((list(r) + [0, 0, 0])[:3])] for r in neigh], dtype=np.complex128)
        h_of_k = self._h_of_k_factory(h0, neigh, hops)
        hks = h_of_k(self._kfrac_to_kcart(k_path))
        energies = np.linalg.eigvalsh(hks) if config.hermitian else np.sort(np.linalg.eigvals(hks))

        dos_energy = None
        dos_components = None
        if config.DOS in (1, 2, 3):
            dos_energy, dos_components = self._calculate_dos(h_of_k, kdim)

        return BandResult(k_path, k_axis, high_sym_points, energies, dos_energy, dos_components)

    @staticmethod
    def _periodically_equivalent(first: np.ndarray, last: np.ndarray, atol: float = 1e-10) -> bool:
        difference = np.asarray(last, dtype=float) - np.asarray(first, dtype=float)
        return bool(np.allclose(difference, np.rint(difference), rtol=0.0, atol=atol))

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
        h0 = np.asarray(hoppings[(0, 0, 0)], dtype=np.complex128)
        neigh = np.asarray(config.neighbor, dtype=int)
        hops = np.asarray([hoppings[tuple((list(r) + [0, 0, 0])[:3])] for r in neigh], dtype=np.complex128)
        hks = self._h_of_k_factory(h0, neigh, hops)(self._kfrac_to_kcart(k_flat))
        eigvals, eigvecs = np.linalg.eigh(hks)
        result.bz_eigvals = eigvals.reshape(*nk_shape, int(config.band_calc_num))
        result.bz_eigvecs = eigvecs.reshape(*nk_shape, int(config.band_calc_num), int(config.band_calc_num))
        result.groups = self.group_bands(result.bz_eigvals, delta_rel=1e-2)

    def _kfrac_to_kcart(self, kfrac: np.ndarray) -> np.ndarray:
        reciprocal = np.asarray(self.config.reciprocal_lattice_vectors, dtype=float)[: self.config.kdim, :]
        return (kfrac @ reciprocal) * (2.0 * np.pi / float(self.config.lattice_const))

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
        coords = list(map(int, r))
        all_zero = True
        for axis, n_axis in enumerate(kshape):
            val = coords[axis] % int(n_axis)
            if val != 0:
                all_zero = False
            if int(n_axis) % 2 == 0:
                if (2 * val) % int(n_axis) != 0:
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
        return np.asarray(out, dtype=int)

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
