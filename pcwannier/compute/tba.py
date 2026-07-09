from __future__ import annotations

from itertools import product

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

from ..data import BandResult
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
        sign = -1 if config.dataset_type.lower() == "comsol" else 1
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
        transform = state.get_transform(True if config.disable_orth else False)
        for pos, (i, j, k) in enumerate(state.k_indices()):
            umat = transform[i, j, k] @ self.ctx.initializer.matV[i, j, k] @ self.ctx.gradient.U[i, j, k]
            energy = np.asarray(state.E[i, j, k], dtype=np.complex128)
            projected[pos] = np.conj(umat).T @ (energy[:, None] * umat)
            k_cart[pos] = get_kxyz(config, [i, j, k])[:dim]
        self._projected_k_cart = k_cart
        self._projected_hamiltonians = projected
        return k_cart, projected

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
        for idx, point in enumerate(config.k_path):
            high_sym_points.append([point["name"], total])
            num = int(point["num"])
            start = np.asarray(point["point"], dtype=float)[:kdim]
            stop = np.asarray(config.k_path[(idx + 1) % len(config.k_path)]["point"], dtype=float)[:kdim]
            if num > 0:
                seg = np.stack([np.linspace(start[axis], stop[axis], num + 1)[1:] for axis in range(kdim)], axis=1)
                k_list_parts.append(seg)
            total += num
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
        if config.DOS in (1, 2):
            dos_energy = np.linspace(np.min(np.real(energies)), np.max(np.real(energies)), int(config.DOS_num))
            comp_count = 1 if config.DOS == 1 else int(config.band_calc_num)
            dos_components = np.zeros((comp_count, int(config.DOS_num)), dtype=np.complex128)
            ident = np.eye(int(config.band_calc_num), dtype=np.complex128)
            for eidx, energy in enumerate(dos_energy):
                green = np.linalg.inv(hks - (energy - 1j * config.DOS_eps) * ident)
                diag = np.real((-1 / np.pi) * np.imag(np.diagonal(green, axis1=-2, axis2=-1)))
                if config.DOS == 1:
                    dos_components[0, eidx] = diag.sum()
                else:
                    dos_components[:, eidx] = diag.sum(axis=0)
        elif config.DOS == 3:
            mesh = np.array(config.DOS_Brillouin_mesh, dtype=int)[:kdim]
            axes = [np.arange(m, dtype=float) / m for m in mesh]
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, kdim)
            hgrid = h_of_k(self._kfrac_to_kcart(grid))
            dos_energy = np.linspace(np.min(np.real(energies)), np.max(np.real(energies)), int(config.DOS_num))
            dos_components = np.zeros((1, int(config.DOS_num)), dtype=np.complex128)
            ident = np.eye(int(config.band_calc_num), dtype=np.complex128)
            for eidx, energy in enumerate(dos_energy):
                green = np.linalg.inv(hgrid - (energy - 1j * config.DOS_eps) * ident)
                dos_components[0, eidx] = (-1 / np.pi) * np.imag(np.diagonal(green, axis1=-2, axis2=-1)).sum()

        return BandResult(k_path, k_axis, high_sym_points, energies, dos_energy, dos_components)

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

        def h_of_k(k_cart: np.ndarray) -> np.ndarray:
            if neigh.size == 0:
                return np.broadcast_to(h0, (k_cart.shape[0], band_count, band_count)).copy()
            phase = np.exp(1j * (k_cart @ delta_r.T))
            if np.any(~nyq_mask):
                hi = np.einsum("mn,nab->mab", phase[:, ~nyq_mask], hops[~nyq_mask])
            else:
                hi = np.zeros((k_cart.shape[0], band_count, band_count), dtype=np.complex128)
            hq = np.einsum("mn,nab->mab", phase[:, nyq_mask], hops[nyq_mask]) if np.any(nyq_mask) else 0.0
            return h0 + hi + np.conjugate(np.swapaxes(hi, -2, -1)) + hq

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
        ranges = []
        for n_axis in kshape:
            half = int(n_axis) // 2
            lo = -half + (1 if int(n_axis) % 2 == 0 else 0)
            ranges.append(range(lo, half + 1))
        out = []
        for coords in product(*ranges):
            if all(c == 0 for c in coords):
                continue
            keep = False
            for coord in coords:
                if coord != 0:
                    keep = coord > 0
                    break
            if keep:
                out.append((coords + (0, 0, 0))[:3])
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
