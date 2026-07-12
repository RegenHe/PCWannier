from __future__ import annotations

import logging

import numpy as np

from ..matrix_io import load_cell_matrix
from .integration import integrate_overlap_matrix, integrate_weighted_abs2_columns, validated_real
from .kspace import neighbor_reciprocal_lattice_vectors
from .matrix import MSet
from .parallel import parallel_map
from .state import StateCollection

LOGGER = logging.getLogger(__name__)


class StateBases:
    @staticmethod
    def Angular(l: int = 0):
        table = {
            0: StateBases.s,
            1: StateBases.px,
            -1: StateBases.py,
            2: StateBases.dx2_y2,
            -2: StateBases.dxy,
            3: StateBases.fxx2_3y2,
            -3: StateBases.fy3x2_y2,
        }
        if l not in table:
            raise ValueError(f"Invalid angular state: {l}")
        return table[l]

    @staticmethod
    def s(phi):
        return 1 / np.sqrt(4 * np.pi)

    @staticmethod
    def px(phi):
        return np.sqrt(3 / (4 * np.pi)) * np.cos(phi)

    @staticmethod
    def py(phi):
        return np.sqrt(3 / (4 * np.pi)) * np.sin(phi)

    @staticmethod
    def dx2_y2(phi):
        return np.sqrt(15 / (16 * np.pi)) * np.cos(2 * phi)

    @staticmethod
    def dxy(phi):
        return np.sqrt(15 / (16 * np.pi)) * np.sin(2 * phi)

    @staticmethod
    def fxx2_3y2(phi):
        return np.sqrt(35 / (32 * np.pi)) * (np.cos(phi) ** 2 - 3 * np.sin(phi) ** 2) * np.cos(phi)

    @staticmethod
    def fy3x2_y2(phi):
        return np.sqrt(35 / (32 * np.pi)) * (3 * np.cos(phi) ** 2 - np.sin(phi) ** 2) * np.sin(phi)

    @staticmethod
    def Radial(n: int, l: int | None = None):
        if l is None:
            l = n - 1
        name = f"r{n}{abs(l)}"
        try:
            return getattr(StateBases, name)
        except AttributeError as exc:
            raise ValueError(f"No radial function defined for (n={n}, l={l}).") from exc

    @staticmethod
    def r10(r, alpha=1.0):
        return 2 * alpha ** (3 / 2) * np.exp(-alpha * r / 2)

    @staticmethod
    def r20(r, alpha=1.0):
        return 1 / np.sqrt(2) * alpha ** (3 / 2) * (1 - 0.5 * alpha * r) * np.exp(-alpha * r / 2)

    @staticmethod
    def r21(r, alpha=1.0):
        return 1 / (2 * np.sqrt(6)) * alpha ** (3 / 2) * alpha * r * np.exp(-alpha * r / 2)

    @staticmethod
    def r30(r, alpha=1.0):
        return np.sqrt(4 / 27) * alpha ** (3 / 2) * (1 - 2 * alpha * r / 3 + 2 * alpha**2 * r**2 / 27) * np.exp(-alpha * r / 3)

    @staticmethod
    def r31(r, alpha=1.0):
        return 8 / (27 * np.sqrt(6)) * alpha ** (3 / 2) * (1 - alpha * r / 6) * alpha * r * np.exp(-alpha * r / 3)

    @staticmethod
    def r32(r, alpha=1.0):
        return 4 / (81 * np.sqrt(30)) * alpha ** (3 / 2) * alpha**2 * r**2 * np.exp(-alpha * r / 3)

    @staticmethod
    def r40(r, alpha=1.0):
        return 0.25 * alpha ** (3 / 2) * (1 - 0.75 * alpha * r + 0.125 * alpha**2 * r**2 - alpha**3 * r**3 / 192) * np.exp(-alpha * r / 4)

    @staticmethod
    def r41(r, alpha=1.0):
        return 5 / (16 * np.sqrt(15)) * alpha ** (3 / 2) * (1 - alpha * r / 4 + alpha**2 * r**2 / 80) * alpha * r * np.exp(-alpha * r / 4)

    @staticmethod
    def r42(r, alpha=1.0):
        return 1 / (64 * np.sqrt(5)) * alpha ** (3 / 2) * (1 - alpha * r / 12) * alpha**2 * r**2 * np.exp(-alpha * r / 4)

    @staticmethod
    def r43(r, alpha=1.0):
        return 1 / (768 * np.sqrt(35)) * alpha ** (3 / 2) * alpha**3 * r**3 * np.exp(-alpha * r / 4)


class StateInitializer:
    def __init__(self, state: StateCollection, mset: MSet, threads: int = 1):
        self.state = state
        self.mset = mset
        self.config = state.config
        self.threads = threads
        band_count = int(self.config.band_calc_num)
        factory = lambda i, j, k: np.zeros((len(self.state.E_idx[i, j, k]), band_count), dtype=np.complex128)
        self.matC = self.state.gen_matrix_on_kmesh(factory)
        self.matZ = self.state.gen_matrix_on_kmesh(factory)
        self.last_matZ = self.state.gen_matrix_on_kmesh(lambda *_: None)
        self.lambda_ = self.state.gen_matrix_on_kmesh(lambda *_: None)
        self.diagII_sum = self.state.gen_matrix_on_kmesh(lambda *_: 0.0)
        self.matA = self.state.gen_matrix_on_kmesh(factory)
        self.matV = self.state.gen_matrix_on_kmesh(factory)
        self.I_idx = self.state.gen_matrix_on_kmesh(lambda *_: None)
        self.O_idx = self.state.gen_matrix_on_kmesh(lambda *_: None)
        self.alpha = 0.5

    def iter(self, err_diff: float, max_iter: int) -> None:
        has_cached_a = "A" in self.config.use_cached_data
        has_cached_v = "V" in self.config.use_cached_data
        if has_cached_a:
            path = self.config.input_path(self.config.A_file)
            if path is None:
                raise ValueError("A cache requested, but A_file is disabled.")
            self.matA = load_cell_matrix(path, self.state.k_shape)
            self._validate_cached_matrix("A", self.matA, require_semiunitary=False)
        if has_cached_v:
            path = self.config.input_path(self.config.V_file)
            if path is None:
                raise ValueError("V cache requested, but V_file is disabled.")
            self.matV = load_cell_matrix(path, self.state.k_shape)
            self._validate_cached_matrix("V", self.matV, require_semiunitary=True)

        if has_cached_a and not has_cached_v:
            band_count = int(self.config.band_calc_num)
            for i, j, k in self.state.k_indices():
                u, _, vh = np.linalg.svd(self.matA[i, j, k])
                self.matC[i, j, k] = u @ np.eye(len(self.state.E_idx[i, j, k]), band_count) @ vh
            self.matV = self.matC.copy()
        elif not has_cached_v:
            self.projection()
            if self.config.inner_window is False:
                self.matV = self.matC.copy()

        if has_cached_a or has_cached_v:
            self.set_window_indices()
        min_len, max_len = self.get_min_max_len_idx(self.state.E_idx)
        if min_len == self.config.band_calc_num == max_len or not self.config.proj_iter:
            self.mset.initial(self.matV)
            return

        last_omega = np.inf
        omega = np.inf
        for idx in range(max_iter):
            self.update_Z()
            self.sort_Z()
            omega = self.get_omega_I()
            LOGGER.info("initializer iter %s omega=%s err=%s", idx, abs(omega), abs(omega - last_omega))
            if abs(omega - last_omega) < err_diff:
                break
            last_omega = omega
        if has_cached_a or not has_cached_v:
            band_count = int(self.config.band_calc_num)
            for i, j, k in self.state.k_indices():
                tmat = np.conj(self.matV[i, j, k]).T @ self.matA[i, j, k]
                u, _, vh = np.linalg.svd(tmat)
                self.matV[i, j, k] = self.matV[i, j, k] @ (u @ np.eye(band_count, band_count) @ vh)
            if self.config.v_proj:
                self.V_projection()
        else:
            LOGGER.info("V-only cache: projection-gauge alignment skipped because A is unavailable")
        self.mset.initial(self.matV)

    def projection(self) -> None:
        if self.state.extention_mesh is None or self.state.extention_epsilon is None:
            raise ValueError("StateCollection must be extended before projection.")
        band_count = int(self.config.band_calc_num)
        min_len, _ = self.get_min_max_len_idx(self.state.E_idx)
        if band_count > min_len:
            raise ValueError(f"Calculated bands exceed band window: {band_count} > {min_len}.")

        h_columns = []
        for projection in self.config.projections:
            frac = projection["frac_position"]
            cart_position = (
                frac[0] * np.asarray(self.config.real_lattice_vectors[0])
                + frac[1] * np.asarray(self.config.real_lattice_vectors[1])
                + np.asarray(self.config.origin)
            ) * float(self.config.lattice_const)
            for state_spec in projection["states"]:
                if isinstance(state_spec, dict) and "lc_states" in state_spec:
                    lc_states = state_spec["lc_states"]
                    lc_coeffs = state_spec["lc_coeffs"]

                    def fn(r, phi, _states=lc_states, _coeffs=lc_coeffs):
                        total = 0.0 + 0.0j
                        for (n, l, z), coeff in zip(_states, _coeffs):
                            rr = r / float(self.config.lattice_const)
                            total += coeff * StateBases.Radial(n, l)(rr, z) * StateBases.Angular(l)(phi)
                        return total

                else:
                    n, l, z = state_spec

                    def fn(r, phi, _n=n, _l=l, _z=z):
                        rr = r / float(self.config.lattice_const)
                        return StateBases.Radial(_n, _l)(rr, _z) * StateBases.Angular(_l)(phi)

                h_columns.append(self.state.extention_mesh.rfunc(fn, cart_position, projection["xaxis_angluar"]))

        hmat = np.column_stack(h_columns)
        norms = np.atleast_1d(
            integrate_weighted_abs2_columns(
                self.state.extention_mesh,
                self.state.extention_epsilon,
                hmat,
                chunk_size=2048,
                backend=self.state.compute_backend,
                mode=self.config.integration_mode,
            )
        )
        norms = validated_real(norms, "projection basis norms")
        norms = np.where(norms == 0, 1.0, norms)
        gmat = hmat / np.sqrt(norms)[None, :]

        def calc_idx(idx):
            i, j, k = idx
            phase = self.state.get_extention_phase(i, j, k)
            fields = self.state.get_extention_block(i, j, k)
            fields *= phase[None, :]
            amat = integrate_overlap_matrix(
                self.state.extention_integral_view,
                fields,
                gmat.T,
                self.state.extention_epsilon,
                chunk_size=64,
                backend=self.state.compute_backend,
                mode=self.config.integration_mode,
            )
            if self.config.proj_binarize:
                amat = self.binarize(amat)
            u, _, vh = np.linalg.svd(amat)
            cmat = u @ np.eye(len(self.state.E_idx[i, j, k]), band_count) @ vh
            return idx, amat, cmat

        for idx, amat, cmat in parallel_map(self.state.k_indices(), calc_idx, self.threads):
            self.matA[idx] = amat
            self.matC[idx] = cmat

        if self.config.inner_window is not False:
            self.inner_projection()
        else:
            self.set_window_indices()

    def set_window_indices(self) -> None:
        for i, j, k in self.state.k_indices():
            self.I_idx[i, j, k] = self.map_inner_to_local(self.state.E_idx[i, j, k], self.state.inner_E_idx[i, j, k])
            self.O_idx[i, j, k] = np.setdiff1d(
                np.arange(len(self.state.E_idx[i, j, k])),
                self.I_idx[i, j, k],
                assume_unique=True,
            )

    def inner_projection(self) -> None:
        band_count = int(self.config.band_calc_num)
        tol = 1e-10
        for i, j, k in self.state.k_indices():
            n_k = len(self.state.E_idx[i, j, k])
            self.I_idx[i, j, k] = self.map_inner_to_local(self.state.E_idx[i, j, k], self.state.inner_E_idx[i, j, k])
            self.O_idx[i, j, k] = np.setdiff1d(np.arange(n_k), self.I_idx[i, j, k], assume_unique=True)
            m_k = self.I_idx[i, j, k].size
            p = band_count - m_k
            if p < 0 or p > self.O_idx[i, j, k].size:
                raise ValueError("Inner window is incompatible with the projection band count.")
            u, s, _ = np.linalg.svd(self.matA[i, j, k], full_matrices=False)
            rank = min(int(np.sum(s > tol)), band_count)
            p_g = np.zeros((n_k, n_k), dtype=np.complex128) if rank == 0 else u[:, :rank] @ u[:, :rank].conj().T
            ui = np.eye(n_k, dtype=np.complex128)[:, self.I_idx[i, j, k]]
            p_g_oo = 0.5 * (
                p_g[np.ix_(self.O_idx[i, j, k], self.O_idx[i, j, k])]
                + p_g[np.ix_(self.O_idx[i, j, k], self.O_idx[i, j, k])].conj().T
            )
            evals, vecs = np.linalg.eigh(p_g_oo)
            order = np.argsort(evals)[::-1]
            uopt = np.zeros((n_k, p), dtype=np.complex128)
            uopt[self.O_idx[i, j, k], :] = vecs[:, order][:, :p]
            self.matV[i, j, k] = np.concatenate([ui, uopt], axis=1)

    def V_projection(self) -> None:
        for i, j, k in self.state.k_indices():
            amat = self.matV[i, j, k].conj().T @ self.matA[i, j, k]
            u, _, vh = np.linalg.svd(amat)
            self.matV[i, j, k] = self.matV[i, j, k] @ (u @ np.eye(u.shape[0], vh.shape[0]) @ vh)

    def update_Z(self) -> None:
        b_count = len(self.config.composition_of_b)

        def calc_idx(idx):
            i, j, k = idx
            zmat = np.zeros((len(self.O_idx[i, j, k]), len(self.O_idx[i, j, k])), dtype=np.complex128)
            diag_ii = 0.0
            for b in range(b_count):
                mmat = self.mset.get_M0(i, j, k, b)
                ik, _ = neighbor_reciprocal_lattice_vectors(self.config, [i, j, k], b)
                cb = mmat @ self.matV[ik]
                cb_o = cb[self.O_idx[i, j, k], :]
                zmat += self.config.wb[b] * (cb_o @ cb_o.conj().T)
                if self.I_idx[i, j, k].size > 0:
                    ci = cb[self.I_idx[i, j, k], :]
                    diag_ii += self.config.wb[b] * np.sum(np.abs(ci) ** 2)
            zmat = 0.5 * (zmat + zmat.conj().T)
            if self.last_matZ[i, j, k] is None:
                next_last = zmat
            else:
                zmat = self.alpha * zmat + (1 - self.alpha) * self.last_matZ[i, j, k]
                next_last = zmat
            return idx, zmat, next_last, diag_ii

        for idx, zmat, next_last, diag_ii in parallel_map(self.state.k_indices(), calc_idx, self.threads):
            self.matZ[idx] = zmat
            self.last_matZ[idx] = next_last
            self.diagII_sum[idx] = diag_ii

    def sort_Z(self) -> None:
        band_count = int(self.config.band_calc_num)

        def calc_idx(idx):
            i, j, k = idx
            evals, vecs = np.linalg.eigh(self.matZ[i, j, k])
            p = band_count - len(self.I_idx[i, j, k])
            vp = vecs[:, -p:] if p > 0 else np.zeros((self.matZ[i, j, k].shape[0], 0), dtype=np.complex128)
            lambda_value = float(np.sum(evals[-p:])) if p > 0 else 0.0
            n_k = len(self.state.E_idx[i, j, k])
            uopt = np.zeros((n_k, p), dtype=np.complex128)
            uopt[self.O_idx[i, j, k], :] = vp
            ei = np.eye(n_k, dtype=np.complex128)[:, self.I_idx[i, j, k]] if self.I_idx[i, j, k].size else np.zeros((n_k, 0), dtype=np.complex128)
            return idx, lambda_value, np.concatenate([ei, uopt], axis=1)[:, :band_count]

        for idx, lambda_value, matv in parallel_map(self.state.k_indices(), calc_idx, self.threads):
            self.lambda_[idx] = lambda_value
            self.matV[idx] = matv

    def get_omega_I(self):
        total = 0.0
        s_n_wb = self.config.band_calc_num * np.sum(self.config.wb)
        for idx in self.state.k_indices():
            total += s_n_wb - (self.diagII_sum[idx] + self.lambda_[idx])
        return total / self.state.get_k_num()

    @staticmethod
    def binarize(amat):
        strength = np.abs(np.asarray(amat)).astype(float)
        n, m = strength.shape
        order = np.argsort(strength.ravel(), kind="mergesort")[::-1]
        rows, cols = np.unravel_index(order, (n, m))
        used_r = np.zeros(n, dtype=bool)
        used_c = np.zeros(m, dtype=bool)
        out = np.zeros((n, m), dtype=np.complex128)
        for i, j in zip(rows, cols):
            if not used_r[i] and not used_c[j] and np.isfinite(strength[i, j]):
                out[i, j] = 1.0
                used_r[i] = True
                used_c[j] = True
        return out

    @staticmethod
    def map_inner_to_local(outer_idx, inner_idx):
        positions = {int(band_id): pos for pos, band_id in enumerate(np.asarray(outer_idx, dtype=int))}
        return np.unique([positions[int(band_id)] for band_id in np.asarray(inner_idx, dtype=int) if int(band_id) in positions])

    @staticmethod
    def get_min_max_len_idx(e_idx) -> tuple[int, int]:
        lengths = [len(e_idx[idx]) for idx in np.ndindex(e_idx.shape)]
        if not lengths:
            raise ValueError("E_idx is empty.")
        return min(lengths), max(lengths)

    def _validate_cached_matrix(self, name: str, matrix: np.ndarray, *, require_semiunitary: bool) -> None:
        band_count = int(self.config.band_calc_num)
        for idx in self.state.k_indices():
            cell = np.asarray(matrix[idx], dtype=np.complex128)
            expected = (len(self.state.E_idx[idx]), band_count)
            if cell.shape != expected:
                raise ValueError(f"Cached {name} matrix at k={idx} has shape {cell.shape}; expected {expected}.")
            if not np.all(np.isfinite(cell)):
                raise ValueError(f"Cached {name} matrix at k={idx} contains non-finite values.")
            if require_semiunitary:
                residual = np.linalg.norm(cell.conj().T @ cell - np.eye(band_count), ord="fro")
                if residual > 1e-6:
                    raise ValueError(f"Cached {name} matrix at k={idx} is not semi-unitary (residual={residual:.6g}).")
