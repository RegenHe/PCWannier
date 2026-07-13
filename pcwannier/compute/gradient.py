from __future__ import annotations

import logging

import numpy as np
import scipy.linalg

from ..matrix_io import load_cell_matrix
from .matrix import MSet
from .parallel import parallel_map
from .state import StateCollection

LOGGER = logging.getLogger(__name__)


class Gradient:
    def __init__(self, state: StateCollection, mset: MSet, threads: int = 1):
        self.state = state
        self.mset = mset
        self.config = state.config
        self.threads = threads
        band_count = int(self.config.band_calc_num)
        self.U = self.state.gen_matrix_on_kmesh(lambda *_: np.eye(band_count, dtype=np.complex128))
        self.G = self.state.gen_matrix_on_kmesh(lambda *_: np.zeros((band_count, band_count), dtype=np.complex128))
        self.dW = self.state.gen_matrix_on_kmesh(lambda *_: np.zeros((band_count, band_count), dtype=np.complex128))
        self.omega = np.array([np.nan, np.nan, np.nan], dtype=float)
        self.epsilon = 0.01
        self.rn = np.zeros((self.config.kdim, band_count), dtype=np.complex128)

    def iter(self, err_diff: float, max_iter: int, epsilon: float = 0.01) -> None:
        if "U" in self.config.use_cached_data:
            path = self.config.input_path(self.config.U_file)
            if path is None:
                raise ValueError("U cache requested, but U_file is disabled.")
            self.U = load_cell_matrix(path, self.state.k_shape)
            self._validate_cached_u()
        self.epsilon = epsilon
        last_omega = np.inf
        err = np.inf
        if max_iter == 0:
            self.evaluate_current()
            return
        self.mset.update(self.U)
        for iteration in range(max_iter):
            self.calc()
            gradient_norm = max(float(np.linalg.norm(self.G[idx], ord="fro")) for idx in self.state.k_indices())
            self.mset.update(self.U)
            self.update()
            total = float(np.sum(self.omega))
            if not np.isfinite(total) or not np.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Gradient optimization produced a non-finite value at iteration {iteration + 1}: "
                    f"omega={total}, gradient_norm={gradient_norm}."
                )
            err = abs(last_omega - total)
            LOGGER.info(
                "gradient iter %s omega=%s omega_I=%s omega_OD=%s omega_D=%s err=%s max_gradient_norm=%s",
                iteration + 1,
                total,
                float(self.omega[0]),
                float(self.omega[1]),
                float(self.omega[2]),
                err,
                gradient_norm,
            )
            if err < err_diff:
                break
            if np.isfinite(last_omega) and total > last_omega + max(err_diff, abs(last_omega) * 1e-12):
                self.epsilon *= 0.5
                LOGGER.warning("Omega increased; gradient step reduced to %s", self.epsilon)
            if err < self.epsilon * 1e-1:
                self.epsilon *= 0.1
            last_omega = total
        if err > err_diff:
            LOGGER.warning("Gradient iteration reached the limit with err=%s", err)
        self.update()

    def evaluate_current(self) -> None:
        """Evaluate spread diagnostics without changing the current gauge."""
        self.mset.update(self.U)
        self.update()
        self.calc(is_update=False)

    def calc(self, is_update: bool = True) -> None:
        band_count = int(self.config.band_calc_num)
        b_count = len(self.config.composition_of_b)
        self.generateRn()

        def calc_idx(idx):
            i, j, k = idx
            gmat = np.zeros((band_count, band_count), dtype=np.complex128)
            for b in range(b_count):
                mmat = self.mset.get(i, j, k, b)
                diag = self._checked_diagonal(mmat, (i, j, k), b)
                mr = mmat * np.conj(diag)[None, :]
                phase = np.imag(np.log(diag)) + np.dot(self.config.b_vectors[b, :], self.rn)
                mt = (mmat / diag[None, :]) * phase[None, :]
                gmat += self.config.wb[b] * (self.operator_A(mr) - self.operator_S(mt))
            gmat *= 4
            dw = self.epsilon * gmat
            if not np.all(np.isfinite(dw)):
                raise FloatingPointError(f"Non-finite gradient update at k={idx}.")
            step_norm = float(np.linalg.norm(dw, ord="fro"))
            if step_norm > 100.0:
                raise FloatingPointError(
                    f"Gradient step is too large at k={idx} (Frobenius norm={step_norm:.6g}); "
                    "reduce epsilon."
                )
            umat = self.U[i, j, k]
            if is_update:
                if not np.isclose(abs(np.trace(umat @ umat.conj().T)), band_count, rtol=1e-8):
                    u, _, vh = np.linalg.svd(umat)
                    umat = u @ vh
                umat = umat @ scipy.linalg.expm(dw)
            return idx, gmat, dw, umat

        for idx, gmat, dw, umat in parallel_map(self.state.k_indices(), calc_idx, self.threads):
            self.G[idx] = gmat
            self.dW[idx] = dw
            self.U[idx] = umat

    def generateRn(self):
        band_count = int(self.config.band_calc_num)
        b_count = len(self.config.composition_of_b)
        rn = np.zeros((self.config.kdim, band_count), dtype=np.complex128)

        def calc_idx(idx):
            i, j, k = idx
            local = np.zeros((self.config.kdim, band_count), dtype=np.complex128)
            for b in range(b_count):
                mmat = self.mset.get(i, j, k, b)
                diag = self._checked_diagonal(mmat, (i, j, k), b)
                local -= (
                    self.config.wb[b]
                    * self.config.b_vectors[b, :, None]
                    * np.imag(np.log(diag))[None, :]
                )
            return local

        for local in parallel_map(self.state.k_indices(), calc_idx, self.threads):
            rn += local
        self.rn = rn / self.state.get_k_num()
        return self.rn

    def update(self) -> None:
        band_count = int(self.config.band_calc_num)
        b_count = len(self.config.composition_of_b)
        self.generateRn()
        omega = np.zeros(3, dtype=np.complex128)

        def calc_idx(idx):
            i, j, k = idx
            local = np.zeros(3, dtype=np.complex128)
            for b in range(b_count):
                mmat = self.mset.get(i, j, k, b)
                diag = self._checked_diagonal(mmat, (i, j, k), b)
                abs2 = np.abs(mmat) ** 2
                diag_abs2 = np.abs(diag) ** 2
                temp_i = band_count - np.sum(abs2)
                temp_od = np.sum(abs2) - np.sum(diag_abs2)
                temp_d = np.sum(
                    (
                        -np.imag(np.log(diag))
                        - np.dot(self.config.b_vectors[b, :], self.rn)
                    )
                    ** 2
                )
                local[0] += temp_i * self.config.wb[b]
                local[1] += temp_od * self.config.wb[b]
                local[2] += temp_d * self.config.wb[b]
            return local

        for local in parallel_map(self.state.k_indices(), calc_idx, self.threads):
            omega += local
        self.omega = np.real(omega) / self.state.get_k_num()

    @staticmethod
    def operator_A(a):
        return (a - np.conj(a).T) / 2

    @staticmethod
    def operator_S(a):
        return (a + np.conj(a).T) / (2j)

    @staticmethod
    def _checked_diagonal(matrix: np.ndarray, k_index: tuple[int, int, int], direction: int) -> np.ndarray:
        diag = np.diag(matrix)
        scale = max(float(np.linalg.norm(matrix, ord="fro")), 1.0)
        threshold = np.finfo(float).eps * scale * 128.0
        if not np.all(np.isfinite(diag)) or np.any(np.abs(diag) <= threshold):
            minimum = float(np.min(np.abs(diag))) if diag.size else 0.0
            raise FloatingPointError(
                f"M diagonal is singular or non-finite at k={k_index}, direction={direction} "
                f"(min_abs={minimum:.6g}, threshold={threshold:.6g})."
            )
        return diag

    def _validate_cached_u(self) -> None:
        band_count = int(self.config.band_calc_num)
        identity = np.eye(band_count)
        for idx in self.state.k_indices():
            matrix = np.asarray(self.U[idx], dtype=np.complex128)
            if matrix.shape != (band_count, band_count):
                raise ValueError(f"Cached U matrix at k={idx} has shape {matrix.shape}; expected {identity.shape}.")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"Cached U matrix at k={idx} contains non-finite values.")
            residual = np.linalg.norm(matrix.conj().T @ matrix - identity, ord="fro")
            if residual > 1e-6:
                raise ValueError(f"Cached U matrix at k={idx} is not unitary (residual={residual:.6g}).")
