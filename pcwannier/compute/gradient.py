from __future__ import annotations

import logging

import numpy as np
import scipy.linalg

from ..matrix_io import load_cell_matrix
from .kspace import get_kxyz
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
        self.epsilon = epsilon
        last_omega = np.inf
        err = np.inf
        if max_iter == 0:
            self.mset.update(self.U)
            self.update()
            self.calc(is_update=False)
            return
        self.mset.update(self.U)
        for iteration in range(max_iter):
            self.calc()
            self.mset.update(self.U)
            self.update()
            total = float(np.sum(self.omega))
            err = abs(last_omega - total)
            LOGGER.info("gradient iter %s omega=%s err=%s", iteration + 1, total, err)
            if err < err_diff:
                break
            if err < self.epsilon * 1e-1:
                self.epsilon *= 0.1
            last_omega = total
        if err > err_diff:
            LOGGER.warning("Gradient iteration reached the limit with err=%s", err)
        self.update()

    def calc(self, is_update: bool = True) -> None:
        band_count = int(self.config.band_calc_num)
        b_count = len(self.config.composition_of_b)
        self.generateRn()

        def calc_idx(idx):
            i, j, k = idx
            gmat = np.zeros((band_count, band_count), dtype=np.complex128)
            for b in range(b_count):
                mmat = self.mset.get(i, j, k, b)
                mr = np.zeros((band_count, band_count), dtype=np.complex128)
                mt = np.zeros((band_count, band_count), dtype=np.complex128)
                for m in range(band_count):
                    for n in range(band_count):
                        mr[m, n] = mmat[m, n] * np.conj(mmat[n, n])
                        mt[m, n] = (
                            mmat[m, n]
                            / mmat[n, n]
                            * (np.imag(np.log(mmat[n, n])) + np.dot(self.config.b_vectors[b, :], self.rn[:, n]))
                        )
                gmat += self.config.wb[b] * (self.operator_A(mr) - self.operator_S(mt))
            gmat *= 4
            dw = self.epsilon * gmat
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
                for n in range(band_count):
                    local[:, n] -= self.config.wb[b] * self.config.b_vectors[b, :] * np.imag(np.log(mmat[n, n]))
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
                temp_i = band_count
                temp_od = 0
                temp_d = 0
                for m in range(band_count):
                    for n in range(band_count):
                        temp_i -= np.abs(mmat[m, n]) ** 2
                        if m != n:
                            temp_od += np.abs(mmat[m, n]) ** 2
                    temp_d += (
                        -np.imag(np.log(mmat[m, m])) - np.dot(self.config.b_vectors[b, :], self.rn[:, m])
                    ) ** 2
                local[0] += temp_i * self.config.wb[b]
                local[1] += temp_od * self.config.wb[b]
                local[2] += temp_d * self.config.wb[b]
            return local

        for local in parallel_map(self.state.k_indices(), calc_idx, self.threads):
            omega += local
        self.omega = np.real(omega) / self.state.get_k_num()

    def set_center(self, center):
        self.generateRn()
        for i, j, k in self.state.k_indices():
            rmat = (self.rn.T - center) @ np.asarray(self.config.real_lattice_vectors) * float(self.config.lattice_const)
            kxyz = get_kxyz(self.config, [i, j, k])
            sign = -1 if self.config.dataset_type.lower() == "comsol" else 1
            phase = np.diag(np.exp(-1j * sign * np.dot(kxyz[: self.config.kdim], rmat.T)))
            self.U[i, j, k] = phase @ self.U[i, j, k]
        self.mset.update(self.U)

    @staticmethod
    def operator_A(a):
        return (a - np.conj(a).T) / 2

    @staticmethod
    def operator_S(a):
        return (a + np.conj(a).T) / (2j)
