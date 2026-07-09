from __future__ import annotations

import numpy as np

from ..matrix_io import load_cell_matrix
from .integration import integrate_overlap_matrix
from .kspace import neighbor_reciprocal_lattice_vectors
from .parallel import parallel_map
from .state import StateCollection


class MSet:
    def __init__(self, state: StateCollection, threads: int = 1):
        self.state = state
        self.config = state.config
        self.threads = threads
        self.mM0: np.ndarray | None = None
        self.mMInitial: np.ndarray | None = None
        self.mM: np.ndarray | None = None

    def init_M0(self) -> None:
        b_half = len(self.config.composition_of_b) // 2
        band_count = int(self.config.band_calc_num)
        if self.config.M_in or "M" in self.config.use_cached_data:
            path = self.config.input_path(self.config.M_file)
            if path is None:
                raise ValueError("M cache requested, but M_file is disabled.")
            self.mM0 = load_cell_matrix(path, self.state.k_shape + (b_half,))
            self.mMInitial = self.state.gen_matrix_on_kmesh(
                lambda *_: [np.zeros((band_count, band_count), dtype=np.complex128) for _ in range(b_half)]
            )
            self.mM = self.state.gen_matrix_on_kmesh(
                lambda *_: [np.zeros((band_count, band_count), dtype=np.complex128) for _ in range(b_half)]
            )
            self.state.turn_to_bloch()
            return

        self.state.turn_to_bloch()
        self.mM0 = self.state.gen_matrix_on_kmesh(lambda *_: np.empty(b_half, dtype=object))

        for i, j, k in self.state.k_indices():
            for b in range(b_half):
                ik, _ = neighbor_reciprocal_lattice_vectors(self.config, [i, j, k], b)
                self.mM0[i, j, k][b] = np.zeros(
                    (len(self.state.E_idx[i, j, k]), len(self.state.E_idx[ik])), dtype=np.complex128
                )

        def calc_for_index(idx):
            i, j, k = idx
            left = self.state.get_block(i, j, k)
            result = []
            for b in range(b_half):
                ik, k_raw = neighbor_reciprocal_lattice_vectors(self.config, [i, j, k], b)
                right = self.state.get_block(*ik)
                if k_raw is not None:
                    phase1 = self.state.get_phase(*ik)
                    phase2 = self.state.get_phase(*k_raw)
                    right = right * (phase1 * np.conj(phase2))[None, :]
                result.append(
                    integrate_overlap_matrix(
                        self.state.integral_view,
                        left,
                        right,
                        self.state.epsilon,
                        chunk_size=64,
                        backend=self.state.compute_backend,
                    )
                )
            return idx, result

        for idx, result in parallel_map(self.state.k_indices(), calc_for_index, self.threads):
            i, j, k = idx
            for b, mat in enumerate(result):
                self.mM0[i, j, k][b] = mat

        self.mMInitial = self.state.gen_matrix_on_kmesh(
            lambda *_: [np.zeros((band_count, band_count), dtype=np.complex128) for _ in range(b_half)]
        )
        self.mM = self.state.gen_matrix_on_kmesh(
            lambda *_: [np.zeros((band_count, band_count), dtype=np.complex128) for _ in range(b_half)]
        )

    def get_M0(self, i: int, j: int, k: int, b: int) -> np.ndarray:
        b_half = len(self.config.composition_of_b) // 2
        if b < b_half:
            mat = self.mM0[i, j, k][b]
        else:
            ik, _ = neighbor_reciprocal_lattice_vectors(self.config, [i, j, k], b)
            mat = np.conj(self.mM0[ik][b - b_half]).T
        transform = self.state.get_transform()
        ik, _ = neighbor_reciprocal_lattice_vectors(self.config, [i, j, k], b)
        return transform[i, j, k] @ mat @ transform[ik]

    def get(self, i: int, j: int, k: int, b: int) -> np.ndarray:
        b_half = len(self.config.composition_of_b) // 2
        if b < b_half:
            return self.mM[i, j, k][b]
        ik, _ = neighbor_reciprocal_lattice_vectors(self.config, [i, j, k], b)
        return np.conj(self.mM[ik][b - b_half]).T

    def initial(self, vmat: np.ndarray) -> None:
        b_half = len(self.config.composition_of_b) // 2
        for i, j, k in self.state.k_indices():
            for b in range(b_half):
                ik, _ = neighbor_reciprocal_lattice_vectors(self.config, [i, j, k], b)
                self.mMInitial[i, j, k][b] = np.conj(vmat[i, j, k]).T @ self.get_M0(i, j, k, b) @ vmat[ik]

    def update(self, umat: np.ndarray) -> None:
        b_half = len(self.config.composition_of_b) // 2

        def update_idx(idx):
            i, j, k = idx
            out = []
            for b in range(b_half):
                ik, _ = neighbor_reciprocal_lattice_vectors(self.config, [i, j, k], b)
                out.append(np.conj(umat[i, j, k]).T @ self.mMInitial[i, j, k][b] @ umat[ik])
            return idx, out

        for idx, out in parallel_map(self.state.k_indices(), update_idx, self.threads):
            self.mM[idx] = out
