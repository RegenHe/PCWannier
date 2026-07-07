from __future__ import annotations

import math

import numpy as np

from ..data import FieldData, InputBundle, Mesh
from .integration import integrate_over_mesh
from .kspace import get_kxyz


class StateCollection:
    def __init__(self, bundle: InputBundle, *, backend: str = "python"):
        self.config = bundle.config
        self.compute_backend = backend
        self.mesh = bundle.mesh
        self.fields = bundle.fields
        self.epsilon = np.asarray(bundle.epsilon)
        self.E = bundle.energies
        self.E_idx = bundle.band_indices
        self.inner_E_idx = bundle.inner_band_indices
        self.energy_matrix = bundle.energy_matrix
        self.kdim = int(self.config.kdim)
        self.k_shape = self.fields.shape

        self.S: np.ndarray | None = None
        self.transform: np.ndarray | None = None
        self.transform_correction: np.ndarray | None = None
        self.is_bloch = False
        self.is_orthogonalized = False
        self.extention_mesh: Mesh | None = None
        self.space_to_original_mapping: np.ndarray | None = None
        self.extention_epsilon: np.ndarray | None = None

    def k_indices(self):
        yield from np.ndindex(self.k_shape)

    def n_indices(self, i: int, j: int, k: int):
        return range(len(self.fields[i, j, k]))

    def get_k_num(self) -> int:
        nk1, nk2, nk3 = self.k_shape
        if self.kdim == 1:
            return nk1
        if self.kdim == 2:
            return nk1 * nk2
        return nk1 * nk2 * nk3

    def gen_matrix_on_kmesh(self, factory):
        arr = np.empty(self.k_shape, dtype=object)
        for idx in self.k_indices():
            arr[idx] = factory(*idx)
        return arr

    def norm_ijkn(self, *idx) -> tuple[int, int, int, int]:
        n = int(idx[-1])
        coords = list(idx[:-1])
        coords += [0] * (self.kdim - len(coords))
        coords = (coords + [0, 0, 0])[:3]
        nk1, nk2, nk3 = self.k_shape
        return int(coords[0]) % nk1, int(coords[1]) % nk2, int(coords[2]) % nk3, n

    def get(self, *idx) -> np.ndarray:
        i, j, k, n = self.norm_ijkn(*idx)
        return self.fields[i, j, k][n]

    def get_block(self, i: int, j: int, k: int) -> np.ndarray:
        block = self.fields[i, j, k]
        if block is None:
            raise ValueError(f"Empty field block at k=({i}, {j}, {k}).")
        return np.asarray(np.vstack(block), dtype=np.complex128)

    def ensure_identity_transform(self) -> None:
        if self.transform_correction is not None:
            return
        self.transform = self.gen_matrix_on_kmesh(lambda i, j, k: np.eye(len(self.fields[i, j, k]), dtype=np.complex128))
        self.transform_correction = self.gen_matrix_on_kmesh(
            lambda i, j, k: np.eye(len(self.fields[i, j, k]), dtype=np.complex128)
        )

    def orthogonalize(self, tol_rel: float = 1e-6, atol_abs: float = 1e-12) -> None:
        self.S = self.gen_matrix_on_kmesh(lambda *_: None)
        self.transform = self.gen_matrix_on_kmesh(lambda *_: None)
        self.transform_correction = self.gen_matrix_on_kmesh(lambda *_: None)
        eps = np.asarray(self.epsilon)
        nv = self.mesh.vertices.shape[0]

        for i, j, k in self.k_indices():
            wblock = self.get_block(i, j, k)
            if wblock.ndim == 1:
                wblock = wblock[None, :]
            elif wblock.shape[1] != nv:
                wblock = wblock.T
            nwin = wblock.shape[0]
            smat = np.empty((nwin, nwin), dtype=np.complex128)
            for a in self.n_indices(i, j, k):
                left = np.conj(self.get(i, j, k, a)) * eps
                fmat = (left[:, None] * wblock.T).astype(np.complex128, copy=False)
                vals = np.atleast_1d(
                    integrate_over_mesh(
                        FieldData("S", self.mesh, fmat),
                        chunk_size=2048,
                        backend=self.compute_backend,
                    )
                )
                smat[a, :] = vals[:]

            evals, vecs = np.linalg.eigh(smat)
            lam = evals.real
            tau = max(tol_rel * np.max(lam), atol_abs)
            invsqrt = 1.0 / np.sqrt(np.maximum(lam, tau))
            tfull = vecs @ np.diag(invsqrt) @ vecs.conj().T
            tdiag = np.diag(np.diag(tfull))
            for n in range(len(self.fields[i, j, k])):
                if tdiag[n, n] == 0.0:
                    raise ValueError(f"Normalization failed at ({i}, {j}, {k}, {n}).")
                self.fields[i, j, k][n] *= tdiag[n, n]
            self.S[i, j, k] = smat
            self.transform[i, j, k] = tfull
            self.transform_correction[i, j, k] = np.linalg.inv(tdiag) @ tfull
        self.is_orthogonalized = True

    def check_orthogonality(self) -> tuple[np.ndarray, bool]:
        eps = np.asarray(self.epsilon)
        report = np.zeros(self.k_shape + (6,), dtype=float)
        need_orth = False
        nv = self.mesh.vertices.shape[0]
        for i, j, k in self.k_indices():
            wblock = self.get_block(i, j, k)
            if wblock.ndim == 1:
                wblock = wblock[None, :]
            elif wblock.shape[1] != nv:
                wblock = wblock.T
            nwin = wblock.shape[0]
            smat = np.empty((nwin, nwin), dtype=np.complex128)
            for a in self.n_indices(i, j, k):
                left = np.conj(self.get(i, j, k, a)) * eps
                fmat = (left[:, None] * wblock.T).astype(np.complex128, copy=False)
                vals = np.atleast_1d(
                    integrate_over_mesh(
                        FieldData("S", self.mesh, fmat),
                        chunk_size=2048,
                        backend=self.compute_backend,
                    )
                )
                smat[a, :] = vals[:]
            if self.is_orthogonalized and self.transform_correction is not None:
                tcorr = self.transform_correction[i, j, k]
                smat = tcorr.conj().T @ smat @ tcorr
            herm_res = float(np.linalg.norm(smat - smat.conj().T, ord="fro"))
            diag = np.real(np.diag(smat))
            diag_max_err = float(np.max(np.abs(diag - 1.0))) if nwin > 0 else 0.0
            offdiag = smat - np.diag(np.diag(smat))
            offdiag_max = float(np.max(np.abs(offdiag))) if nwin > 1 else 0.0
            frob_res = float(np.linalg.norm(smat - np.eye(nwin), ord="fro"))
            evals = np.linalg.eigvalsh(smat)
            lam_min = float(np.min(evals))
            lam_max = float(np.max(evals))
            cond = float(lam_max / max(lam_min, 1e-16)) if lam_max > 0 else np.inf
            report[i, j, k] = [herm_res, diag_max_err, offdiag_max, frob_res, lam_min, cond]
            if herm_res > 1e-8 or diag_max_err > 1e-3 or offdiag_max > 1e-3 or lam_min < -1e-6:
                need_orth = True
        return report, need_orth

    def turn_to_bloch(self) -> None:
        if self.is_bloch:
            return
        for i, j, k in self.k_indices():
            phase = self.get_phase(i, j, k)
            for n in self.n_indices(i, j, k):
                self.fields[i, j, k][n] = np.conj(phase) * self.fields[i, j, k][n]
        self.is_bloch = True

    def get_transform(self, zero: bool = False) -> np.ndarray:
        self.ensure_identity_transform()
        if self.is_orthogonalized and self.transform_correction is not None and not zero:
            return self.transform_correction
        return self.gen_matrix_on_kmesh(lambda i, j, k: np.eye(len(self.fields[i, j, k]), dtype=np.complex128))

    def get_phase(self, i: int, j: int, k: int) -> np.ndarray:
        sign = -1 if self.config.dataset_type.lower() == "comsol" else 1
        kvec = get_kxyz(self.config, [i, j, k])[: self.config.kdim]
        return np.exp(1j * sign * np.dot(self.mesh.vertices, kvec))

    def get_extention_phase(self, i: int, j: int, k: int) -> np.ndarray:
        if self.extention_mesh is None:
            raise ValueError("The field has not been extended.")
        sign = -1 if self.config.dataset_type.lower() == "comsol" else 1
        kvec = get_kxyz(self.config, [i, j, k])[: self.config.kdim]
        return np.exp(1j * sign * np.dot(self.extention_mesh.vertices, kvec))

    def extention(self, n: list[int]) -> None:
        self.extention_mesh = self.mesh.__deepcopy__()
        self.space_to_original_mapping = self.extention_mesh.extension(
            n,
            self.config.real_lattice_vectors,
            float(self.config.lattice_const),
        )
        self.get_extention_epsilon()

    def get_extention_field(self, i: int, j: int, k: int, n: int) -> np.ndarray:
        if self.extention_mesh is None or self.space_to_original_mapping is None:
            raise ValueError("The field has not been extended.")
        scale = math.sqrt(float(np.prod(self.config.extension[: self.config.kdim])))
        base = self.fields[i, j, k][n]
        return np.asarray([base[p] / scale for p in self.space_to_original_mapping], dtype=np.complex128)

    def get_extention_epsilon(self) -> np.ndarray:
        if self.extention_mesh is None or self.space_to_original_mapping is None:
            raise ValueError("The field has not been extended.")
        if self.extention_epsilon is None:
            self.extention_epsilon = np.asarray([self.epsilon[p] for p in self.space_to_original_mapping])
        return self.extention_epsilon
