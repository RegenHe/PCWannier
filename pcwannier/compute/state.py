from __future__ import annotations

import numpy as np

from ..data import InputBundle, Mesh
from .integration import integrate_overlap_matrix, mesh_integral_view
from .kspace import get_kxyz
from .parallel import parallel_map


class StateCollection:
    def __init__(self, bundle: InputBundle, *, backend: str = "python", threads: int = 1):
        self.config = bundle.config
        self.compute_backend = backend
        self.integration_mode = self.config.integration_mode
        self.threads = max(1, int(threads))
        self.mesh = bundle.mesh
        self.fields = bundle.fields
        self._normalize_field_blocks()
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
        self.integral_view = mesh_integral_view(self.mesh)
        self.extention_integral_view = None
        self.space_to_original_mapping: np.ndarray | None = None
        self.extention_epsilon: np.ndarray | None = None
        self._identity_transform: np.ndarray | None = None
        self._phase_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _normalize_field_blocks(self) -> None:
        for idx in np.ndindex(self.fields.shape):
            block = np.asarray(self.fields[idx], dtype=np.complex128)
            if block.ndim == 1:
                block = block.reshape(1, -1)
            self.fields[idx] = block

    def k_indices(self):
        yield from np.ndindex(self.k_shape)

    def n_indices(self, i: int, j: int, k: int):
        return range(self.get_block(i, j, k).shape[0])

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
        return self.get_block(i, j, k)[n]

    def get_block(self, i: int, j: int, k: int) -> np.ndarray:
        raw_block = self.fields[i, j, k]
        if raw_block is None:
            raise ValueError(f"Empty field block at k=({i}, {j}, {k}).")
        block = np.asarray(raw_block, dtype=np.complex128)
        if block.ndim == 1:
            block = block.reshape(1, -1)
        nv = self.mesh.vertices.shape[0]
        if block.shape[1] != nv and block.shape[0] == nv:
            block = block.T
        if block.shape[1] != nv:
            raise ValueError(f"Field block at k=({i}, {j}, {k}) has invalid shape {block.shape}.")
        self.fields[i, j, k] = block
        return block

    def ensure_identity_transform(self) -> None:
        if self.transform_correction is not None:
            return
        self.transform = self.gen_matrix_on_kmesh(lambda i, j, k: np.eye(self.get_block(i, j, k).shape[0], dtype=np.complex128))
        self.transform_correction = self.gen_matrix_on_kmesh(
            lambda i, j, k: np.eye(self.get_block(i, j, k).shape[0], dtype=np.complex128)
        )
        self._identity_transform = self.transform

    def orthogonalize(self, tol_rel: float = 1e-6, atol_abs: float = 1e-12) -> None:
        self.S = self.gen_matrix_on_kmesh(lambda *_: None)
        self.transform = self.gen_matrix_on_kmesh(lambda *_: None)
        self.transform_correction = self.gen_matrix_on_kmesh(lambda *_: None)

        def calc_idx(idx):
            i, j, k = idx
            smat = self._overlap_matrix(i, j, k)
            evals, vecs = np.linalg.eigh(smat)
            lam = evals.real
            tau = max(tol_rel * np.max(lam), atol_abs)
            invsqrt = 1.0 / np.sqrt(np.maximum(lam, tau))
            tfull = vecs @ np.diag(invsqrt) @ vecs.conj().T
            tdiag = np.diag(np.diag(tfull))
            block = self.get_block(i, j, k)
            scaled = np.empty_like(block)
            for n in range(block.shape[0]):
                if tdiag[n, n] == 0.0:
                    raise ValueError(f"Normalization failed at ({i}, {j}, {k}, {n}).")
                scaled[n] = block[n] * tdiag[n, n]
            return idx, smat, tfull, np.linalg.inv(tdiag) @ tfull, scaled

        for idx, smat, tfull, tcorr, scaled in parallel_map(self.k_indices(), calc_idx, self.configured_threads):
            i, j, k = idx
            self.fields[i, j, k] = scaled
            self.S[i, j, k] = smat
            self.transform[i, j, k] = tfull
            self.transform_correction[i, j, k] = tcorr
        self._identity_transform = None
        self.is_orthogonalized = True

    def check_orthogonality(self, *, apply_transform: bool = True) -> tuple[np.ndarray, bool]:
        report = np.zeros(self.k_shape + (6,), dtype=float)
        need_orth = False

        def calc_idx(idx):
            i, j, k = idx
            smat = self._overlap_matrix(i, j, k)
            if apply_transform and self.is_orthogonalized and self.transform_correction is not None:
                tcorr = self.transform_correction[i, j, k]
                smat = tcorr.conj().T @ smat @ tcorr
            nwin = smat.shape[0]
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
            row = np.array([herm_res, diag_max_err, offdiag_max, frob_res, lam_min, cond], dtype=float)
            local_need = herm_res > 1e-8 or diag_max_err > 1e-3 or offdiag_max > 1e-3 or lam_min < -1e-6
            return idx, row, local_need

        for idx, row, local_need in parallel_map(self.k_indices(), calc_idx, self.configured_threads):
            report[idx] = row
            need_orth = need_orth or local_need
        return report, need_orth

    @property
    def configured_threads(self) -> int:
        return self.threads

    def _overlap_matrix(self, i: int, j: int, k: int) -> np.ndarray:
        wblock = self.get_block(i, j, k)
        smat = integrate_overlap_matrix(
            self.integral_view,
            wblock,
            wblock,
            self.epsilon,
            chunk_size=64,
            backend=self.compute_backend,
            mode=self.integration_mode,
        )
        scale = max(float(np.linalg.norm(smat, ord="fro")), 1.0)
        residual = float(np.linalg.norm(smat - smat.conj().T, ord="fro"))
        if not np.isfinite(residual) or residual > 1e-10 * scale:
            raise FloatingPointError(
                f"Overlap matrix at k={(i, j, k)} is not Hermitian "
                f"(residual={residual:.6g}, scale={scale:.6g})."
            )
        return 0.5 * (smat + smat.conj().T)

    def turn_to_bloch(self) -> None:
        """Convert stored full Bloch fields to periodic parts.

        The historical method name is retained for compatibility. For COMSOL,
        psi_k = exp(-i k.r) u_k, so multiplying by exp(+i k.r) stores u_k.
        """
        if self.is_bloch:
            return
        for i, j, k in self.k_indices():
            phase = self.get_phase(i, j, k)
            self.fields[i, j, k] = self.get_block(i, j, k) * np.conj(phase)[None, :]
        self.is_bloch = True

    @property
    def stores_periodic_bloch_parts(self) -> bool:
        return self.is_bloch

    def get_transform(self, zero: bool = False) -> np.ndarray:
        self.ensure_identity_transform()
        if self.is_orthogonalized and self.transform_correction is not None and not zero:
            return self.transform_correction
        if self._identity_transform is None:
            self._identity_transform = self.gen_matrix_on_kmesh(
                lambda i, j, k: np.eye(self.get_block(i, j, k).shape[0], dtype=np.complex128)
            )
        return self._identity_transform

    def get_phase(self, i: int, j: int, k: int) -> np.ndarray:
        key = (int(i), int(j), int(k))
        if key in self._phase_cache:
            return self._phase_cache[key]
        sign = -1 if self.config.dataset_type.lower() == "comsol" else 1
        kvec = get_kxyz(self.config, [i, j, k])[: self.config.kdim]
        phase = np.exp(1j * sign * np.dot(self.mesh.vertices, kvec))
        self._phase_cache[key] = phase
        return phase

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
        self.extention_integral_view = mesh_integral_view(self.extention_mesh)
        self.get_extention_epsilon()

    def get_extention_field(self, i: int, j: int, k: int, n: int) -> np.ndarray:
        if self.extention_mesh is None or self.space_to_original_mapping is None:
            raise ValueError("The field has not been extended.")
        scale = np.sqrt(float(np.prod(self.config.extension[: self.config.kdim])))
        base = self.get_block(i, j, k)[n]
        return base[self.space_to_original_mapping] / scale

    def get_extention_block(self, i: int, j: int, k: int) -> np.ndarray:
        if self.extention_mesh is None or self.space_to_original_mapping is None:
            raise ValueError("The field has not been extended.")
        scale = np.sqrt(float(np.prod(self.config.extension[: self.config.kdim])))
        return self.get_block(i, j, k)[:, self.space_to_original_mapping] / scale

    def get_extention_epsilon(self) -> np.ndarray:
        if self.extention_mesh is None or self.space_to_original_mapping is None:
            raise ValueError("The field has not been extended.")
        if self.extention_epsilon is None:
            self.extention_epsilon = np.asarray(self.epsilon)[self.space_to_original_mapping]
        return self.extention_epsilon
