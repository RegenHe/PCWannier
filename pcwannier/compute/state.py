from __future__ import annotations

import numpy as np

from ..data import InputBundle, Mesh
from ..matrix_io import load_cell_matrix
from .integration import (
    integrate_overlap_matrix,
    integrate_weighted_abs2_columns,
    mesh_integral_view,
    validated_real,
)
from .kspace import get_kxyz
from .parallel import parallel_map


class StateCollection:
    def __init__(self, bundle: InputBundle, *, backend: str = "python", threads: int = 1):
        self.config = bundle.config
        self.maxwell = bundle.maxwell
        configured_maxwell = getattr(self.config, "maxwell_problem", None)
        if configured_maxwell is not None and configured_maxwell != self.maxwell:
            raise ValueError(
                "InputBundle Maxwell metadata does not match the calculation config."
            )
        self.compute_backend = backend
        self.integration_mode = self.config.integration_mode
        self.threads = max(1, int(threads))
        self.mesh = bundle.mesh
        self.fields = bundle.fields
        self._normalize_field_blocks()
        raw_metric = np.asarray(bundle.metric_material)
        if np.iscomplexobj(raw_metric) and np.any(np.abs(raw_metric.imag) > 1.0e-12):
            raise ValueError("Metric material must contain real values.")
        self.metric_material = np.asarray(raw_metric.real, dtype=float)
        expected_metric_shape = (self.mesh.vertices.shape[0],)
        if self.metric_material.shape != expected_metric_shape or not np.all(
            np.isfinite(self.metric_material)
        ):
            raise ValueError(
                f"Metric material must contain one finite value per mesh vertex; "
                f"expected {expected_metric_shape}, got {self.metric_material.shape}."
            )
        self.E = bundle.energies
        self.E_idx = bundle.band_indices
        self.inner_E_idx = bundle.inner_band_indices
        self.energy_matrix = bundle.energy_matrix
        self.kdim = int(self.config.kdim)
        self.k_shape = self.fields.shape

        self.S: np.ndarray | None = None
        self.transform: np.ndarray | None = None
        self.normalization_transform: np.ndarray | None = None
        self.transform_correction: np.ndarray | None = None
        self.is_bloch = False
        self.is_orthogonalized = False
        self.extention_mesh: Mesh | None = None
        self.integral_view = mesh_integral_view(self.mesh)
        self.extention_integral_view = None
        self.space_to_original_mapping: np.ndarray | None = None
        self.extended_metric_material: np.ndarray | None = None
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
        self.normalization_transform = self.gen_matrix_on_kmesh(
            lambda i, j, k: np.eye(self.get_block(i, j, k).shape[0], dtype=np.complex128)
        )
        self.transform_correction = self.gen_matrix_on_kmesh(
            lambda i, j, k: np.eye(self.get_block(i, j, k).shape[0], dtype=np.complex128)
        )
        self._identity_transform = self.transform

    def orthogonalize(self, tol_rel: float = 1e-6, atol_abs: float = 1e-12) -> None:
        self._ensure_raw_overlap()
        self.transform = self.gen_matrix_on_kmesh(lambda *_: None)
        self.normalization_transform = self.gen_matrix_on_kmesh(lambda *_: None)
        self.transform_correction = self.gen_matrix_on_kmesh(lambda *_: None)

        def calc_idx(idx):
            i, j, k = idx
            smat = np.asarray(self.S[idx], dtype=np.complex128)
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
            return idx, tfull, tdiag, np.linalg.inv(tdiag) @ tfull, scaled

        for idx, tfull, tdiag, tcorr, scaled in parallel_map(self.k_indices(), calc_idx, self.configured_threads):
            i, j, k = idx
            self.fields[i, j, k] = scaled
            self.transform[i, j, k] = tfull
            self.normalization_transform[i, j, k] = tdiag
            self.transform_correction[i, j, k] = tcorr
        self._identity_transform = None
        self.is_orthogonalized = True

    def check_orthogonality(self, *, apply_transform: bool = True) -> tuple[np.ndarray, bool]:
        self._ensure_raw_overlap()
        report = np.zeros(self.k_shape + (6,), dtype=float)
        need_orth = False

        def calc_idx(idx):
            i, j, k = idx
            smat = np.asarray(self.S[idx], dtype=np.complex128)
            if self.is_orthogonalized:
                if apply_transform and self.transform is not None:
                    transform = self.transform[idx]
                elif not apply_transform and self.normalization_transform is not None:
                    transform = self.normalization_transform[idx]
                else:
                    transform = None
                if transform is not None:
                    smat = transform.conj().T @ smat @ transform
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

    def _ensure_raw_overlap(self) -> None:
        if self.S is not None:
            return
        use_cached = {str(name).upper() for name in getattr(self.config, "use_cached_data", ())}
        if "S" in use_cached:
            path_value = getattr(self.config, "S_file", None)
            input_path = getattr(self.config, "input_path", None)
            path = input_path(path_value) if callable(input_path) else None
            if path is None:
                raise ValueError("S cache requested, but S_file is disabled.")
            raw = load_cell_matrix(path, self.k_shape)
            self.S = self._validate_raw_overlap(raw, source=f"Cached S matrix {path}")
            return

        raw = self.gen_matrix_on_kmesh(lambda *_: None)

        def calc_idx(idx):
            return idx, self._overlap_matrix(*idx)

        for idx, smat in parallel_map(self.k_indices(), calc_idx, self.configured_threads):
            raw[idx] = smat
        self.S = self._validate_raw_overlap(raw, source="Calculated S matrix")

    def _validate_raw_overlap(self, matrix: np.ndarray, *, source: str) -> np.ndarray:
        if np.asarray(matrix).dtype != object or np.asarray(matrix).shape != self.k_shape:
            raise ValueError(
                f"{source} has k-grid shape {np.asarray(matrix).shape}; expected {self.k_shape}."
            )
        validated = np.empty(self.k_shape, dtype=object)
        for idx in self.k_indices():
            cell = np.asarray(matrix[idx], dtype=np.complex128)
            band_count = self.get_block(*idx).shape[0]
            expected = (band_count, band_count)
            if cell.shape != expected:
                raise ValueError(f"{source} at k={idx} has shape {cell.shape}; expected {expected}.")
            if not np.all(np.isfinite(cell)):
                raise ValueError(f"{source} at k={idx} contains non-finite values.")
            scale = max(float(np.linalg.norm(cell, ord="fro")), 1.0)
            residual = float(np.linalg.norm(cell - cell.conj().T, ord="fro"))
            if residual > 1.0e-8 * scale:
                raise ValueError(
                    f"{source} at k={idx} is not Hermitian "
                    f"(residual={residual:.6g}, scale={scale:.6g})."
                )
            validated[idx] = 0.5 * (cell + cell.conj().T)
        return validated

    @property
    def configured_threads(self) -> int:
        return self.threads

    def _overlap_matrix(self, i: int, j: int, k: int) -> np.ndarray:
        wblock = self.get_block(i, j, k)
        smat = self.metric_overlap(
            wblock,
            wblock,
            chunk_size=64,
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
        self.get_extended_metric_material()

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

    def get_extended_metric_material(self) -> np.ndarray:
        if self.extention_mesh is None or self.space_to_original_mapping is None:
            raise ValueError("The field has not been extended.")
        if self.extended_metric_material is None:
            self.extended_metric_material = self.metric_material[
                self.space_to_original_mapping
            ]
        return self.extended_metric_material

    def metric_overlap(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        extended: bool = False,
        view=None,
        conjugate_left: bool = True,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        if extended:
            weights = self.get_extended_metric_material()
            integral_view = self.extention_integral_view if view is None else view
        else:
            weights = self.metric_material
            integral_view = self.integral_view if view is None else view
        return integrate_overlap_matrix(
            integral_view,
            left,
            right,
            weights,
            conjugate_left=conjugate_left,
            chunk_size=chunk_size,
            backend=self.compute_backend,
            mode=self.integration_mode,
        )

    def metric_norms(
        self,
        values: np.ndarray,
        *,
        extended: bool = False,
        view=None,
        chunk_size: int | None = None,
        name: str = "metric norms",
    ) -> np.ndarray:
        if extended:
            weights = self.get_extended_metric_material()
            integral_view = self.extention_integral_view if view is None else view
        else:
            weights = self.metric_material
            integral_view = self.integral_view if view is None else view
        result = integrate_weighted_abs2_columns(
            integral_view,
            weights,
            values,
            chunk_size=chunk_size,
            backend=self.compute_backend,
            mode=self.integration_mode,
        )
        return validated_real(np.atleast_1d(result), name)

    def metric_field_norm(
        self,
        field: np.ndarray,
        *,
        extended: bool = False,
        view=None,
        name: str = "metric field norm",
    ) -> float:
        values = self.metric_norms(
            np.asarray(field).reshape(-1, 1),
            extended=extended,
            view=view,
            chunk_size=1,
            name=name,
        )
        return float(values[0])
