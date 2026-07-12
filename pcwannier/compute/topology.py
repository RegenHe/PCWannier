from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import IncarConfig
from ..data import BandResult, TopologyResult

LOGGER = logging.getLogger(__name__)


class Topology2D:
    def __init__(self):
        self.U: np.ndarray | None = None
        self._source_id: int | None = None

    def construct_parallel_transport(self, eigvecs: np.ndarray):
        n_k1, n_k2, dim, occ = eigvecs.shape
        self.U = np.empty((2, n_k1, n_k2, occ, occ), dtype=np.complex128)
        mats_0 = np.einsum("ijda,ijdb->ijab", eigvecs.conj(), np.roll(eigvecs, -1, axis=0), optimize=True)
        mats_1 = np.einsum("ijda,ijdb->ijab", eigvecs.conj(), np.roll(eigvecs, -1, axis=1), optimize=True)
        for direction, mats in enumerate((mats_0, mats_1)):
            u, _, vh = np.linalg.svd(mats.reshape(n_k1 * n_k2, occ, occ))
            self.U[direction] = (u @ vh).reshape(n_k1, n_k2, occ, occ)
        self._source_id = id(eigvecs)

    def hybrid_Wilson_loop(self, eigvecs: np.ndarray, direction: int = 0):
        if self.U is None or self._source_id != id(eigvecs):
            self.construct_parallel_transport(eigvecs)
        umat = self.U[direction]
        umat = umat if direction == 0 else umat.transpose((1, 0, 2, 3))
        s_loop, s_param, dim, _ = umat.shape
        k_param = np.linspace(0.0, 1.0, s_param, endpoint=False)
        raw_centers = np.empty((s_param, dim), dtype=float)
        for j in range(s_param):
            wilson = np.eye(dim, dtype=np.complex128)
            for i in range(s_loop):
                wilson = wilson @ umat[i, j]
            raw_centers[j] = np.sort((np.angle(np.linalg.eigvals(wilson)) / (2 * np.pi)) % 1.0)
        centers = self._track_periodic_branches(raw_centers)
        z2 = self._z2_from_tracked_centers(centers)
        return centers, k_param, z2

    @staticmethod
    def _track_periodic_branches(raw_centers: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_centers, dtype=float)
        if raw.ndim != 2 or raw.shape[0] == 0:
            raise ValueError("Wilson centers must be a non-empty two-dimensional array.")
        tracked = np.empty_like(raw)
        tracked[0] = np.sort(raw[0] % 1.0)
        for idx in range(1, raw.shape[0]):
            previous = tracked[idx - 1] % 1.0
            current = raw[idx] % 1.0
            delta = ((current[None, :] - previous[:, None] + 0.5) % 1.0) - 0.5
            cost = np.abs(delta)
            rows, cols = linear_sum_assignment(cost)
            ordered = np.empty(raw.shape[1], dtype=float)
            ordered[rows] = current[cols]
            step = ((ordered - previous + 0.5) % 1.0) - 0.5
            tracked[idx] = tracked[idx - 1] + step
        return tracked

    @classmethod
    def _z2_from_tracked_centers(cls, centers: np.ndarray) -> int | None:
        centers = np.asarray(centers, dtype=float)
        if centers.shape[1] == 0 or centers.shape[1] % 2:
            LOGGER.warning("Z2 unavailable: the Wilson subspace must contain a positive even number of bands")
            return None
        if centers.shape[0] % 2:
            LOGGER.warning("Z2 unavailable: the transverse k mesh must have an even number of points")
            return None
        half = centers.shape[0] // 2
        trajectory = centers[: half + 1]
        endpoint_residual = max(cls._kramers_pairing_residual(trajectory[0]), cls._kramers_pairing_residual(trajectory[-1]))
        if endpoint_residual > 5e-2:
            LOGGER.warning("Z2 unavailable: Kramers pairing residual at a TRIM is %.6g", endpoint_residual)
            return None

        samples = trajectory % 1.0
        candidate_count = max(64, 8 * samples.shape[0] * samples.shape[1])
        candidates = (np.arange(candidate_count, dtype=float) + 0.5) / candidate_count
        distances = np.abs(((samples.reshape(-1, 1) - candidates[None, :] + 0.5) % 1.0) - 0.5)
        reference = float(candidates[np.argmax(np.min(distances, axis=0))])
        crossings = 0
        for start, stop in zip(trajectory[:-1], trajectory[1:]):
            crossings += int(np.sum(np.abs(np.floor(stop - reference) - np.floor(start - reference))))
        return crossings % 2

    @staticmethod
    def _kramers_pairing_residual(centers: np.ndarray) -> float:
        values = np.sort(np.asarray(centers, dtype=float) % 1.0)
        if values.size == 0 or values.size % 2:
            return np.inf

        def periodic_distance(a, b):
            return abs(((a - b + 0.5) % 1.0) - 0.5)

        direct = max(periodic_distance(values[idx], values[idx + 1]) for idx in range(0, values.size, 2))
        shifted_pairs = [(values[-1], values[0])]
        shifted_pairs.extend((values[idx], values[idx + 1]) for idx in range(1, values.size - 1, 2))
        shifted = max(periodic_distance(a, b) for a, b in shifted_pairs)
        return float(min(direct, shifted))

    def Chern_number(self, eigvecs: np.ndarray):
        n_k1, n_k2, _, _ = eigvecs.shape
        if self.U is None or self._source_id != id(eigvecs):
            self.construct_parallel_transport(eigvecs)
        u1 = self.U[0]
        u2 = self.U[1]
        loop = (
            u1
            @ np.roll(u2, -1, axis=0)
            @ np.swapaxes(np.roll(u1, -1, axis=1).conj(), -2, -1)
            @ np.swapaxes(u2.conj(), -2, -1)
        )
        flux = np.angle(np.linalg.det(loop))
        total = float(np.sum(flux))
        return flux, float(total / (2 * np.pi))


def calculate_topology(band: BandResult, config: IncarConfig) -> TopologyResult | None:
    if band.bz_eigvecs is None:
        return None
    if int(config.kdim) != 2:
        raise NotImplementedError("Topology calculations currently support two-dimensional models only.")
    if not config.hermitian:
        raise NotImplementedError("Topology calculations currently support Hermitian models only.")
    topo_result = TopologyResult()
    eigvecs = band.bz_eigvecs
    for gid, group in enumerate(band.groups):
        topo = Topology2D()
        subspace = np.take(eigvecs, np.asarray(group, dtype=int), axis=-1)
        topo.construct_parallel_transport(subspace)
        if config.hybrid_Wilson_loop:
            for direction in (0, 1):
                topo_result.wilson[(gid, direction)] = topo.hybrid_Wilson_loop(subspace, direction)
        if config.Chern_number:
            key = str(gid)
            bands = tuple(int(index) + 1 for index in group)
            topo_result.chern[key] = topo.Chern_number(subspace)
            topo_result.chern_bands[key] = bands
            LOGGER.info(
                "Chern result: group=%s bands=%s chern=%.10g",
                gid,
                _format_band_numbers(bands),
                topo_result.chern[key][1],
            )
    topo = Topology2D()
    topo.construct_parallel_transport(eigvecs)
    if config.Chern_number:
        topo_result.chern["all"] = topo.Chern_number(eigvecs)
        all_bands = tuple(range(1, eigvecs.shape[-1] + 1))
        topo_result.chern_bands["all"] = all_bands
        LOGGER.info(
            "Chern result: group=all bands=%s chern=%.10g",
            _format_band_numbers(all_bands),
            topo_result.chern["all"][1],
        )
    return topo_result


def _format_band_numbers(bands: tuple[int, ...]) -> str:
    if not bands:
        return "none"
    if bands == tuple(range(bands[0], bands[-1] + 1)):
        return str(bands[0]) if len(bands) == 1 else f"{bands[0]}-{bands[-1]}"
    return ",".join(str(band) for band in bands)
