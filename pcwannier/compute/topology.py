from __future__ import annotations

import numpy as np

from ..config import IncarConfig
from ..data import BandResult, TopologyResult


class Topology2D:
    def __init__(self):
        self.U: np.ndarray | None = None

    def construct_parallel_transport(self, eigvecs: np.ndarray):
        n_k1, n_k2, dim, occ = eigvecs.shape
        self.U = np.empty((2, n_k1, n_k2, occ, occ), dtype=np.complex128)
        mats_0 = np.einsum("ijda,ijdb->ijab", eigvecs.conj(), np.roll(eigvecs, -1, axis=0), optimize=True)
        mats_1 = np.einsum("ijda,ijdb->ijab", eigvecs.conj(), np.roll(eigvecs, -1, axis=1), optimize=True)
        for direction, mats in enumerate((mats_0, mats_1)):
            u, _, vh = np.linalg.svd(mats.reshape(n_k1 * n_k2, occ, occ))
            self.U[direction] = (u @ vh).reshape(n_k1, n_k2, occ, occ)

    def hybrid_Wilson_loop(self, eigvecs: np.ndarray, direction: int = 0):
        if self.U is None:
            self.construct_parallel_transport(eigvecs)
        umat = self.U[direction]
        umat = umat if direction == 0 else umat.transpose((1, 0, 2, 3))
        s_loop, s_param, dim, _ = umat.shape
        k_param = np.linspace(0.0, 1.0, s_param, endpoint=False)
        centers = np.empty((s_param, dim), dtype=float)
        for j in range(s_param):
            wilson = np.eye(dim, dtype=np.complex128)
            for i in range(s_loop):
                wilson = wilson @ umat[i, j]
            theta = np.unwrap(np.angle(np.sort(np.linalg.eigvals(wilson))))
            centers[j] = theta / (2 * np.pi)
        centers = centers - np.floor(centers.min(axis=0))
        half = k_param.size // 2
        crossing = 0
        for band in range(centers.shape[1]):
            shifted = centers[: half + 1, band] % 1 - 0.5
            crossing += np.sum(np.abs(np.diff(np.signbit(shifted))))
        return centers, k_param, int(crossing % 2)

    def Chern_number(self, eigvecs: np.ndarray):
        n_k1, n_k2, _, _ = eigvecs.shape
        if self.U is None:
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
    topo_result = TopologyResult()
    eigvecs = band.bz_eigvecs
    for gid, group in enumerate(band.groups):
        topo = Topology2D()
        subspace = eigvecs[:, :, :, group[0] : group[-1] + 1]
        topo.construct_parallel_transport(subspace)
        if config.hybrid_Wilson_loop:
            for direction in (0, 1):
                topo_result.wilson[(gid, direction)] = topo.hybrid_Wilson_loop(subspace, direction)
        if config.Chern_number:
            topo_result.chern[str(gid)] = topo.Chern_number(subspace)
    topo = Topology2D()
    topo.construct_parallel_transport(eigvecs)
    if config.Chern_number:
        topo_result.chern["all"] = topo.Chern_number(eigvecs)
    return topo_result
