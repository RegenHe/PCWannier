import os

import numpy as np
import scipy
import matplotlib.pyplot as plt

from .Log import Logger
from .IO import IO
from .Timer import Timer, timer
from .Utils import global_data

from .Utils import WannierTools, FieldData

class Topo:
    def __init__(self):
        self.U = None
    
    def construct_parallel_transport(self, eigvecs: np.ndarray):
        n_k1, n_k2, dim, occ = eigvecs.shape
        self.U = np.empty((2, n_k1, n_k2, occ, occ), dtype=complex)

        for i in range(n_k1):
            for j in range(n_k2):
                M = eigvecs[i, j, :, :].conj().T @ eigvecs[(i + 1) % n_k1, j, :, :]
                mU, _, mVh = np.linalg.svd(M)
                self.U[0, i, j] = mU @ mVh

                M = eigvecs[i, j, :, :].conj().T @ eigvecs[i, (j + 1) % n_k2, :, :]
                mU, _, mVh = np.linalg.svd(M)
                self.U[1, i, j] = mU @ mVh

    def hybrid_Wilson_loop(self, eigvecs: np.ndarray, direction: int=0):
        if self.U is None:
            self.construct_parallel_transport(eigvecs)

        U = self.U[direction]
        U = U if direction == 0 else U.transpose((1, 0, 2, 3))

        s_loop, s_param, dim, _ = U.shape
        k_param = np.linspace(0.0, 1.0, s_param, endpoint=False)
        x_centers = np.empty((s_param, dim))

        for j in range(s_param):
            W = np.eye(dim, dtype=complex)
            for i in range(s_loop):
                W =  W @ U[i, j]

            eigvals = np.linalg.eigvals(W)

            theta = np.angle(np.sort(eigvals))
            theta = np.unwrap(theta)
            x_centers[j] = (1 / (2 * np.pi)) * theta

        x_centers = x_centers - np.floor(x_centers.min(axis=0))

        half = k_param.size // 2
        Ncross = 0
        for b in range(x_centers.shape[1]):
            s = x_centers[:half + 1, b] % 1 - 0.5
            Ncross += np.sum(np.abs(np.diff(np.signbit(s))))

        return x_centers, k_param, Ncross % 2
    
    def save_hybrid_Wilson_loop(self, filename: str, eigvecs: np.ndarray, direction: int=0):
        x_centers, k_param, Z2 = self.hybrid_Wilson_loop(eigvecs, direction)

        fig, ax = plt.subplots()
        for band in range(x_centers.shape[1]):
            ax.plot(k_param, x_centers[:, band] % 1)
        
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.8)
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.8)
        ax.set_xlabel(r"$k (2\pi / a)$")
        ax.set_ylabel(r"$x$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Wilson loop (direction = {direction}, Z2 = {Z2})")
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        Logger.info(f"figure successfully saved to {filename}")
        
        return Z2

    def Chern_number(self, eigvecs: np.ndarray, filename: str):
        n_k1, n_k2, dim, _ = eigvecs.shape
        if self.U is None:
            self.construct_parallel_transport(eigvecs)

        U1 = self.U[0]
        U2 = self.U[1]

        flux_sum = 0.0
        F = np.zeros((n_k1, n_k2), dtype=float)
        for i in range(n_k1):
            for j in range(n_k2):
                Wp = U1[i, j] @ U2[(i + 1) % n_k1, j] @ U1[i, (j + 1) % n_k2].conj().T @ U2[i, j].conj().T
                F[i, j] = np.angle(np.linalg.det(Wp))
                flux_sum += F[i, j]

        self.Chern = flux_sum / (2 * np.pi)

        fig, ax = plt.subplots()
        img = ax.imshow(F.T / (2 * np.pi) * (n_k1 * n_k2), origin="lower", extent=[-0.5, 0.5, -0.5, 0.5])
        cbar = fig.colorbar(img, ax=ax)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel(r"$k_1 (2\pi / a)$")
        ax.set_ylabel(r"$k_2 (2\pi / a)$")
        ax.set_title(f"Chern number = {self.Chern:.4f}")

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        Logger.info(f"figure successfully saved to {filename}")
        return self.Chern
