import os

from collections import defaultdict
from itertools import product

import numpy as np
from math import factorial
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .Log import Logger
from .IO import IO
from .Timer import Timer, timer
from .Utils import global_data

from .Utils import WannierTools, FieldData

class Finite2D:
    def __init__(self, nx: int, ny: int, gen_hopping, neighbors_half):
        self.nx = nx
        self.ny = ny
        self._gen = gen_hopping
        self._half = { (int(dx), int(dy)) for dx, dy in neighbors_half }
        self._build_HR_pair_complete()
        self.norb = self.HR[(0, 0)].shape[0]
        
    def _build_HR_pair_complete(self):
        self.HR = {}
        H00 = self._gen([0, 0])
        self.HR[(0, 0)] = 0.5 * (H00 + H00.conj().T)

        for (dx, dy) in self._half:
            H = self._gen([dx, dy])
            self.HR[(dx, dy)] = H
            self.HR[(-dx, -dy)] = H.conj().T
        return self.HR
    
    def _blocks_along_axis(self, axis: str, k: float, pos: list[int]):
        A0 = np.zeros_like(self.HR[(0, 0)], dtype=np.complex128)
        B = defaultdict(lambda: np.zeros_like(A0, dtype=np.complex128))
        
        for (dx, dy), H in self.HR.items():
            if axis == 'x':
                i_alpha_next = pos[0] + dx
                if (self.nx is not None) and not (0 <= i_alpha_next < self.nx):
                    continue
                d_alpha, d_beta = dx, dy
            else:
                i_alpha_next = pos[1] + dy
                if (self.ny is not None) and not (0 <= i_alpha_next < self.ny):
                    continue
                d_alpha, d_beta = dy, dx

            if (dx, dy) == (0, 0):
                A0 += H
                continue

            phase = np.exp(1j * k * d_beta)
            if d_alpha == 0:
                if d_beta > 0:
                    A0 += H * phase + H.conj().T * np.conjugate(phase)
            elif d_alpha > 0:
                B[d_alpha] += H * phase

        return A0, B
    
    def build_stripe_H(self, k: float):
        if (self.nx is not None) and (self.ny is None):
            axis, layers = 'x', int(self.nx)
        elif (self.nx is None) and (self.ny is not None):
            axis, layers = 'y', int(self.ny)
        else:
            Logger.error("build_stripe_H only supports one direction finite or two directions finite.")
            raise
        
        norb, nlayer = self.norb, layers
        n = norb * nlayer

        H = np.zeros((n, n), dtype=np.complex128)
        for i in range(nlayer):
            if axis == 'x':
                pos = [i, 0]
            else:
                pos = [0, i]
            A0, B = self._blocks_along_axis(axis, k, pos)
            i0 = i * norb
            H[i0 : i0 + norb, i0 : i0 + norb] += A0
            for d_alpha, H_block in B.items():
                if i + d_alpha < nlayer:
                    j0 = (i + d_alpha) * norb
                    H[i0 : i0 + norb, j0 : j0 + norb] += H_block
                    H[j0 : j0 + norb, i0 : i0 + norb] += H_block.conj().T
        return H
    
    def build_finite_H(self):
        if (self.nx is None) or (self.ny is None):
            Logger.error("build_finite_H only supports two directions finite.")
            raise

        n = self.norb * self.nx * self.ny

        H = np.zeros((n, n), dtype=np.complex128)

        def flat_index(ix: int, iy: int) -> int:
            return (ix * self.ny + iy) * self.norb

        for (dx, dy), Hblk in self.HR.items():
            if Hblk is None:
                continue

            ix0 = max(0, -dx)
            ix1 = min(self.nx, self.nx - dx)
            if ix0 >= ix1:
                continue

            iy0 = max(0, -dy)
            iy1 = min(self.ny, self.ny - dy)
            if iy0 >= iy1:
                continue

            for ix in range(ix0, ix1):
                jx = ix + dx
                i0 = flat_index(ix, 0)
                j0_base = flat_index(jx, 0)
                for iy in range(iy0, iy1):
                    jy = iy + dy
                    i00 = i0 + iy * self.norb
                    j00 = j0_base + jy * self.norb
                    H[i00:i00 + self.norb, j00:j00 + self.norb] += Hblk
        H = 0.5 * (H + H.conj().T)
        return H
    
    def bands_stripe(self, k_list=None):
        if (self.nx is not None) and (self.ny is None):
            nlayer = self.nx
        elif (self.nx is None) and (self.ny is not None):
            nlayer = self.ny
        elif (self.nx is not None) and (self.ny is not None):
            nlayer = self.nx * self.ny
            k_list = None
        else:
            Logger.error("build_stripe_H only supports one direction finite or two directions finite.")
            raise

        n = int(nlayer) * self.norb
        # use_sparse = (sp is not None) and (n >= 512)

        if k_list is None:
            H = self.build_finite_H()
            E, v = np.linalg.eigh(H)
            return None, E, v
        else:
            k_list = np.asarray(k_list, dtype=float)

            evals = []
            vlist = []
            for k in k_list:
                H = self.build_stripe_H(k)
                w, v = np.linalg.eigh(H)
                evals.append(w)
                vlist.append(v)

            maxlen = max(len(w) for w in evals)
            E = np.full((len(evals), maxlen), np.nan, dtype=float)
            V = np.zeros((len(evals), maxlen, n), dtype=np.complex128)
            for i, w in enumerate(evals):
                E[i, :len(w)] = w
                V[i, :len(w), :] = vlist[i]
            return k_list, E, V
        
    def _half_blocks_with_layer(self, axis: str, k: float, L: int):
        norb = self.norb
        n = L * norb
        H00 = np.zeros((n, n), dtype=np.complex128)
        H01 = np.zeros((n, n), dtype=np.complex128)

        for (dx, dy), H in self.HR.items():
            d_alpha, d_beta = (dx, dy) if axis == 'x' else (dy, dx)
            phase = np.exp(1j * k * d_beta)

            for a in range(L):
                j = a + d_alpha
                q = j // L
                a2 = j - q * L
                if not (0 <= a2 < L):
                    continue

                I = slice(a * norb, (a + 1) * norb)
                J = slice(a2 * norb, (a2 + 1) * norb)

                if q == 0:
                    H00[I, J] += H * phase
                elif q == 1:
                    H01[I, J] += H * phase
                elif q == -1:
                    pass
                else:
                    if not hasattr(self, '__layer_warninged__') or not self.__layer_warninged__:
                        self.__layer_warninged__ = True
                        warn = f"layer number {L} is too small to include hopping."
                        Logger.warning(warn)

        H00 = 0.5 * (H00 + H00.conj().T)
        return H00, H01
        
    def half_infinte_H(self, k: float, layernum: int = 1):
        if (self.nx is not None) and (self.ny is None):
            axis = 'x'
        elif (self.nx is None) and (self.ny is not None):
            axis = 'y'
        else:
            Logger.error("build_stripe_H only supports one direction finite or two directions finite.")
            raise
        
        return self._half_blocks_with_layer(axis, k, int(layernum))
    
    def _sancho_T(self, H00: np.ndarray, H01: np.ndarray, z: complex, tol: float = 1e-12, max_iter: int = 200):
        n = H00.shape[0]
        I = np.eye(n, dtype=np.complex128)
        H10 = H01.conj().T

        A = z * I - H00
        t = np.linalg.solve(A, H10)
        t_tilde = np.linalg.solve(A, H01)

        T_eff = t.copy()
        prod_tilde = t_tilde.copy()

        for _ in range(max_iter):
            Den = I - t @ t_tilde - t_tilde @ t
            t_new = np.linalg.solve(Den, t @ t)
            t_tilde_new = np.linalg.solve(Den, t_tilde @ t_tilde)

            T_eff += prod_tilde @ t_new
            prod_tilde = prod_tilde @ t_tilde_new

            if max(np.linalg.norm(t_new, ord='fro'), np.linalg.norm(t_tilde_new, ord='fro')) < tol:
                break
            t, t_tilde = t_new, t_tilde_new

        return T_eff

    def _surface_greens(self, H00: np.ndarray, H01: np.ndarray, z: complex):
        n = H00.shape[0]
        I = np.eye(n, dtype=np.complex128)
        T = self._sancho_T(H00, H01, z)
        Sigma = H01 @ T
        return np.linalg.solve(z * I - H00 - Sigma, I)
    
    @timer("Half-infinite DOS calculation - ")
    def half_infinte_DOS(self, layernum: int, k_list, eta: float, energy: np.ndarray):
        k_list = np.atleast_1d(np.asarray(k_list, dtype=float))
        energy = np.asarray(energy, dtype=float)

        dos = np.empty((k_list.size, energy.size), dtype=float)

        for ik, k in enumerate(k_list):
            H00, H01 = self.half_infinte_H(k, layernum=int(layernum))
            for ie, E in enumerate(energy):
                z = E + 1j * float(eta)
                gS = self._surface_greens(H00, H01, z)
                dos[ik, ie] = -(1.0 / np.pi) * np.imag(np.trace(gS))

        return k_list, dos


