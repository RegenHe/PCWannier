import numpy as np
import matplotlib.pyplot as plt
import scipy

from typing import List, Tuple

import scipy.linalg

from .Log import Logger
from .Timer import Timer, timer
from .IO import IO

from .GlobalData import global_data

from .Utils import FieldData, StateCollection, WannierTools

class Gradient:
    def __init__(self):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])
        B = global_data.incar.band_calc_num

        self.U = [[np.eye(B, B, dtype=complex) for _ in range(k2_sz)]for _ in range(k1_sz)]
        self.G = [[np.zeros((B, B), dtype=complex) for _ in range(k2_sz)]for _ in range(k1_sz)]
        self.dW = [[np.zeros((B, B), dtype=complex) for _ in range(k2_sz)]for _ in range(k1_sz)]

        self.omega = [1e6, 1e6, 1e6]

        self.epsilon = 0.01

        self.rn = np.zeros((2, B), dtype=complex)

    @timer("Gradient iter - ")
    def iter(self, err_diff: float, max_iter: int, epsilon: float=0.01):
        Logger.info('Starting Gradient iteration')
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])

        if 'U' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - U")
            self.U = IO.load_cell_matrix(global_data.incar.U_file, shape=(k1_sz, k2_sz))
        
        lastOmega = +np.inf
        if max_iter == 0:
            Logger.info(f'iter n = 0')
            self.update()
            Logger.info(f"Omega: {np.sum(self.omega)},\t Omega_I: {self.omega[0]},\t Omega_OD: {self.omega[1]},\t Omega_D: {self.omega[2]}")
            self.calc(False)
            return
        
        global_data.m_set.update(self.U)

        for n in range(max_iter):
            Logger.info(f'iter n = {n + 1}')
            self.calc()
            global_data.m_set.update(self.U)
            self.update()
            err = np.abs(lastOmega - np.sum(self.omega))
            Logger.info(f"Omega: {np.sum(self.omega)},\t Omega_I: {self.omega[0]},\t Omega_OD: {self.omega[1]},\t Omega_D: {self.omega[2]}")
            if err < err_diff:
                Logger.info(f"Convergence criterion met, err_diff = {np.abs(lastOmega - np.sum(self.omega))}, total iterations: {n + 1}")
                break
            if err < self.epsilon * 1e-1:
                self.epsilon *= 0.1
                Logger.info(f"err_diff = {err}, set epsilon to {self.epsilon}")
            lastOmega = np.sum(self.omega)
        if err > err_diff:
            Logger.warning(f"Convergence criteria not met, iteration limit reached, err_diff = {err}, total iterations: {n + 1}")
        self.update()
        Logger.info(f"iter n = {n + 1} - end, err_diff = {err}")
        Logger.info(f"Omega: {np.sum(self.omega)},\t Omega_I: {self.omega[0]},\t Omega_OD: {self.omega[1]},\t Omega_D: {self.omega[2]}")
        Logger.info('Gradient iteration completed')


    def calc(self, isUpdate=True):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])
        B = global_data.incar.band_calc_num
        b_sz = len(global_data.incar.composition_of_b)
        self.generateRn()
        for i in range(k1_sz):
            for j in range(k2_sz):
                self.G[i][j] = np.zeros((B, B), dtype=complex)
                for b in range(b_sz):
                    mM = global_data.m_set.get(i, j, b)

                    mR = np.zeros((B, B), dtype=complex)
                    mT = np.zeros((B, B), dtype=complex)
                    for m in range(B):
                        for n in range(B):
                            mR[m, n] = mM[m, n] * np.conj(mM[n, n])
                            mT[m, n] = mM[m, n] / mM[n, n] * (np.imag(np.log(mM[n, n])) + np.dot(global_data.incar.b_vectors[b, :], self.rn[:, n]))
                    self.G[i][j] += global_data.incar.wb[b] * (self.operator_A(mR) - self.operator_S(mT))
                self.G[i][j] = 4 * self.G[i][j]
                self.dW[i][j] = self.epsilon * self.G[i][j]
                if isUpdate:
                    if not np.isclose(np.abs(np.trace(self.U[i][j] @ self.U[i][j].conj().T)), B, rtol=1e-8):
                        Logger.warning(f"||U[{i}][{j}]|| = {np.abs(np.trace(self.U[i][j] @ self.U[i][j].conj().T))}, updating it")
                        u, s, vh = np.linalg.svd(self.U[i][j])
                        self.U[i][j] = u @ vh
                    self.U[i][j] = self.U[i][j] @ scipy.linalg.expm(self.dW[i][j])


    def generateRn(self):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])
        B = global_data.incar.band_calc_num
        b_sz = len(global_data.incar.composition_of_b)

        self.rn = np.zeros((2, B), dtype=complex)
        for i in range(k1_sz):
            for j in range(k2_sz):
                for b in range(b_sz):
                    mM = global_data.m_set.get(i, j, b)
                    for n in range(B):
                        self.rn[:, n] -= global_data.incar.wb[b] * global_data.incar.b_vectors[b, :] * np.imag(np.log(mM[n, n]))
        self.rn = self.rn / (k1_sz * k2_sz)
        return self.rn

    def update(self):
        k1_sz = len(global_data.incar.k_points[0])
        k2_sz = len(global_data.incar.k_points[1])
        B = global_data.incar.band_calc_num
        b_sz = len(global_data.incar.composition_of_b)

        self.generateRn()
        self.omega = [0, 0, 0]

        for i in range(k1_sz):
            for j in range(k2_sz):
                for b in range(b_sz):
                    mM = global_data.m_set.get(i, j, b)

                    temp_I = B
                    temp_OD = 0
                    temp_D = 0
                    for m in range(B):
                        for n in range(B):
                            temp_I = temp_I - np.abs(mM[m, n]) ** 2
                            if m != n:
                                temp_OD += np.abs(mM[m, n]) ** 2
                        temp_D += (-np.imag(np.log(mM[m, m])) - np.dot(global_data.incar.b_vectors[b, :], self.rn[:, m])) ** 2
                    self.omega[0] += temp_I * global_data.incar.wb[b]
                    self.omega[1] += temp_OD * global_data.incar.wb[b]
                    self.omega[2] += temp_D * global_data.incar.wb[b]
        self.omega = np.real(self.omega) / (k1_sz * k2_sz)

    def set_center(self, center):
        self.generateRn()
        phase = np.zeros((len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])), dtype=object)
        for i in range(len(global_data.incar.k_points[0])):
            for j in range(len(global_data.incar.k_points[1])):
                r = (self.rn.T - center) @ np.array(global_data.incar.real_lattice_vectors) * global_data.incar.lattice_const
                k = WannierTools.get_kx_ky([i, j])
                sign = 1
                if global_data.incar.dataset_type.lower() == 'comsol':
                    sign = -1
                phase[i, j] = np.diag(np.exp(-1j * sign * np.dot(k, r.T)))
                self.U[i][j] = phase[i, j] @ self.U[i][j]
        global_data.m_set.update(self.U)
        return phase
    
    def save_as(self, filename):
        IO.save_to_txt(filename, self.U, (len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])))


    @staticmethod
    def operator_A(a):
        return (a - np.conj(a).T) / 2
    @staticmethod
    def operator_S(a):
        return (a + np.conj(a).T) / (2j)
