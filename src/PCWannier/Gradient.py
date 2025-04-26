import numpy as np
import matplotlib.pyplot as plt
import scipy

from typing import List, Tuple

from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager

import scipy.linalg

from PCWannier.Timer import Timer, timer
from PCWannier.IO import IO

from .GlobalData import global_data
from .CallableWrapper import CallableWrapper

from .Utils import FieldData, StateCollection, WannierTools

class Gradient:
    def __init__(self):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), global_data.incar.band_calc_num]
        self.U = [[np.eye(shape[2], shape[2], dtype=complex) for _ in range(shape[1])]for _ in range(shape[0])]
        self.G = [[np.zeros((shape[2], shape[2]), dtype=complex) for _ in range(shape[1])]for _ in range(shape[0])]
        self.dW = [[np.zeros((shape[2], shape[2]), dtype=complex) for _ in range(shape[1])]for _ in range(shape[0])]

        self.omega = [1e6, 1e6, 1e6]

        self.epsilon = 0.01

        self.rn = np.zeros((2, shape[2]), dtype=complex)

    def iter(self, err_diff: float, max_iter: int, epsilon: float=0.01):
        lastOmega = 1e6
        if max_iter == 0:
            print(f'iter n = 0')
            self.update()
            print(f"Omega: {np.sum(self.omega)},\t Omega_I: {self.omega[0]},\t Omega_OD: {self.omega[1]},\t Omega_D: {self.omega[2]}")
            self.calc(False)
            return
        
        global_data.m_set.update(self.U)

        for n in range(max_iter):
            print(f'iter n = {n + 1}')
            self.calc()
            global_data.m_set.update(self.U)
            self.update()
            err = np.abs(lastOmega - np.sum(self.omega))
            print(f"Omega: {np.sum(self.omega)},\t Omega_I: {self.omega[0]},\t Omega_OD: {self.omega[1]},\t Omega_D: {self.omega[2]}")
            if err < err_diff:
                print(f"Convergence criterion met, err_diff = {np.abs(lastOmega - np.sum(self.omega))}, total iterations: {n + 1}")
                break
            lastOmega = np.sum(self.omega)
        self.update()
        print(f"iter n = {n + 1} - end, err_diff = {err}")
        print(f"Omega: {np.sum(self.omega)},\t Omega_I: {self.omega[0]},\t Omega_OD: {self.omega[1]},\t Omega_D: {self.omega[2]}")


    def calc(self, isUpdate=True):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), global_data.incar.band_calc_num, int(len(global_data.incar.composition_of_b))]
        self.generateRn()
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.G[i][j] = np.zeros((shape[2], shape[2]), dtype=complex)
                for b in range(shape[3]):
                    mM = global_data.m_set.get(i, j, b)

                    mR = np.zeros((shape[2], shape[2]), dtype=complex)
                    mT = np.zeros((shape[2], shape[2]), dtype=complex)
                    for m in range(shape[2]):
                        for n in range(shape[2]):
                            mR[m, n] = mM[m, n] * np.conj(mM[n, n])
                            mT[m, n] = mM[m, n] / mM[n, n] * (np.imag(np.log(mM[n, n])) + np.dot(global_data.incar.b_vectors[b, :], self.rn[:, n]))
                    self.G[i][j] += global_data.incar.wb[b] * (self.operator_A(mR) - self.operator_S(mT))
                self.G[i][j] = 4 * self.G[i][j]
                self.dW[i][j] = self.epsilon * self.G[i][j]
                if isUpdate:
                    self.U[i][j] = self.U[i][j] @ scipy.linalg.expm(self.dW[i][j])


    def generateRn(self):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), global_data.incar.band_calc_num, int(len(global_data.incar.composition_of_b))]
        self.rn = np.zeros((2, shape[2]), dtype=complex)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for b in range(shape[3]):
                    mM = global_data.m_set.get(i, j, b)
                    for n in range(shape[2]):
                        self.rn[:, n] -= global_data.incar.wb[b] * global_data.incar.b_vectors[b, :] * np.imag(np.log(mM[n, n]))
        self.rn = self.rn / (shape[0] * shape[1])

    def update(self):
        shape = [len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1]), global_data.incar.band_calc_num, int(len(global_data.incar.composition_of_b))]
        self.generateRn()
        self.omega = [0, 0, 0]

        for i in range(shape[0]):
            for j in range(shape[1]):
                for b in range(shape[3]):
                    mM = global_data.m_set.get(i, j, b)

                    temp_I = shape[2]
                    temp_OD = 0
                    temp_D = 0
                    for m in range(shape[2]):
                        for n in range(shape[2]):
                            temp_I = temp_I - np.abs(mM[m, n]) ** 2
                            if m != n:
                                temp_OD += np.abs(mM[m, n]) ** 2
                        temp_D += (-np.imag(np.log(mM[m, m])) - np.dot(global_data.incar.b_vectors[b, :], self.rn[:, m])) ** 2
                    self.omega[0] += temp_I * global_data.incar.wb[b]
                    self.omega[1] += temp_OD * global_data.incar.wb[b]
                    self.omega[2] += temp_D * global_data.incar.wb[b]
        self.omega = np.real(self.omega) / (shape[0] * shape[1])
    
    def save_as(self, filename):
        IO.save_to_txt(filename, self.U, (len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])))


    @staticmethod
    def operator_A(a):
        return (a - np.conj(a).T) / 2
    @staticmethod
    def operator_S(a):
        return (a + np.conj(a).T) / (2j)
