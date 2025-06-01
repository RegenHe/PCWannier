import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree

import copy

from typing import List, Tuple

from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager

from .IO import IO
from .Log import Logger
from .Timer import Timer, timer
from .GlobalData import global_data
from .CallableWrapper import CallableWrapper

from .IncarParser import IncarData

class Mesh:
    def __init__(self, vertices: np.ndarray, elements: np.ndarray, edge: np.ndarray=None) -> None:
        if vertices is None:
            self.vertices = None
            self.mindist = None
        else:
            self.vertices: np.ndarray = np.array(vertices)
            tree = cKDTree(self.vertices)
            dists, idxs = tree.query(self.vertices, k=2)
            self.mindist = np.min(dists[:, 1])
        
        if elements is None:
            self.elements = None
        else:
            self.elements: np.ndarray = np.array(elements)
        
        if edge is None:
            self.edge = None
        else:
            self.edge: np.ndarray = np.array(edge)
        # self.edge_index: np.ndarray = np.unique(self.edge.flatten())

    def func(self, f, offset=[0, 0]):
        return [f(p[0] - offset[0], p[1] - offset[1]) for p in self.vertices]
    
    def rfunc(self, f, offset=[0, 0], ang=0):
        return [f(np.sqrt((p[0] - offset[0]) ** 2 + (p[1] - offset[1]) ** 2), np.atan2(p[0] - offset[0], p[1] - offset[1]) - np.radians(ang)) for p in self.vertices]
        

    def __repr__(self) -> str:
        return f"Mesh(vertices={self.vertices}, elements={self.elements})"
    
    def plot(self) -> None:
        fig, ax = plt.subplots()
        for element in self.elements:
            triangle_vertices = self.vertices[element]
            
            triangle = plt.Polygon(triangle_vertices, edgecolor='black', fill=None)
            ax.add_patch(triangle)

        if self.edge is not None:
            for line in self.edge:
                point1, point2 = self.vertices[line[0]], self.vertices[line[1]]
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='red', linewidth=2)

        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        min_x, max_x = np.min(self.vertices[:, 0]), np.max(self.vertices[:, 0])
        min_y, max_y = np.min(self.vertices[:, 1]), np.max(self.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)

        plt.show()
    def save_fig(self, filename):
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        fig, ax = plt.subplots()
        for element in self.elements:
            triangle_vertices = self.vertices[element]
            
            triangle = plt.Polygon(triangle_vertices, edgecolor='black', fill=None)
            ax.add_patch(triangle)

        if self.edge is not None:
            for line in self.edge:
                point1, point2 = self.vertices[line[0]], self.vertices[line[1]]
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='red', linewidth=2)

        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        min_x, max_x = np.min(self.vertices[:, 0]), np.max(self.vertices[:, 0])
        min_y, max_y = np.min(self.vertices[:, 1]), np.max(self.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    def extension(self, n: list) -> List:
        if n[0] < 1 or n[1] < 1:
            err_msg = "n must be greater than 1"
            Logger.error(err_msg)
            raise ValueError(err_msg)
        if self.vertices is None:
            err_msg = "Mesh must be initialized"
            Logger.error(err_msg)
            raise ValueError(err_msg)
        
        original_vertices = self.vertices.copy()
        original_elements = self.elements.copy()

        # TODO: using the edge to find the same vertices
        # original_index = self.edge_index.copy()
        
        index_map = None
        space_to_original_mapping = list(range(len(original_vertices)))

        for i in range(n[0]):
            for j in range(n[1]):
                if i == 0 and j == 0:
                    continue
                offset_x = (global_data.incar.real_lattice_vectors[0][0] * i + global_data.incar.real_lattice_vectors[1][0] * j) * global_data.incar.lattice_const
                offset_y = (global_data.incar.real_lattice_vectors[0][1] * i + global_data.incar.real_lattice_vectors[1][1] * j) * global_data.incar.lattice_const

                new_elements = original_elements + np.max(self.elements) + 1
                # new_vertices = original_vertices + np.array([offset_x + 1e-3 * i, offset_y + 1e-3 * j])
                new_vertices = original_vertices + np.array([offset_x, offset_y])
                vertex_idx_in_new_vertices, vertex_idx_in_vertices = self.match(new_vertices, self.vertices)
                for k in range(len(vertex_idx_in_new_vertices)):
                    new_elements[new_elements == (vertex_idx_in_new_vertices[k] + np.max(self.elements) + 1)] = vertex_idx_in_vertices[k]
                
                self.elements = np.vstack((self.elements, new_elements))
                self.vertices = np.vstack((self.vertices, new_vertices))

                space_to_original_mapping = np.hstack((space_to_original_mapping, np.arange(len(original_vertices))))

                index_map, space_to_original_mapping = self.rebuild_index(space_to_original_mapping)
        
        offset_x = (global_data.incar.real_lattice_vectors[0][0] * np.floor((n[0] - 1) / 2) + global_data.incar.real_lattice_vectors[1][0] * np.floor((n[1] - 1) / 2)) * global_data.incar.lattice_const
        offset_y = (global_data.incar.real_lattice_vectors[0][1] * np.floor((n[0] - 1) / 2) + global_data.incar.real_lattice_vectors[1][1] * np.floor((n[1] - 1) / 2)) * global_data.incar.lattice_const
        self.vertices = self.vertices - np.array([offset_x, offset_y])
        return space_to_original_mapping

    def match(self, new_vertices, vertices) -> Tuple[np.ndarray, np.ndarray]:
        tree = cKDTree(new_vertices)
        dists, idxs = tree.query(vertices, k=1)
        vertex_idx_in_vertices = np.where(dists < self.mindist * 0.5)[0]

        vertex_idx_in_new_vertices = []
        if vertex_idx_in_vertices.size > 0:
                t_tree = cKDTree(new_vertices)
                for i in vertex_idx_in_vertices:
                    t_dists, t_idxs = t_tree.query(vertices[i], k=1)
                    vertex_idx_in_new_vertices.append(t_idxs)
        else:
            Logger.warning(f"No points found")

        return np.array(vertex_idx_in_new_vertices), vertex_idx_in_vertices
    
    def rebuild_index(self, space_to_original_mapping=None):
        used_indices = set(self.elements.flatten())
        
        new_vertices = [self.vertices[i] for i in sorted(used_indices)]
        new_vertices = np.array(new_vertices)

        old_to_new_index = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_indices))}
        if space_to_original_mapping is None:
            space_to_original_mapping = list(range(len(self.vertices)))
        else:
            space_to_original_mapping = [space_to_original_mapping[i] for i in sorted(old_to_new_index.keys())]
        # space_to_original_mapping = list(sorted(used_indices))
        
        new_elements = []
        for element in self.elements:
            new_element = [old_to_new_index[idx] for idx in element]
            new_elements.append(new_element)
        
        self.vertices = new_vertices
        self.elements = np.array(new_elements)

        return old_to_new_index, space_to_original_mapping
    
    def __deepcopy__(self, memo=None):
        return Mesh(copy.deepcopy(self.vertices, memo), copy.deepcopy(self.elements, memo), copy.deepcopy(self.edge, memo))


class RawData:
    def __init__(self, point_matrix: np.ndarray, value_matrix: np.ndarray) -> None:
        self.point_matrix: np.ndarray = np.array(point_matrix)
        self.value_matrix: np.ndarray = np.array(value_matrix)

    def __repr__(self) -> str:
        return f"RawData(point_matrix={self.point_matrix}, value_matrix={self.value_matrix})"
    
class FieldData:
    def __init__(self, name: str, mesh: Mesh, value: np.ndarray) -> None:
        self.name: str = name
        self.mesh: Mesh = mesh
        self.field: np.ndarray = np.array(value)

    def __repr__(self) -> str:
        return f"FieldData(point_matrix={self.mesh}, value_matrix={self.field})"
    
    def plot(self) -> None:
        fig, ax = plt.subplots()
        triang = Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements)

        plt.tricontourf(triang, np.real(self.field), levels=255, cmap='bwr')
        plt.clim(-max(np.abs(self.field)), max(np.abs(self.field)))
        plt.colorbar(label='Real Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.mesh.vertices[:, 0]), np.max(self.mesh.vertices[:, 0])
        min_y, max_y = np.min(self.mesh.vertices[:, 1]), np.max(self.mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.show()
    def save_fig(self, filename):
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        fig, ax = plt.subplots()
        triang = Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements)

        plt.tricontourf(triang, np.real(self.field), levels=255, cmap='bwr')
        plt.clim(-max(np.abs(self.field)), max(np.abs(self.field)))
        plt.colorbar(label='Real Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.mesh.vertices[:, 0]), np.max(self.mesh.vertices[:, 0])
        min_y, max_y = np.min(self.mesh.vertices[:, 1]), np.max(self.mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        Logger.info(f"figure successfully saved to {filename}")


class StateCollection:
    def __init__(self, name: str, mesh: Mesh) -> None:
        self.name: str = name
        self.mesh: Mesh = mesh
        self.field: list = []
        self.epsilon: np.array = None
        self.normalization: float = 0.0

        self.is_normalized = False
        self.is_bloch = False

        self.extention_mesh = None
        self.space_to_original_mapping = None

        self.extention_epsilon: np.array = None

        self.E: list = []

    def add_field(self, state: np.ndarray, i: int, j: int, k: int) -> None:
        while len(self.field) <= i:
            self.field.append([])
        while len(self.field[i]) <= j:
            self.field[i].append([])
        while len(self.field[i][j]) <= k:
            self.field[i][j].append(None)
        
        self.field[i][j][k] = state

    def __getitem__(self, index: tuple) -> np.ndarray:
        if self.field is None:
            err_msg = "Field data is not initialized"
            Logger.error(err_msg)
            raise IndexError(err_msg)

        i, j, n = index
        if not (0 <= i < len(self.mesh) and 0 <= j < len(self.mesh) and 0 <= n < len(self.field)):
            err_msg = "Index out of range"
            Logger.error(err_msg)
            raise IndexError(err_msg)

        return self.field[i][j][n]
    
    @timer("Normalize - ")
    def normalize(self) -> None:
        if self.field is None:
            err_msg = "Field data is not initialized"
            Logger.error(err_msg)
            raise ValueError(err_msg)
        
        if 'N' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - N")
            self.normalization = IO.load_cell_matrix(global_data.incar.N_file, shape=(len(self.field[0][0]),))
            self.normalization = np.transpose(np.array([p for p in self.normalization[:]]), (1, 2, 0))
            self.is_normalized = True
            for i in range(len(self.field)):
                for j in range(len(self.field[0])):
                    for n in range(len(self.field[0][0])):
                        if self.normalization[i][j][n] == 0.0:
                            raise ValueError(f"Normalization failed for field ({i}, {j}, {n})")
                        self.field[i][j][n] /= np.sqrt(self.normalization[i][j][n])
            return
        
        self.normalization = [[[None for _ in range(len(self.field[0][0]))] for _ in range(len(self.field[0]))] for _ in range(len(self.field))]
        
        futures = []
        result_queue = Manager().Queue()
        def process_batch(i, j, n_range, result_queue):
            for n in n_range:
                fd = FieldData(self.name, self.mesh, np.abs(self.field[i][j][n]) ** 2 * self.epsilon)
                result_queue.put((i, j, n, WannierTools.integrate_over_mesh(fd)))

        with ProcessPoolExecutor(max_workers=global_data.threads) as executor:
            for i in range(len(self.field)):
                for j in range(len(self.field[i])):
                    futures.append(
                        executor.submit(
                            CallableWrapper(process_batch), i, j, range(len(self.field[i][j])), result_queue
                            ))
            
            wait(futures, return_when='ALL_COMPLETED')
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    raise e

        while not result_queue.empty():
            i, j, n, result = result_queue.get()
            self.normalization[i][j][n] = result
            # Logger.info(f"Normalization for field ({i}, {j}, {n}) => {result}")
        
        self.is_normalized = True
        for i in range(len(self.field)):
            for j in range(len(self.field[0])):
                for n in range(len(self.field[0][0])):
                    if self.normalization[i][j][n] == 0.0:
                        raise ValueError(f"Normalization failed for field ({i}, {j}, {n})")
                    self.field[i][j][n] /= np.sqrt(self.normalization[i][j][n])
        
        if not global_data.incar.N_file.lower == "false":
            IO.save_to_txt(global_data.incar.N_file, np.transpose(self.normalization, (2, 0, 1)), (len(self.field[0][0])))
        
    def turn_to_Bloch(self) -> None:
        if self.field is None:
            err_msg = "Field data is not initialized"
            Logger.error(err_msg)
            raise ValueError(err_msg)
        
        if self.is_bloch:
            Logger.info("Field data is already in Bloch form")
            return
        self.is_bloch = True

        for i in range(len(self.field)):
            for j in range(len(self.field[0])):
                for n in range(len(self.field[0][0])):
                    phase = self.get_phase(i, j)
                    self.field[i][j][n] = np.conj(phase) * self.field[i][j][n]
    
    def get_phase(self, i: int, j: int):
        if global_data.incar.dataset_type.lower() == 'comsol':
            sign = -1
        k = WannierTools.get_kx_ky([i, j])
        return np.exp(1j * sign * np.dot(self.mesh.vertices, k))
    
    def get_phase_k(self, k: np.ndarray):
        if global_data.incar.dataset_type.lower() == 'comsol':
            sign = -1
        return np.exp(1j * sign * np.dot(self.mesh.vertices, k))
    
    def get_extention_phase(self, i: int, j: int):
        if global_data.incar.dataset_type.lower() == 'comsol':
            sign = -1
        k = WannierTools.get_kx_ky([i, j])
        return np.exp(1j * sign * np.dot(self.extention_mesh.vertices, k))
    
    @timer("Extention - ")
    def extention(self, n: List) -> None:
        self.extention_mesh = copy.deepcopy(self.mesh)
        self.space_to_original_mapping = self.extention_mesh.extension(n)
        if self.epsilon is not None:
            self.get_extention_epsilon()
    
    def get_extention_field(self, i: int, j: int, n: int) -> List:
        if self.extention_mesh is None:
            err_msg = "The field has not been extended"
            Logger.error(err_msg)
            raise ValueError(err_msg)
        return np.array([self.field[i][j][n][k] / np.sqrt(global_data.incar.extension[0] * global_data.incar.extension[1]) for k in self.space_to_original_mapping])
    
    def get_extention_epsilon(self) -> List:
        if self.extention_mesh is None:
            err_msg = "The field has not been extended"
            Logger.error(err_msg)
            raise ValueError(err_msg)
        if self.extention_epsilon is None:
            self.extention_epsilon = np.array([self.epsilon[k] for k in self.space_to_original_mapping])
        return self.extention_epsilon
        
    def get_zero_field(self):
        return [0.0 + 0.0j] * self.mesh.vertices.shape[0]
    
    def get_zero_extension_field(self):
        return [0.0 + 0.0j] * self.extention_mesh.vertices.shape[0]
    
    def plot_field(self, i: int, j: int, n: int) -> None:
        fig, ax = plt.subplots()
        triang = Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements)

        plt.tricontourf(triang, np.real(self.field[i][j][n]), levels=255, cmap='jet')
        plt.colorbar(label='Real Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.mesh.vertices[:, 0]), np.max(self.mesh.vertices[:, 0])
        min_y, max_y = np.min(self.mesh.vertices[:, 1]), np.max(self.mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.show()
    
    def plot_extention_field(self, i: int, j: int, n: int) -> None:
        fig, ax = plt.subplots()
        triang = Triangulation(self.extention_mesh.vertices[:, 0], self.extention_mesh.vertices[:, 1], self.extention_mesh.elements)

        field = self.get_extention_field(i, j, n)
        plt.tricontourf(triang, np.real(field), levels=255, cmap='jet')
        plt.colorbar(label='Real Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.extention_mesh.vertices[:, 0]), np.max(self.extention_mesh.vertices[:, 0])
        min_y, max_y = np.min(self.extention_mesh.vertices[:, 1]), np.max(self.extention_mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.show()

    def plot_epsilon(self) -> None:
        fig, ax = plt.subplots()
        triang = Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements)

        plt.tricontourf(triang, np.real(self.epsilon), levels=255, cmap='jet')
        plt.colorbar(label='Real Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.mesh.vertices[:, 0]), np.max(self.mesh.vertices[:, 0])
        min_y, max_y = np.min(self.mesh.vertices[:, 1]), np.max(self.mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.show()
    
    def plot_extention_epsilon(self) -> None:
        fig, ax = plt.subplots()
        triang = Triangulation(self.extention_mesh.vertices[:, 0], self.extention_mesh.vertices[:, 1], self.extention_mesh.elements)

        plt.tricontourf(triang, np.real(self.extention_epsilon), levels=255, cmap='jet')
        plt.colorbar(label='Real Part')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        min_x, max_x = np.min(self.extention_mesh.vertices[:, 0]), np.max(self.extention_mesh.vertices[:, 0])
        min_y, max_y = np.min(self.extention_mesh.vertices[:, 1]), np.max(self.extention_mesh.vertices[:, 1])
        
        margin = 0.1
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        plt.show()

    def __deepcopy__(self, memo=None):
        sc = StateCollection(copy.deepcopy(self.name, memo), copy.deepcopy(self.mesh, memo))
        sc.field = copy.deepcopy(self.field)
        sc.epsilon = copy.deepcopy(self.epsilon)
        sc.normalization = copy.deepcopy(self.normalization)

        sc.is_normalized = copy.deepcopy(self.is_normalized)
        sc.is_bloch = copy.deepcopy(self.is_bloch)
        return sc
    
    def __repr__(self) -> str:
        return f"StateCollection(point_matrix={self.mesh}, field_number={len(self.field)})"



class WannierTools:
    def __init__(self) -> None:
        self.init = False

    def preprocess(self) -> None:
        if np.array_equal(global_data.incar.reciprocal_lattice_vectors, np.array([[0, 0], [0, 0]])):
            v = np.linalg.inv(global_data.incar.real_lattice_vectors) @ np.eye(len(global_data.incar.real_lattice_vectors))
            Logger.info(f"reciprocal_lattice_vectors will be set to: {v.T}")
            global_data.incar.reciprocal_lattice_vectors = v.T
        
        if self.init is False:
            self.init = True
            global_data.incar.composition_of_b = [global_data.incar.composition_of_b, [[-v for v in sublist] for sublist in global_data.incar.composition_of_b]]
            global_data.incar.composition_of_b = [item for sublist in global_data.incar.composition_of_b for item in sublist]

            global_data.incar.b_vectors = []
            for i in range(len(global_data.incar.composition_of_b)):
                global_data.incar.b_vectors.append((global_data.incar.composition_of_b[i][0] * global_data.incar.reciprocal_lattice_vectors[0] / len(global_data.incar.k_points[0]) * 2 * np.pi / global_data.incar.lattice_const
                                            + global_data.incar.composition_of_b[i][1] * global_data.incar.reciprocal_lattice_vectors[1] / len(global_data.incar.k_points[1]) * 2 * np.pi / global_data.incar.lattice_const).tolist())
                
            global_data.incar.b_vectors = np.array(global_data.incar.b_vectors)
            mat_a = np.eye(2).reshape(-1, 1)
            mat_b = np.zeros((2 ** 2, len(global_data.incar.composition_of_b)))
            for i in range(2):
                for j in range(2):
                    for k in range(len(global_data.incar.composition_of_b)):
                        mat_b[i * 2 + j][k] = global_data.incar.b_vectors[k][i] * global_data.incar.b_vectors[k][j]
            
            global_data.incar.wb = np.linalg.pinv(mat_b) @ mat_a
            global_data.incar.wb = [item for sublist in global_data.incar.wb for item in sublist]
        
        global_data.incar.band_calc_num = 0
        for p in global_data.incar.projections:
            global_data.incar.band_calc_num += len(p['states'])

    def set_incar(self, incar_data: IncarData) -> None:
        global_data.incar = incar_data

    @staticmethod
    def neighbor_reciprocal_lattice_vectors(k: list, direction: int) -> np.ndarray:
        n_k1_idx = int(np.mod(k[0] + global_data.incar.composition_of_b[direction][0], len(global_data.incar.k_points[0])))
        n_k2_idx = int(np.mod(k[1] + global_data.incar.composition_of_b[direction][1], len(global_data.incar.k_points[1])))
        k_ = None
        if n_k1_idx != k[0] + global_data.incar.composition_of_b[direction][0] or n_k2_idx != k[1] + global_data.incar.composition_of_b[direction][1]:
            k_ = [0, 0]
            k_[0] = int(k[0] + global_data.incar.composition_of_b[direction][0])
            k_[1] = int(k[1] + global_data.incar.composition_of_b[direction][1])
        return n_k1_idx, n_k2_idx, k_
    
    @staticmethod
    def get_kx_ky(k: list) -> np.ndarray:
        if k[0] >= global_data.incar.k_points[0].size or k[1] >= global_data.incar.k_points[1].size or k[0] < 0 or k[1] < 0:
            kx = (((global_data.incar.k_points[0][1] - global_data.incar.k_points[0][0]) * k[0] + global_data.incar.k_points[0][0]) * global_data.incar.reciprocal_lattice_vectors[0][0] + ((global_data.incar.k_points[1][1] - global_data.incar.k_points[1][0]) * k[1] + global_data.incar.k_points[1][0]) * global_data.incar.reciprocal_lattice_vectors[1][0]) * 2 * np.pi / global_data.incar.lattice_const
            ky = (((global_data.incar.k_points[0][1] - global_data.incar.k_points[0][0]) * k[0] + global_data.incar.k_points[0][0]) * global_data.incar.reciprocal_lattice_vectors[0][1] + ((global_data.incar.k_points[1][1] - global_data.incar.k_points[1][0]) * k[1] + global_data.incar.k_points[1][0]) * global_data.incar.reciprocal_lattice_vectors[1][1]) * 2 * np.pi / global_data.incar.lattice_const
            return np.array([kx, ky])
        kx = global_data.incar.k_points[0][k[0]] * global_data.incar.reciprocal_lattice_vectors[0][0] * 2 * np.pi / global_data.incar.lattice_const + global_data.incar.k_points[1][k[1]] * global_data.incar.reciprocal_lattice_vectors[1][0] * 2 * np.pi / global_data.incar.lattice_const
        ky = global_data.incar.k_points[0][k[0]] * global_data.incar.reciprocal_lattice_vectors[0][1] * 2 * np.pi / global_data.incar.lattice_const + global_data.incar.k_points[1][k[1]] * global_data.incar.reciprocal_lattice_vectors[1][1] * 2 * np.pi / global_data.incar.lattice_const
        return np.array([kx, ky])
    
    @staticmethod
    def integrate_over_triangle(vertices: np.ndarray, data_on_triangle: np.ndarray) -> complex:
        jacobian = np.array([[vertices[1, 0] - vertices[0, 0], vertices[2, 0] - vertices[0, 0]], [vertices[1, 1] - vertices[0, 1], vertices[2, 1] - vertices[0, 1]]])
        return np.sum(data_on_triangle) * np.abs(np.linalg.det(jacobian)) / 6.0
    
    @staticmethod
    def integrate_over_mesh(data: FieldData) -> complex:
        total_integral = 0.0 + 0.0j
        for idx in range(len(data.mesh.elements)):
            element = data.mesh.elements[idx]
            vertices = data.mesh.vertices[element, :]
            data_on_triangle = data.field[element]
            total_integral += WannierTools.integrate_over_triangle(vertices, data_on_triangle)
        return total_integral


if __name__ == "__main__":
    from PCWannier import IncarParser
    parser_data = IncarParser.IncarParser("examples/incar")
    wtools = WannierTools()
    wtools.set_incar(parser_data.parse_file())
    wtools.preprocess()
    Logger.info(global_data.incar)