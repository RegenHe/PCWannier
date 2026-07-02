import os
import math

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

import copy

from .GlobalData import global_data
from .Log import Logger
from .Timer import Timer, timer
from .IO import IO

from . import MeshData

from .StateInitializer import StateBases
from .Utils import FieldData, StateCollection, WannierTools

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

class Fatband:
    def __init__(self):
        self.state_collection = None
        self.is_bloch = False
        self.is_normalized = False

    def parser(self, input_file: str):
        self.incar = IncarParser(input_file).parse_file()
        self.incar.validate()
        Logger.info(self.incar)
        global_data.incar = self.incar

        if np.array_equal(self.incar.reciprocal_lattice_vectors, np.array([[0, 0], [0, 0]])):
            v = np.linalg.inv(self.incar.real_lattice_vectors) @ np.eye(len(self.incar.real_lattice_vectors))
            Logger.info(f"reciprocal_lattice_vectors will be set to: {v.T}")
            self.incar.reciprocal_lattice_vectors = v.T
        self.build_klist()
        
    def run(self):
        if self.incar.dataset_type.lower() == "comsol":
            self.mesh = MeshData.load_comsol_mesh(self.incar.mesh_file)
            self.raw_data = MeshData.load_comsol_data(self.incar.dataset_file)
            self.epsilon = MeshData.load_comsol_data(self.incar.dielectric_file)
            self.epsilon.value_matrix = self.epsilon.value_matrix.flatten()
            if self.incar.E_file.lower() != 'false':
                self.E_raw_data = MeshData.load_comsol_data(self.incar.E_file)
        
        self.state_collection = StateCollection("psi", self.mesh)
        idxs, _ = MeshData.match_data_to_mesh(self.mesh, self.raw_data)
        self.raw_data.value_matrix = self.raw_data.value_matrix[idxs]
        
        idxs, _ = MeshData.match_data_to_mesh(self.mesh, self.epsilon)
        self.state_collection.epsilon = self.epsilon.value_matrix[idxs].flatten()

        t_feild = np.zeros((self.raw_data.value_matrix.shape[0], self.incar.kpoint_num, self.raw_data.value_matrix.shape[1] // self.incar.kpoint_num), dtype=complex)
        for p in range(self.raw_data.value_matrix.shape[0]):
            t_feild[p] = self.raw_data.value_matrix[p].reshape((self.incar.kpoint_num, -1), order='C')

        self.state_collection.Rfield = [[None for _ in range(self.raw_data.value_matrix.shape[1] // self.incar.kpoint_num)] for _ in range(self.incar.kpoint_num)]
        for i in range(self.incar.kpoint_num):
            for k in range(self.raw_data.value_matrix.shape[1] // self.incar.kpoint_num):
                self.state_collection.Rfield[i][k] = t_feild[:, i, k]

        self.state_collection.E = self.E_raw_data.value_matrix[0].reshape((self.incar.kpoint_num, -1), order='C')
        
        self.turn_to_Bloch()
        self.normalize()

        self.state_collection.extention_mesh = copy.deepcopy(self.state_collection.mesh)
        self.state_collection.space_to_original_mapping = self.state_collection.extention_mesh.extension(self.incar.extension, self.incar.real_lattice_vectors, self.incar.lattice_const)
        self.state_collection.get_extention_epsilon()

        H_list = []
        for p in self.incar.projections:
            for state in p['states']:
                if isinstance(state, dict) and 'lc_states' in state:
                    lc_states = state['lc_states']
                    lc_coeffs = state['lc_coeffs']

                    def f(r, phi, _lc_states=lc_states, _lc_coeffs=lc_coeffs):
                        s = 0.0 + 0.0j
                        for (n, l, z), c in zip(_lc_states, _lc_coeffs):
                            if self.incar.radius_func:
                                s += c * StateBases.Radial(n, l)(r, z) * StateBases.Angular(l)(phi)
                            else:
                                s += c * 1 * StateBases.Angular(l)(phi)
                        return s
                else:
                    n, l, z = state
                    def f(r, phi, _n=n, _l=l, _z=z):
                        if self.incar.radius_func:
                            return StateBases.Radial(_n, _l)(r, _z) * StateBases.Angular(_l)(phi)
                        else:
                            return 1 * StateBases.Angular(_l)(phi)

                cart_position = (p['frac_position'][0] * np.array(self.incar.real_lattice_vectors[0]) +
                                p['frac_position'][1] * np.array(self.incar.real_lattice_vectors[1]) +
                                np.array(self.incar.origin)) * self.incar.lattice_const

                h = self.state_collection.extention_mesh.rfunc(f, cart_position, p['xaxis_angluar'])
                H_list.append(np.asarray(h, dtype=np.complex128))
        H = np.column_stack(H_list)

        eps = self.state_collection.extention_epsilon
        abs2 = np.abs(H)**2
        if np.isscalar(eps):
            F = (abs2 * float(eps)).astype(np.complex128, copy=False)
        else:
            eps_arr = np.asarray(eps, dtype=np.complex128)
            F = (abs2 * eps_arr[:, None]).astype(np.complex128, copy=False)

        fd = FieldData('', self.state_collection.extention_mesh, F)
        norms = WannierTools.integrate_over_mesh(fd, chunk_size=2048)
        norms = np.where(norms == 0, 1.0, norms)

        G = H / np.sqrt(norms)[None, :]

        g = [G[:, k] for k in range(G.shape[1])]
        self.matA = np.zeros((self.incar.kpoint_num, self.raw_data.value_matrix.shape[1] // self.incar.kpoint_num, G.shape[1]), dtype=complex)
        for i in range(self.incar.kpoint_num):
            Nv = self.state_collection.extention_mesh.vertices.shape[0]

            G = np.column_stack(g).astype(np.complex128, copy=False)
            if G.shape[0] != Nv:
                G = G.T

            for m in range(self.raw_data.value_matrix.shape[1] // self.incar.kpoint_num):
                field = np.array([self.state_collection.Rfield[i][m][k] for k in self.state_collection.space_to_original_mapping])
                
                base = self.state_collection.extention_epsilon * np.conj(field)

                F = (base[:, None] * G)
                fd = FieldData('', self.state_collection.extention_mesh, F)
                vals = WannierTools.integrate_over_mesh(fd, chunk_size=2048)

                self.matA[i][m, :vals.shape[0]] = vals
        Logger.info(f"matA shape: {self.matA.shape}")

        idx = 0
        for p in self.incar.projections:
            for state in p['states']:
                self.linecollection_fatband(np.arange(0, len(self.klist)), self.state_collection.E, np.abs(self.matA[:, :, idx])**2)
                if not os.path.exists(self.incar.fatband_path):
                    os.makedirs(self.incar.fatband_path)
                plt.savefig(self.incar.fatband_path + f"/state-{idx}-real.png", dpi=300, bbox_inches='tight')
                plt.close()
                idx += 1
    
    def linecollection_fatband(self, kdist, E, w, cmap='Reds', vmin=0.0, vmax=1.0, lw=2.0):
        fig, ax = plt.subplots(figsize=(7, 4))
        norm = Normalize(vmin=vmin, vmax=vmax)

        Nk, Nband = E.shape
        for n in range(Nband):
            x = kdist
            y = E[:, n]
            points = np.column_stack([x, y]).reshape(-1, 1, 2)
            segs = np.concatenate([points[:-1], points[1:]], axis=1)

            cvals = 0.5 * (w[:-1, n] + w[1:, n])

            lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=lw)
            lc.set_array(cvals)
            ax.add_collection(lc)

        ax.set_xlim(kdist.min(), kdist.max())
        ax.set_xlabel(r'$k$-path')
        ax.set_ylabel('E')
        ax.autoscale(enable=True, axis='y', tight=True)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Projection weight')
        plt.tight_layout()
        

    def turn_to_Bloch(self) -> None:
        if self.is_bloch:
            Logger.info("Field data is already in Bloch form")
            return
        self.is_bloch = True

        for i in range(len(self.state_collection.Rfield)):
            for n in range(len(self.state_collection.Rfield[0])):
                phase = self.get_phase(i)
                self.state_collection.Rfield[i][n] = np.conj(phase) * self.state_collection.Rfield[i][n]

    def normalize(self) -> None:
        self.normalization = [[None for _ in range(len(self.state_collection.Rfield[0]))] for _ in range(len(self.state_collection.Rfield))]

        Nv = self.mesh.vertices.shape[0]
        arr0 = np.asarray(self.state_collection.Rfield[0][0])
        if arr0.ndim == 1:
            arr0 = arr0[None, :]
        elif arr0.ndim == 2 and arr0.shape[1] != Nv:
            arr0 = arr0.T

        eps = np.asarray(self.state_collection.epsilon)
        if eps.shape != arr0.shape:
            if eps.ndim == 2 and eps.T.shape == arr0.shape:
                eps = eps.T
            elif eps.ndim == 1 and eps.shape[0] == Nv:
                eps = np.broadcast_to(eps[None, :], arr0.shape)
            else:
                raise ValueError(f"epsilon shape {eps.shape} != field shape {arr0.shape}")
        
        for i in range(len(self.state_collection.Rfield)):
            arr = np.asarray(self.state_collection.Rfield[i])
            F = (np.abs(arr)**2 * eps).T

            fd = FieldData(self.state_collection.name, self.mesh, F.astype(np.complex128, copy=False))
            vals = WannierTools.integrate_over_mesh(fd, chunk_size=2048)

            # A = FieldData("A", self.mesh, np.conj(arr * eps).astype(np.complex128, copy=False))
            # B = FieldData("B", self.mesh, arr.astype(np.complex128, copy=False))

            # vals = WannierTools.integrate_over_mesh(A, other=B, chunk_size=2048)
            self.normalization[i][:vals.shape[0]] = vals
        
        self.is_normalized = True
        for i in range(len(self.state_collection.Rfield)):
            for n in range(len(self.state_collection.Rfield[0])):
                if self.normalization[i][n] == 0.0:
                    raise ValueError(f"Normalization failed for field ({i}, {n})")
                self.state_collection.Rfield[i][n] /= np.sqrt(self.normalization[i][n])
    
    def turn_to_Bloch(self) -> None:
        if self.is_bloch:
            Logger.info("Field data is already in Bloch form")
            return
        self.is_bloch = True

        for i in range(len(self.state_collection.Rfield)):
            for n in range(len(self.state_collection.Rfield[0])):
                phase = self.get_phase(i)
                self.state_collection.Rfield[i][n] = np.conj(phase) * self.state_collection.Rfield[i][n]
    def get_kx_ky(self, k1, k2) -> Tuple[float, float]:
        kx = self.incar.reciprocal_lattice_vectors[0, 0] * k1 + self.incar.reciprocal_lattice_vectors[1, 0] * k2
        ky = self.incar.reciprocal_lattice_vectors[0, 1] * k1 + self.incar.reciprocal_lattice_vectors[1, 1] * k2
        return kx, ky
    
    def get_phase(self, i):
        k = self.klist[i]
        kx, ky = self.get_kx_ky(k[0], k[1])
        if self.incar.dataset_type.lower() == 'comsol':
            sign = -1
        phase = np.exp(1j * sign * (self.mesh.vertices @ np.array([kx, ky])))
        return phase
    
    def build_klist(self):
        self.klist = []
        path = self.incar.k_path
        S = len(path)
        for s in range(S):
            a = np.array(path[s]['point'], dtype=float)
            b = np.array(path[(s+1) % S]['point'], dtype=float)
            n = int(path[s]['num'])
            if n < 2:
                raise ValueError(f"num must be >=2 at segment {s}")
            m = n if s != S - 1 else n - 1
            for l in range(m):
                t = l / (n - 1)
                self.klist.append(((1.0 - t) * a + t * b).tolist())

        self.klist = np.array(self.klist, dtype=float)

        first = np.array(path[0]['point'], dtype=float)[None, :]
        self.klist = np.vstack([first, self.klist, first])


def evaluate_math_expression(expr: str) -> float:
    try:
        return eval(expr, {"__builtins__": None}, vars(math))
    except Exception as e:
        raise ValueError(f"Invalid expression: '{expr}'. Error: {e}")


class IncarData:
    def __init__(self):
        self.name: str = None
        self.lattice_const: float = None
        self.real_lattice_vectors: list = None
        self.dataset_type: str = None
        self.dataset_file: str = None
        self.dielectric_file: str = None
        self.E_file: str = None
        self.kpoint_num: int = None
        self.radius_func: bool = True

    def __repr__(self):
        class_name = self.__class__.__name__
        lines = []
        for key in sorted(self.__dict__):
            value = getattr(self, key)
            lines.append(f"  {key}={value!r},")
        body = "\n".join(lines)
        return f"{class_name} =>\n{body}"
    
    def validate(self):
        missing = [k for k, v in vars(self).items() if v is None]
        if missing:
            err_msg = f"Missing required IncarData fields: {', '.join(missing)}"
            Logger.error(err_msg)
            raise ValueError(err_msg)

class IncarParser:
    DEFAULTS = {
        "name": "Wannier",
        "reciprocal_lattice_vectors": np.array([[0, 0], [0, 0]]),
        "dataset_type":  "comsol",
        "origin": [0, 0],
        "radius_func": True,
        }

    def __init__(self, filename: str):
        self.filename = filename

    def parse_value(self, key: str, value: str):
        value = value.strip()
        if key in ["name", "dataset_type", "fatband_path"]:
            return value
        elif key in ["epsilon", "err_diff", "DOS_eps"]:
            return float(value.strip())
        elif key in ["kpoint_num"]:
            return int(value.strip())
        elif key in ["extension", "k_num", "DOS_Brillouin_mesh"]:
            return [int(x) for x in value.split(',')]
        elif key in ["origin", "w_center", "eff_k"]:
            return [float(x) for x in value.split(',')]
        elif key == "lattice_const":
            return float(evaluate_math_expression(value.strip()))
        elif key in ["real_lattice_vectors", "reciprocal_lattice_vectors", "composition_of_b"]:
            parts = value.split(',')
            vectors = []
            for part in parts:
                vec = [float(evaluate_math_expression(x)) for x in part.strip().split()]
                vectors.append(vec)
            return vectors
        elif key in ["k_points"]:
            parts = value.split(',')
            ranges = []
            for part in parts:
                part = part.strip()
                tokens = part.split(':')
                if len(tokens) == 3:
                    start, step, stop = map(float, tokens)
                    ranges.append(np.arange(start, stop, step))
                else:
                    raise ValueError(f"Invalid k_points range format: '{part}'")
            return ranges
        elif key in ["hopping_state"]:
            parts = value.split(',')
            ranges = []
            for part in parts:
                part = part.strip()
                tokens = part.split(':')
                if len(tokens) == 2:
                    start, stop = map(int, tokens)
                    ranges.append(np.arange(start, stop))
                else:
                    raise ValueError(f"Invalid hopping_state range format: '{part}'")
            return ranges
        elif key in ["band_window", "band_calc"]:
            tokens = value.split(':')
            if len(tokens) == 2:
                start, stop = map(int, tokens)
                return np.arange(start, stop)
            else:
                raise ValueError(f"Invalid band_window format: '{value}'")
        elif key == "dataset_order":
            return [x.strip() for x in value.split(',')]
        elif key == "projections":
            def _complex(s: str):
                s = s.strip()
                s_mod = s.replace("i", "j")
                if s_mod == "j":
                    s_mod = "1j"
                elif s_mod == "-j":
                    s_mod = "-1j"
                try:
                    return complex(s_mod)
                except Exception:
                    return complex(evaluate_math_expression(s))

            def _extract_brace_blocks(token: str):
                blocks, depth, start = [], 0, None
                for i, ch in enumerate(token):
                    if ch == "{":
                        if depth == 0:
                            start = i + 1
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and start is not None:
                            blocks.append(token[start:i].strip())
                            start = None
                return blocks

            def _extract_bracket_groups(s: str):
                groups, depth, start = [], 0, None
                for i, ch in enumerate(s):
                    if ch == "[":
                        if depth == 0:
                            start = i + 1
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                        if depth == 0 and start is not None:
                            groups.append(s[start:i].strip())
                            start = None
                return groups

            def _parse_linear_combo(token: str):
                blocks = _extract_brace_blocks(token)
                if len(blocks) != 2:
                    raise ValueError(f"Invalid linear-combo block: '{token}'")
                state_groups = _extract_bracket_groups(blocks[0])
                if not state_groups:
                    raise ValueError(f"Empty states in linear-combo: '{token}'")
                lc_states = []
                for g in state_groups:
                    term = [t.strip() for t in g.split(",")]
                    if len(term) != 3:
                        raise ValueError(f"State must be [n,l,z]: '[{g}]'")
                    n = int(term[0]); l = int(term[1]); z = float(evaluate_math_expression(term[2]))
                    lc_states.append([n, l, z])
                coeff_strs = [c.strip() for c in blocks[1].split(",") if c.strip()]
                if not coeff_strs:
                    raise ValueError(f"Empty coeffs in linear-combo: '{token}'")
                lc_coeffs = [_complex(c) for c in coeff_strs]
                if len(lc_coeffs) != len(lc_states):
                    raise ValueError(f"#coeffs != #states in linear-combo: '{token}'")
                return {"lc_states": lc_states, "lc_coeffs": lc_coeffs}
            
            projections = []
            value = value.strip().strip('end').strip()
            lines = value.splitlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line:
                    projections_dict = {}
                    parts = line.split(';')

                    state_list = []
                    for i in range(len(parts)):
                        if i == 0:
                            projections_dict['atom'] = parts[i].strip()
                        elif i == 1:
                            groups = _extract_bracket_groups(parts[i].strip())
                            if len(groups) != 1:
                                raise ValueError(f"Invalid positon in projections: '{parts[i].strip()}'")
                            term = groups[0]
                            projections_dict['frac_position'] = [float(evaluate_math_expression(v.strip())) for v in term.split(',')]
                        elif i == 2:
                            projections_dict['xaxis_angluar'] = float(evaluate_math_expression(parts[i].strip()))
                        else:
                            token = parts[i].strip()
                            if token.startswith('{'):
                                state_list.append(_parse_linear_combo(token))
                            elif token.startswith('['):
                                groups = _extract_bracket_groups(token)
                                if len(groups) != 1:
                                    raise ValueError(f"Invalid state block: '{token}'")
                                g = groups[0]
                                vals = [t.strip() for t in g.split(',')]
                                if len(vals) != 3:
                                    raise ValueError(f"State must be [n,l,z]: '[{g}]'")
                                n = int(vals[0]); l = int(vals[1]); z = float(evaluate_math_expression(vals[2]))
                                state_list.append([n, l, z])
                            else:
                                raise ValueError(f"Invalid states in projections: '{token}'")

                    projections_dict['states'] = state_list
                    projections.append(projections_dict)

            return projections
        elif key == "k_path":
            k_path = []
            value = value.strip().strip('end').strip()
            lines = value.splitlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line:
                    k_path_dict = {}
                    parts = line.split(';')

                    for i in range(len(parts)):
                        if i == 0:
                            k_path_dict['name'] = parts[i].strip()
                        elif i == 1:
                            k_path_dict['point'] = [float(evaluate_math_expression(p)) for p in parts[i].strip().split(',')]
                        elif i == 2:
                            k_path_dict['num'] = int(parts[i].strip())
                    k_path.append(k_path_dict)
            return k_path
        elif key in ["M_in", "hermitian", "proj_iter", "hybrid_Wilson_loop", "Chern_number", "symmetry", "decompose", "disable_orth", "radius_func"]:
            if value.strip().lower() == "true":
                return True
            else:
                return False
        elif key in ["neighbor"]:
            neighbor = []
            parts = value.strip().split(',')
            for part in parts:
                neighbor.append([int(p.strip()) for p in part.strip().split(' ')])
            return neighbor
        elif key in ["use_cached_data"]:
            return [p.strip().upper() for p in value.split(',')]
        else:
            return value

    def parse_file(self) -> IncarData:
        incar_data = IncarData()
        with open(self.filename, 'r', encoding='utf-8') as f:
            inside_projections = False
            projections_data = ''
            inside_k_path = False
            k_path_data = ''

            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if "projections" in line:
                    inside_projections = True
                    continue
                elif "end" in line and inside_projections:
                    inside_projections = False
                    setattr(incar_data, 'projections', self.parse_value('projections', projections_data))
                    continue
                
                if inside_projections:
                    projections_data += '\n' + line

                if "k_path" in line:
                    inside_k_path = True
                    continue
                elif "end" in line and inside_k_path:
                    inside_k_path = False
                    setattr(incar_data, 'k_path', self.parse_value('k_path', k_path_data))
                    continue
                
                if inside_k_path:
                    k_path_data += '\n' + line

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    parsed_value = self.parse_value(key, value)
                    setattr(incar_data, key, parsed_value)
                    
        for key, default_val in self.DEFAULTS.items():
            if getattr(incar_data, key, None) is None:
                setattr(incar_data, key, default_val)
        return incar_data