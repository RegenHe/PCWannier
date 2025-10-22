from typing import NamedTuple

import numpy as np
import math

from .Log import Logger

def evaluate_math_expression(expr: str) -> float:
    try:
        return eval(expr, {"__builtins__": None}, vars(math))
    except Exception as e:
        raise ValueError(f"Invalid expression: '{expr}'. Error: {e}")

class EnergyWindow(NamedTuple):
    emin: float
    emax: float

class IncarData:
    def __init__(self):
        self.name: str = None
        self.lattice_const: float = None
        self.real_lattice_vectors: list = None
        self.reciprocal_lattice_vectors: list = None
        self.k_points: list = None
        self.dataset_type: str = None
        self.dataset_file: str = None
        self.dataset_order: list = None
        self.dielectric_file: str = None
        self.N_file: str = None
        self.U_file: str = None
        self.V_file: str = None
        self.M_file: str = None
        self.A_file: str = None
        self.O_file: str = None
        self.E_is_real: bool = None
        self.E_file: str = None
        self.band_file: str = None
        self.hopping_file: str = None
        self.wannier_file: str = None
        self.wannier_figures: str = None

        self.mesh_file: str = None

        self.b_vectors: list = None
        self.composition_of_b: list = None
        self.wb: list = None

        self.origin: list = None
        self.band_window: list = None
        self.inner_window: list = None
        self.proj_iter: bool = None
        self.proj_binarize: bool = None
        self.v_proj: bool = None
        self.projections: list = None
        self.w_center: list = None
        self.M_in: str = None

        self.epsilon: float = None
        self.err_diff: float = None
        self.max_iter: float = None

        self.extension: list = None

        self.band_calc_num: int = None
        self.hopping_state: list = None

        self.neighbor: list = None
        self.k_path: list = None
        self.band_figure: str = None

        self.use_cached_data: list = None

        self.DOS = None
        self.DOS_eps = None
        self.DOS_num = None

        self.topo_output: str = None
        self.k_num: list = None
        self.hybrid_Wilson_loop: bool = None
        self.Chern_number: bool = None

        self.symmetry: bool = None

        self.eff_k: list = None
        self.eff_order: int = None
        self.eff_file: str = None

        self.decompose: bool = None
        self.decompose_file: str = None

        self.finite: list = None
        self.finite_k: list = None
        self.finite_band_figure: str = None
        self.finite_band_file: str = None
        self.finite_wavefunction_file: str = None

        self.finite_DOS_file: str = None
        self.finite_DOS_figure: str = None
        self.finite_DOS_eps: float = None
        self.finite_DOS_num: int = None
        self.finite_layer_num: int = None

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
        "dataset_order": ["k1", "k2", "E"],
        "N_file" :  "./N.txt",
        "U_file": "./U.txt",
        "V_file": "./V.txt",
        "M_file": "./M.txt",
        "A_file": "./A.txt",
        "O_file": "./O.txt",
        "disable_orth": True,
        "E_is_real": True,
        "band_file": "./band.txt",
        "hopping_file": "./hopping.txt",
        "wannier_file": "./wannier.txt",
        "wannier_figures": "./wanniers",
        "proj_iter": True,
        "v_proj": True,
        "inner_window": False,
        "proj_binarize": False,
        "w_center": False,
        "origin": [0, 0],
        "M_in": False,
        "epsilon": 0.01,
        "err_diff": 1e-6,
        "max_iter": 2000,
        "neighbor": [],
        "band_figure": "./band.png",

        "use_cached_data": ["False"],
        "DOS": 0,
        "DOS_eps": 0.01,
        "DOS_num": 200,
        "DOS_Brillouin_mesh": [100, 100],
        "topo_output": "./topo",
        "k_num": [100, 100],
        "hybrid_Wilson_loop": False,
        "Chern_number": False,

        "symmetry": False,

        "eff_order": 2,
        "eff_k": False,
        "eff_file": "./H_eff.txt",
        "decompose": False,
        "decompose_file": "./decompose.txt",

        "finite": False,
        "finite_k": [0, 1, 100],
        "finite_band_figure": "./finite_band.png",
        "finite_band_file": "./finite_band.txt",
        "finite_wavefunction_file": "./finite_wavefunctions.txt",
        "finite_DOS_file": "./finite_DOS.txt",
        "finite_DOS_figure": "./finite_DOS.png",
        "finite_DOS_eps": 0.01,
        "finite_DOS_num": False,
        "finite_layer_num": 3,
        }

    def __init__(self, filename: str):
        self.filename = filename

    def parse_value(self, key: str, value: str):
        value = value.strip()
        if key in ["name", "dataset_type", "dataset_file", "dielectric_file", "U_file", "V_file", "A_file", "hopping_file", "wannier_file", "wannier_figure", "mesh_file", "M_file", "E_file", "band_figure", "band_file", "N_file", "topo_output", "eff_file", "decompose_file", "finite_band_figure", "finite_band_file", "finite_wavefunction_file", "finite_DOS_file", "finite_DOS_figure"]:
            return value
        elif key in ["epsilon", "err_diff", "DOS_eps", "finite_DOS_eps"]:
            return float(evaluate_math_expression(value.strip()))
        elif key in ["max_iter", "DOS", "DOS_num", "eff_order", "finite_DOS_num", "finite_layer_num"]:
            return int(evaluate_math_expression(value.strip()))
        elif key in ["extension", "k_num", "DOS_Brillouin_mesh"]:
            return [int(evaluate_math_expression(x)) for x in value.split(',')]
        elif key in ["origin", "w_center", "eff_k", "finite_k"]:
            return [float(evaluate_math_expression(x)) for x in value.split(',')]
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
        elif key in ["band_window", "band_calc", "inner_window"]:
            v = value.strip()
            if ':' in v:
                start_str, stop_str = v.split(':', 1)
                start = int(start_str.strip())
                stop = int(stop_str.strip())
                return np.arange(start, stop)

            if ',' in v:
                left, right = [t.strip() for t in v.split(',', 1)]
                emin = float(evaluate_math_expression(left))
                emax = float(evaluate_math_expression(right))
                if emin > emax:
                    emin, emax = emax, emin
                return EnergyWindow(emin, emax)
            raise ValueError(f"Invalid band_window format: '{v}'")
        # elif key == "finite_DOS_range":
        #     v = value.strip()
        #     if ',' in v:
        #         left, right = [t.strip() for t in v.split(',', 1)]
        #         emin = float(evaluate_math_expression(left))
        #         emax = float(evaluate_math_expression(right))
        #         if emin > emax:
        #             emin, emax = emax, emin
        #         return EnergyWindow(emin, emax)
        #     raise ValueError(f"Invalid energy range format: '{v}'")
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
        elif key in ["M_in", "E_is_real", "proj_iter", "hybrid_Wilson_loop", "Chern_number", "symmetry", "decompose", "disable_orth", "proj_binarize", "v_proj"]:
            if value.strip().lower() == "true":
                return True
            else:
                return False
        elif key in ["neighbor"]:
            neighbor = []
            if value == '':
                return neighbor
            parts = value.strip().split(',')
            for part in parts:
                neighbor.append([int(p.strip()) for p in part.strip().split(' ')])
            return neighbor
        elif key in ["use_cached_data"]:
            return [p.strip().upper() for p in value.split(',')]
        elif key == "finite":
            s = value.strip()
            parts = [p.strip() for p in s.split(',')[:2]]
            if len(parts) < 2:
                parts.append('')
            out = []
            for t in parts:
                if t == '':
                    out.append(None)
                else:
                    n = int(t)
                    if n < 1:
                        raise ValueError("finite ≥ 1")
                    out.append(n)
            return tuple(out)
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
