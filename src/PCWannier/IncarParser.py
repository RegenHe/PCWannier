import numpy as np
import math

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

        self.band_window: list = None
        self.proj_iter: bool = None
        self.projections: list = None
        self.M_in: str = None

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

    def __repr__(self):
        class_name = self.__class__.__name__
        lines = []
        for key in sorted(self.__dict__):
            value = getattr(self, key)
            lines.append(f"  {key}={value!r},")
        body = "\n".join(lines)
        return f"{class_name} =>\n{body}"
    
    


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
    "E_is_real": True,
    "band_file": "./band.txt",
    "hopping_file": "./hopping.txt",
    "wannier_file": "./wannier.txt",
    "wannier_figures": "./wanniers",
    "proj_iter": True,
    "M_in": False,
    "err_diff": 1e-6,
    "max_iter": 2000,
    "band_figure": "./band.png",

    "use_cached_data": ["False"],
    "DOS": 2,
    "DOS_eps": 0.01,
    "DOS_num": 200,
    }

    def __init__(self, filename: str):
        self.filename = filename

    def parse_value(self, key: str, value: str):
        value = value.strip()
        if key in ["name", "dataset_type", "dataset_file", "dielectric_file", "U_file", "V_file", "A_file", "hopping_file", "wannier_file", "wannier_figure", "mesh_file", "M_file", "E_file", "band_figure", "band_file", "N_file"]:
            return value
        elif key in ["err_diff", "DOS_eps"]:
            return float(value.strip())
        elif key in ["max_iter", "DOS", "DOS_num"]:
            return int(value.strip())
        elif key in ["extension",]:
            return [int(x) for x in value.split(',')]
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
                            if '[' in parts[i].strip() and ']' in parts[i].strip():
                                coefficient, term = parts[i].split('[')
                                coefficient = coefficient.strip()
                                term = term.split(']')[0].strip()
                                projections_dict['position'] = [float(evaluate_math_expression(v.strip())) for v in term.split(',')]
                            else:
                                raise ValueError(f"Invalid positon in projections: '{parts[i].strip()}'")
                        elif i == 2:
                            projections_dict['xaxis_angluar'] = float(evaluate_math_expression(parts[i].strip()))
                        else:
                            if '[' in parts[i].strip() and ']' in parts[i].strip():
                                coefficient, term = parts[i].split('[')
                                coefficient = coefficient.strip()
                                term = term.split(']')[0].strip().split(',')
                                state_list.append([int(term[0].strip()), int(term[1].strip()), float(evaluate_math_expression(term[2].strip()))])
                            else:
                                raise ValueError(f"Invalid states in projections: '{parts[i].strip()}'")

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
        elif key in ["M_in", "E_is_real", "proj_iter"]:
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
    


