import numpy as np
from .Utils import IncarData


class IncarParser:
    def __init__(self, filename: str):
        self.filename = filename

    def parse_value(self, key: str, value: str):
        value = value.strip()
        if key in ["name", "dataset_type", "dataset_file", "dielectric_file", "U_file", "V_file", "hopping_file", "wannier_file", "wannier_figure", "mesh_file", "M_file", "E_file"]:
            return value
        elif key in ["err_diff"]:
            return float(value.strip())
        elif key in ["max_iter"]:
            return int(value.strip())
        elif key in ["extension",]:
            return [int(x) for x in value.split(',')]
        elif key == "lattice_const":
            return float(value.strip())
        elif key in ["real_lattice_vectors", "reciprocal_lattice_vectors", "composition_of_b"]:
            parts = value.split(',')
            vectors = []
            for part in parts:
                vec = [float(x) for x in part.strip().split()]
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
                return np.arange(start, stop + 1)
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
                            if '(' in parts[i].strip() and ')' in parts[i].strip():
                                coefficient, term = parts[i].split('(')
                                coefficient = coefficient.strip()
                                term = term.split(')')[0].strip()
                                projections_dict['position'] = [float(v.strip()) for v in term.split(',')]
                            else:
                                raise ValueError(f"Invalid positon in projections: '{parts[i].strip()}'")
                        elif i == 2:
                            projections_dict['xaxis_angluar'] = float(parts[i].strip())
                        else:
                            if '(' in parts[i].strip() and ')' in parts[i].strip():
                                coefficient, term = parts[i].split('(')
                                coefficient = coefficient.strip()
                                term = term.split(')')[0].strip().split(',')
                                state_list.append([int(term[0].strip()), int(term[1].strip()), float(term[2].strip())])
                            else:
                                raise ValueError(f"Invalid states in projections: '{parts[i].strip()}'")

                    projections_dict['states'] = state_list
                    projections.append(projections_dict)

            return projections
        elif key in ["M_in", "E_is_real"]:
            if value.strip().lower() == "true":
                return True
            else:
                return False
        else:
            return value

    def parse_file(self) -> IncarData:
        incar_data = IncarData()
        with open(self.filename, 'r', encoding='utf-8') as f:
            inside_projections = False
            projections_data = ''

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

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    parsed_value = self.parse_value(key, value)
                    setattr(incar_data, key, parsed_value)
        return incar_data
