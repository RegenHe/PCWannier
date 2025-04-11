import numpy as np
from .Utils import IncarData


class IncarParser:
    def __init__(self, filename: str):
        self.filename = filename

    def parse_value(self, key: str, value: str):
        value = value.strip()
        if key in ["name", "dataset_type", "dataset_file", "dielectric_file", "U_file", "hopping_file", "wannier_file", "wannier_figure", "mesh_file"]:
            return value
        elif key == "lattice_const":
            return [float(x) for x in value.split()]
        elif key in ["real_lattice_vectors", "reciprocal_lattice_vectors", "composition_of_b"]:
            parts = value.split(',')
            vectors = []
            for part in parts:
                vec = [float(x) for x in part.strip().split()]
                vectors.append(vec)
            return vectors
        elif key == "k_points":
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
        elif key in ["band_window", "band_calc"]:
            tokens = value.split(':')
            if len(tokens) == 2:
                start, stop = map(int, tokens)
                return np.arange(start, stop + 1)
            else:
                raise ValueError(f"Invalid band_window format: '{value}'")
        elif key == "dataset_order":
            return [x.strip() for x in value.split(',')]
        else:
            return value

    def parse_file(self) -> IncarData:
        incar_data = IncarData()
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    parsed_value = self.parse_value(key, value)
                    setattr(incar_data, key, parsed_value)
        return incar_data
