import numpy as np

class IncarData:
    def __init__(self):
        self.name = None
        self.lattice_params = None
        self.real_lattice_vectors = None
        self.reciprocal_lattice_vectors = None
        self.k_points = None
        self.dataset_type = None
        self.dataset_file = None
        self.dataset_order = None
        self.dielectric_file = None
        self.U_file = None
        self.hopping_file = None
        self.wannier_file = None
        self.wannier_figure = None

    def __repr__(self):
        return (
            f"IncarData(\n"
            f"  name={self.name},\n"
            f"  lattice_params={self.lattice_params},\n"
            f"  real_lattice_vectors={self.real_lattice_vectors},\n"
            f"  reciprocal_lattice_vectors={self.reciprocal_lattice_vectors},\n"
            f"  k_points={self.k_points},\n"
            f"  dataset_type={self.dataset_type},\n"
            f"  dataset_file={self.dataset_file},\n"
            f"  dataset_order={self.dataset_order},\n"
            f"  dielectric_file={self.dielectric_file},\n"
            f"  U_file={self.U_file},\n"
            f"  hopping_file={self.hopping_file},\n"
            f"  wannier_file={self.wannier_file},\n"
            f"  wannier_figure={self.wannier_figure}\n)"
        )


class IncarParser:
    def __init__(self, filename: str):
        self.filename = filename

    def parse_value(self, key: str, value: str):
        value = value.strip()
        if key in ["name", "dataset_type", "dataset_file", "dielectric_file", "U_file", "hopping_file", "wannier_file", "wannier_figure"]:
            return value
        elif key == "lattice_params":
            return [float(x) for x in value.split()]
        elif key in ["real_lattice_vectors", "reciprocal_lattice_vectors"]:
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
