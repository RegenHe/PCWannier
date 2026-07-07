from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple
import math

import numpy as np


class EnergyWindow(NamedTuple):
    emin: float
    emax: float


@dataclass
class IncarConfig:
    name: str = "Wannier"
    lattice_const: float | None = None
    real_lattice_vectors: list[list[float]] | None = None
    reciprocal_lattice_vectors: list[list[float]] | np.ndarray = field(
        default_factory=lambda: np.array([[0.0, 0.0], [0.0, 0.0]])
    )
    kdim: int | None = None
    k_points: list[np.ndarray] | None = None
    dataset_type: str = "comsol"
    dataset_file: str | bool | None = None
    dataset_order: list[str] = field(default_factory=lambda: ["k1", "k2", "E"])
    dielectric_file: str | bool | None = None
    mesh_file: str | bool | None = None
    E_file: str | bool = "./E.txt"
    E_is_real: bool = True
    compute_backend: str = "python"

    N_file: str = "./N.txt"
    U_file: str = "./U.txt"
    V_file: str = "./V.txt"
    M_file: str = "./M.txt"
    A_file: str = "./A.txt"
    S_file: str = "./S.txt"
    band_file: str = "./band.txt"
    hopping_file: str = "./hopping.txt"
    wannier_file: str = "./wannier.txt"
    wannier_figures: str = "./wanniers"
    band_figure: str = "./band.png"
    topo_output: str = "./topo"

    left_dataset_file: str | bool = False
    hermitian: bool = True
    disable_orth: bool = True
    M_in: bool = False
    use_cached_data: list[str] = field(default_factory=lambda: ["FALSE"])

    composition_of_b: list[list[float]] | None = None
    b_vectors: np.ndarray | None = None
    wb: np.ndarray | None = None

    origin: list[float] = field(default_factory=lambda: [0.0, 0.0])
    band_window: np.ndarray | EnergyWindow | None = None
    inner_window: np.ndarray | EnergyWindow | bool = False
    projections: list[dict[str, Any]] | None = None
    proj_iter: bool = True
    proj_binarize: bool = False
    v_proj: bool = True
    w_center: list[float] | bool = False

    epsilon: float = 0.01
    err_diff: float = 1e-6
    max_iter: int = 2000
    extension: list[int] | None = None
    band_calc_num: int | None = None
    hopping_state: list[np.ndarray] | None = None
    neighbor: list[list[int]] = field(default_factory=list)
    k_path: list[dict[str, Any]] | None = None

    DOS: int = 0
    DOS_eps: float = 0.01
    DOS_num: int = 200
    DOS_Brillouin_mesh: list[int] = field(default_factory=lambda: [100, 100])
    k_num: list[int] = field(default_factory=lambda: [100, 100])
    hybrid_Wilson_loop: bool = False
    Chern_number: bool = False

    symmetry: bool = False
    eff_k: list[float] | bool = False
    eff_order: int = 2
    eff_file: str = "./H_eff.txt"
    decompose: bool = False
    decompose_file: str = "./decompose.txt"
    finite: tuple[int | None, int | None] | bool = False
    finite_k: list[float] = field(default_factory=lambda: [0.0, 1.0, 100.0])
    finite_band_figure: str = "./finite_band.png"
    finite_band_file: str = "./finite_band.txt"
    finite_wavefunction_file: str = "./finite_wavefunctions.txt"
    finite_DOS_file: str = "./finite_DOS.txt"
    finite_DOS_figure: str = "./finite_DOS.png"
    finite_DOS_eps: float = 0.01
    finite_DOS_num: int | bool = False
    finite_layer_num: int = 3

    base_dir: Path = field(default_factory=Path.cwd)
    _preprocessed: bool = False

    def input_path(self, value: str | bool | None) -> Path | None:
        if value is None or value is False:
            return None
        text = str(value)
        if text.lower() == "false":
            return None
        path = Path(text)
        return path if path.is_absolute() else self.base_dir / path

    def validate_runtime_scope(self) -> None:
        if self.dataset_type.lower() != "comsol":
            raise NotImplementedError("Only COMSOL input is implemented in PCWannier v1.")
        if not self.hermitian:
            raise NotImplementedError("Non-Hermitian left/right fields are not implemented in PCWannier v1.")
        if self.symmetry:
            raise NotImplementedError("Symmetry adaptation is not implemented in PCWannier v1.")
        if self.finite is not False:
            raise NotImplementedError("Finite-system calculations are not implemented in PCWannier v1.")
        if self.eff_k is not False:
            raise NotImplementedError("Effective Hamiltonian expansion is not implemented in PCWannier v1.")

    def validate_required(self) -> None:
        missing = [
            name
            for name in (
                "lattice_const",
                "real_lattice_vectors",
                "k_points",
                "composition_of_b",
                "band_window",
                "projections",
                "extension",
                "dataset_file",
                "dielectric_file",
                "mesh_file",
                "E_file",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"Missing required incar fields: {', '.join(missing)}")


def evaluate_math_expression(expr: str) -> float:
    return float(eval(expr, {"__builtins__": None}, vars(math)))


class IncarParser:
    def __init__(self, filename: str | Path):
        self.filename = Path(filename)

    def parse_file(self) -> IncarConfig:
        cfg = IncarConfig(base_dir=self.filename.resolve().parent)
        inside_projections = False
        inside_k_path = False
        projections_data: list[str] = []
        k_path_data: list[str] = []

        with self.filename.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                if line == "projections":
                    inside_projections = True
                    continue
                if inside_projections:
                    if line == "end":
                        inside_projections = False
                        cfg.projections = self.parse_value("projections", "\n".join(projections_data))
                        projections_data.clear()
                    else:
                        projections_data.append(line)
                    continue

                if line == "k_path":
                    inside_k_path = True
                    continue
                if inside_k_path:
                    if line == "end":
                        inside_k_path = False
                        cfg.k_path = self.parse_value("k_path", "\n".join(k_path_data))
                        k_path_data.clear()
                    else:
                        k_path_data.append(line)
                    continue

                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                setattr(cfg, key, self.parse_value(key, value))

        preprocess_config(cfg)
        cfg.validate_required()
        cfg.validate_runtime_scope()
        return cfg

    def parse_value(self, key: str, value: str):
        value = value.strip()
        string_keys = {
            "name",
            "dataset_type",
            "compute_backend",
            "dataset_file",
            "left_dataset_file",
            "dielectric_file",
            "S_file",
            "U_file",
            "V_file",
            "A_file",
            "hopping_file",
            "wannier_file",
            "wannier_figure",
            "wannier_figures",
            "mesh_file",
            "M_file",
            "E_file",
            "band_figure",
            "band_file",
            "N_file",
            "topo_output",
            "eff_file",
            "decompose_file",
            "finite_band_figure",
            "finite_band_file",
            "finite_wavefunction_file",
            "finite_DOS_file",
            "finite_DOS_figure",
        }
        if key in string_keys:
            return value
        if key in {"epsilon", "err_diff", "DOS_eps", "finite_DOS_eps"}:
            return float(evaluate_math_expression(value))
        if key in {"max_iter", "DOS", "DOS_num", "eff_order", "finite_layer_num"}:
            return int(evaluate_math_expression(value))
        if key == "finite_DOS_num":
            return False if value.lower() == "false" else int(evaluate_math_expression(value))
        if key in {"extension", "k_num", "DOS_Brillouin_mesh"}:
            return [int(evaluate_math_expression(x.strip())) for x in value.split(",")]
        if key in {"origin", "w_center", "eff_k", "finite_k"}:
            if value.lower() == "false":
                return False
            return [float(evaluate_math_expression(x.strip())) for x in value.split(",")]
        if key == "lattice_const":
            return float(evaluate_math_expression(value))
        if key in {"real_lattice_vectors", "reciprocal_lattice_vectors", "composition_of_b"}:
            return [
                [float(evaluate_math_expression(x)) for x in part.strip().split()]
                for part in value.split(",")
            ]
        if key == "k_points":
            ranges = []
            for part in value.split(","):
                tokens = [float(x) for x in part.strip().split(":")]
                if len(tokens) != 3:
                    raise ValueError(f"Invalid k_points range: {part!r}")
                start, step, stop = tokens
                ranges.append(np.arange(start, stop, step))
            return ranges
        if key == "hopping_state":
            ranges = []
            for part in value.split(","):
                tokens = [int(x) for x in part.strip().split(":")]
                if len(tokens) != 2:
                    raise ValueError(f"Invalid hopping_state range: {part!r}")
                ranges.append(np.arange(tokens[0], tokens[1]))
            return ranges
        if key in {"band_window", "band_calc", "inner_window"}:
            if value.lower() == "false":
                return False
            if ":" in value:
                start, stop = [int(x.strip()) for x in value.split(":", 1)]
                return np.arange(start, stop)
            if "," in value:
                emin, emax = [float(evaluate_math_expression(x.strip())) for x in value.split(",", 1)]
                return EnergyWindow(min(emin, emax), max(emin, emax))
            raise ValueError(f"Invalid {key} format: {value!r}")
        if key == "dataset_order":
            return [x.strip() for x in value.split(",")]
        if key == "projections":
            return self._parse_projections(value)
        if key == "k_path":
            return self._parse_k_path(value)
        if key in {
            "M_in",
            "hermitian",
            "proj_iter",
            "hybrid_Wilson_loop",
            "Chern_number",
            "symmetry",
            "decompose",
            "disable_orth",
            "proj_binarize",
            "v_proj",
            "E_is_real",
        }:
            return value.lower() == "true"
        if key == "neighbor":
            if not value:
                return []
            return [[int(p.strip()) for p in part.strip().split()] for part in value.split(",")]
        if key == "use_cached_data":
            return [p.strip().upper() for p in value.split(",")]
        if key == "finite":
            if value.lower() == "false":
                return False
            parts = [p.strip() for p in value.split(",")[:2]]
            while len(parts) < 2:
                parts.append("")
            out = []
            for item in parts:
                out.append(None if item == "" else int(item))
            return tuple(out)
        return value

    def _parse_projections(self, value: str) -> list[dict[str, Any]]:
        def extract_bracket_groups(text: str) -> list[str]:
            groups = []
            depth = 0
            start = None
            for idx, ch in enumerate(text):
                if ch == "[":
                    if depth == 0:
                        start = idx + 1
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0 and start is not None:
                        groups.append(text[start:idx].strip())
                        start = None
            return groups

        def extract_brace_blocks(text: str) -> list[str]:
            blocks = []
            depth = 0
            start = None
            for idx, ch in enumerate(text):
                if ch == "{":
                    if depth == 0:
                        start = idx + 1
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        blocks.append(text[start:idx].strip())
                        start = None
            return blocks

        def parse_complex(text: str) -> complex:
            normalized = text.strip().replace("i", "j")
            if normalized == "j":
                normalized = "1j"
            if normalized == "-j":
                normalized = "-1j"
            try:
                return complex(normalized)
            except ValueError:
                return complex(evaluate_math_expression(text))

        def parse_linear_combo(token: str) -> dict[str, Any]:
            blocks = extract_brace_blocks(token)
            if len(blocks) != 2:
                raise ValueError(f"Invalid linear-combo projection: {token!r}")
            states = []
            for group in extract_bracket_groups(blocks[0]):
                n, l, z = [x.strip() for x in group.split(",")]
                states.append([int(n), int(l), float(evaluate_math_expression(z))])
            coeffs = [parse_complex(x) for x in blocks[1].split(",") if x.strip()]
            if len(states) != len(coeffs):
                raise ValueError("Projection linear-combo state and coefficient counts differ.")
            return {"lc_states": states, "lc_coeffs": coeffs}

        projections = []
        for line in value.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 4:
                raise ValueError(f"Invalid projection line: {line!r}")
            groups = extract_bracket_groups(parts[1])
            if len(groups) != 1:
                raise ValueError(f"Invalid projection position: {parts[1]!r}")
            entry: dict[str, Any] = {
                "atom": parts[0],
                "frac_position": [float(evaluate_math_expression(v.strip())) for v in groups[0].split(",")],
                "xaxis_angluar": float(evaluate_math_expression(parts[2])),
                "states": [],
            }
            for token in parts[3:]:
                if token.startswith("{"):
                    entry["states"].append(parse_linear_combo(token))
                else:
                    state_groups = extract_bracket_groups(token)
                    if len(state_groups) != 1:
                        raise ValueError(f"Invalid projection state: {token!r}")
                    n, l, z = [x.strip() for x in state_groups[0].split(",")]
                    entry["states"].append([int(n), int(l), float(evaluate_math_expression(z))])
            projections.append(entry)
        return projections

    def _parse_k_path(self, value: str) -> list[dict[str, Any]]:
        path = []
        for line in value.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) != 3:
                raise ValueError(f"Invalid k_path line: {line!r}")
            path.append(
                {
                    "name": parts[0],
                    "point": [float(evaluate_math_expression(p)) for p in parts[1].split(",")],
                    "num": int(evaluate_math_expression(parts[2])),
                }
            )
        return path


def preprocess_config(cfg: IncarConfig) -> IncarConfig:
    if cfg._preprocessed:
        return cfg
    if cfg.real_lattice_vectors is None:
        return cfg
    cfg.kdim = len(cfg.real_lattice_vectors)
    reciprocal = np.asarray(cfg.reciprocal_lattice_vectors, dtype=float)
    zeros = np.zeros((cfg.kdim, cfg.kdim), dtype=float)
    if reciprocal.shape[:2] == zeros.shape and np.allclose(reciprocal, zeros):
        reciprocal = (np.linalg.inv(np.asarray(cfg.real_lattice_vectors, dtype=float)) @ np.eye(cfg.kdim)).T
    cfg.reciprocal_lattice_vectors = reciprocal

    if cfg.composition_of_b is not None and cfg.k_points is not None:
        positive = [list(v) for v in cfg.composition_of_b]
        negative = [[-x for x in v] for v in positive]
        cfg.composition_of_b = positive + negative
        b_vectors = []
        for comp in cfg.composition_of_b:
            vec = np.zeros_like(reciprocal[0], dtype=float)
            for axis in range(cfg.kdim):
                vec += (
                    comp[axis]
                    * reciprocal[axis]
                    / len(cfg.k_points[axis])
                    * 2.0
                    * np.pi
                    / float(cfg.lattice_const)
                )
            b_vectors.append(vec.tolist())
        cfg.b_vectors = np.array(b_vectors, dtype=float) * float(cfg.lattice_const)

        mat_a = np.eye(cfg.kdim).reshape(-1, 1)
        mat_b = np.zeros((cfg.kdim**cfg.kdim, len(cfg.composition_of_b)))
        for i in range(cfg.kdim):
            for j in range(cfg.kdim):
                for k, bvec in enumerate(cfg.b_vectors):
                    mat_b[i * cfg.kdim + j, k] = bvec[i] * bvec[j]
        cfg.wb = (np.linalg.pinv(mat_b) @ mat_a).flatten()

    if cfg.projections is not None:
        cfg.band_calc_num = sum(len(p["states"]) for p in cfg.projections)
    cfg._preprocessed = True
    return cfg


def load_config(path: str | Path) -> IncarConfig:
    return IncarParser(path).parse_file()
