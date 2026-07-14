from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
import math

import numpy as np

if TYPE_CHECKING:
    from .symmetry import SymmetryContext


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
    integration_mode: str = "nodal"
    symmetry_file: str | bool = False
    symmetry_constrained: bool = False
    symmetry_output_basis: str = "strict"
    symmetry_tolerance: float = 1.0e-8
    symmetry_max_iter: int = 20
    symmetry_svd_tolerance: float = 1.0e-10
    symmetry_validate_wannier: bool = True
    symmetry_real_space_tolerance: float = 1.0e-6
    symmetry_minimum_retained_norm: float = 0.99
    representation_field_kind: str = "scalar"
    representation_degeneracy_absolute: float = 1.0e-6
    representation_degeneracy_relative: float = 1.0e-8
    wannier_targets: list[dict[str, Any]] | None = None
    representation_analysis: list[dict[str, Any]] | None = None
    symmetry_resolved_path: Path | None = field(default=None, init=False)
    disentangle_max_iter: int | None = None
    disentangle_err_diff: float | None = None
    disentangle_mixing: float = 0.5
    symmetry_context: SymmetryContext | None = field(default=None, init=False, repr=False)

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


_MATH_NAMES = {name: value for name, value in vars(math).items() if not name.startswith("_")}
_MATH_NAMES.update(
    {
        "abs": abs,
        "max": max,
        "min": min,
        "pow": pow,
        "round": round,
        "math": math,
    }
)
_BLOCKED_EXPR_TOKENS = (
    "__",
    "import",
    "open",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "input",
)


def evaluate_math_expression(expr: str) -> float:
    """Evaluate a small math expression used by incar numeric fields."""
    expr = str(expr).strip()
    lowered = expr.lower()
    if any(token in lowered for token in _BLOCKED_EXPR_TOKENS):
        raise ValueError(f"Invalid numeric expression: {expr!r}")
    try:
        return float(eval(expr, {"__builtins__": {}}, _MATH_NAMES))
    except Exception as exc:
        raise ValueError(f"Invalid numeric expression: {expr!r}") from exc


class IncarParser:
    def __init__(self, filename: str | Path):
        self.filename = Path(filename)

    def parse_file(self) -> IncarConfig:
        cfg = IncarConfig(base_dir=self.filename.resolve().parent)
        inside_projections = False
        inside_k_path = False
        inside_wannier_targets = False
        inside_representation_analysis = False
        projections_data: list[str] = []
        k_path_data: list[str] = []
        wannier_targets_data: list[str] = []
        representation_analysis_data: list[str] = []

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

                if line == "wannier_targets":
                    inside_wannier_targets = True
                    continue
                if inside_wannier_targets:
                    if line == "end":
                        inside_wannier_targets = False
                        cfg.wannier_targets = self._parse_wannier_targets(wannier_targets_data)
                        wannier_targets_data.clear()
                    else:
                        wannier_targets_data.append(line)
                    continue

                if line == "representation_analysis":
                    inside_representation_analysis = True
                    continue
                if inside_representation_analysis:
                    if line == "end":
                        inside_representation_analysis = False
                        cfg.representation_analysis = self._parse_representation_analysis(
                            representation_analysis_data
                        )
                        representation_analysis_data.clear()
                    else:
                        representation_analysis_data.append(line)
                    continue

                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "w_center":
                    raise ValueError(
                        "The w_center input has been removed because forcing Wannier centers is not a physical operation."
                    )
                if key == "symmetry":
                    raise ValueError("The boolean symmetry input has been removed; use symmetry_file = ./sym.yaml.")
                setattr(cfg, key, self.parse_value(key, value))

        unterminated = [
            name
            for name, active in (
                ("projections", inside_projections),
                ("k_path", inside_k_path),
                ("wannier_targets", inside_wannier_targets),
                ("representation_analysis", inside_representation_analysis),
            )
            if active
        ]
        if unterminated:
            raise ValueError(f"Unterminated incar block(s): {', '.join(unterminated)}.")

        preprocess_config(cfg)
        cfg.validate_required()
        cfg.validate_runtime_scope()
        if cfg.symmetry_file is not False and str(cfg.symmetry_file).lower() != "false":
            from .symmetry import (
                DegeneracyTolerance,
                FieldKind,
                RepresentationAnalysisSpec,
                RepresentationPointSpec,
                SymmetryCalculationSpec,
                SymmetryGaugeSpec,
                WannierTargetSpec,
                build_symmetry_context,
                cartesian_field_matrix,
                compose_symmetry_model,
                load_symmetry,
                resolve_symmetry_file,
            )

            requested_symmetry_path = cfg.input_path(cfg.symmetry_file)
            if requested_symmetry_path is None:
                raise ValueError("symmetry_file is enabled but no path was supplied.")
            symmetry_path = resolve_symmetry_file(str(cfg.symmetry_file), cfg.base_dir)
            cfg.symmetry_resolved_path = symmetry_path
            model = load_symmetry(symmetry_path, tolerance=cfg.symmetry_tolerance)
            target_specs = None
            if cfg.wannier_targets is not None:
                target_specs = tuple(
                    WannierTargetSpec(item["name"], item["center"], item["site_irrep"])
                    for item in cfg.wannier_targets
                )
            analysis = None
            if cfg.representation_analysis is not None:
                degeneracy = DegeneracyTolerance(
                    cfg.representation_degeneracy_absolute,
                    cfg.representation_degeneracy_relative,
                )
                points = tuple(
                    RepresentationPointSpec(
                        item["name"],
                        item["k"],
                        item["bands"],
                        item["targets"],
                        degeneracy,
                    )
                    for item in cfg.representation_analysis
                )
                analysis = RepresentationAnalysisSpec(
                    FieldKind(cfg.representation_field_kind), degeneracy, points
                )
            gauge = None
            if cfg.symmetry_constrained:
                gauge = SymmetryGaugeSpec(
                    enabled=True,
                    tolerance=cfg.symmetry_tolerance,
                    max_iterations=cfg.symmetry_max_iter,
                    svd_relative_tolerance=cfg.symmetry_svd_tolerance,
                    validate_wannier=cfg.symmetry_validate_wannier,
                    real_space_tolerance=cfg.symmetry_real_space_tolerance,
                    minimum_retained_norm=cfg.symmetry_minimum_retained_norm,
                )
            model = compose_symmetry_model(
                model,
                SymmetryCalculationSpec(target_specs, analysis, gauge),
            )
            if model.dimension != cfg.kdim:
                raise ValueError(
                    f"Symmetry dimension {model.dimension} does not match incar kdim={cfg.kdim}."
                )
            for operation in model.group.operations:
                cartesian_field_matrix(
                    operation,
                    cfg.real_lattice_vectors,
                    FieldKind.ELECTRIC_POLAR_VECTOR,
                    model.tolerance,
                )
            target_dimension = sum(target.wannier_dimension for target in model.targets)
            if model.targets and target_dimension != cfg.band_calc_num:
                raise ValueError(
                    f"Symmetry Wannier targets define {target_dimension} functions, but incar projections "
                    f"define band_calc_num={cfg.band_calc_num}."
                )
            cfg.symmetry_context = build_symmetry_context(model, cfg.k_points)
        elif cfg.wannier_targets is not None or cfg.representation_analysis is not None:
            raise ValueError(
                "Symmetry calculation blocks require symmetry_file; symmetry_constrained=true also requires symmetry_file."
            )
        if cfg.symmetry_constrained:
            if cfg.symmetry_context is None:
                raise ValueError("symmetry_constrained=true requires symmetry_file = ./sym.yaml.")
            gauge = cfg.symmetry_context.model.symmetry_gauge
            if gauge is None or not gauge.enabled:
                raise ValueError(
                    "symmetry_constrained=true requires a valid symmetry gauge configuration."
                )
            if not cfg.symmetry_context.model.targets:
                raise ValueError("symmetry_constrained=true requires at least one Wannier target.")
        return cfg

    def parse_value(self, key: str, value: str):
        value = value.strip()
        string_keys = {
            "name",
            "dataset_type",
            "compute_backend",
            "integration_mode",
            "representation_field_kind",
            "symmetry_file",
            "symmetry_output_basis",
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
        if key in {
            "epsilon",
            "err_diff",
            "disentangle_err_diff",
            "disentangle_mixing",
            "DOS_eps",
            "finite_DOS_eps",
            "symmetry_tolerance",
            "symmetry_svd_tolerance",
            "symmetry_real_space_tolerance",
            "symmetry_minimum_retained_norm",
            "representation_degeneracy_absolute",
            "representation_degeneracy_relative",
        }:
            return float(evaluate_math_expression(value))
        if key in {
            "max_iter",
            "disentangle_max_iter",
            "symmetry_max_iter",
            "DOS",
            "DOS_num",
            "eff_order",
            "finite_layer_num",
        }:
            return int(evaluate_math_expression(value))
        if key == "finite_DOS_num":
            return False if value.lower() == "false" else int(evaluate_math_expression(value))
        if key in {"extension", "k_num", "DOS_Brillouin_mesh"}:
            return [int(evaluate_math_expression(x.strip())) for x in value.split(",")]
        if key in {"origin", "eff_k", "finite_k"}:
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
                tokens = [float(evaluate_math_expression(x.strip())) for x in part.strip().split(":")]
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
            "decompose",
            "disable_orth",
            "proj_binarize",
            "v_proj",
            "E_is_real",
            "symmetry_constrained",
            "symmetry_validate_wannier",
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

    def _parse_wannier_targets(self, lines: list[str]) -> list[dict[str, Any]]:
        if not lines:
            raise ValueError("wannier_targets block must not be empty.")
        output = []
        names = set()
        for line in lines:
            parts = [part.strip() for part in line.split(";")]
            if len(parts) != 3 or not all(parts):
                raise ValueError(f"Invalid wannier_targets line: {line!r}")
            name, center_text, site_irrep = parts
            if name in names:
                raise ValueError(f"Duplicate Wannier target name: {name!r}.")
            names.add(name)
            center = [
                float(evaluate_math_expression(value.strip()))
                for value in center_text.strip("[] ").split(",")
            ]
            output.append({"name": name, "center": center, "site_irrep": site_irrep})
        return output

    def _parse_representation_analysis(self, lines: list[str]) -> list[dict[str, Any]]:
        if not lines:
            raise ValueError("representation_analysis block must not be empty.")
        output = []
        names = set()
        for line in lines:
            parts = [part.strip() for part in line.split(";")]
            if len(parts) not in {3, 4}:
                raise ValueError(f"Invalid representation_analysis line: {line!r}")
            name = parts[0]
            if not name or name in names:
                raise ValueError(f"Duplicate or empty representation-analysis point name: {name!r}.")
            names.add(name)
            kpoint = [
                float(evaluate_math_expression(value.strip()))
                for value in parts[1].strip("[] ").split(",")
            ]
            bands = self._parse_analysis_bands(parts[2])
            targets = None
            if len(parts) == 4 and parts[3]:
                targets = tuple(value.strip() for value in parts[3].split(",") if value.strip())
                if not targets or len(targets) != len(set(targets)):
                    raise ValueError(f"Invalid target list in representation-analysis line: {line!r}")
            output.append({"name": name, "k": kpoint, "bands": bands, "targets": targets})
        return output

    @staticmethod
    def _parse_analysis_bands(value: str) -> tuple[int, ...] | None:
        text = value.strip()
        if text in {"", "*"}:
            return None
        if ":" in text:
            parts = text.split(":")
            if len(parts) not in {2, 3} or any(part.strip() == "" for part in parts[:2]):
                raise ValueError(f"Invalid representation-analysis band slice: {value!r}")
            start = int(evaluate_math_expression(parts[0].strip()))
            stop = int(evaluate_math_expression(parts[1].strip()))
            step = 1 if len(parts) == 2 or not parts[2].strip() else int(
                evaluate_math_expression(parts[2].strip())
            )
            bands = tuple(range(start, stop, step))
        else:
            bands = tuple(
                int(evaluate_math_expression(part.strip())) for part in text.split(",")
            )
        if not bands or any(index < 0 for index in bands) or len(bands) != len(set(bands)):
            raise ValueError("Representation-analysis bands must be unique non-negative indices.")
        return bands


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
        reconstructed = mat_b @ cfg.wb.reshape(-1, 1)
        residual = float(np.linalg.norm(reconstructed - mat_a) / max(np.linalg.norm(mat_a), np.finfo(float).tiny))
        if residual > 1e-10:
            rank = int(np.linalg.matrix_rank(mat_b))
            raise ValueError(
                "composition_of_b cannot reproduce the isotropic finite-difference tensor: "
                f"relative residual={residual:.6g}, rank={rank}. Add independent neighbor directions."
            )

    if cfg.integration_mode not in {"nodal", "quadratic"}:
        raise ValueError("integration_mode must be 'nodal' or 'quadratic'.")
    cfg.symmetry_output_basis = str(cfg.symmetry_output_basis).strip().lower()
    if cfg.symmetry_output_basis not in {"strict", "fem"}:
        raise ValueError("symmetry_output_basis must be 'strict' or 'fem'.")
    if cfg.disentangle_max_iter is not None and cfg.disentangle_max_iter < 0:
        raise ValueError("disentangle_max_iter must be non-negative.")
    if cfg.disentangle_err_diff is not None and (
        not np.isfinite(cfg.disentangle_err_diff) or cfg.disentangle_err_diff < 0.0
    ):
        raise ValueError("disentangle_err_diff must be finite and non-negative.")
    if not np.isfinite(cfg.disentangle_mixing) or not 0.0 < cfg.disentangle_mixing <= 1.0:
        raise ValueError("disentangle_mixing must lie in (0, 1].")
    if not np.isfinite(cfg.symmetry_tolerance) or cfg.symmetry_tolerance <= 0.0:
        raise ValueError("symmetry_tolerance must be positive and finite.")
    if cfg.symmetry_max_iter <= 0:
        raise ValueError("symmetry_max_iter must be positive.")
    if not np.isfinite(cfg.symmetry_svd_tolerance) or cfg.symmetry_svd_tolerance <= 0.0:
        raise ValueError("symmetry_svd_tolerance must be positive and finite.")
    if (
        not np.isfinite(cfg.symmetry_real_space_tolerance)
        or cfg.symmetry_real_space_tolerance <= 0.0
    ):
        raise ValueError("symmetry_real_space_tolerance must be positive and finite.")
    if not 0.0 < cfg.symmetry_minimum_retained_norm <= 1.0:
        raise ValueError("symmetry_minimum_retained_norm must lie in (0, 1].")

    if cfg.projections is not None:
        cfg.band_calc_num = sum(len(p["states"]) for p in cfg.projections)
    cfg._preprocessed = True
    return cfg


def load_config(path: str | Path) -> IncarConfig:
    return IncarParser(path).parse_file()
