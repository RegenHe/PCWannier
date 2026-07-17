from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
import math

import numpy as np

if TYPE_CHECKING:
    from .maxwell import MaxwellProblem
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
    field_components: str = "Ez"
    metric_file: str | bool | None = None
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
    symmetry_boundary_tolerance: float = 1.0e-6
    representation_degeneracy_absolute: float = 1.0e-6
    representation_degeneracy_relative: float = 1.0e-8
    representation_leakage_tolerance: float | None = None
    wannier_targets: list[dict[str, Any]] | None = None
    representation_analysis: list[dict[str, Any]] | None = None
    symmetry_resolved_path: Path | None = field(default=None, init=False)
    disentangle_max_iter: int | None = None
    disentangle_err_diff: float | None = None
    disentangle_projector_tolerance: float | None = None
    disentangle_mixing: float = 0.5
    symmetry_context: SymmetryContext | None = field(default=None, init=False, repr=False)
    maxwell_problem: MaxwellProblem | None = field(default=None, init=False, repr=False)

    N_file: str = "./N.txt"
    U_file: str = "./U.txt"
    V_file: str = "./V.txt"
    M_file: str = "./M.txt"
    A_file: str = "./A.txt"
    S_file: str = "./S.txt"
    D_file: str = "./D.txt"
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
    projection_rank_tolerance: float = 1.0e-10
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
        from .maxwell import FieldComponents

        if self.dataset_type.lower() != "comsol":
            raise NotImplementedError("Only COMSOL input is implemented in PCWannier v1.")
        if not self.hermitian:
            raise NotImplementedError("Non-Hermitian left/right fields are not implemented in PCWannier v1.")
        if self.finite is not False:
            raise NotImplementedError("Finite-system calculations are not implemented in PCWannier v1.")
        if self.eff_k is not False:
            raise NotImplementedError("Effective Hamiltonian expansion is not implemented in PCWannier v1.")
        if FieldComponents.parse(self.field_components) == FieldComponents.FULL_VECTOR:
            raise NotImplementedError(
                "field_components=full_vector is not implemented; the current COMSOL "
                "reader supports scalar Ez and Hz fields only."
            )

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
                "metric_file",
                "mesh_file",
                "E_file",
            )
            if getattr(self, name) is None
            or (name == "metric_file" and getattr(self, name) is False)
        ]
        if missing:
            raise ValueError(f"Missing required incar fields: {', '.join(missing)}")


_INTERNAL_CONFIG_FIELDS = {
    "base_dir",
    "kdim",
    "b_vectors",
    "wb",
    "band_calc_num",
    "symmetry_resolved_path",
    "symmetry_context",
    "maxwell_problem",
    "_preprocessed",
}


def _user_config_fields() -> set[str]:
    return {
        item.name
        for item in fields(IncarConfig)
        if item.init and item.name not in _INTERNAL_CONFIG_FIELDS
    }


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
        assigned_fields: set[str] = set()
        allowed_fields = _user_config_fields()

        def reserve_field(name: str) -> None:
            if name not in allowed_fields:
                raise ValueError(f"Unknown incar field: {name!r}.")
            if name in assigned_fields:
                raise ValueError(f"Duplicate incar field or block: {name!r}.")
            assigned_fields.add(name)

        with self.filename.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                if line == "projections":
                    reserve_field("projections")
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
                    reserve_field("k_path")
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
                    reserve_field("wannier_targets")
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
                    reserve_field("representation_analysis")
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
                    raise ValueError(
                        f"Malformed incar line {line_number}: expected 'key = value' or a known block, got {line!r}."
                    )
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "w_center":
                    raise ValueError(
                        "The w_center input has been removed because forcing Wannier centers is not a physical operation."
                    )
                if key == "symmetry":
                    raise ValueError("The boolean symmetry input has been removed; use symmetry_file = ./sym.yaml.")
                reserve_field(key)
                if key in {"projections", "k_path", "wannier_targets", "representation_analysis"}:
                    raise ValueError(f"incar field {key!r} must use its block form terminated by 'end'.")
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

        cfg.validate_required()
        preprocess_config(cfg)
        cfg.validate_runtime_scope()
        if cfg.symmetry_file is not False and str(cfg.symmetry_file).lower() != "false":
            from .symmetry import (
                BlochConvention,
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
                    cfg.maxwell_problem.symmetry_field_kind,
                    degeneracy,
                    points,
                    (
                        cfg.symmetry_tolerance
                        if cfg.representation_leakage_tolerance is None
                        else cfg.representation_leakage_tolerance
                    ),
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
                SymmetryCalculationSpec(
                    target_specs=target_specs,
                    representation_analysis=analysis,
                    symmetry_gauge=gauge,
                    bloch_convention=BlochConvention.for_dataset(cfg.dataset_type),
                    boundary_tolerance=cfg.symmetry_boundary_tolerance,
                ),
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
                "field_components",
            "symmetry_file",
            "symmetry_output_basis",
            "dataset_file",
            "left_dataset_file",
                "metric_file",
            "S_file",
            "D_file",
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
            "disentangle_projector_tolerance",
            "disentangle_mixing",
            "DOS_eps",
            "finite_DOS_eps",
            "symmetry_tolerance",
            "symmetry_svd_tolerance",
            "symmetry_real_space_tolerance",
            "symmetry_minimum_retained_norm",
            "symmetry_boundary_tolerance",
            "representation_degeneracy_absolute",
            "representation_degeneracy_relative",
            "representation_leakage_tolerance",
            "projection_rank_tolerance",
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
                if not np.isfinite(step) or step == 0.0:
                    raise ValueError(f"Invalid k_points range with zero or non-finite step: {part!r}")
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
            normalized = value.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(
                    f"Boolean incar field {key!r} must be 'true' or 'false', got {value!r}."
                )
            return normalized == "true"
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


def _validate_band_window(name: str, window, *, allow_false: bool) -> None:
    if window is None:
        return
    if window is False:
        if allow_false:
            return
        raise ValueError(f"{name} must not be false.")
    if isinstance(window, EnergyWindow):
        if not np.isfinite(window.emin) or not np.isfinite(window.emax) or window.emin >= window.emax:
            raise ValueError(f"{name} energy bounds must be finite and strictly increasing.")
        return
    values = np.asarray(window)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} band indices must be a non-empty one-dimensional list.")
    numeric = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} band indices must be finite integers.")
    indices = numeric.astype(int)
    if np.any(indices < 0) or np.unique(indices).size != indices.size:
        raise ValueError(f"{name} band indices must be unique and non-negative.")


def _validate_vector_list(name: str, vectors, dimension: int, *, reject_opposites: bool) -> None:
    if vectors is None:
        return
    seen: list[np.ndarray] = []
    for index, raw in enumerate(vectors):
        vector = np.asarray(raw, dtype=float)
        if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name}[{index}] must be a finite vector of dimension {dimension}.")
        if np.linalg.norm(vector) == 0.0:
            raise ValueError(f"{name}[{index}] must be non-zero.")
        for previous in seen:
            if np.array_equal(vector, previous) or (
                reject_opposites and np.array_equal(vector, -previous)
            ):
                qualifier = " (including the automatically generated opposite direction)" if reject_opposites else ""
                raise ValueError(f"{name} contains a duplicate vector{qualifier}: {vector.tolist()}.")
        seen.append(vector)


def _validate_config_inputs(cfg: IncarConfig) -> None:
    dimension = None
    if cfg.lattice_const is not None and (
        not np.isfinite(cfg.lattice_const) or float(cfg.lattice_const) <= 0.0
    ):
        raise ValueError("lattice_const must be positive and finite.")
    if cfg.real_lattice_vectors is not None:
        lattice = np.asarray(cfg.real_lattice_vectors, dtype=float)
        if lattice.ndim != 2 or lattice.shape[0] == 0 or lattice.shape[0] != lattice.shape[1]:
            raise ValueError("real_lattice_vectors must be a non-empty square matrix.")
        if not np.all(np.isfinite(lattice)):
            raise ValueError("real_lattice_vectors must contain only finite values.")
        if np.linalg.matrix_rank(lattice) != lattice.shape[0]:
            raise ValueError("real_lattice_vectors must be invertible.")
        dimension = int(lattice.shape[0])
        origin = np.asarray(cfg.origin, dtype=float)
        if origin.shape != (dimension,) or not np.all(np.isfinite(origin)):
            raise ValueError(f"origin must be a finite vector of dimension {dimension}.")

    if cfg.k_points is not None:
        if dimension is not None and len(cfg.k_points) != dimension:
            raise ValueError(f"k_points must define exactly {dimension} axes.")
        for axis_index, raw_axis in enumerate(cfg.k_points):
            axis = np.asarray(raw_axis, dtype=float)
            if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
                raise ValueError(f"k_points axis {axis_index} must be non-empty and finite.")
            if axis.size > 1:
                spacing = np.diff(axis)
                scale = max(float(np.max(np.abs(axis))), 1.0)
                tolerance = max(128.0 * np.finfo(float).eps * scale, 1.0e-12)
                if abs(float(spacing[0])) <= tolerance or not np.allclose(
                    spacing, spacing[0], rtol=1.0e-10, atol=tolerance
                ):
                    raise ValueError(f"k_points axis {axis_index} must be uniformly sampled with non-zero spacing.")
                for left in range(axis.size):
                    for right in range(left + 1, axis.size):
                        difference = float(axis[left] - axis[right])
                        if abs(difference - round(difference)) <= tolerance:
                            raise ValueError(
                                f"k_points axis {axis_index} contains periodic duplicate samples "
                                f"{axis[left]} and {axis[right]}."
                            )

    _validate_band_window("band_window", cfg.band_window, allow_false=False)
    _validate_band_window("inner_window", cfg.inner_window, allow_false=True)
    if (
        isinstance(cfg.band_window, np.ndarray)
        and isinstance(cfg.inner_window, np.ndarray)
        and not set(np.asarray(cfg.inner_window, dtype=int)).issubset(
            set(np.asarray(cfg.band_window, dtype=int))
        )
    ):
        raise ValueError("All frozen inner_window bands must belong to band_window.")
    if isinstance(cfg.band_window, EnergyWindow) and isinstance(cfg.inner_window, EnergyWindow):
        if cfg.inner_window.emin < cfg.band_window.emin or cfg.inner_window.emax > cfg.band_window.emax:
            raise ValueError("inner_window energy bounds must lie inside band_window.")

    if cfg.max_iter < 0:
        raise ValueError("max_iter must be non-negative.")
    if cfg.disentangle_max_iter is not None and cfg.disentangle_max_iter < 0:
        raise ValueError("disentangle_max_iter must be non-negative.")
    for name, value, strictly_positive in (
        ("epsilon", cfg.epsilon, True),
        ("err_diff", cfg.err_diff, False),
        ("disentangle_err_diff", cfg.disentangle_err_diff, False),
        ("disentangle_projector_tolerance", cfg.disentangle_projector_tolerance, True),
    ):
        if value is None:
            continue
        valid = np.isfinite(value) and (value > 0.0 if strictly_positive else value >= 0.0)
        if not valid:
            relation = "positive" if strictly_positive else "non-negative"
            raise ValueError(f"{name} must be finite and {relation}.")
    if not np.isfinite(cfg.projection_rank_tolerance) or not 0.0 < cfg.projection_rank_tolerance < 1.0:
        raise ValueError("projection_rank_tolerance must lie in (0, 1).")

    if dimension is not None:
        if cfg.extension is not None:
            values = np.asarray(cfg.extension)
            if values.shape != (dimension,) or not np.all(values == np.floor(values)) or np.any(values <= 0):
                raise ValueError(f"extension must contain {dimension} positive integers.")
        _validate_vector_list(
            "composition_of_b", cfg.composition_of_b, dimension, reject_opposites=True
        )
        _validate_vector_list("neighbor", cfg.neighbor, dimension, reject_opposites=False)
        if cfg.k_path is not None:
            if not cfg.k_path:
                raise ValueError("k_path must not be empty when specified.")
            for index, point in enumerate(cfg.k_path):
                coordinates = np.asarray(point.get("point", ()), dtype=float)
                if coordinates.shape != (dimension,) or not np.all(np.isfinite(coordinates)):
                    raise ValueError(f"k_path point {index} must be finite and have dimension {dimension}.")
                count = point.get("num")
                if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or int(count) <= 0:
                    raise ValueError(f"k_path point {index} must have a positive integer segment count.")

        if cfg.projections is not None:
            if not cfg.projections:
                raise ValueError("projections must not be empty.")
            for projection_index, projection in enumerate(cfg.projections):
                center = np.asarray(projection.get("frac_position", ()), dtype=float)
                if center.shape != (dimension,) or not np.all(np.isfinite(center)):
                    raise ValueError(
                        f"projection {projection_index} center must be finite and have dimension {dimension}."
                    )
                angle = float(projection.get("xaxis_angluar", np.nan))
                if not np.isfinite(angle):
                    raise ValueError(f"projection {projection_index} x-axis angle must be finite.")
                states = projection.get("states")
                if not states:
                    raise ValueError(f"projection {projection_index} must define at least one state.")
                for state_index, state in enumerate(states):
                    definitions = state.get("lc_states") if isinstance(state, dict) else [state]
                    coefficients = state.get("lc_coeffs") if isinstance(state, dict) else None
                    if coefficients is not None and len(definitions) != len(coefficients):
                        raise ValueError(
                            f"projection {projection_index} state {state_index} has mismatched linear-combination data."
                        )
                    for definition in definitions:
                        values = np.asarray(definition, dtype=float)
                        if values.shape != (3,) or not np.all(np.isfinite(values)):
                            raise ValueError(
                                f"projection {projection_index} state {state_index} must use [n, l, z] definitions."
                            )

    if cfg.integration_mode not in {"nodal", "quadratic"}:
        raise ValueError("integration_mode must be 'nodal' or 'quadratic'.")
    cfg.symmetry_output_basis = str(cfg.symmetry_output_basis).strip().lower()
    if cfg.symmetry_output_basis not in {"strict", "fem"}:
        raise ValueError("symmetry_output_basis must be 'strict' or 'fem'.")
    if not np.isfinite(cfg.disentangle_mixing) or not 0.0 < cfg.disentangle_mixing <= 1.0:
        raise ValueError("disentangle_mixing must lie in (0, 1].")
    if not np.isfinite(cfg.symmetry_tolerance) or cfg.symmetry_tolerance <= 0.0:
        raise ValueError("symmetry_tolerance must be positive and finite.")
    if not np.isfinite(cfg.symmetry_boundary_tolerance) or cfg.symmetry_boundary_tolerance <= 0.0:
        raise ValueError("symmetry_boundary_tolerance must be positive and finite.")
    if cfg.representation_leakage_tolerance is not None and (
        not np.isfinite(cfg.representation_leakage_tolerance)
        or cfg.representation_leakage_tolerance <= 0.0
    ):
        raise ValueError("representation_leakage_tolerance must be positive and finite.")
    if cfg.symmetry_max_iter <= 0:
        raise ValueError("symmetry_max_iter must be positive.")
    if not np.isfinite(cfg.symmetry_svd_tolerance) or cfg.symmetry_svd_tolerance <= 0.0:
        raise ValueError("symmetry_svd_tolerance must be positive and finite.")
    if not np.isfinite(cfg.symmetry_real_space_tolerance) or cfg.symmetry_real_space_tolerance <= 0.0:
        raise ValueError("symmetry_real_space_tolerance must be positive and finite.")
    if not 0.0 < cfg.symmetry_minimum_retained_norm <= 1.0:
        raise ValueError("symmetry_minimum_retained_norm must lie in (0, 1].")


def preprocess_config(cfg: IncarConfig) -> IncarConfig:
    if cfg._preprocessed:
        return cfg
    _validate_config_inputs(cfg)
    from .maxwell import FieldComponents, MaxwellProblem

    components = FieldComponents.parse(cfg.field_components)
    cfg.field_components = components.value
    if components != FieldComponents.FULL_VECTOR:
        cfg.maxwell_problem = MaxwellProblem.for_components(components)
    if cfg.real_lattice_vectors is None:
        return cfg
    cfg.kdim = len(cfg.real_lattice_vectors)
    reciprocal = np.asarray(cfg.reciprocal_lattice_vectors, dtype=float)
    if reciprocal.size > 0 and np.allclose(reciprocal, 0.0):
        reciprocal = (np.linalg.inv(np.asarray(cfg.real_lattice_vectors, dtype=float)) @ np.eye(cfg.kdim)).T
    cfg.reciprocal_lattice_vectors = reciprocal
    if reciprocal.shape != (cfg.kdim, cfg.kdim) or not np.all(np.isfinite(reciprocal)):
        raise ValueError(
            f"reciprocal_lattice_vectors must be a finite {cfg.kdim}x{cfg.kdim} matrix."
        )
    if np.linalg.matrix_rank(reciprocal) != cfg.kdim:
        raise ValueError("reciprocal_lattice_vectors must be invertible.")

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

    if cfg.projections is not None:
        cfg.band_calc_num = sum(len(p["states"]) for p in cfg.projections)
    cfg._preprocessed = True
    return cfg


def load_config(path: str | Path) -> IncarConfig:
    return IncarParser(path).parse_file()
