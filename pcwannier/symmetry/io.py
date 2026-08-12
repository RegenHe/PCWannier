from __future__ import annotations

import ast
import cmath
from functools import lru_cache
from importlib import resources
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import spglib
import yaml

from .definition import (
    FiniteGroupDefinition,
    FiniteGroupLibrary,
    SpaceGroupDefinition,
    build_group_irrep,
    validate_irrep_table,
)
from .group import SpaceGroup, SpaceGroupOperation
from .representation import (
    SymmetryModel,
    build_wannier_target_from_group_irrep,
)
from .specs import SymmetryCalculationSpec
from .tables import ConcreteFiniteGroup, FiniteGroupTable


LOGGER = logging.getLogger(__name__)


_FINITE_GROUP_FILES = (
    "C1.yaml",
    "C2.yaml",
    "C3.yaml",
    "C4.yaml",
    "C6.yaml",
    "Cs.yaml",
    "C2v.yaml",
    "C3v.yaml",
    "C4v.yaml",
    "C6v.yaml",
)

_SPACE_GROUP_ALIASES = {
    "cm.yaml": "c1m1.yaml",
    "cmm.yaml": "c2mm.yaml",
    "pmm.yaml": "p2mm.yaml",
    "pmg.yaml": "p2mg.yaml",
    "pgg.yaml": "p2gg.yaml",
    "p4m.yaml": "p4mm.yaml",
    "p4g.yaml": "p4gm.yaml",
    "p6m.yaml": "p6mm.yaml",
}


def resolve_symmetry_file(path: str | Path, base_dir: str | Path) -> Path:
    requested = Path(path)
    requested_candidates = [requested]
    if requested.suffix == "":
        requested_candidates.append(requested.with_suffix(".yaml"))
    for candidate in requested_candidates:
        explicit = candidate if candidate.is_absolute() else Path(base_dir) / candidate
        if explicit.is_file():
            return explicit.resolve()
    resource_name = requested_candidates[-1].name.lower()
    resource_name = _SPACE_GROUP_ALIASES.get(resource_name, resource_name)
    library = resources.files("pcwannier.symmetry").joinpath("space_groups", resource_name)
    if library.is_file():
        return Path(str(library))
    explicit = requested if requested.is_absolute() else Path(base_dir) / requested
    raise FileNotFoundError(
        f"Space-group file {explicit} does not exist, and {requested.name!r} was not found "
        "in the built-in space_groups library."
    )


@lru_cache(maxsize=None)
def _hall_number_for_symbol(symbol: str) -> int:
    requested = str(symbol).strip()
    if requested.lower().startswith("hall:"):
        try:
            hall_number = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid spglib Hall reference {requested!r}.") from exc
        if spglib.get_spacegroup_type(hall_number) is None:
            raise ValueError(f"Unknown spglib Hall number {hall_number}.")
        return hall_number

    normalized = requested.replace(" ", "").lower()
    matches = []
    for hall_number in range(1, 531):
        group_type = spglib.get_spacegroup_type(hall_number)
        if group_type is None:
            continue
        names = {
            str(group_type.international_short).replace(" ", "").lower(),
            str(group_type.international).replace(" ", "").lower(),
            str(group_type.hall_symbol).replace(" ", "").lower(),
        }
        if normalized in names:
            matches.append(hall_number)
    if not matches:
        raise FileNotFoundError(
            f"{requested!r} is neither a symmetry YAML file nor a recognized spglib space-group symbol."
        )
    if len(matches) != 1:
        choices = ", ".join(
            f"hall:{value} (choice {spglib.get_spacegroup_type(value).choice or 'default'})"
            for value in matches
        )
        raise ValueError(
            f"Space-group symbol {requested!r} has multiple Hall settings: {choices}. "
            "Use hall:<number> to select the setting explicitly."
        )
    return int(matches[0])


def load_space_group_from_spglib(
    symbol: str,
    *,
    tolerance: float = 1.0e-8,
    algebra_tolerance: float = 1.0e-10,
    finite_groups: FiniteGroupLibrary | None = None,
    lattice_vectors=None,
) -> SpaceGroupDefinition:
    """Load a finite set of Seitz representatives from spglib's database."""

    hall_number = _hall_number_for_symbol(symbol)
    group_type = spglib.get_spacegroup_type(hall_number)
    symmetry = spglib.get_symmetry_from_database(hall_number)
    if group_type is None or symmetry is None:
        raise ValueError(f"spglib could not load Hall number {hall_number}.")
    rotations = np.asarray(symmetry["rotations"], dtype=np.int64)
    translations = np.mod(np.asarray(symmetry["translations"], dtype=float), 1.0)
    translations[np.abs(translations - 1.0) <= tolerance] = 0.0
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("spglib returned invalid space-group rotations.")
    if translations.shape != (rotations.shape[0], 3):
        raise ValueError("spglib returned invalid space-group translations.")

    if lattice_vectors is not None:
        rotations, translations, setting = _operations_for_lattice(
            rotations,
            translations,
            lattice_vectors,
            str(group_type.international_short),
            tolerance,
        )
        LOGGER.info(
            "spglib Hall %d operations adapted to %s lattice setting: count=%d",
            hall_number,
            setting,
            len(rotations),
        )

    order = sorted(
        range(len(rotations)),
        key=lambda index: (
            0 if np.array_equal(rotations[index], np.eye(3, dtype=np.int64))
            and np.allclose(translations[index], 0.0, rtol=0.0, atol=tolerance) else 1,
            tuple(int(value) for value in rotations[index].reshape(-1)),
            tuple(float(value) for value in np.round(translations[index], 12)),
        ),
    )
    operations = []
    for output_index, source_index in enumerate(order):
        identity = (
            np.array_equal(rotations[source_index], np.eye(3, dtype=np.int64))
            and np.allclose(translations[source_index], 0.0, rtol=0.0, atol=tolerance)
        )
        operations.append(
            SpaceGroupOperation(
                rotations[source_index],
                translations[source_index],
                "E" if identity else f"g{output_index:02d}",
            )
        )
    group = SpaceGroup(tuple(operations), tolerance)
    return SpaceGroupDefinition(
        str(group_type.international_short),
        3,
        float(tolerance),
        group,
        finite_groups or load_builtin_finite_groups(),
        float(algebra_tolerance),
        hall_number=hall_number,
    )


def load_symmetry_from_spglib(
    symbol: str,
    *,
    tolerance: float = 1.0e-8,
    algebra_tolerance: float = 1.0e-10,
    lattice_vectors=None,
) -> SymmetryModel:
    definition = load_space_group_from_spglib(
        symbol,
        tolerance=tolerance,
        algebra_tolerance=algebra_tolerance,
        lattice_vectors=lattice_vectors,
    )
    return SymmetryModel(
        definition.dimension,
        definition.tolerance,
        definition.group,
        (),
        None,
        None,
        definition,
        algebra_tolerance=definition.algebra_tolerance,
    )


def _operations_for_lattice(
    rotations: np.ndarray,
    translations: np.ndarray,
    lattice_vectors,
    international_symbol: str,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Express database Seitz operations in the configured lattice basis.

    spglib's database uses the conventional Hall setting.  A calculation may
    instead use a primitive cell, in which centering translations become
    ordinary lattice translations and must be removed from the representative
    set.
    """

    lattice = np.asarray(lattice_vectors, dtype=float)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("A finite 3x3 lattice is required to adapt spglib operations.")
    if abs(float(np.linalg.det(lattice))) <= np.finfo(float).tiny:
        raise ValueError("The lattice used to adapt spglib operations is singular.")

    direct_residual = _maximum_isometry_residual(rotations, lattice)
    threshold = max(1.0e-8, 100.0 * float(tolerance))
    if direct_residual <= threshold:
        return rotations, translations, "conventional"

    centering = international_symbol.strip()[:1].upper()
    primitive_basis = _conventional_primitive_basis(centering)
    if primitive_basis is None:
        raise ValueError(
            f"spglib operations for {international_symbol!r} are not isometries of the "
            f"configured lattice (residual={direct_residual:.6g}), and centering "
            f"{centering!r} has no supported primitive transformation."
        )

    inverse = np.linalg.inv(primitive_basis)
    transformed_rotations = []
    transformed_translations = []
    for rotation, translation in zip(rotations, translations):
        raw_rotation = inverse @ rotation @ primitive_basis
        integer_rotation = np.rint(raw_rotation).astype(np.int64)
        if not np.allclose(
            raw_rotation,
            integer_rotation,
            rtol=0.0,
            atol=threshold,
        ):
            raise ValueError(
                f"Hall operation cannot be represented by an integer rotation in the "
                f"configured {centering}-primitive basis: {raw_rotation.tolist()}."
            )
        transformed_rotations.append(integer_rotation)
        transformed_translations.append(inverse @ translation)

    unique_rotations, unique_translations = _deduplicate_seitz_representatives(
        np.asarray(transformed_rotations, dtype=np.int64),
        np.asarray(transformed_translations, dtype=float),
        tolerance,
    )
    primitive_residual = _maximum_isometry_residual(unique_rotations, lattice)
    if primitive_residual > threshold:
        raise ValueError(
            f"Neither the conventional nor the standard {centering}-primitive Hall "
            f"setting matches the configured lattice. Isometry residuals are "
            f"{direct_residual:.6g} and {primitive_residual:.6g}, respectively. "
            "Use a symmetry YAML expressed in the calculation's fractional basis."
        )
    return unique_rotations, unique_translations, f"{centering}-primitive"


def _maximum_isometry_residual(rotations: np.ndarray, lattice: np.ndarray) -> float:
    basis = lattice.T
    inverse = np.linalg.inv(basis)
    identity = np.eye(3)
    return max(
        float(np.linalg.norm((basis @ rotation @ inverse).T @ (basis @ rotation @ inverse) - identity))
        for rotation in rotations
    )


def _conventional_primitive_basis(centering: str) -> np.ndarray | None:
    """Return columns of a standard primitive basis in conventional fractions."""

    matrices = {
        "F": np.asarray(
            [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
        ),
        "I": np.asarray(
            [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]
        ),
        "A": np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.5, -0.5], [0.0, 0.5, 0.5]]
        ),
        "B": np.asarray(
            [[0.5, -0.5, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]
        ),
        "C": np.asarray(
            [[0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
        ),
    }
    return matrices.get(centering)


def _deduplicate_seitz_representatives(
    rotations: np.ndarray,
    translations: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    unique_rotations = []
    unique_translations = []
    for rotation, translation in zip(rotations, translations):
        reduced_translation = np.mod(translation, 1.0)
        reduced_translation[
            np.abs(reduced_translation - 1.0) <= tolerance
        ] = 0.0
        duplicate = False
        for previous_rotation, previous_translation in zip(
            unique_rotations, unique_translations
        ):
            if not np.array_equal(rotation, previous_rotation):
                continue
            difference = reduced_translation - previous_translation
            if np.allclose(
                difference - np.rint(difference),
                0.0,
                rtol=0.0,
                atol=tolerance,
            ):
                duplicate = True
                break
        if not duplicate:
            unique_rotations.append(rotation)
            unique_translations.append(reduced_translation)
    return (
        np.asarray(unique_rotations, dtype=np.int64),
        np.asarray(unique_translations, dtype=float),
    )


@lru_cache(maxsize=1)
def _spglib_point_group_hall_numbers() -> dict[str, int]:
    """Return one deterministic crystallographic setting for each 3D point group."""

    output: dict[str, int] = {}
    for hall_number in range(1, 531):
        group_type = spglib.get_spacegroup_type(hall_number)
        if group_type is None:
            continue
        keys = {
            str(group_type.pointgroup_international),
            str(group_type.pointgroup_schoenflies),
        }
        for key in keys:
            normalized = _normalize_point_group_name(key)
            output.setdefault(normalized, hall_number)
    return output


def _normalize_point_group_name(value: str) -> str:
    return str(value).strip().replace(" ", "").replace("_", "").casefold()


@lru_cache(maxsize=None)
def load_point_group_from_spglib(
    symbol: str,
    *,
    tolerance: float = 1.0e-8,
) -> FiniteGroupDefinition:
    """Generate a finite-group character table from a 3D point-group symbol.

    spglib supplies a standard crystallographic rotation set.  The existing
    finite-group library then identifies the group and delegates irrep
    enumeration to spgrep, exactly as it does for calculated little groups.
    """

    requested = str(symbol).strip()
    if not requested:
        raise ValueError("Point-group symbol must be non-empty.")
    hall_number = _spglib_point_group_hall_numbers().get(
        _normalize_point_group_name(requested)
    )
    if hall_number is None:
        raise ValueError(
            f"Unknown crystallographic point-group symbol {requested!r}. "
            "Use a Hermann-Mauguin symbol such as 'm-3m' or a Schoenflies "
            "symbol such as 'Oh'."
        )

    group_type = spglib.get_spacegroup_type(hall_number)
    symmetry = spglib.get_symmetry_from_database(hall_number)
    if group_type is None or symmetry is None:
        raise ValueError(
            f"spglib could not construct point group {requested!r} from Hall {hall_number}."
        )

    unique: dict[tuple[int, ...], np.ndarray] = {}
    for raw_rotation in np.asarray(symmetry["rotations"], dtype=np.int64):
        key = tuple(int(value) for value in raw_rotation.reshape(-1))
        unique.setdefault(key, raw_rotation.copy())
    rotations = sorted(
        unique.values(),
        key=lambda rotation: (
            0 if np.array_equal(rotation, np.eye(3, dtype=np.int64)) else 1,
            tuple(int(value) for value in rotation.reshape(-1)),
        ),
    )
    operations = tuple(
        SpaceGroupOperation(
            rotation,
            np.zeros(3, dtype=float),
            "E" if index == 0 else f"g{index:02d}",
        )
        for index, rotation in enumerate(rotations)
    )
    concrete = ConcreteFiniteGroup.from_space_group(
        SpaceGroup(operations, tolerance),
        name=str(group_type.pointgroup_international),
    )
    definition = load_builtin_finite_groups().identify(concrete).canonical
    expected = _normalize_point_group_name(group_type.pointgroup_international)
    if _normalize_point_group_name(definition.point_group_symbol) != expected:
        raise RuntimeError(
            "Generated finite-group catalog has an inconsistent point-group symbol: "
            f"expected {group_type.pointgroup_international!r}, got "
            f"{definition.point_group_symbol!r}."
        )
    return definition


def load_space_group(
    path: str | Path,
    *,
    tolerance: float = 1.0e-8,
    algebra_tolerance: float = 1.0e-10,
    finite_groups: FiniteGroupLibrary | None = None,
) -> SpaceGroupDefinition:
    raw = _read_yaml(path, "Space-group")
    _require_keys(raw, {"name", "dimension", "operations"}, "space-group file")
    name = _nonempty_string(raw["name"], "space-group name")
    dimension = _positive_int(raw["dimension"], "dimension")
    operations_raw = raw["operations"]
    if not isinstance(operations_raw, list) or not operations_raw:
        raise ValueError("operations must be a non-empty YAML list.")
    operations = tuple(
        _parse_space_operation(value, dimension, index)
        for index, value in enumerate(operations_raw)
    )
    group = SpaceGroup(operations, tolerance)
    return SpaceGroupDefinition(
        name,
        dimension,
        float(tolerance),
        group,
        finite_groups or load_builtin_finite_groups(),
        float(algebra_tolerance),
    )


def load_symmetry(
    path: str | Path,
    *,
    tolerance: float = 1.0e-8,
    algebra_tolerance: float = 1.0e-10,
) -> SymmetryModel:
    definition = load_space_group(
        path,
        tolerance=tolerance,
        algebra_tolerance=algebra_tolerance,
    )
    return SymmetryModel(
        definition.dimension,
        definition.tolerance,
        definition.group,
        (),
        None,
        None,
        definition,
        algebra_tolerance=definition.algebra_tolerance,
    )


def load_finite_group(path: str | Path) -> FiniteGroupDefinition:
    raw = _read_yaml(path, "Finite-group")
    _require_keys(
        raw,
        {"name", "dimension", "point_group_symbol", "elements", "irreps"},
        "finite-group file",
        optional={"multiplication"},
    )
    name = _nonempty_string(raw["name"], "finite-group name")
    dimension = _positive_int(raw["dimension"], "dimension")
    point_group_symbol = _nonempty_string(
        raw["point_group_symbol"],
        "point_group_symbol",
    )
    elements_raw = raw["elements"]
    if not isinstance(elements_raw, list) or not elements_raw:
        raise ValueError("finite-group elements must be a non-empty YAML list.")
    element_names = []
    point_actions = []
    has_action = []
    for index, element_raw in enumerate(elements_raw):
        if not isinstance(element_raw, dict):
            raise ValueError(f"elements[{index}] must be a YAML mapping.")
        _require_keys(element_raw, {"name"}, f"elements[{index}]", optional={"point_action"})
        element_names.append(_nonempty_string(element_raw["name"], f"elements[{index}].name"))
        action_raw = element_raw.get("point_action")
        has_action.append(action_raw is not None)
        point_actions.append(
            None
            if action_raw is None
            else _integer_matrix(action_raw, dimension, f"elements[{index}].point_action")
        )
    if len(element_names) != len(set(element_names)):
        raise ValueError("Finite-group element names must be unique.")
    if not all(has_action):
        raise ValueError(
            "Finite-group label catalogs require point_action for every canonical element."
        )
    multiplication = _multiplication_from_actions(tuple(point_actions))
    if "multiplication" in raw:
        declared = _parse_multiplication(raw["multiplication"], tuple(element_names))
        if not np.array_equal(multiplication, declared):
            raise ValueError("Declared finite-group multiplication disagrees with point_action products.")
    actions = tuple(point_actions)
    table = FiniteGroupTable(tuple(element_names), multiplication, name=name)
    irreps = _parse_irreps(raw["irreps"], table)
    validate_irrep_table(table, irreps, actions, dimension=dimension)
    return FiniteGroupDefinition(
        name,
        dimension,
        point_group_symbol,
        table,
        actions,
        irreps,
    )


@lru_cache(maxsize=1)
def load_builtin_finite_groups() -> FiniteGroupLibrary:
    root = resources.files("pcwannier.symmetry").joinpath("finite_groups")
    definitions = []
    for filename in _FINITE_GROUP_FILES:
        resource = root.joinpath(filename)
        if not resource.is_file():
            raise FileNotFoundError(f"Built-in finite-group resource {filename!r} is missing.")
        definitions.append(load_finite_group(Path(str(resource))))
    return FiniteGroupLibrary(tuple(definitions))


def compose_symmetry_model(
    base: SymmetryModel,
    calculation: SymmetryCalculationSpec,
) -> SymmetryModel:
    convention = (
        base.bloch_convention
        if calculation.bloch_convention is None
        else calculation.bloch_convention
    )
    boundary_tolerance = (
        base.boundary_tolerance
        if calculation.boundary_tolerance is None
        else calculation.boundary_tolerance
    )
    targets = base.targets
    if calculation.target_specs is not None:
        if base.group_definition is None:
            raise ValueError("Wannier targets require a loaded space-group definition.")
        from .group import build_crystallographic_orbit

        built_targets = []
        names = set()
        for target_spec in calculation.target_specs:
            if target_spec.name in names:
                raise ValueError(f"Duplicate Wannier target name: {target_spec.name!r}.")
            names.add(target_spec.name)
            orbit = build_crystallographic_orbit(base.group, target_spec.center)
            site_indices = tuple(
                element.source_operation_index for element in orbit.site_symmetry.elements
            )
            resolved_irrep = base.group_definition.site_irrep(
                site_indices, target_spec.site_irrep
            )
            built_targets.append(
                build_wannier_target_from_group_irrep(
                    target_spec.name,
                    base.group,
                    target_spec.center,
                    resolved_irrep,
                    convention,
                )
            )
        targets = tuple(built_targets)
    analysis = (
        calculation.representation_analysis
        if calculation.representation_analysis is not None
        else base.representation_analysis
    )
    gauge = (
        calculation.symmetry_gauge
        if calculation.symmetry_gauge is not None
        else base.symmetry_gauge
    )
    if analysis is not None:
        target_names = {target.name for target in targets}
        for point in analysis.points:
            if point.target_names is None:
                continue
            unknown = sorted(set(point.target_names) - target_names)
            if unknown:
                raise ValueError(
                    f"Representation point {point.name!r} references unknown Wannier targets {unknown}."
                )
    if gauge is not None and gauge.enabled and not targets:
        raise ValueError("Symmetry-constrained gauge construction requires Wannier targets.")
    return SymmetryModel(
        base.dimension,
        base.tolerance,
        base.group,
        targets,
        analysis,
        gauge,
        base.group_definition,
        convention,
        boundary_tolerance,
        base.magnetic_bias_direction,
        base.algebra_tolerance,
    )


def _read_yaml(path, description: str) -> dict[str, Any]:
    filename = Path(path)
    try:
        with filename.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid {description.lower()} YAML in {filename}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{description} file {filename} must contain a YAML mapping.")
    return raw


def _parse_space_operation(raw: Any, dimension: int, index: int) -> SpaceGroupOperation:
    if not isinstance(raw, dict):
        raise ValueError(f"operations[{index}] must be a YAML mapping.")
    _require_keys(raw, {"rotation", "translation"}, f"operations[{index}]", optional={"name"})
    rotation = _integer_matrix(raw["rotation"], dimension, f"operations[{index}].rotation")
    translation = _vector(raw["translation"], dimension, f"operations[{index}].translation")
    name = raw.get("name", f"g{index}")
    return SpaceGroupOperation(rotation, translation, _nonempty_string(name, f"operations[{index}].name"))


def _parse_irreps(raw: Any, table: FiniteGroupTable):
    if not isinstance(raw, dict) or not raw:
        raise ValueError("finite-group irreps must be a non-empty mapping.")
    output = []
    for irrep_name, irrep_raw in raw.items():
        if not isinstance(irrep_raw, dict):
            raise ValueError(f"Irrep {irrep_name!r} must be a YAML mapping.")
        _require_keys(
            irrep_raw,
            {"dimension", "characters"},
            f"irrep {irrep_name!r}",
            optional={"generators", "matrices"},
        )
        dimension = _positive_int(irrep_raw["dimension"], f"irrep {irrep_name!r}.dimension")
        characters = _parse_characters(irrep_raw["characters"], f"irrep {irrep_name!r}.characters")
        generators = (
            None
            if "generators" not in irrep_raw
            else _parse_named_matrices(
                irrep_raw["generators"], dimension, f"irrep {irrep_name!r}.generators"
            )
        )
        matrices = (
            None
            if "matrices" not in irrep_raw
            else _parse_named_matrices(
                irrep_raw["matrices"], dimension, f"irrep {irrep_name!r}.matrices"
            )
        )
        output.append(
            build_group_irrep(
                table,
                str(irrep_name),
                dimension,
                characters=characters,
                generators=generators,
                matrices=matrices,
            )
        )
    return tuple(output)


def _multiplication_from_actions(actions: tuple[np.ndarray, ...]) -> np.ndarray:
    output = np.empty((len(actions), len(actions)), dtype=np.int64)
    for left, left_action in enumerate(actions):
        for right, right_action in enumerate(actions):
            value = left_action @ right_action
            matches = [index for index, candidate in enumerate(actions) if np.array_equal(value, candidate)]
            if len(matches) != 1:
                raise ValueError("Finite-group point_action elements are not closed or are duplicated.")
            output[left, right] = matches[0]
    return output


def _parse_multiplication(raw: Any, names: tuple[str, ...]) -> np.ndarray:
    if not isinstance(raw, dict) or set(raw) != set(names):
        raise ValueError("multiplication must contain one row for every finite-group element.")
    name_to_index = {name: index for index, name in enumerate(names)}
    output = np.empty((len(names), len(names)), dtype=np.int64)
    for row_index, row_name in enumerate(names):
        row = raw[row_name]
        if not isinstance(row, list) or len(row) != len(names) or any(value not in name_to_index for value in row):
            raise ValueError(f"multiplication row {row_name!r} must list valid element names in canonical order.")
        output[row_index] = [name_to_index[value] for value in row]
    return output


def _parse_characters(raw: Any, description: str) -> dict[str, complex]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{description} must be a non-empty mapping.")
    return {str(name): _complex_scalar(value, description) for name, value in raw.items()}


def _parse_named_matrices(raw: Any, dimension: int, description: str):
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{description} must be a non-empty mapping.")
    return {
        str(name): _complex_matrix(value, dimension, f"{description}.{name}")
        for name, value in raw.items()
    }


def _require_keys(raw: dict, required: set[str], description: str, *, optional: set[str] | None = None) -> None:
    optional = optional or set()
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required - optional)
    if missing or unknown:
        raise ValueError(f"{description} keys are invalid; missing={missing}, forbidden={unknown}.")


def _positive_int(value: Any, description: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be a positive integer.") from exc
    if integer <= 0 or value != integer:
        raise ValueError(f"{description} must be a positive integer.")
    return integer


def _nonempty_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string.")
    return value.strip()


def _integer_matrix(value: Any, dimension: int, description: str) -> np.ndarray:
    raw = _real_matrix(value, dimension, description)
    matrix = np.rint(raw).astype(np.int64)
    if not np.allclose(raw, matrix, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{description} must contain integers.")
    return matrix


def _vector(value: Any, dimension: int, description: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{description} must be a finite vector with length {dimension}.")
    array = np.asarray(
        [parse_real_scalar(entry, f"{description}[{index}]") for index, entry in enumerate(value)],
        dtype=float,
    )
    return array


def _real_matrix(value: Any, dimension: int, description: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{description} must have shape {(dimension, dimension)}.")
    matrix = np.empty((dimension, dimension), dtype=float)
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != dimension:
            raise ValueError(f"{description} must have shape {(dimension, dimension)}.")
        for column_index, scalar in enumerate(row):
            matrix[row_index, column_index] = parse_real_scalar(
                scalar, f"{description}[{row_index},{column_index}]"
            )
    return matrix


def _complex_matrix(value: Any, dimension: int, description: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{description} must contain {dimension} rows.")
    matrix = np.empty((dimension, dimension), dtype=np.complex128)
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != dimension:
            raise ValueError(f"{description} row {row_index} must contain {dimension} entries.")
        for column_index, scalar in enumerate(row):
            matrix[row_index, column_index] = _complex_scalar(scalar, description)
    return matrix


def _complex_scalar(value: Any, description: str) -> complex:
    if isinstance(value, bool):
        raise ValueError(f"{description} contains an invalid boolean entry.")
    if isinstance(value, (int, float, complex, np.number)):
        result = complex(value)
    elif isinstance(value, str):
        text = value.strip().replace("−", "-")
        direct_text = text.replace("i", "j")
        if direct_text == "j":
            direct_text = "1j"
        elif direct_text == "-j":
            direct_text = "-1j"
        try:
            result = complex(direct_text)
        except ValueError:
            result = _evaluate_scalar_expression(text, description)
    else:
        raise ValueError(f"{description} contains unsupported value {value!r}.")
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f"{description} contains a non-finite value.")
    return result


def parse_real_scalar(value: Any, description: str) -> float:
    """Parse a finite real scalar, including exact YAML expressions such as ``1/3``."""

    result = _complex_scalar(value, description)
    if abs(result.imag) > 1.0e-12:
        raise ValueError(f"{description} must evaluate to a real number, got {result!r}.")
    return float(result.real)


def _evaluate_scalar_expression(expression: str, description: str) -> complex:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_scalar_node(tree.body)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(
            f"{description} contains invalid scalar expression {expression!r}."
        ) from exc
    return complex(result)


def _evaluate_scalar_node(node: ast.AST) -> complex:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float, complex)):
            raise ValueError("Only numeric constants are allowed.")
        return complex(node.value)
    if isinstance(node, ast.Name):
        constants = {
            "pi": complex(math.pi),
            "e": complex(math.e),
            "i": 1.0j,
            "j": 1.0j,
        }
        if node.id not in constants:
            raise ValueError(f"Unknown scalar name {node.id!r}.")
        return constants[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_scalar_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_scalar_node(node.left)
        right = _evaluate_scalar_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ValueError("Only +, -, *, and / are allowed.")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sqrt"
        and len(node.args) == 1
        and not node.keywords
    ):
        return cmath.sqrt(_evaluate_scalar_node(node.args[0]))
    raise ValueError("Unsupported scalar expression.")
