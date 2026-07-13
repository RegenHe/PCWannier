from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .group import SpaceGroup, SpaceGroupOperation
from .representation import (
    SiteIrrepGenerator,
    SiteIrrepSpec,
    SymmetryModel,
    build_wannier_target,
)
from .specs import (
    DegeneracyTolerance,
    FieldKind,
    IrrepCharacterSpec,
    RepresentationAnalysisSpec,
    RepresentationPointSpec,
    SymmetryGaugeSpec,
)


def load_symmetry(path: str | Path) -> SymmetryModel:
    filename = Path(path)
    try:
        with filename.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid symmetry YAML in {filename}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Symmetry file {filename} must contain a YAML mapping at its root.")

    dimension = _positive_int(raw.get("dimension"), "dimension")
    tolerance = float(raw.get("tolerance", 1e-8))
    operations_raw = raw.get("symmetry_operations")
    if not isinstance(operations_raw, list) or not operations_raw:
        raise ValueError("symmetry_operations must be a non-empty YAML list.")
    operations = []
    for index, operation_raw in enumerate(operations_raw):
        operation = _parse_operation(operation_raw, dimension, default_name=f"g{index}")
        operations.append(operation)
    group = SpaceGroup(operations, tolerance)

    targets_raw = raw.get("wannier_targets", [])
    if targets_raw is None:
        targets_raw = []
    if not isinstance(targets_raw, list):
        raise ValueError("wannier_targets must be a YAML list.")
    targets = []
    target_names = set()
    for target_index, target_raw in enumerate(targets_raw):
        if not isinstance(target_raw, dict):
            raise ValueError(f"wannier_targets[{target_index}] must be a mapping.")
        name = str(target_raw.get("name", f"target{target_index}"))
        if name in target_names:
            raise ValueError(f"Duplicate Wannier target name: {name!r}.")
        target_names.add(name)
        center = _vector(target_raw.get("center"), dimension, f"wannier_targets[{target_index}].center")
        irrep_spec = _parse_site_irrep(target_raw.get("site_irrep"), group, dimension, target_index)
        targets.append(build_wannier_target(name, group, center, irrep_spec))
    analysis = _parse_representation_analysis(
        raw.get("representation_analysis"), group, dimension, target_names
    )
    gauge = _parse_symmetry_gauge(raw.get("symmetry_gauge"))
    if gauge is not None and gauge.enabled and not targets:
        raise ValueError("symmetry_gauge requires at least one Wannier target.")
    return SymmetryModel(dimension, tolerance, group, tuple(targets), analysis, gauge)


def _parse_symmetry_gauge(raw: Any) -> SymmetryGaugeSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("symmetry_gauge must be a YAML mapping.")
    enabled = raw.get("enabled", True)
    validate_wannier = raw.get("validate_wannier", True)
    if not isinstance(enabled, bool) or not isinstance(validate_wannier, bool):
        raise ValueError("symmetry_gauge enabled and validate_wannier must be booleans.")
    return SymmetryGaugeSpec(
        enabled=enabled,
        tolerance=float(raw.get("tolerance", 1.0e-8)),
        max_iterations=_positive_int(raw.get("max_iterations", 20), "symmetry_gauge.max_iterations"),
        svd_relative_tolerance=float(raw.get("svd_relative_tolerance", 1.0e-10)),
        validate_wannier=validate_wannier,
        real_space_tolerance=float(raw.get("real_space_tolerance", 1.0e-6)),
        minimum_retained_norm=float(raw.get("minimum_retained_norm", 0.99)),
    )


def _parse_representation_analysis(
    raw: Any,
    group: SpaceGroup,
    dimension: int,
    target_names: set[str],
) -> RepresentationAnalysisSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("representation_analysis must be a YAML mapping.")
    try:
        field_kind = FieldKind(str(raw.get("field_kind", "scalar")))
    except ValueError as exc:
        choices = ", ".join(kind.value for kind in FieldKind)
        raise ValueError(f"representation_analysis.field_kind must be one of: {choices}.") from exc
    default_tolerance = _parse_degeneracy_tolerance(
        raw.get("degeneracy_tolerance"), DegeneracyTolerance(), "representation_analysis"
    )
    points_raw = raw.get("points")
    if not isinstance(points_raw, list) or not points_raw:
        raise ValueError("representation_analysis.points must be a non-empty YAML list.")
    points = []
    point_names = set()
    for point_index, point_raw in enumerate(points_raw):
        description = f"representation_analysis.points[{point_index}]"
        if not isinstance(point_raw, dict):
            raise ValueError(f"{description} must be a mapping.")
        name = str(point_raw.get("name", f"k{point_index}"))
        if name in point_names:
            raise ValueError(f"Duplicate representation-analysis point name: {name!r}.")
        point_names.add(name)
        kpoint = _vector(point_raw.get("k"), dimension, f"{description}.k")
        bands_raw = point_raw.get("bands")
        bands = None if bands_raw is None else _nonnegative_integer_tuple(bands_raw, f"{description}.bands")
        targets_raw = point_raw.get("targets")
        if targets_raw is None:
            targets = None
        else:
            if not isinstance(targets_raw, list) or any(not isinstance(value, str) for value in targets_raw):
                raise ValueError(f"{description}.targets must be a list of target names.")
            targets = tuple(targets_raw)
            unknown = sorted(set(targets) - target_names)
            if unknown:
                raise ValueError(f"{description} references unknown Wannier targets {unknown}.")
        tolerance = _parse_degeneracy_tolerance(
            point_raw.get("degeneracy_tolerance"), default_tolerance, description
        )
        classes = _parse_conjugacy_classes(point_raw.get("conjugacy_classes"), group, description)
        irreps = _parse_character_table(point_raw.get("irreps"), classes, description)
        _validate_analysis_character_table(group, kpoint, classes, irreps, description)
        points.append(
            RepresentationPointSpec(name, kpoint, bands, targets, tolerance, classes, irreps)
        )
    return RepresentationAnalysisSpec(field_kind, default_tolerance, tuple(points))


def _parse_degeneracy_tolerance(
    raw: Any,
    default: DegeneracyTolerance,
    description: str,
) -> DegeneracyTolerance:
    if raw is None:
        return default
    if not isinstance(raw, dict):
        raise ValueError(f"{description}.degeneracy_tolerance must be a mapping.")
    absolute = float(raw.get("absolute", default.absolute))
    relative = float(raw.get("relative", default.relative))
    return DegeneracyTolerance(absolute, relative)


def _parse_conjugacy_classes(
    raw: Any,
    group: SpaceGroup,
    description: str,
) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{description}.conjugacy_classes must be a non-empty mapping.")
    classes: dict[str, tuple[str, ...]] = {}
    for class_name, members_raw in raw.items():
        if not isinstance(members_raw, list) or not members_raw or any(
            not isinstance(value, str) for value in members_raw
        ):
            raise ValueError(f"Conjugacy class {class_name!r} must contain operation names.")
        members = tuple(members_raw)
        for name in members:
            group.operation_by_name(name)
        classes[str(class_name)] = members
    return classes


def _parse_character_table(
    raw: Any,
    classes: dict[str, tuple[str, ...]],
    description: str,
) -> tuple[IrrepCharacterSpec, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{description}.irreps must be a non-empty mapping.")
    irreps = []
    for irrep_name, irrep_raw in raw.items():
        if not isinstance(irrep_raw, dict):
            raise ValueError(f"Irrep {irrep_name!r} must be a mapping.")
        element_raw = irrep_raw.get("characters")
        class_raw = irrep_raw.get("class_characters")
        if (element_raw is None) == (class_raw is None):
            raise ValueError(
                f"Irrep {irrep_name!r} must define exactly one of characters or class_characters."
            )
        if class_raw is not None and not classes:
            raise ValueError(f"Irrep {irrep_name!r} uses class_characters without conjugacy_classes.")
        element = _parse_character_mapping(element_raw, f"irrep {irrep_name!r} characters")
        class_characters = _parse_character_mapping(
            class_raw, f"irrep {irrep_name!r} class_characters"
        )
        irreps.append(IrrepCharacterSpec(str(irrep_name), element, class_characters))
    return tuple(irreps)


def _parse_character_mapping(raw: Any, description: str) -> dict[str, complex]:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{description} must be a non-empty mapping.")
    return {str(name): _complex_scalar(value, description) for name, value in raw.items()}


def _validate_analysis_character_table(
    group: SpaceGroup,
    kpoint: np.ndarray,
    classes: dict[str, tuple[str, ...]],
    irreps: tuple[IrrepCharacterSpec, ...],
    description: str,
) -> None:
    if not irreps:
        return
    little_indices = []
    for index, operation in enumerate(group.operations):
        displacement = operation.act_reciprocal(kpoint) - kpoint
        if np.allclose(displacement, np.rint(displacement), rtol=0.0, atol=group.tolerance):
            little_indices.append(index)
    little_names = {
        group.operations[index].name
        for index in little_indices
        if group.operations[index].name is not None
    }
    if len(little_names) != len(little_indices):
        raise ValueError(f"{description} requires names for all little-group operations.")

    class_mode = all(bool(irrep.class_characters) for irrep in irreps)
    element_mode = all(bool(irrep.characters) for irrep in irreps)
    if not (class_mode or element_mode):
        raise ValueError(f"{description}.irreps must use one consistent character-table format.")
    if element_mode:
        for irrep in irreps:
            if set(irrep.characters) != little_names:
                raise ValueError(
                    f"{description} irrep {irrep.name!r} must define every little-group operation."
                )
        return

    listed = [name for members in classes.values() for name in members]
    if len(listed) != len(set(listed)) or set(listed) != little_names:
        raise ValueError(
            f"{description}.conjugacy_classes must partition every little-group operation exactly once."
        )
    for irrep in irreps:
        if set(irrep.class_characters) != set(classes):
            raise ValueError(
                f"{description} irrep {irrep.name!r} must define every configured conjugacy class."
            )

    little_set = set(little_indices)
    for class_name, members in classes.items():
        member_indices = {group.operation_index(group.operation_by_name(name)) for name in members}
        for member_index in member_indices:
            member = group.operations[member_index]
            for conjugator_index in little_indices:
                conjugator = group.operations[conjugator_index]
                conjugate = group.operation_index(conjugator * member * conjugator.inverse())
                if conjugate not in little_set or conjugate not in member_indices:
                    raise ValueError(
                        f"{description} class {class_name!r} is not closed under little-group conjugation."
                    )


def _nonnegative_integer_tuple(raw: Any, description: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{description} must be a non-empty integer list.")
    output = []
    for value in raw:
        if isinstance(value, bool):
            raise ValueError(f"{description} must contain non-negative integers.")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{description} must contain non-negative integers.") from exc
        if integer < 0 or value != integer:
            raise ValueError(f"{description} must contain non-negative integers.")
        output.append(integer)
    if len(set(output)) != len(output):
        raise ValueError(f"{description} must not contain duplicates.")
    return tuple(output)


def _parse_site_irrep(raw: Any, group: SpaceGroup, dimension: int, target_index: int) -> SiteIrrepSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"wannier_targets[{target_index}].site_irrep must be a mapping.")
    name = str(raw.get("name", "unnamed"))
    irrep_dimension = _positive_int(raw.get("dimension"), f"wannier_targets[{target_index}].site_irrep.dimension")
    matrices = raw.get("matrices", {})
    if not isinstance(matrices, dict):
        raise ValueError(f"wannier_targets[{target_index}].site_irrep.matrices must be a mapping.")
    identity = matrices.get("identity", np.eye(irrep_dimension))
    identity_matrix = _complex_matrix(identity, irrep_dimension, "site-irrep identity")
    generators_raw = matrices.get("generators", [])
    if generators_raw is None:
        generators_raw = []
    if not isinstance(generators_raw, list):
        raise ValueError("site-irrep generators must be a list.")
    generators = []
    for generator_index, generator_raw in enumerate(generators_raw):
        if not isinstance(generator_raw, dict) or "operation" not in generator_raw or "matrix" not in generator_raw:
            raise ValueError(f"site-irrep generator {generator_index} must contain operation and matrix.")
        operation_raw = generator_raw["operation"]
        if isinstance(operation_raw, str):
            operation = group.operation_by_name(operation_raw)
            match_modulo_lattice = True
        else:
            operation = _parse_operation(operation_raw, dimension)
            match_modulo_lattice = False
        matrix = _complex_matrix(generator_raw["matrix"], irrep_dimension, f"site-irrep generator {generator_index}")
        generators.append(SiteIrrepGenerator(operation, matrix, match_modulo_lattice))
    return SiteIrrepSpec(name, irrep_dimension, identity_matrix, tuple(generators))


def _parse_operation(raw: Any, dimension: int, default_name: str | None = None) -> SpaceGroupOperation:
    if not isinstance(raw, dict):
        raise ValueError("A symmetry operation must be a YAML mapping.")
    if "rotation" not in raw or "translation" not in raw:
        raise ValueError("A symmetry operation must contain rotation and translation.")
    rotation = np.asarray(raw["rotation"])
    if rotation.shape != (dimension, dimension):
        raise ValueError(f"Symmetry rotation has shape {rotation.shape}; expected {(dimension, dimension)}.")
    translation = _vector(raw["translation"], dimension, "symmetry translation")
    name = raw.get("name", default_name)
    return SpaceGroupOperation(rotation, translation, None if name is None else str(name))


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


def _vector(value: Any, dimension: int, description: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{description} must be a finite vector with length {dimension}.")
    return array


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
        raise ValueError(f"{description} contains an invalid boolean matrix entry.")
    if isinstance(value, (int, float, complex, np.number)):
        result = complex(value)
    elif isinstance(value, str):
        text = value.strip().replace("−", "-").replace("i", "j")
        if text == "j":
            text = "1j"
        elif text == "-j":
            text = "-1j"
        try:
            result = complex(text)
        except ValueError as exc:
            raise ValueError(f"{description} contains invalid complex value {value!r}.") from exc
    else:
        raise ValueError(f"{description} contains unsupported matrix entry {value!r}.")
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f"{description} contains a non-finite matrix entry.")
    return result
