from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .models import EBRCatalog, EBRDefinition, EBRKPoint


_BUILTIN_ALIASES = {
    "221": "sg221.yaml",
    "sg221": "sg221.yaml",
    "224": "sg224.yaml",
    "sg224": "sg224.yaml",
    "p4mm": "p4mm.yaml",
    "wp11": "p4mm.yaml",
}


def load_ebr_catalog(
    path_or_alias: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> EBRCatalog:
    """Load a strict EBR-generation catalog from a path or built-in alias."""

    raw_value = str(path_or_alias).strip()
    if not raw_value:
        raise ValueError("EBR catalog path or alias must not be empty.")
    path = Path(raw_value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        candidate = Path(base_dir) / path
        if candidate.is_file():
            path = candidate
    if path.is_file():
        source = str(path.resolve())
        raw = _read_yaml(path)
    else:
        key = Path(raw_value).stem.casefold()
        filename = _BUILTIN_ALIASES.get(key)
        if filename is None:
            available = ", ".join(sorted(_BUILTIN_ALIASES))
            raise FileNotFoundError(
                f"EBR catalog {path_or_alias!r} was not found; built-in aliases: {available}."
            )
        resource = resources.files("pcwannier.ebr").joinpath("catalogs", filename)
        source = f"builtin:{filename}"
        try:
            raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid built-in EBR catalog {filename}: {exc}") from exc
    return _parse_catalog(raw, source)


def infer_builtin_catalog_alias(space_group: int | str) -> str:
    key = str(space_group).strip().casefold()
    if key not in _BUILTIN_ALIASES:
        raise ValueError(
            f"No built-in EBR catalog is available for space group {space_group!r}; "
            "set ebr_catalog to a custom YAML file."
        )
    return Path(_BUILTIN_ALIASES[key]).stem


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid EBR catalog YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"EBR catalog {path} must contain a YAML mapping.")
    return raw


def _parse_catalog(raw: Any, source: str) -> EBRCatalog:
    if not isinstance(raw, dict):
        raise ValueError("EBR catalog must contain a YAML mapping.")
    common = {"name", "k_points", "ebrs"}
    identity_keys = {"space_group_number", "hall_number", "space_group_name"}
    missing = sorted(common - set(raw))
    unknown = sorted(set(raw) - common - identity_keys)
    if missing or unknown:
        raise ValueError(
            f"EBR catalog keys are invalid; missing={missing}, forbidden={unknown}."
        )
    has_hall_identity = "space_group_number" in raw or "hall_number" in raw
    has_named_identity = "space_group_name" in raw
    if has_hall_identity and (
        "space_group_number" not in raw or "hall_number" not in raw
    ):
        raise ValueError(
            "EBR catalog space_group_number and hall_number must be provided together."
        )
    if has_hall_identity == has_named_identity:
        raise ValueError(
            "EBR catalog must define exactly one identity: "
            "space_group_number+hall_number or space_group_name."
        )
    points_raw = raw["k_points"]
    ebrs_raw = raw["ebrs"]
    if not isinstance(points_raw, list) or not points_raw:
        raise ValueError("EBR catalog k_points must be a non-empty list.")
    if not isinstance(ebrs_raw, list) or not ebrs_raw:
        raise ValueError("EBR catalog ebrs must be a non-empty list.")
    points = []
    for index, item in enumerate(points_raw):
        if not isinstance(item, dict):
            raise ValueError(f"k_points[{index}] must be a mapping.")
        _require_keys(item, {"name", "k"}, f"k_points[{index}]")
        points.append(EBRKPoint(_text(item["name"], "k-point name"), _vector(item["k"])))
    ebrs = []
    for index, item in enumerate(ebrs_raw):
        if not isinstance(item, dict):
            raise ValueError(f"ebrs[{index}] must be a mapping.")
        _require_keys(
            item,
            {"name", "wyckoff", "center", "site_irrep"},
            f"ebrs[{index}]",
        )
        ebrs.append(
            EBRDefinition(
                _text(item["name"], "EBR name"),
                _text(item["wyckoff"], "Wyckoff label"),
                _vector(item["center"]),
                _text(item["site_irrep"], "site irrep"),
            )
        )
    return EBRCatalog(
        _text(raw["name"], "catalog name"),
        (
            _integer(raw["space_group_number"], "space_group_number")
            if has_hall_identity
            else None
        ),
        _integer(raw["hall_number"], "hall_number") if has_hall_identity else None,
        tuple(points),
        tuple(ebrs),
        source,
        _text(raw["space_group_name"], "space_group_name")
        if has_named_identity
        else None,
    )


def _require_keys(raw: dict, required: set[str], description: str) -> None:
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required)
    if missing or unknown:
        raise ValueError(
            f"{description} keys are invalid; missing={missing}, forbidden={unknown}."
        )


def _text(value: Any, description: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{description} must not be empty.")
    return text


def _integer(value: Any, description: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be an integer.") from exc
    if result != value:
        raise ValueError(f"{description} must be an integer.")
    return result


def _vector(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError("EBR catalog coordinates must be finite non-empty vectors.")
    return result
