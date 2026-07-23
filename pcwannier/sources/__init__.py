from __future__ import annotations

from pathlib import Path

from ..conventions import BlochConvention
from .base import SourceAdapter
from .comsol import COMSOL_SOURCE, load_comsol_data, load_comsol_mesh, match_data_to_mesh

_SOURCES = {COMSOL_SOURCE.name: COMSOL_SOURCE}


def resolve_source(name: str) -> SourceAdapter:
    key = str(name).strip().lower()
    try:
        return _SOURCES[key]
    except KeyError as exc:
        available = ", ".join(sorted(_SOURCES))
        raise ValueError(
            f"Unknown data source {name!r}; available sources: {available}."
        ) from exc


def load_input(config):
    source = resolve_source(config.dataset_type)
    source.validate_field_components(config.field_components)
    bundle = source.input_loader(config)
    convention = getattr(bundle, "bloch_convention", None)
    if not isinstance(convention, BlochConvention):
        raise ValueError(
            f"Data source {source.name!r} returned an input bundle without a valid "
            "BlochConvention."
        )
    if convention != source.bloch_convention:
        raise ValueError(
            f"Data source {source.name!r} returned Bloch convention "
            f"{convention}, expected {source.bloch_convention}."
        )
    return bundle


def load_mesh(config):
    source = resolve_source(config.dataset_type)
    path = config.input_path(config.mesh_file)
    if path is None:
        raise ValueError("mesh_file is required to load a source mesh.")
    return source.mesh_loader(Path(path))


__all__ = [
    "SourceAdapter",
    "load_comsol_data",
    "load_comsol_mesh",
    "load_input",
    "load_mesh",
    "match_data_to_mesh",
    "resolve_source",
]
