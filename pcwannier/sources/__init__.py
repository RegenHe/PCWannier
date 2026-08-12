from __future__ import annotations

from copy import copy
from pathlib import Path

from ..conventions import (
    BlochConvention,
    BlochFieldRepresentation,
    SpatialDiscretization,
)
from ..data import InputBundle
from .base import LoadedSourceChannel, LoadedSourceData, SourceAdapter
from .comsol import COMSOL_SOURCE
from .mpb import MPB_SOURCE

_SOURCES = {
    COMSOL_SOURCE.name: COMSOL_SOURCE,
    MPB_SOURCE.name: MPB_SOURCE,
}


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
    loaded = source.input_loader(config)
    if not isinstance(loaded, LoadedSourceData):
        raise ValueError(
            f"Data source {source.name!r} did not return LoadedSourceData."
        )
    return _assemble_bundle(config, source, loaded)


def _assemble_bundle(config, source: SourceAdapter, loaded: LoadedSourceData) -> InputBundle:
    discretization = getattr(loaded.mesh, "discretization", None)
    if discretization is not source.discretization:
        raise ValueError(
            f"Data source {source.name!r} declared discretization "
            f"{source.discretization.value!r}, but returned {discretization!r}."
        )
    auxiliary_loaders = {}
    for name, channel in (loaded.auxiliary_channels or {}).items():
        def load_channel(channel=channel):
            channel_config = copy(config)
            if channel.S_file is not None:
                channel_config.S_file = channel.S_file
            if channel.D_file is not None:
                channel_config.D_file = channel.D_file
            return _assemble_bundle(channel_config, source, channel.loader())

        auxiliary_loaders[str(name)] = load_channel
    return InputBundle(
        config=config,
        maxwell=config.maxwell_problem,
        bloch_convention=source.bloch_convention,
        mesh=loaded.mesh,
        fields=loaded.fields,
        metric_material=loaded.metric_material,
        energies=loaded.energies,
        band_indices=loaded.band_indices,
        inner_band_indices=loaded.inner_band_indices,
        energy_matrix=loaded.energy_matrix,
        field_representation=source.field_representation,
        symmetry=config.symmetry_context,
        zero_modes=loaded.zero_modes,
        band_channels=dict(loaded.band_channels or {}),
        auxiliary_bundle_loaders=auxiliary_loaders,
        base_hamiltonians=loaded.base_hamiltonians,
    )


def load_mesh(config):
    source = resolve_source(config.dataset_type)
    path = config.input_path(config.mesh_file)
    if path is None:
        raise ValueError("mesh_file is required to load a source mesh.")
    mesh = source.mesh_loader(Path(path))
    discretization = getattr(mesh, "discretization", None)
    if discretization is not source.discretization:
        raise ValueError(
            f"Data source {source.name!r} declared discretization "
            f"{source.discretization.value!r}, but its mesh loader returned "
            f"{discretization!r}."
        )
    return mesh


__all__ = [
    "BlochConvention",
    "BlochFieldRepresentation",
    "LoadedSourceChannel",
    "LoadedSourceData",
    "SourceAdapter",
    "SpatialDiscretization",
    "load_input",
    "load_mesh",
    "resolve_source",
]
