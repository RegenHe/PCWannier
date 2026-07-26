from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..conventions import BlochConvention
from ..maxwell import FieldComponents

if TYPE_CHECKING:
    from ..config import IncarConfig
    from ..data import InputBundle, Mesh, PeriodicGrid


@dataclass(frozen=True)
class SourceAdapter:
    name: str
    bloch_convention: BlochConvention
    supported_field_components: frozenset[FieldComponents]
    input_loader: Callable[[IncarConfig], InputBundle]
    mesh_loader: Callable[[str | Path], Mesh | PeriodicGrid]
    required_config_fields: tuple[str, ...] = (
        "mesh_file",
        "dataset_file",
        "metric_file",
        "E_file",
    )
    supported_dimensions: frozenset[int] = frozenset({2})

    def validate_field_components(self, value: str | FieldComponents) -> None:
        components = FieldComponents.parse(value)
        if components not in self.supported_field_components:
            supported = ", ".join(
                item.value for item in sorted(self.supported_field_components, key=lambda item: item.value)
            )
            description = (
                "scalar Ez and Hz"
                if self.supported_field_components
                == frozenset({FieldComponents.EZ, FieldComponents.HZ})
                else supported
            )
            raise NotImplementedError(
                f"Data source {self.name!r} does not support field_components={components.value}; "
                f"it currently supports {description} (available values: {supported})."
            )

    def validate_dimension(self, dimension: int) -> None:
        if int(dimension) not in self.supported_dimensions:
            supported = ", ".join(str(value) for value in sorted(self.supported_dimensions))
            raise NotImplementedError(
                f"Data source {self.name!r} does not support dimension={dimension}; "
                f"available dimensions: {supported}."
            )
