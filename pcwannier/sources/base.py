from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..conventions import BlochConvention
from ..maxwell import FieldComponents

if TYPE_CHECKING:
    from ..config import IncarConfig
    from ..data import InputBundle, Mesh


@dataclass(frozen=True)
class SourceAdapter:
    name: str
    bloch_convention: BlochConvention
    supported_field_components: frozenset[FieldComponents]
    input_loader: Callable[[IncarConfig], InputBundle]
    mesh_loader: Callable[[str | Path], Mesh]

    def validate_field_components(self, value: str | FieldComponents) -> None:
        components = FieldComponents.parse(value)
        if components not in self.supported_field_components:
            supported = ", ".join(
                item.value for item in sorted(self.supported_field_components, key=lambda item: item.value)
            )
            raise NotImplementedError(
                f"Data source {self.name!r} does not support field_components={components.value}; "
                f"it currently supports scalar Ez and Hz (available values: {supported})."
            )
