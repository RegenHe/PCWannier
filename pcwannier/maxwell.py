from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

class FieldComponents(str, Enum):
    EZ = "Ez"
    HZ = "Hz"
    FULL_VECTOR = "full_vector"

    @classmethod
    def parse(cls, value: str | FieldComponents) -> FieldComponents:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {item.value.lower(): item for item in cls}
        try:
            return aliases[normalized]
        except KeyError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"field_components must be one of {allowed}; got {value!r}."
            ) from exc


class PrimaryField(str, Enum):
    ELECTRIC = "electric"
    MAGNETIC = "magnetic"


class MaterialKind(str, Enum):
    EPSILON = "epsilon"
    MU = "mu"


class FieldKind(str, Enum):
    SCALAR = "scalar"
    ELECTRIC_Z = "electric_z"
    MAGNETIC_AXIAL_Z = "magnetic_axial_z"
    ELECTRIC_POLAR_VECTOR = "electric_polar_vector"
    MAGNETIC_AXIAL_VECTOR = "magnetic_axial_vector"


@dataclass(frozen=True)
class MaxwellProblem:
    field_components: FieldComponents
    primary_field: PrimaryField
    metric_material: MaterialKind
    curl_material: MaterialKind
    symmetry_field_kind: FieldKind

    @classmethod
    def for_components(cls, value: str | FieldComponents) -> MaxwellProblem:
        components = FieldComponents.parse(value)
        if components == FieldComponents.EZ:
            return cls(
                components,
                PrimaryField.ELECTRIC,
                MaterialKind.EPSILON,
                MaterialKind.MU,
                FieldKind.ELECTRIC_Z,
            )
        if components == FieldComponents.HZ:
            return cls(
                components,
                PrimaryField.MAGNETIC,
                MaterialKind.MU,
                MaterialKind.EPSILON,
                FieldKind.MAGNETIC_AXIAL_Z,
            )
        raise NotImplementedError(
            "field_components=full_vector is not implemented; the current COMSOL "
            "reader supports scalar Ez and Hz fields only."
        )

    def apply_time_reversal(self, values):
        """Apply spinless Maxwell time reversal to the configured primary field."""

        array = np.asarray(values)
        if self.field_components == FieldComponents.EZ:
            return np.conj(array)
        if self.field_components == FieldComponents.HZ:
            return -np.conj(array)
        raise NotImplementedError(
            "Time reversal for field_components=full_vector is not implemented."
        )
