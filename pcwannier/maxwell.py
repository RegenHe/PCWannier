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

    @classmethod
    def parse(cls, value: str | PrimaryField) -> PrimaryField:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"primary_field must be one of {allowed}; got {value!r}."
            ) from exc


class MaterialKind(str, Enum):
    EPSILON = "epsilon"
    MU = "mu"


class FieldKind(str, Enum):
    SCALAR = "scalar"
    PSEUDOSCALAR = "pseudoscalar"
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
    def for_components(
        cls,
        value: str | FieldComponents,
        primary_field: str | PrimaryField | None = None,
    ) -> MaxwellProblem:
        components = FieldComponents.parse(value)
        if components == FieldComponents.EZ:
            if primary_field is not None and PrimaryField.parse(primary_field) is not PrimaryField.ELECTRIC:
                raise ValueError("field_components=Ez requires primary_field=electric.")
            return cls(
                components,
                PrimaryField.ELECTRIC,
                MaterialKind.EPSILON,
                MaterialKind.MU,
                FieldKind.ELECTRIC_Z,
            )
        if components == FieldComponents.HZ:
            if primary_field is not None and PrimaryField.parse(primary_field) is not PrimaryField.MAGNETIC:
                raise ValueError("field_components=Hz requires primary_field=magnetic.")
            return cls(
                components,
                PrimaryField.MAGNETIC,
                MaterialKind.MU,
                MaterialKind.EPSILON,
                FieldKind.MAGNETIC_AXIAL_Z,
            )
        if primary_field is None:
            raise ValueError("field_components=full_vector requires primary_field=electric or magnetic.")
        primary = PrimaryField.parse(primary_field)
        if primary is PrimaryField.ELECTRIC:
            return cls(
                components,
                primary,
                MaterialKind.EPSILON,
                MaterialKind.MU,
                FieldKind.ELECTRIC_POLAR_VECTOR,
            )
        return cls(
            components,
            primary,
            MaterialKind.MU,
            MaterialKind.EPSILON,
            FieldKind.MAGNETIC_AXIAL_VECTOR,
        )

    def apply_time_reversal(self, values):
        """Apply spinless Maxwell time reversal to the configured primary field."""

        array = np.asarray(values)
        if self.primary_field is PrimaryField.ELECTRIC:
            return np.conj(array)
        if self.primary_field is PrimaryField.MAGNETIC:
            return -np.conj(array)
        raise RuntimeError("Unknown Maxwell primary field.")
