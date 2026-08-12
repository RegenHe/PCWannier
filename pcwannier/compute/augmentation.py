from __future__ import annotations

from dataclasses import replace
import logging

import numpy as np

from ..data import BandChannelReference, InputBundle
from ..maxwell import FieldComponents, PrimaryField
from ..symmetry.representation import build_symmetry_context


LOGGER = logging.getLogger(__name__)


def combine_transverse_longitudinal(
    physical: InputBundle,
    longitudinal: InputBundle,
) -> InputBundle:
    """Build a source-neutral augmented T+L field space."""

    config = physical.config
    if (
        physical.mesh.dimension != 3
        or physical.maxwell.field_components is not FieldComponents.FULL_VECTOR
        or physical.maxwell.primary_field is not PrimaryField.MAGNETIC
    ):
        raise NotImplementedError(
            "wannier_subspace=T + L requires a 3D full-vector magnetic calculation."
        )
    _validate_compatible_channels(physical, longitudinal)
    k_shape = physical.fields.shape
    h_count = int(physical.energy_matrix.shape[-1])
    l_count = int(longitudinal.energy_matrix.shape[-1])
    combined_fields = np.empty(k_shape, dtype=object)
    combined_energies = np.empty(k_shape, dtype=object)
    combined_bands = np.empty(k_shape, dtype=object)
    combined_inner = np.empty(k_shape, dtype=object)
    combined_zero = np.empty(k_shape, dtype=object)
    longitudinal_zero_bands = np.empty(k_shape, dtype=object)

    for index in np.ndindex(k_shape):
        h_fields = np.asarray(physical.fields[index], dtype=np.complex128)
        l_fields = np.asarray(longitudinal.fields[index], dtype=np.complex128)
        h_energy = np.asarray(physical.energies[index], dtype=float)
        l_energy = np.asarray(longitudinal.energies[index], dtype=float)
        h_zero = h_energy <= config.gamma_zero_mode_tolerance
        l_zero = l_energy <= config.gamma_zero_mode_tolerance
        if _at_gamma(config, index) and (np.any(h_zero) or np.any(l_zero)):
            h_fields, l_fields = _regularize_gamma_constant_fields(
                physical,
                h_fields,
                l_fields,
                h_zero,
                l_zero,
            )

        selected_h = np.asarray(physical.band_indices[index], dtype=int)
        selected_l = np.asarray(longitudinal.band_indices[index], dtype=int)
        combined_fields[index] = np.ascontiguousarray(
            np.concatenate((h_fields, l_fields), axis=0)
        )
        combined_energies[index] = np.concatenate((h_energy, l_energy))
        combined_bands[index] = np.concatenate((selected_h, h_count + selected_l)).tolist()
        combined_inner[index] = (
            [int(value) for value in physical.inner_band_indices[index]]
            + [h_count + int(value) for value in longitudinal.inner_band_indices[index]]
        )
        combined_zero[index] = np.concatenate((h_zero, l_zero))
        longitudinal_zero_bands[index] = selected_l[l_zero].tolist()

    channels = {
        index: BandChannelReference("H", index) for index in range(h_count)
    }
    channels.update(
        {
            h_count + index: BandChannelReference("L", index)
            for index in range(l_count)
        }
    )
    context = resolve_channel_analysis_context(physical, l_offset=h_count)
    LOGGER.info(
        "T+L outer space prepared: H bands retain ids [0,%s), L offset=%s",
        h_count,
        h_count,
    )
    return replace(
        physical,
        fields=combined_fields,
        energies=combined_energies,
        band_indices=combined_bands,
        inner_band_indices=combined_inner,
        energy_matrix=np.concatenate(
            (physical.energy_matrix, longitudinal.energy_matrix), axis=-1
        ),
        symmetry=context,
        zero_modes=combined_zero,
        band_channels=channels,
        auxiliary_zero_mode_bands={"longitudinal": longitudinal_zero_bands},
        auxiliary_bundle_loaders={},
    )


def resolve_channel_analysis_context(bundle: InputBundle, *, l_offset: int):
    context = bundle.symmetry
    if context is None or context.model.representation_analysis is None:
        return context
    raw_points = bundle.config.representation_analysis or ()
    specification = context.model.representation_analysis
    if len(raw_points) != len(specification.points):
        raise RuntimeError("Representation-analysis config and symmetry model are inconsistent.")
    changed = False
    resolved_points = []
    for raw, point in zip(raw_points, specification.points):
        selectors = raw.get("channel_bands")
        if not selectors:
            resolved_points.append(point)
            continue
        bands = [int(value) for value in selectors.get("H", ())]
        bands.extend(l_offset + int(value) for value in selectors.get("L", ()))
        if not bands or len(set(bands)) != len(bands):
            raise ValueError(
                f"Representation point {point.name!r} has an empty or duplicate H/L selector."
            )
        resolved_points.append(replace(point, band_indices=tuple(bands)))
        changed = True
    if not changed:
        return context
    analysis = replace(specification, points=tuple(resolved_points))
    model = replace(context.model, representation_analysis=analysis)
    return build_symmetry_context(model, context.k_points)


def zero_mode_band_indices(bundle: InputBundle) -> np.ndarray:
    """Return actual band ids selected as zero modes in one channel."""

    result = np.empty(bundle.band_indices.shape, dtype=object)
    for index in np.ndindex(bundle.band_indices.shape):
        mask = (
            np.zeros(len(bundle.band_indices[index]), dtype=bool)
            if bundle.zero_modes is None
            else np.asarray(bundle.zero_modes[index], dtype=bool)
        )
        selected = np.asarray(bundle.band_indices[index], dtype=int)
        if mask.shape != selected.shape:
            raise ValueError(f"Zero-mode mask has an invalid shape at k={index}.")
        result[index] = selected[mask].tolist()
    return result


def exclude_zero_modes(bundle: InputBundle) -> InputBundle:
    """Remove non-normalizable zero modes from an independently analyzed channel."""

    if bundle.zero_modes is None:
        return bundle
    fields = np.empty(bundle.fields.shape, dtype=object)
    energies = np.empty(bundle.energies.shape, dtype=object)
    bands = np.empty(bundle.band_indices.shape, dtype=object)
    inner = np.empty(bundle.inner_band_indices.shape, dtype=object)
    zero_modes = np.empty(bundle.zero_modes.shape, dtype=object)
    for index in np.ndindex(bundle.fields.shape):
        retained_mask = ~np.asarray(bundle.zero_modes[index], dtype=bool)
        selected = np.asarray(bundle.band_indices[index], dtype=int)
        fields[index] = np.asarray(bundle.fields[index])[retained_mask]
        energies[index] = np.asarray(bundle.energies[index])[retained_mask]
        bands[index] = selected[retained_mask].tolist()
        retained = set(bands[index])
        inner[index] = [
            int(value) for value in bundle.inner_band_indices[index]
            if int(value) in retained
        ]
        zero_modes[index] = np.zeros(np.count_nonzero(retained_mask), dtype=bool)
    return replace(
        bundle,
        fields=fields,
        energies=energies,
        band_indices=bands,
        inner_band_indices=inner,
        zero_modes=zero_modes,
    )


def _validate_compatible_channels(physical: InputBundle, auxiliary: InputBundle) -> None:
    if physical.mesh is not auxiliary.mesh:
        if (
            type(physical.mesh) is not type(auxiliary.mesh)
            or tuple(physical.mesh.shape) != tuple(auxiliary.mesh.shape)
            or not np.allclose(physical.mesh.vertices, auxiliary.mesh.vertices)
        ):
            raise ValueError("Physical and longitudinal channels use incompatible spatial grids.")
    if physical.fields.shape != auxiliary.fields.shape:
        raise ValueError("Physical and longitudinal channels use different k grids.")
    if physical.bloch_convention != auxiliary.bloch_convention:
        raise ValueError("Physical and longitudinal channels use different Bloch conventions.")
    if physical.field_representation is not auxiliary.field_representation:
        raise ValueError("Physical and longitudinal channels store different field representations.")
    if not np.allclose(physical.metric_material, auxiliary.metric_material):
        raise ValueError("Physical and longitudinal channels use different metric material data.")


def _at_gamma(config, index: tuple[int, ...]) -> bool:
    k_fractional = np.asarray(
        [config.k_points[axis][index[axis]] for axis in range(3)], dtype=float
    )
    return bool(
        np.allclose(
            k_fractional - np.rint(k_fractional),
            0.0,
            rtol=0.0,
            atol=max(config.symmetry_tolerance, 1.0e-12),
        )
    )


def _regularize_gamma_constant_fields(
    bundle: InputBundle,
    h_fields: np.ndarray,
    l_fields: np.ndarray,
    h_zero: np.ndarray,
    l_zero: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h_positions = np.flatnonzero(h_zero)
    l_positions = np.flatnonzero(l_zero)
    if h_positions.size != 2 or l_positions.size != 1:
        raise ValueError(
            "The Gamma T+L zero-frequency regularization requires exactly two selected "
            f"physical T modes and one selected L mode; got T={h_positions.size}, L={l_positions.size}."
        )
    metric_integral = bundle.mesh.cell_volume * float(
        np.mean(bundle.metric_material, dtype=np.float64)
    )
    if not np.isfinite(metric_integral) or metric_integral <= 0.0:
        raise ValueError("The magnetic metric gives a non-positive constant-field norm.")
    constants = np.zeros((3, bundle.mesh.point_count, 3), dtype=np.complex128)
    constants[:] = np.eye(3, dtype=np.complex128)[:, None, :] / np.sqrt(metric_integral)
    h_result = h_fields.copy()
    l_result = l_fields.copy()
    h_result[h_positions[0]] = constants[0]
    h_result[h_positions[1]] = constants[1]
    l_result[l_positions[0]] = constants[2]
    return h_result, l_result
