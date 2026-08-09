from __future__ import annotations

import numpy as np

from .context import CalculationContext
from .kspace import get_kxyz, is_complete_uniform_k_mesh
from .parallel import memory_limited_threads, parallel_map

_WANNIER_POINT_CHUNK_SIZE = 131072


def _uniform_grid_wannier_sum(
    ctx: CalculationContext,
    r_use: list[int],
) -> np.ndarray:
    state = ctx.state
    config = ctx.config
    base_mesh = state.mesh
    extended = state.extended_mesh
    mapping = np.asarray(state.space_to_original_mapping, dtype=np.intp)
    dimension = int(base_mesh.dimension)
    band_count = int(config.band_calc_num)
    vector_field = state.get_block(0, 0, 0).ndim == 3
    components = 3 if vector_field else 1
    k_shape = tuple(int(value) for value in state.k_shape)
    k_count = int(np.prod(k_shape))
    base_count = int(base_mesh.point_count)
    fractional_delta = (
        np.asarray(extended.fractional_vertices, dtype=float)
        - np.asarray(base_mesh.fractional_vertices, dtype=float)[mapping]
    )
    tile_shifts = np.rint(fractional_delta).astype(np.int64)
    if not np.allclose(
        fractional_delta, tile_shifts, rtol=0.0, atol=1.0e-10
    ):
        raise RuntimeError("Periodic-grid extension is not an integer lattice tiling.")
    unique_tiles, tile_inverse = np.unique(
        tile_shifts, axis=0, return_inverse=True
    )
    lookup = np.full((len(unique_tiles), base_count), -1, dtype=np.intp)
    lookup[tile_inverse, mapping] = np.arange(mapping.size, dtype=np.intp)
    if np.any(lookup < 0):
        raise RuntimeError("Periodic-grid extension does not contain every base point per tile.")

    result_shape = (
        (mapping.size, band_count, 3)
        if vector_field
        else (mapping.size, band_count)
    )
    output = np.empty(result_shape, dtype=np.complex128)
    k_fractional = np.asarray(
        [
            [config.k_points[axis][index[axis]] for axis in range(dimension)]
            for index in np.ndindex(k_shape)
        ],
        dtype=float,
    )
    sign = int(state.bloch_sign)
    r_fractional = np.asarray(r_use[:dimension], dtype=float)
    tile_phase = np.exp(
        sign
        * 2j
        * np.pi
        * ((unique_tiles - r_fractional[None, :]) @ k_fractional.T)
    )
    use_fft = (
        tuple(int(value) for value in extended.tile_shape) == k_shape[:dimension]
        and len(unique_tiles) == k_count
    )
    bytes_per_point = max(
        1,
        k_count
        * band_count
        * components
        * np.dtype(np.complex128).itemsize
        * 2,
    )
    point_chunk = max(1, min(base_count, (256 << 20) // bytes_per_point))
    axes = tuple(range(dimension))
    k0 = np.asarray([config.k_points[axis][0] for axis in range(dimension)])

    for start in range(0, base_count, point_chunk):
        stop = min(start + point_chunk, base_count)
        trailing = (
            (stop - start, band_count, 3)
            if vector_field
            else (stop - start, band_count)
        )
        values = np.empty(k_shape + trailing, dtype=np.complex128)
        for index in np.ndindex(k_shape):
            block = state.get_block(*index)
            coefficients = ctx.output_state_coefficients_at(*index)
            if vector_field:
                mixed = np.einsum(
                    "npc,ni->pic",
                    block[:, start:stop, :],
                    coefficients,
                    optimize=True,
                )
                local_phase = state.get_phase(*index)[start:stop, None, None]
            else:
                mixed = block[:, start:stop].T @ coefficients
                local_phase = state.get_phase(*index)[start:stop, None]
            values[index] = mixed * local_phase

        if use_fft:
            transformed = (
                np.fft.ifftn(values, axes=axes) * k_count
                if sign > 0
                else np.fft.fftn(values, axes=axes)
            )
            for tile_index, tile in enumerate(unique_tiles):
                translated_tile = tile - r_fractional
                residue = tuple(
                    int(translated_tile[axis]) % k_shape[axis]
                    for axis in range(dimension)
                )
                storage_residue = residue + (0,) * (len(k_shape) - dimension)
                offset = np.exp(
                    sign * 2j * np.pi * float(np.dot(k0, translated_tile))
                )
                output[lookup[tile_index, start:stop]] = (
                    transformed[storage_residue] * offset
                )
        else:
            flat = values.reshape((k_count,) + trailing)
            tile_values = np.einsum(
                "tk,kp...->tp...", tile_phase, flat, optimize=True
            )
            for tile_index in range(len(unique_tiles)):
                output[lookup[tile_index, start:stop]] = tile_values[tile_index]

    extension_scale = np.sqrt(
        float(np.prod(config.extension[: config.kdim]))
    )
    output /= np.sqrt(float(k_count)) * extension_scale
    return output


def generate_wannier(ctx: CalculationContext, r: list[int] | None = None):
    config = ctx.config
    state = ctx.state
    if r is None:
        r = [0, 0, 0]
    avec = np.asarray(config.real_lattice_vectors)
    dim = avec.shape[1]
    r_use = (list(r) + [0, 0, 0])[: config.kdim]
    r_cart = np.zeros(dim, dtype=float)
    for axis in range(config.kdim):
        r_cart += r_use[axis] * avec[axis, :]
    r_cart *= float(config.lattice_const)

    band_count = int(config.band_calc_num)
    nv = state.extended_mesh.vertices.shape[0]
    vector_field = state.get_block(0, 0, 0).ndim == 3
    wsum_shape = (nv, band_count, 3) if vector_field else (nv, band_count)
    if (
        getattr(getattr(state, "mesh", None), "integration_family", None)
        == "uniform_grid"
        and is_complete_uniform_k_mesh(config.k_points)
    ):
        wsum = _uniform_grid_wannier_sum(ctx, r_use)
    else:
        wsum = np.zeros(wsum_shape, dtype=np.complex128)
    sign = state.bloch_sign
    if state.space_to_original_mapping is None:
        raise RuntimeError("Extended field mapping has not been initialized.")
    mapping = np.asarray(state.space_to_original_mapping, dtype=np.intp)
    vertices = np.asarray(state.extended_mesh.vertices, dtype=float)
    if (
        getattr(getattr(state, "mesh", None), "integration_family", None)
        != "uniform_grid"
        or not is_complete_uniform_k_mesh(config.k_points)
    ):
        extension_scale = np.sqrt(
            float(np.prod(config.extension[: config.kdim]))
        )
        k_data = []
        for i, j, k in state.k_indices():
            k_vec = get_kxyz(config, [i, j, k])[:dim]
            phase_scalar = np.exp(1j * (-(sign) * np.dot(k_vec, r_cart)))
            k_data.append(
                (
                    k_vec,
                    phase_scalar,
                    state.get_block(i, j, k),
                    ctx.output_state_coefficients_at(i, j, k),
                )
            )

        chunks = tuple(
            (start, min(start + _WANNIER_POINT_CHUNK_SIZE, nv))
            for start in range(0, nv, _WANNIER_POINT_CHUNK_SIZE)
        )
        components = 3 if vector_field else 1
        bytes_per_worker = (
            _WANNIER_POINT_CHUNK_SIZE
            * band_count
            * components
            * np.dtype(np.complex128).itemsize
            * 3
        )
        workers = memory_limited_threads(
            getattr(state, "configured_threads", 1),
            bytes_per_worker,
        )

        def calc_chunk(bounds):
            start, stop = bounds
            local_shape = (
                (stop - start, band_count, 3)
                if vector_field
                else (stop - start, band_count)
            )
            local_sum = np.zeros(local_shape, dtype=np.complex128)
            local_mapping = mapping[start:stop]
            local_vertices = vertices[start:stop]
            for k_vec, phase_scalar, base_block, coeff in k_data:
                phase = np.exp(
                    1j * sign * (local_vertices @ k_vec)
                ) * phase_scalar
                if vector_field:
                    block = base_block[:, local_mapping, :] / extension_scale
                    mixed = np.einsum("npc,ni->pic", block, coeff, optimize=True)
                    local_sum += mixed * phase[:, None, None]
                else:
                    block = base_block[:, local_mapping] / extension_scale
                    local_sum += (block.T @ coeff) * phase[:, None]
            return start, stop, local_sum

        for start, stop, local_sum in parallel_map(chunks, calc_chunk, workers):
            wsum[start:stop] = local_sum
        wsum /= np.sqrt(float(state.get_k_num()))
    if state.extended_inner_product is None:
        raise RuntimeError("Extended metric inner product has not been initialized.")
    norm_values = np.swapaxes(wsum, 1, 2) if vector_field else wsum
    norms = state.extended_inner_product.norms(
        norm_values,
        chunk_size=2048,
        name="Wannier norms",
    )
    return tuple(r_use), wsum, norms
