from __future__ import annotations

import numpy as np

from .context import CalculationContext
from .kspace import get_kxyz
from .parallel import memory_limited_threads, parallel_map

_WANNIER_POINT_CHUNK_SIZE = 131072


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
    wsum = np.zeros(wsum_shape, dtype=np.complex128)
    sign = state.bloch_sign
    if state.space_to_original_mapping is None:
        raise RuntimeError("Extended field mapping has not been initialized.")
    mapping = np.asarray(state.space_to_original_mapping, dtype=np.intp)
    vertices = np.asarray(state.extended_mesh.vertices, dtype=float)
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
