from __future__ import annotations

import numpy as np

from .context import CalculationContext
from .kspace import get_kxyz

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
    nv = state.extention_mesh.vertices.shape[0]
    vector_field = state.get_block(0, 0, 0).ndim == 3
    wsum_shape = (nv, band_count, 3) if vector_field else (nv, band_count)
    wsum = np.zeros(wsum_shape, dtype=np.complex128)
    sign = state.bloch_sign
    if state.space_to_original_mapping is None:
        raise RuntimeError("Extended field mapping has not been initialized.")
    mapping = np.asarray(state.space_to_original_mapping, dtype=np.intp)
    vertices = np.asarray(state.extention_mesh.vertices, dtype=float)
    extension_scale = np.sqrt(
        float(np.prod(config.extension[: config.kdim]))
    )
    for i, j, k in state.k_indices():
        k_vec = get_kxyz(config, [i, j, k])[:dim]
        phase_scalar = np.exp(1j * (-(sign) * np.dot(k_vec, r_cart)))
        base_block = state.get_block(i, j, k)
        coeff = ctx.output_state_coefficients_at(i, j, k)
        for start in range(0, nv, _WANNIER_POINT_CHUNK_SIZE):
            stop = min(start + _WANNIER_POINT_CHUNK_SIZE, nv)
            local_mapping = mapping[start:stop]
            phase = np.exp(
                1j * sign * (vertices[start:stop] @ k_vec)
            ) * phase_scalar
            if vector_field:
                block = base_block[:, local_mapping, :] / extension_scale
                mixed = np.einsum("npc,ni->pic", block, coeff, optimize=True)
                wsum[start:stop] += mixed * phase[:, None, None]
            else:
                block = base_block[:, local_mapping] / extension_scale
                wsum[start:stop] += (block.T @ coeff) * phase[:, None]
    wsum /= np.sqrt(float(state.get_k_num()))
    if state.extended_inner_product is None:
        raise RuntimeError("Extended metric inner product has not been initialized.")
    norms = state.extended_inner_product.norms(
        wsum,
        chunk_size=2048,
        name="Wannier norms",
    )
    return tuple(r_use), wsum, norms
