from __future__ import annotations

import numpy as np

from .context import CalculationContext
from .kspace import get_kxyz


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
    wsum = np.zeros((nv, band_count), dtype=np.complex128)
    sign = state.bloch_sign
    for i, j, k in state.k_indices():
        phase_vec = state.get_extention_phase(i, j, k)
        k_vec = get_kxyz(config, [i, j, k])[:dim]
        phase_scalar = np.exp(1j * (-(sign) * np.dot(k_vec, r_cart)))
        pvec = phase_vec * phase_scalar
        emat = state.get_extention_block(i, j, k).T
        coeff = ctx.output_state_coefficients_at(i, j, k)
        wsum += (emat @ coeff) * pvec[:, None]
    wsum /= np.sqrt(float(state.get_k_num()))
    if state.extended_inner_product is None:
        raise RuntimeError("Extended metric inner product has not been initialized.")
    norms = state.extended_inner_product.norms(
        wsum,
        chunk_size=2048,
        name="Wannier norms",
    )
    return tuple(r_use), wsum, norms
