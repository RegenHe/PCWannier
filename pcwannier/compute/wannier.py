from __future__ import annotations

import numpy as np

from .context import CalculationContext
from .integration import integrate_weighted_abs2_columns
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
    transform = state.get_transform(True if config.disable_orth else False)
    sign = -1 if config.dataset_type.lower() == "comsol" else 1
    for i, j, k in state.k_indices():
        phase_vec = state.get_extention_phase(i, j, k)
        k_vec = get_kxyz(config, [i, j, k])[:dim]
        phase_scalar = np.exp(1j * (-(sign) * np.dot(k_vec, r_cart)))
        pvec = phase_vec * phase_scalar
        emat = state.get_extention_block(i, j, k).T
        coeff = transform[i, j, k] @ ctx.initializer.matV[i, j, k] @ ctx.gradient.U[i, j, k]
        wsum += (emat @ coeff) * pvec[:, None]
    wsum /= np.sqrt(float(state.get_k_num()))
    norms = np.atleast_1d(
        integrate_weighted_abs2_columns(
            state.extention_mesh,
            state.extention_epsilon,
            wsum,
            chunk_size=2048,
            backend=state.compute_backend,
        )
    )
    return tuple(r_use), wsum, norms
