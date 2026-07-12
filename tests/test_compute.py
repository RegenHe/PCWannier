import numpy as np
from types import SimpleNamespace

from pcwannier import load_config
from pcwannier.compute import run_calculation
from pcwannier.compute.tba import TBAModel
from pcwannier.sources.comsol import load_input


def test_smoke_calculation_with_data_incar(tmp_path):
    cfg = load_config("data/incar")
    cfg.max_iter = 0
    cfg.k_num = [6, 6]
    cfg.hybrid_Wilson_loop = False
    cfg.Chern_number = False
    cfg.wannier_figures = "false"
    cfg.band_figure = "false"
    cfg.topo_output = "false"
    cfg.M_file = "false"
    cfg.V_file = "false"
    cfg.A_file = "false"
    cfg.U_file = "false"
    cfg.hopping_file = "false"
    cfg.wannier_file = "false"

    result = run_calculation(load_input(cfg), threads=1)

    assert result.M0.shape == (16, 16, 1)
    assert result.V.shape == (16, 16, 1)
    assert result.U.shape == (16, 16, 1)
    assert result.wanniers[(0, 0)].shape[1] == cfg.band_calc_num
    assert result.wannier_norms.shape == (cfg.band_calc_num,)
    assert np.all(np.isfinite(np.real(result.wannier_norms)))
    assert (0, 0, 0) in result.hoppings
    assert result.hoppings[(0, 0, 0)].shape == (cfg.band_calc_num, cfg.band_calc_num)
    assert result.band is not None
    assert result.band.energies.shape[1] == cfg.band_calc_num


def test_thread_counts_are_numerically_consistent():
    results = [_run_small_calculation(threads) for threads in (1, 2, 4)]
    ref = results[0]
    for result in results[1:]:
        for idx in np.ndindex(ref.M0.shape):
            for b in range(len(ref.M0[idx])):
                assert np.allclose(result.M0[idx][b], ref.M0[idx][b], rtol=1e-10, atol=1e-10)
        for key, hopping in ref.hoppings.items():
            assert np.allclose(result.hoppings[key], hopping, rtol=1e-10, atol=1e-10)
        assert np.allclose(result.band.energies, ref.band.energies, rtol=1e-10, atol=1e-10)


def test_even_kmesh_half_r_set_has_no_inverse_duplicates():
    shape = (8, 8, 1)
    neighbors = TBAModel.R_half_rect(shape)
    residues = {tuple(int(value) % shape[axis] for axis, value in enumerate(row)) for row in neighbors}

    assert neighbors.shape == (33, 3)
    assert sum(TBAModel.is_nyquist(row, shape) for row in neighbors) == 3
    for residue in residues:
        negative = tuple((-value) % shape[axis] for axis, value in enumerate(residue))
        if residue != negative:
            assert negative not in residues


def test_hopping_fourier_roundtrip_is_hermitian_on_rectangular_lattice():
    rng = np.random.default_rng(1234)
    shape = (4, 4, 1)
    avec = np.array([[1.0, 0.0], [0.0, np.sqrt(3.0)]])
    config = SimpleNamespace(
        band_calc_num=3,
        real_lattice_vectors=avec,
        lattice_const=1.0,
        dataset_type="comsol",
    )
    tba = object.__new__(TBAModel)
    tba.config = config
    tba.state = SimpleNamespace(k_shape=shape)

    k_axis = np.arange(-shape[0] // 2, shape[0] // 2, dtype=float) / shape[0]
    kfrac = np.stack(np.meshgrid(k_axis, k_axis, indexing="ij"), axis=-1).reshape(-1, 2)
    reciprocal = np.linalg.inv(avec).T
    k_cart = (kfrac @ reciprocal) * (2.0 * np.pi)
    raw = rng.normal(size=(k_cart.shape[0], 3, 3)) + 1j * rng.normal(size=(k_cart.shape[0], 3, 3))
    sampled = 0.5 * (raw + np.conjugate(np.swapaxes(raw, -2, -1)))

    neighbors = TBAModel.R_half_rect(shape)
    h0 = np.mean(sampled, axis=0)
    hops = []
    for row in neighbors:
        r_cart = row[:2] @ avec
        phase = np.exp(1j * (k_cart @ r_cart))
        hops.append(np.mean(sampled * phase[:, None, None], axis=0))
    hops = np.asarray(hops)

    h_of_k = tba._h_of_k_factory(h0, neighbors, hops)
    reconstructed = h_of_k(k_cart)
    off_grid = h_of_k(np.array([[0.17, -0.31], [1.13, 0.29]]))

    assert np.allclose(reconstructed, sampled, rtol=0.0, atol=1e-12)
    assert np.allclose(off_grid, np.conjugate(np.swapaxes(off_grid, -2, -1)), rtol=0.0, atol=1e-12)


def _run_small_calculation(threads: int):
    cfg = load_config("data/incar")
    cfg.max_iter = 0
    cfg.extension = [1, 1]
    cfg.k_num = [4, 4]
    cfg.hybrid_Wilson_loop = False
    cfg.Chern_number = False
    cfg.wannier_figures = "false"
    cfg.band_figure = "false"
    cfg.topo_output = "false"
    cfg.M_file = "false"
    cfg.V_file = "false"
    cfg.A_file = "false"
    cfg.U_file = "false"
    cfg.hopping_file = "false"
    cfg.wannier_file = "false"
    return run_calculation(load_input(cfg), threads=threads)
