import numpy as np

from pcwannier import load_config
from pcwannier.compute import run_calculation
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

    assert result.M0.shape == (10, 10, 1)
    assert result.V.shape == (10, 10, 1)
    assert result.U.shape == (10, 10, 1)
    assert result.wanniers[(0, 0)].shape[1] == cfg.band_calc_num
    assert result.wannier_norms.shape == (cfg.band_calc_num,)
    assert np.all(np.isfinite(np.real(result.wannier_norms)))
    assert (0, 0, 0) in result.hoppings
    assert result.hoppings[(0, 0, 0)].shape == (cfg.band_calc_num, cfg.band_calc_num)
    assert result.band is not None
    assert result.band.energies.shape[1] == cfg.band_calc_num
