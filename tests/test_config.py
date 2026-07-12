import numpy as np
import pytest

from pcwannier import EnergyWindow, load_config
from pcwannier.config import evaluate_math_expression


def test_load_data_incar_defaults_and_preprocess():
    cfg = load_config("data/incar")

    assert cfg.dataset_type == "comsol"
    assert cfg.hermitian is True
    assert cfg.kdim == 2
    assert [len(axis) for axis in cfg.k_points] == [16, 16]
    assert np.array_equal(cfg.band_window, np.arange(0, 4))
    assert cfg.band_calc_num == 4
    assert len(cfg.composition_of_b) == 4
    assert cfg.b_vectors.shape == (4, 2)
    assert cfg.wb.shape == (4,)
    assert len(cfg.projections) == 4
    assert cfg.compute_backend == "python"


def test_energy_window_parser(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0, 0 1",
                "reciprocal_lattice_vectors = 0 0, 0 0",
                "k_points = 0:1:1, 0:1:1",
                "composition_of_b = 1 0, 0 1",
                "band_window = 0.1, 0.9",
                "dataset_file = ./Ez.txt",
                "dielectric_file = ./eps.txt",
                "mesh_file = ./mesh.mphtxt",
                "E_file = ./E.txt",
                "compute_backend = auto",
                "extension = 1, 1",
                "projections",
                "a; [0, 0]; 0; [1, 0, 5]",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(incar)

    assert isinstance(cfg.band_window, EnergyWindow)
    assert cfg.band_window.emin == 0.1
    assert cfg.band_window.emax == 0.9
    assert cfg.compute_backend == "auto"


def test_math_expression_parser_is_limited():
    assert np.isclose(evaluate_math_expression("sqrt(4) + pi / pi"), 3.0)
    with pytest.raises(ValueError):
        evaluate_math_expression("__import__('os').system('echo nope')")
