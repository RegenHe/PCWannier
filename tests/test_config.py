import numpy as np
import pytest
from pathlib import Path

from pcwannier import EnergyWindow, load_config
from pcwannier.config import evaluate_math_expression


@pytest.mark.requires_dataset
def test_load_data_incar_defaults_and_preprocess():
    cfg = load_config("datasets/c4v/incar")

    assert cfg.dataset_type == "comsol"
    assert cfg.hermitian is True
    assert cfg.kdim == 2
    assert [len(axis) for axis in cfg.k_points] == [10, 10]
    assert np.array_equal(cfg.band_window, np.arange(0, 3))
    assert cfg.band_calc_num == 3
    assert len(cfg.composition_of_b) == 4
    assert cfg.b_vectors.shape == (4, 2)
    assert cfg.wb.shape == (4,)
    assert len(cfg.projections) == 1
    assert len(cfg.projections[0]["states"]) == 3
    assert cfg.compute_backend == "python"
    assert cfg.integration_mode == "nodal"
    assert cfg.symmetry_constrained is True
    assert cfg.symmetry_context is not None
    assert cfg.symmetry_context.model.symmetry_gauge.enabled


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
                "disentangle_max_iter = 25",
                "disentangle_err_diff = 1e-7",
                "disentangle_mixing = 0.75",
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
    assert cfg.symmetry_constrained is False
    assert cfg.disentangle_max_iter == 25
    assert cfg.disentangle_err_diff == pytest.approx(1e-7)
    assert cfg.disentangle_mixing == pytest.approx(0.75)


def test_math_expression_parser_is_limited():
    assert np.isclose(evaluate_math_expression("sqrt(4) + pi / pi"), 3.0)
    with pytest.raises(ValueError):
        evaluate_math_expression("__import__('os').system('echo nope')")


@pytest.mark.requires_dataset
def test_symmetry_constrained_requires_symmetry_file(tmp_path):
    source = Path("datasets/c4v/incar").read_text(encoding="utf-8")
    source = source.replace("symmetry_file = ../../symmetries/c4v.yaml", "symmetry_file = false")
    incar = tmp_path / "incar"
    incar.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="requires symmetry_file"):
        load_config(incar)


def test_rank_deficient_b_vectors_are_rejected(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0, 0 2",
                "reciprocal_lattice_vectors = 0 0, 0 0",
                "k_points = 0:0.5:1, 0:0.5:1",
                "composition_of_b = 1 0",
                "band_window = 0:1",
                "dataset_file = Ez.txt",
                "dielectric_file = eps.txt",
                "mesh_file = mesh.mphtxt",
                "E_file = E.txt",
                "extension = 1, 1",
                "projections",
                "a; [0, 0]; 0; [1, 0, 1]",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Add independent neighbor directions"):
        load_config(incar)


@pytest.mark.requires_dataset
def test_removed_w_center_input_is_rejected(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(Path("datasets/c4v/incar").read_text(encoding="utf-8") + "\nw_center = 0, 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="w_center input has been removed"):
        load_config(incar)
