import shutil
from pathlib import Path

import numpy as np
import pytest

from pcwannier.cli import main
from pcwannier.cli import parse_args
from pcwannier.sources.comsol import load_comsol_mesh


def test_cli_smoke_writes_outputs(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    for name in ["incar", "mesh.mphtxt", "Ez.txt", "eps.txt", "E.txt"]:
        shutil.copy2(Path("data") / name, case / name)

    incar = case / "incar"
    text = incar.read_text(encoding="utf-8")
    replacements = {
        "max_iter = 1000": "max_iter = 0",
        "wannier_figures = ./wanniers/": "wannier_figures = false",
        "band_figure = ./band.png": "band_figure = false",
        "hybrid_Wilson_loop = true": "hybrid_Wilson_loop = false",
        "Chern_number = true": "Chern_number = false",
        "topo_output = ./topo/": "topo_output = false",
        "extension = 10, 10": "extension = 1, 1",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    incar.write_text(text, encoding="utf-8")

    interp_points = case / "interp-points.txt"
    mesh = load_comsol_mesh(case / "mesh.mphtxt")
    np.savetxt(interp_points, mesh.vertices[:2], delimiter=",")

    out = tmp_path / "out"
    interp_wannier = out / "interp-wannier.txt"
    interp_epsilon = out / "interp-epsilon.txt"
    assert (
        main(
            [
                "-i",
                str(incar),
                "--out",
                str(out),
                "-t",
                "1",
                "-l",
                "log.txt",
                "--interp",
                str(interp_points),
                "--interp-wannier",
                str(interp_wannier),
                "--interp-epsilon",
                str(interp_epsilon),
            ]
        )
        == 0
    )

    assert (out / "M0.txt").exists()
    assert (out / "V.txt").exists()
    assert (out / "A.txt").exists()
    assert (out / "U.txt").exists()
    assert (out / "hopping.txt").exists()
    assert (out / "band.txt").exists()
    assert interp_wannier.exists()
    assert interp_epsilon.exists()
    assert len(interp_wannier.read_text(encoding="utf-8").splitlines()) == 2
    assert len(interp_epsilon.read_text(encoding="utf-8").splitlines()) == 2
    log_text = (out / "log.txt").read_text(encoding="utf-8")
    assert "=========  PCWannier v" in log_text
    assert "total runtime:" in log_text
    assert "memory usage:" in log_text
    assert "memory usage: unavailable" not in log_text
    assert "pcwannier.compute.runner" not in log_text

    assert main(["-i", str(incar), "--out", str(out), "-t", "1", "-l", "cache-log.txt", "--cache"]) == 0

    base_out = tmp_path / "base-out"
    assert main(["-i", str(incar), "--out", str(base_out), "-b", "-l", "base-log.txt"]) == 0
    assert (base_out / "base" / "base-0-real.png").exists()


def test_fatband_argument_removed():
    with pytest.raises(SystemExit):
        parse_args(["-i", "data/incar", "--fatband"])
