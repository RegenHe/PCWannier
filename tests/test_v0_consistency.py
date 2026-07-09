import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pcwannier import load_config, write_outputs
from pcwannier.compute import run_calculation
from pcwannier.sources.comsol import load_input


pytestmark = pytest.mark.skipif(
    os.environ.get("PCWANNIER_RUN_CONSISTENCY") != "1",
    reason="Set PCWANNIER_RUN_CONSISTENCY=1 to run the optional v0 consistency comparison.",
)


def test_v0_consistency_smoke_comparison(tmp_path):
    old_src = Path("..") / "old" / "PCWannier" / "src"
    if not old_src.exists():
        pytest.skip("Old PCWannier source tree is not available.")

    work = tmp_path / "case"
    work.mkdir()
    for name in ["incar", "mesh.mphtxt", "Ez.txt", "eps.txt", "E.txt"]:
        shutil.copy2(Path("data") / name, work / name)

    incar = work / "incar"
    text = incar.read_text(encoding="utf-8")
    replacements = {
        "max_iter = 1000": "max_iter = 0",
        "wannier_figures = ./wanniers/": "wannier_figures = false",
        "band_figure = ./band.png": "band_figure = false",
        "hybrid_Wilson_loop = true": "hybrid_Wilson_loop = false",
        "Chern_number = true": "Chern_number = false",
        "topo_output = ./topo/": "topo_output = false",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    incar.write_text(text, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(old_src.resolve())
    subprocess.run(
        [sys.executable, "-m", "PCWannier.main", "-i", str(incar), "-t", "1", "-l", str(work / "old.log")],
        cwd=work,
        env=env,
        check=True,
        timeout=240,
    )

    cfg = load_config(incar)
    result = run_calculation(load_input(cfg), threads=1)
    out = work / "new"
    write_outputs(result, cfg, out)

    old_band = _load_band(work / "band.txt")
    new_band = _load_band(out / "band.txt")
    assert np.allclose(new_band, old_band, rtol=2e-6, atol=2e-6)

    old_h0 = _load_dict_cell(work / "hopping.txt", (0, 0, 0))
    new_h0 = _load_dict_cell(out / "hopping.txt", (0, 0, 0))
    assert np.allclose(new_h0, old_h0, rtol=2e-6, atol=2e-6)


def _parse_complex_token(token: str) -> complex:
    token = token.strip().replace(" ", "").replace("−", "-")
    return complex(token)


def _load_band(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        rows.append([_parse_complex_token(p) for p in parts[2:]])
    return np.asarray(rows, dtype=np.complex128)


def _load_dict_cell(path: Path, key: tuple[int, int, int]) -> np.ndarray:
    wanted = f"CELL({', '.join(str(x) for x in key)})"
    rows = []
    in_cell = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("CELL("):
            if in_cell:
                break
            in_cell = stripped.startswith(wanted)
            continue
        if in_cell and stripped:
            rows.append([_parse_complex_token(p) for p in stripped.split(",") if p.strip()])
    return np.asarray(rows, dtype=np.complex128)
