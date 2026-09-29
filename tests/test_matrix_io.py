from itertools import product
import re
from types import SimpleNamespace

import numpy as np
import pytest

from pcwannier.matrix_io import (
    load_cell_matrix,
    save_cell_matrix,
)
from pcwannier.compute.tba import TBAModel
from pcwannier.outputs import save_hoppings, write_outputs


def test_cell_matrix_roundtrip_preserves_ragged_cells(tmp_path):
    data = np.empty((2,), dtype=object)
    data[0] = np.array([[1.0 + 2.0j, 3.0]])
    data[1] = np.array([[4.0], [5.0 - 1.0j]])
    path = tmp_path / "matrix.txt"

    save_cell_matrix(path, data, data.shape)
    loaded = load_cell_matrix(path, data.shape)

    assert np.allclose(loaded[0], data[0])
    assert np.allclose(loaded[1], data[1])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            "CELL(0) shape=(1, 1):\n1+0j\nCELL(0) shape=(1, 1):\n2+0j\n",
            "Duplicate CELL index",
        ),
        (
            "CELL(root) shape=(1, 1):\n1+0j\nCELL(0) shape=(1, 1):\n2+0j\n",
            "must not be mixed",
        ),
        ("CELL(-1) shape=(1, 1):\n1+0j\n", "negative"),
        ("CELL(0) shape=(1, 2):\n1+0j\n", "declares shape"),
        (
            "CELL(0) shape=(1, 1):\n1+0j\nCELL(0, 0) shape=(1, 1):\n2+0j\n",
            "Duplicate effective CELL index",
        ),
    ],
)
def test_cell_matrix_rejects_ambiguous_or_inconsistent_blocks(tmp_path, content, message):
    path = tmp_path / "invalid.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_cell_matrix(path, (1, 1))


def test_write_outputs_writes_raw_s_in_shared_cell_format(tmp_path):
    smat = np.empty((1, 1, 1), dtype=object)
    smat[0, 0, 0] = np.array([[1.0, 0.2j], [-0.2j, 1.5]])
    projector = np.empty((1, 1, 1), dtype=object)
    projector[0, 0, 0] = np.array([[1.0, 0.0], [0.0, 0.0]])
    config = SimpleNamespace(
        base_dir=tmp_path,
        S_file="S.txt",
        P_file="P.txt",
        M_file=False,
        V_file=False,
        A_file=False,
        U_file=False,
        D_file=False,
        hopping_file=False,
        wannier_file=False,
        wannier_figures=False,
        topo_output=False,
        band_file=False,
        band_figure=False,
        composition_of_b=[],
    )
    result = SimpleNamespace(
        S=smat,
        transverse_projectors=projector,
        band=None,
        topology=None,
        sewing_matrices=None,
    )

    write_outputs(result, config)

    path = tmp_path / "S.txt"
    assert "CELL(0, 0, 0)" in path.read_text(encoding="utf-8")
    loaded = load_cell_matrix(path, smat.shape)
    assert np.allclose(loaded[0, 0, 0], smat[0, 0, 0])

    projector_path = tmp_path / "P.txt"
    text = projector_path.read_text(encoding="utf-8")
    assert "P_T(k) in the final Wannier basis" in text
    loaded_projector = load_cell_matrix(projector_path, projector.shape)
    assert np.allclose(
        loaded_projector[0, 0, 0], projector[0, 0, 0]
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_hopping_text_restores_old_cell_order_and_preserves_off_mesh_hamiltonian(tmp_path, sign):
    # Includes both sides of the even-mesh boundary. They must remain distinct
    # exact displacements even though their FFT residues can coincide.
    rng = np.random.default_rng(104)
    keys = [(0, 0, 0)] + [
        key for key in product(range(-2, 3), repeat=3)
        if key < tuple(-value for value in key)
    ]
    data = {
        key: (rng.integers(-20, 21, (2, 2)) + 1j * rng.integers(-20, 21, (2, 2))) / 16
        for key in keys
    }
    data[(0, 0, 0)] = np.diag([1.0, 2.0])
    path = tmp_path / "hopping.txt"
    save_hoppings(path, data, (4, 4, 4))

    loaded = {}
    current = None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if line.startswith("CELL("):
            current = tuple(int(value) for value in re.match(r"CELL\(([^)]+)\)", line)[1].split(","))
            rows = []
        elif current is not None and line.strip():
            rows.append([complex(value.replace(" ", "")) for value in line.split(",")])
        elif current is not None:
            loaded[current] = np.asarray(rows)
            current = None

    old_order = [(0, 0, 0)] + [tuple(row) for row in TBAModel.R_half_rect((4, 4, 4))]
    assert list(loaded)[:len(old_order)] == old_order
    assert len(loaded) == len(data) == 63
    assert list(data) == keys
    model = TBAModel.__new__(TBAModel)
    model.config = SimpleNamespace(
        kdim=3, band_calc_num=2, neighbor=[], real_lattice_vectors=np.eye(3), lattice_const=1.0,
    )
    model.state = SimpleNamespace(k_shape=(4, 4, 4), bloch_sign=sign)
    queries = rng.uniform(-np.pi, np.pi, (47, 3))
    before = model._band_hamiltonian_factory(data)(queries)
    after = model._band_hamiltonian_factory(loaded)(queries)
    assert np.allclose(before, after, rtol=0, atol=1e-12)
