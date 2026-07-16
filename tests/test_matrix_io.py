from types import SimpleNamespace

import numpy as np
import pytest

from pcwannier.matrix_io import load_cell_matrix, save_cell_matrix
from pcwannier.outputs import write_outputs


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
    config = SimpleNamespace(
        base_dir=tmp_path,
        S_file="S.txt",
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
    result = SimpleNamespace(S=smat, band=None, topology=None, sewing_matrices=None)

    write_outputs(result, config)

    path = tmp_path / "S.txt"
    assert "CELL(0, 0, 0)" in path.read_text(encoding="utf-8")
    loaded = load_cell_matrix(path, smat.shape)
    assert np.allclose(loaded[0, 0, 0], smat[0, 0, 0])
