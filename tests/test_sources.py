import numpy as np

from pcwannier import load_config
from pcwannier.compute import integrate_over_mesh, is_numba_available
from pcwannier.data import FieldData, Mesh
from pcwannier.sources.comsol import load_comsol_data, load_comsol_mesh, load_input, match_data_to_mesh


def test_comsol_data_shapes():
    cfg = load_config("data/incar")
    mesh = load_comsol_mesh(str(cfg.input_path(cfg.mesh_file)))
    ez = load_comsol_data(str(cfg.input_path(cfg.dataset_file)))
    energy = load_comsol_data(str(cfg.input_path(cfg.E_file)))
    eps = load_comsol_data(str(cfg.input_path(cfg.dielectric_file)))

    assert mesh.vertices.shape == (499, 2)
    assert mesh.elements.shape == (936, 3)
    assert ez.value_matrix.shape == (499, 1200)
    assert energy.value_matrix.shape == (1, 1200)
    assert eps.value_matrix.shape[1] == 1


def test_load_input_bundle_distribution():
    bundle = load_input(load_config("data/incar"))

    assert bundle.fields.shape == (10, 10, 1)
    assert len(bundle.fields[0, 0, 0]) == 5
    assert bundle.fields[0, 0, 0][0].shape == (499,)
    assert bundle.epsilon.shape == (499,)
    assert np.array_equal(bundle.band_indices[0, 0, 0], [0, 1, 2, 3, 4])
    assert np.asarray(bundle.energies[0, 0, 0]).shape == (5,)


def test_match_data_to_mesh_and_integral_formula():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(vertices, elements)
    values = np.array([1.0 + 6.0j, 1.0, 2.0, 3.0 + 3.0j])

    result = integrate_over_mesh(FieldData("test", mesh, values), backend="python")

    assert np.isclose(result.real, 5 / 3)
    assert np.isclose(result.imag, 1.5)
    if is_numba_available():
        numba_result = integrate_over_mesh(FieldData("test", mesh, values), backend="numba")
        assert np.isclose(numba_result, result)

    cfg = load_config("data/incar")
    raw = load_comsol_data(str(cfg.input_path(cfg.dataset_file)))
    idxs, dists = match_data_to_mesh(load_comsol_mesh(str(cfg.input_path(cfg.mesh_file))), raw)
    assert len(idxs) == 499
    assert np.min(np.abs(dists)) <= 1e-6
