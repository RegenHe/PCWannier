import numpy as np
import pytest
from types import SimpleNamespace

from pcwannier import load_config
from pcwannier.compute import (
    integrate_overlap_matrix,
    integrate_over_mesh,
    integrate_weighted_abs2_columns,
    integrate_weighted_columns,
    is_numba_available,
)
from pcwannier.data import FieldData, Mesh, RawData
from pcwannier.outputs import _interpolate_real_mesh
from pcwannier.sources.comsol import (
    _validate_header_k_grid,
    _values_on_mesh,
    load_comsol_data,
    load_comsol_mesh,
    load_input,
    match_data_to_mesh,
)


def test_comsol_data_shapes():
    cfg = load_config("data/incar")
    mesh = load_comsol_mesh(str(cfg.input_path(cfg.mesh_file)))
    ez = load_comsol_data(str(cfg.input_path(cfg.dataset_file)))
    energy = load_comsol_data(str(cfg.input_path(cfg.E_file)))
    eps = load_comsol_data(str(cfg.input_path(cfg.dielectric_file)), real_only=True)

    assert mesh.vertices.shape == (1329, 2)
    assert mesh.elements.shape == (2516, 3)
    assert ez.value_matrix.shape[0] == 1329
    assert energy.value_matrix.shape[0] == 1
    assert ez.value_matrix.shape[1] == energy.value_matrix.shape[1]
    assert ez.value_matrix.shape[1] % np.prod([len(axis) for axis in cfg.k_points]) == 0
    assert np.unique(ez.column_parameters["k1"]).size == 16
    assert np.unique(ez.column_parameters["k2"]).size == 16
    assert eps.value_matrix.shape[1] == 1
    assert not np.iscomplexobj(eps.value_matrix)


def test_comsol_real_only_reader_uses_utf8_comments(tmp_path):
    path = tmp_path / "eps.txt"
    path.write_text("% COMSOL ε data\n0 0 1.5\n1 0 2.5\n", encoding="utf-8")

    raw = load_comsol_data(path, real_only=True)

    assert raw.point_matrix.shape == (2, 2)
    assert np.allclose(raw.value_matrix[:, 0], [1.5, 2.5])


def test_comsol_real_only_reader_accepts_complex_tokens(tmp_path):
    path = tmp_path / "E.txt"
    path.write_text(
        "% COMSOL complex-valued real data\n"
        "0 0 23.57556147941945-3.1103526165115537E-9i 4.0+0i\n",
        encoding="utf-8",
    )

    raw = load_comsol_data(path, real_only=True)

    assert raw.value_matrix.shape == (1, 2)
    assert not np.iscomplexobj(raw.value_matrix)
    assert np.allclose(raw.value_matrix[0], [23.57556147941945, 4.0])


def test_comsol_header_k_grid_mismatch_is_rejected():
    cfg = load_config("data/incar")
    raw = load_comsol_data(cfg.input_path(cfg.dataset_file))
    cfg.k_points[0] = cfg.k_points[0][::2]

    with pytest.raises(ValueError, match="COMSOL k-grid mismatch"):
        _validate_header_k_grid(cfg, raw, cfg.input_path(cfg.dataset_file))


def test_comsol_header_validates_band_parameter_and_dataset_order(tmp_path):
    cfg = SimpleNamespace(
        kdim=2,
        k_points=[np.arange(2), np.arange(3)],
        dataset_order=["k1", "k2", "E"],
    )
    shape = (2, 3, 2)
    indices = np.indices(shape)
    parameters = {
        "k1": indices[0].reshape(-1).astype(float),
        "k2": indices[1].reshape(-1).astype(float),
        "lambda": indices[2].reshape(-1).astype(float),
    }
    raw = RawData(np.zeros((1, 2)), np.zeros((1, np.prod(shape))), parameters)

    _validate_header_k_grid(cfg, raw, tmp_path / "E.txt")

    raw.column_parameters = {**parameters, "k2": np.roll(parameters["k2"], 2)}
    with pytest.raises(ValueError, match="dataset_order"):
        _validate_header_k_grid(cfg, raw, tmp_path / "E.txt")


def test_comsol_header_requires_complete_band_parameter(tmp_path):
    cfg = SimpleNamespace(
        kdim=2,
        k_points=[np.arange(2), np.arange(2)],
        dataset_order=["k1", "k2", "E"],
    )
    indices = np.indices((2, 2, 3))
    raw = RawData(
        np.zeros((1, 2)),
        np.zeros((1, 12)),
        {
            "k1": indices[0].reshape(-1).astype(float),
            "k2": indices[1].reshape(-1).astype(float),
        },
    )

    with pytest.raises(ValueError, match="energy/band parameter"):
        _validate_header_k_grid(cfg, raw, tmp_path / "E.txt")


def test_load_input_bundle_distribution():
    bundle = load_input(load_config("data/incar"))

    assert bundle.fields.shape == (16, 16, 1)
    assert len(bundle.fields[0, 0, 0]) == 4
    assert bundle.fields[0, 0, 0].shape == (4, 1329)
    assert bundle.fields[0, 0, 0][0].shape == (1329,)
    assert bundle.epsilon.shape == (1329,)
    assert np.array_equal(bundle.band_indices[0, 0, 0], [0, 1, 2, 3])
    assert np.asarray(bundle.energies[0, 0, 0]).shape == (4,)
    assert np.count_nonzero(np.isclose(bundle.epsilon, 8.0)) == 159


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
    assert len(idxs) == 1329
    assert np.all(dists == 0.0)
    assert np.unique(idxs).size == 1329


def test_values_on_mesh_averages_duplicate_rows_all_columns():
    mesh = Mesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    raw = RawData(
        np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0 + 1.0j, 2.0], [3.0 + 3.0j, 6.0], [5.0, 7.0], [9.0, 11.0]]),
    )

    mapped = _values_on_mesh(mesh, raw)

    assert np.allclose(mapped[0], [2.0 + 2.0j, 4.0])
    assert np.allclose(mapped[1], [5.0, 7.0])
    assert np.allclose(mapped[2], [9.0, 11.0])


def test_match_data_to_mesh_rejects_far_points():
    mesh = Mesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    raw = RawData(np.array([[10.0, 10.0], [11.0, 10.0], [10.0, 11.0]]), np.ones((3, 1)))
    idxs, _ = match_data_to_mesh(mesh, raw)

    assert np.all(idxs < 0)


def test_weighted_columns_matches_explicit_integral_matrix():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(vertices, elements)
    left = np.array([1.0 + 1.0j, 2.0, -0.5j, 3.0 - 2.0j])
    right = np.array(
        [
            [1.0, 2.0 - 1.0j],
            [0.5j, -2.0],
            [3.0, 0.25],
            [-1.0 + 1.0j, 0.75j],
        ],
        dtype=np.complex128,
    )

    expected = integrate_over_mesh(FieldData("explicit", mesh, left[:, None] * right), backend="python")
    actual = integrate_weighted_columns(mesh, left, right, backend="python")

    assert np.allclose(actual, expected)
    if is_numba_available():
        assert np.allclose(integrate_weighted_columns(mesh, left, right, backend="numba"), expected)


def test_weighted_abs2_columns_matches_weighted_columns():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(vertices, elements)
    weights = np.array([1.0, 2.0, 0.5, 1.5])
    values = np.array(
        [
            [1.0 + 2.0j, 2.0 - 1.0j],
            [0.5j, -2.0],
            [3.0, 0.25],
            [-1.0 + 1.0j, 0.75j],
        ],
        dtype=np.complex128,
    )

    expected = integrate_weighted_columns(mesh, weights, np.abs(values) ** 2, backend="python")
    actual = integrate_weighted_abs2_columns(mesh, weights, values, backend="python")

    assert np.allclose(actual, expected)
    if is_numba_available():
        assert np.allclose(integrate_weighted_abs2_columns(mesh, weights, values, backend="numba"), expected)


def test_overlap_matrix_matches_weighted_columns_rows():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    mesh = Mesh(vertices, elements)
    left = np.array([[1.0 + 1.0j, 2.0, -0.5j, 3.0 - 2.0j], [0.5, -1.0j, 2.0, 1.0]])
    right = np.array([[1.0, 0.5j, 3.0, -1.0 + 1.0j], [2.0 - 1.0j, -2.0, 0.25, 0.75j]])
    weights = np.array([1.0, 2.0, 0.5, 1.5])

    expected = np.vstack(
        [integrate_weighted_columns(mesh, np.conj(row) * weights, right.T, backend="python") for row in left]
    )
    actual = integrate_overlap_matrix(mesh, left, right, weights, backend="python")

    assert np.allclose(actual, expected)
    if is_numba_available():
        assert np.allclose(integrate_overlap_matrix(mesh, left, right, weights, backend="numba"), expected)


def test_quadratic_overlap_matches_triangle_mass_matrix():
    mesh = Mesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    left = np.array([[1.0 + 0.5j, -2.0j, 3.0], [0.25, 1.5 - 0.2j, -1.0]])
    right = np.array([[2.0, -1.0j, 0.5 + 0.1j], [1.0j, 2.5, -0.75]])
    mass = (0.5 / 12.0) * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    expected = np.conj(left) @ mass @ right.T

    actual = integrate_overlap_matrix(mesh, left, right, mode="quadratic", backend="python")

    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)
    if is_numba_available():
        numba_actual = integrate_overlap_matrix(mesh, left, right, mode="quadratic", backend="numba")
        assert np.allclose(numba_actual, expected, rtol=1e-13, atol=1e-13)


def test_quadratic_weighted_abs2_is_real_and_exact_for_linear_fields():
    mesh = Mesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    values = np.array([[1.0 + 1.0j], [2.0 - 0.5j], [-1.0j]])
    mass = (0.5 / 12.0) * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    expected = np.array([np.conj(values[:, 0]) @ mass @ values[:, 0]])

    actual = integrate_weighted_abs2_columns(mesh, np.ones(3), values, mode="quadratic", backend="python")

    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_mesh_extension_matches_legacy_incremental_algorithm():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    edge = np.array([[0, 1], [1, 3], [3, 2], [2, 0]])
    avec = [[1.0, 0.0], [0.0, 1.0]]

    for ext in ([1, 1], [2, 2], [3, 2]):
        new_mesh = Mesh(vertices.copy(), elements.copy(), edge.copy())
        new_mapping = new_mesh.extension(ext, avec, 1.0)
        old_mesh = Mesh(vertices.copy(), elements.copy(), edge.copy())
        old_mapping = _legacy_extension(old_mesh, ext, avec, 1.0)

        assert np.allclose(new_mesh.vertices, old_mesh.vertices)
        assert np.array_equal(new_mesh.elements, old_mesh.elements)
        assert np.array_equal(new_mapping, old_mapping)
        assert np.allclose(new_mesh.tri_weights, old_mesh.tri_weights)


def test_data_mesh_extension_full_size():
    cfg = load_config("data/incar")
    mesh = load_comsol_mesh(str(cfg.input_path(cfg.mesh_file)))
    assert np.count_nonzero(mesh._boundary_vertex_mask(mesh.vertices.shape[0])) == 140
    assert np.unique(mesh.edge).size == 299
    base_weights = mesh.tri_weights.copy()
    mapping = mesh.extension(cfg.extension, cfg.real_lattice_vectors, float(cfg.lattice_const))

    assert mesh.vertices.shape == (84881, 2)
    assert mesh.elements.shape == (161024, 3)
    assert mapping.shape == (84881,)
    assert np.allclose(mesh.tri_weights, np.tile(base_weights, np.prod(cfg.extension)), rtol=0.0, atol=1e-14)

    avec = np.asarray(cfg.real_lattice_vectors, dtype=float) * float(cfg.lattice_const)
    base_mesh = load_comsol_mesh(str(cfg.input_path(cfg.mesh_file)))
    base_fractional = base_mesh.vertices @ np.linalg.inv(avec)
    base_field = np.exp(2j * np.pi * base_fractional[:, 0]) + 0.5 * np.exp(2j * np.pi * base_fractional[:, 1])
    extended_fractional = mesh.vertices @ np.linalg.inv(avec)
    expected = np.exp(2j * np.pi * extended_fractional[:, 0]) + 0.5 * np.exp(2j * np.pi * extended_fractional[:, 1])
    assert np.allclose(base_field[mapping], expected, rtol=0.0, atol=1e-10)


def test_extension_does_not_merge_nearby_nonmatching_boundary_nodes():
    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.49], [1.0, 1.0], [0.0, 1.0], [0.0, 0.5], [0.5, 0.5]]
    )
    elements = np.array([[6, i, (i + 1) % 6] for i in range(6)])
    mesh = Mesh(vertices, elements)

    mapping = mesh.extension([2, 1], [[1.0, 0.0], [0.0, 1.0]], 1.0)
    seam = mesh.vertices[np.isclose(mesh.vertices[:, 0], 1.0)]

    assert mesh.vertices.shape[0] == 12
    assert mapping.shape == (12,)
    assert np.allclose(np.sort(seam[:, 1]), [0.0, 0.49, 0.5, 1.0])

    values = vertices[:, 1][mapping]
    points = np.array([[0.25, 0.25], [1.25, 0.25], [1.0, 0.5]])
    interpolated = _interpolate_real_mesh(mesh, values, points, tile_count=2)
    assert np.allclose(interpolated, points[:, 1], rtol=0.0, atol=1e-12)


def _legacy_extension(mesh: Mesh, n, real_lattice_vectors, lattice_const):
    original_vertices = mesh.vertices.copy()
    original_elements = mesh.elements.copy()
    mapping = np.arange(len(original_vertices), dtype=np.intp)

    for i in range(n[0]):
        for j in range(n[1]):
            if i == 0 and j == 0:
                continue
            offset_x = (real_lattice_vectors[0][0] * i + real_lattice_vectors[1][0] * j) * lattice_const
            offset_y = (real_lattice_vectors[0][1] * i + real_lattice_vectors[1][1] * j) * lattice_const
            base_index = int(np.max(mesh.elements)) + 1
            new_elements = original_elements + base_index
            new_vertices = original_vertices + np.array([offset_x, offset_y])

            idx_new, idx_existing = mesh.match(new_vertices, mesh.vertices)
            for new_idx, old_idx in zip(idx_new, idx_existing):
                new_elements[new_elements == (new_idx + base_index)] = old_idx

            mesh.elements = np.vstack((mesh.elements, new_elements))
            mesh.vertices = np.vstack((mesh.vertices, new_vertices))
            mapping = np.hstack((mapping, np.arange(len(original_vertices), dtype=np.intp)))
            _, mapping = mesh.rebuild_index(mapping)

    offset_x = (
        real_lattice_vectors[0][0] * np.floor((n[0] - 1) / 2)
        + real_lattice_vectors[1][0] * np.floor((n[1] - 1) / 2)
    ) * lattice_const
    offset_y = (
        real_lattice_vectors[0][1] * np.floor((n[0] - 1) / 2)
        + real_lattice_vectors[1][1] * np.floor((n[1] - 1) / 2)
    ) * lattice_const
    mesh.vertices = mesh.vertices - np.array([offset_x, offset_y])
    mesh._precompute_tri_weights()
    return np.asarray(mapping, dtype=np.intp)
