import numpy as np
import pytest
from types import SimpleNamespace

from pcwannier import load_config
from pcwannier.compute import is_numba_available, run_calculation
from pcwannier.compute.gradient import Gradient
from pcwannier.compute.initializer import StateInitializer
from pcwannier.compute.matrix import MSet
from pcwannier.compute.state import StateCollection
from pcwannier.data import InputBundle, Mesh
from pcwannier.compute.tba import TBAModel
from pcwannier.sources.comsol import load_input


def test_smoke_calculation_with_data_incar(tmp_path, caplog):
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

    with caplog.at_level("INFO"):
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
    assert "omega_I=" in caplog.text
    assert "omega_OD=" in caplog.text
    assert "omega_D=" in caplog.text
    assert "centers_rn=" in caplog.text


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


@pytest.mark.skipif(not is_numba_available(), reason="numba is not installed")
def test_python_and_numba_backends_are_numerically_consistent():
    python_result = _run_small_calculation(threads=1, backend="python")
    numba_result = _run_small_calculation(threads=1, backend="numba")

    for idx in np.ndindex(python_result.M0.shape):
        for b in range(len(python_result.M0[idx])):
            assert np.allclose(numba_result.M0[idx][b], python_result.M0[idx][b], rtol=1e-10, atol=1e-10)
        assert np.allclose(numba_result.A[idx], python_result.A[idx], rtol=1e-10, atol=1e-10)
        assert np.allclose(numba_result.V[idx], python_result.V[idx], rtol=1e-10, atol=1e-10)
        assert np.allclose(numba_result.U[idx], python_result.U[idx], rtol=1e-10, atol=1e-10)
    for key, hopping in python_result.hoppings.items():
        assert np.allclose(numba_result.hoppings[key], hopping, rtol=1e-10, atol=1e-10)
    assert np.allclose(numba_result.wannier_norms, python_result.wannier_norms, rtol=1e-10, atol=1e-10)
    assert np.allclose(numba_result.band.energies, python_result.band.energies, rtol=1e-10, atol=1e-10)


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


def test_band_path_does_not_reclose_periodically_equivalent_endpoints():
    config = SimpleNamespace(
        kdim=2,
        k_path=[
            {"name": "X", "point": [-0.5, 0.0], "num": 20},
            {"name": "G", "point": [0.0, 0.0], "num": 20},
            {"name": "X", "point": [0.5, 0.0], "num": 20},
        ],
        band_calc_num=1,
        neighbor=[],
        hermitian=True,
        DOS=0,
        real_lattice_vectors=np.eye(2),
        reciprocal_lattice_vectors=np.eye(2),
        lattice_const=1.0,
        dataset_type="comsol",
    )
    tba = object.__new__(TBAModel)
    tba.config = config
    tba.state = SimpleNamespace(k_shape=(4, 4, 1))

    result = tba.gen_hs_bands({(0, 0, 0): np.array([[1.0]])})

    assert result.k_path.shape == (41, 2)
    assert np.allclose(result.k_path[0], [-0.5, 0.0])
    assert np.allclose(result.k_path[-1], [0.5, 0.0])
    assert result.high_sym_points == [["X", 0], ["G", 20], ["X", 40]]
    assert result.k_axis[-1] == 40


def test_band_path_still_closes_distinct_endpoints():
    assert not TBAModel._periodically_equivalent(np.array([0.5, 0.5]), np.array([0.0, 0.0]))
    assert TBAModel._periodically_equivalent(np.array([-0.5, 0.0]), np.array([0.5, 0.0]))


def test_m0_orthogonal_transform_uses_conjugate_transpose():
    correction = np.array([[1.0 + 0.2j, 0.3], [-0.1j, 0.8 - 0.4j]])
    raw = np.array([[0.2 + 0.7j, 1.2 - 0.1j], [-0.5 + 0.3j, 2.0]])
    transforms = np.empty((1, 1, 1), dtype=object)
    transforms[0, 0, 0] = correction
    raw_m = np.empty((1, 1, 1), dtype=object)
    raw_m[0, 0, 0] = np.empty(1, dtype=object)
    raw_m[0, 0, 0][0] = raw
    state = SimpleNamespace(get_transform=lambda: transforms)
    config = SimpleNamespace(
        composition_of_b=[[1], [-1]],
        kdim=1,
        k_points=[np.array([0.0])],
    )
    mset = object.__new__(MSet)
    mset.state = state
    mset.config = config
    mset.mM0 = raw_m

    actual = mset.get_M0(0, 0, 0, 0)

    assert np.allclose(actual, correction.conj().T @ raw @ correction)
    assert not np.allclose(actual, correction @ raw @ correction)


def test_gradient_rejects_zero_m_diagonal():
    with np.testing.assert_raises(FloatingPointError):
        Gradient._checked_diagonal(np.array([[0.0, 1.0], [1.0, 1.0]]), (0, 0, 0), 0)


def test_strict_and_mixed_orthogonality_reports_are_distinct():
    mesh = Mesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    fields = np.empty((1, 1, 1), dtype=object)
    fields[0, 0, 0] = np.array([[1.0, 0.2, 0.1], [0.4, 1.0, 0.3]], dtype=np.complex128)
    indices = np.empty((1, 1, 1), dtype=object)
    indices[0, 0, 0] = [0, 1]
    energies = np.empty((1, 1, 1), dtype=object)
    energies[0, 0, 0] = np.array([1.0, 2.0])
    config = SimpleNamespace(kdim=2, integration_mode="nodal", dataset_type="comsol")
    bundle = InputBundle(
        config=config,
        mesh=mesh,
        fields=fields,
        epsilon=np.ones(3),
        energies=energies,
        band_indices=indices,
        inner_band_indices=indices.copy(),
        energy_matrix=np.array([[[[1.0, 2.0]]]]),
    )
    state = StateCollection(bundle, threads=1)

    _, initially_needs_orth = state.check_orthogonality()
    state.orthogonalize()
    strict_report, strict_needs_orth = state.check_orthogonality(apply_transform=True)
    mixed_report, mixed_needs_orth = state.check_orthogonality(apply_transform=False)

    assert initially_needs_orth
    assert not strict_needs_orth
    assert np.max(strict_report[..., 3]) < 1e-10
    assert mixed_needs_orth
    assert np.max(mixed_report[..., 2]) > 1e-3


def test_inner_window_projection_is_not_overwritten_by_matc():
    e_idx = np.empty((1, 1, 1), dtype=object)
    e_idx[0, 0, 0] = [0, 1]
    state = SimpleNamespace(E_idx=e_idx, k_indices=lambda: iter([(0, 0, 0)]))
    config = SimpleNamespace(
        use_cached_data=[],
        band_calc_num=2,
        inner_window=[0],
        proj_iter=True,
    )
    initializer = object.__new__(StateInitializer)
    initializer.state = state
    initializer.config = config
    initializer.matC = np.empty((1, 1, 1), dtype=object)
    initializer.matV = np.empty((1, 1, 1), dtype=object)
    initializer.matA = np.empty((1, 1, 1), dtype=object)
    projected_v = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    initializer.projection = lambda: (
        initializer.matC.__setitem__((0, 0, 0), np.eye(2)),
        initializer.matV.__setitem__((0, 0, 0), projected_v.copy()),
    )
    captured = {}
    initializer.mset = SimpleNamespace(initial=lambda value: captured.setdefault("V", value[0, 0, 0].copy()))

    initializer.iter(err_diff=1e-6, max_iter=0)

    assert np.array_equal(captured["V"], projected_v)


def _run_small_calculation(threads: int, backend: str = "python"):
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
    return run_calculation(load_input(cfg), threads=threads, backend=backend)
