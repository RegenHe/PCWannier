import numpy as np
import pytest
from types import SimpleNamespace
from threading import Lock

from pcwannier import BlochConvention
from pcwannier.compute.gradient import Gradient
from pcwannier.compute.context import CalculationContext
from pcwannier.compute.band_path import periodically_equivalent_kpoint
from pcwannier.compute.initializer import StateInitializer
from pcwannier.compute.matrix import MSet
from pcwannier.compute.parallel import (
    ExecutionContext,
    memory_limited_threads,
    numba_parallel_allowed,
    parallel_map,
    set_numba_parallel_allowed,
)
from pcwannier.compute.state import StateCollection
from pcwannier.data import BandChannelReference, InputBundle, Mesh
from pcwannier.matrix_io import save_cell_matrix
from pcwannier.maxwell import MaxwellProblem
from pcwannier.compute.tba import TBAModel


def test_gradient_does_not_report_convergence_from_small_spread_change_alone():
    gradient, calls = _synthetic_gradient_optimizer([1.0, 0.9999995, 0.9999990, 0.9999985])

    Gradient.iter(gradient, err_diff=1.0e-6, max_iter=3, epsilon=0.01)

    assert calls["calc"] == 3
    assert gradient.epsilon == pytest.approx(0.01)


def test_gradient_rejects_uphill_step_and_restores_gauge():
    gradient, calls = _synthetic_gradient_optimizer([1.0, 0.9, 1.1])

    Gradient.iter(gradient, err_diff=1.0e-8, max_iter=2, epsilon=0.01)

    assert calls["calc"] == 2
    assert gradient.epsilon == pytest.approx(0.01)
    assert np.allclose(gradient.U[0, 0, 0], [[np.exp(0.01j)]])
    assert not gradient.last_line_search.accepted
    assert gradient.last_line_search.backtracking_steps == 24


def test_gradient_stops_before_dividing_by_a_diagonal_below_the_mv_floor():
    gradient, calls = _synthetic_gradient_optimizer([1.0])
    gradient.config.mv_diagonal_floor = 1.0e-8
    gradient.mset.get = lambda *_args: np.array([[1.0e-10]], dtype=np.complex128)

    Gradient.iter(gradient, err_diff=1.0e-8, max_iter=10, epsilon=0.01)

    assert calls["calc"] == 0
    assert not gradient.converged
    assert np.array_equal(gradient.U[0, 0, 0], np.eye(1))


def test_zero_iteration_mv_evaluation_does_not_form_the_singular_gradient():
    gradient, calls = _synthetic_gradient_optimizer([1.0])
    gradient.mset.get = lambda *_args: np.array([[0.0]], dtype=np.complex128)

    Gradient.iter(gradient, err_diff=1.0e-8, max_iter=0, epsilon=0.01)

    assert calls["calc"] == 0
    assert np.isinf(gradient.omega[2])


def test_calculation_context_separates_internal_and_output_coefficients():
    correction = np.array([[1.2, 0.1], [0.2, 0.8]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    transforms = np.empty((1, 1, 1), dtype=object)
    identities = np.empty((1, 1, 1), dtype=object)
    transforms[0, 0, 0] = correction
    identities[0, 0, 0] = identity
    state = SimpleNamespace(get_transform=lambda zero=False: identities if zero else transforms)
    mat_v = np.empty((1, 1, 1), dtype=object)
    mat_u = np.empty((1, 1, 1), dtype=object)
    mat_v[0, 0, 0] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    mat_u[0, 0, 0] = np.eye(2, dtype=np.complex128)
    config = SimpleNamespace(
        output_basis="strict",
    )
    ctx = CalculationContext(
        config,
        state,
        None,
        SimpleNamespace(matV=mat_v),
        SimpleNamespace(U=mat_u),
        symmetry_gauge=object(),
    )
    gauge = mat_v[0, 0, 0]

    assert np.allclose(ctx.internal_state_coefficients_at(0, 0, 0), correction @ gauge)
    assert np.allclose(ctx.output_state_coefficients_at(0, 0, 0), correction @ gauge)

    config.output_basis = "fem"
    assert np.allclose(ctx.internal_state_coefficients_at(0, 0, 0), correction @ gauge)
    assert np.allclose(ctx.output_state_coefficients_at(0, 0, 0), gauge)

    config.output_basis = "strict"
    assert np.allclose(ctx.output_state_coefficients_at(0, 0, 0), correction @ gauge)


def test_output_spectrum_diagnostics_distinguish_strict_and_fem_bases():
    raw_energies = np.array([1.0, 2.0, 2.0])
    analysis = SimpleNamespace(
        points=(
            SimpleNamespace(
                name="K",
                k_index=(0, 0, 0),
                degenerate_blocks=(SimpleNamespace(band_indices=(1, 2)),),
            ),
        )
    )

    strict = _synthetic_spectrum_model(raw_energies, np.diag([1.0, 1.8, 2.2]), "strict")
    strict_diagnostics = strict.output_spectrum_diagnostics(analysis)
    strict_splitting = strict_diagnostics.degeneracy_splittings[0]
    assert np.isclose(strict_diagnostics.max_eigenvalue_drift, 0.2)
    assert np.isclose(strict_splitting.output_gap, 0.4)

    fem = _synthetic_spectrum_model(raw_energies, np.diag(raw_energies), "fem")
    fem_diagnostics = fem.output_spectrum_diagnostics(analysis)
    fem_splitting = fem_diagnostics.degeneracy_splittings[0]
    assert fem_diagnostics.max_eigenvalue_drift == 0.0
    assert fem_splitting.output_gap == 0.0


def test_tba_inverts_only_longitudinal_energies_before_output_gauge():
    energies = _object_grid(np.array([4.0, 6.0, 9.0]))
    band_indices = _object_grid([0, 1, 10])
    config = SimpleNamespace(
        band_calc_num=3,
        kdim=3,
        real_lattice_vectors=np.eye(3),
        reciprocal_lattice_vectors=np.eye(3),
        lattice_const=1.0,
        k_points=[np.array([0.0]), np.array([0.0]), np.array([0.0])],
        invert_longitudinal_energies=True,
    )
    state = SimpleNamespace(
        E=energies,
        E_idx=band_indices,
        band_channels={
            0: BandChannelReference("H", 0),
            1: BandChannelReference("H", 1),
            10: BandChannelReference("L", 0),
        },
        k_shape=(1, 1, 1),
        k_indices=lambda: iter(((0, 0, 0),)),
        get_k_num=lambda: 1,
        bloch_sign=1,
    )
    unitary = np.array(
        [
            [1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
            [0.0, 1.0, 0.0],
            [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        ],
        dtype=np.complex128,
    )
    ctx = SimpleNamespace(
        config=config,
        state=state,
        output_state_coefficients_at=lambda *_: unitary,
    )
    model = TBAModel(ctx)

    _, projected = model._projected_k_hamiltonians()
    expected = unitary.conj().T @ np.diag([4.0, 6.0, -9.0]) @ unitary

    assert np.allclose(projected[0], expected)
    assert np.allclose(model.gen_hopping((0, 0, 0)), expected)
    assert np.allclose(np.linalg.eigvalsh(projected[0]), [-9.0, 4.0, 6.0])
    assert np.array_equal(energies[0, 0, 0], [4.0, 6.0, 9.0])


def test_tba_uses_full_source_basis_hamiltonian_before_wannier_gauge():
    energies = _object_grid(np.array([0.0, 1.0, 2.0]))
    band_indices = _object_grid([0, 1, 2])
    base = np.array(
        [[1.0, 0.2j, 0.0], [-0.2j, 2.0, 0.3], [0.0, 0.3, 0.0]],
        dtype=np.complex128,
    )
    gauge, _ = np.linalg.qr(
        np.array(
            [[1.0, 1.0j, 0.2], [0.3j, 1.0, -0.4], [0.1, 0.2j, 1.0]],
            dtype=np.complex128,
        )
    )
    config = SimpleNamespace(
        band_calc_num=3,
        kdim=3,
        real_lattice_vectors=np.eye(3),
        reciprocal_lattice_vectors=np.eye(3),
        lattice_const=1.0,
        k_points=[np.array([0.0]), np.array([0.0]), np.array([0.0])],
        invert_longitudinal_energies=False,
    )
    state = SimpleNamespace(
        E=energies,
        E_idx=band_indices,
        base_hamiltonian_at=lambda _index: base,
        k_shape=(1, 1, 1),
        k_indices=lambda: iter(((0, 0, 0),)),
        get_k_num=lambda: 1,
        bloch_sign=1,
    )
    model = TBAModel(
        SimpleNamespace(
            config=config,
            state=state,
            output_state_coefficients_at=lambda *_: gauge,
        )
    )

    _, projected = model._projected_k_hamiltonians()

    assert np.allclose(projected[0], gauge.conj().T @ base @ gauge)
    assert np.allclose(
        np.linalg.eigvalsh(projected[0]), np.linalg.eigvalsh(base), atol=1.0e-13
    )


def test_tba_bands_use_hoppings_and_export_sampled_transverse_projectors():
    shape = (2, 1, 1)
    energies = np.empty(shape, dtype=object)
    indices = np.empty(shape, dtype=object)
    overlaps = np.empty(shape, dtype=object)
    bases = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        energies[index] = np.asarray([2.0, 5.0, 7.0])
        indices[index] = [0, 1, 2]
        overlaps[index] = np.eye(3, dtype=np.complex128)
        bases[index] = np.diag([2.0, 5.0, 7.0]).astype(np.complex128)
    angle = 0.7
    gauges = (
        np.eye(3, dtype=np.complex128),
        np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=np.complex128,
        ),
    )
    config = SimpleNamespace(
        band_calc_num=3,
        kdim=3,
        real_lattice_vectors=np.eye(3),
        reciprocal_lattice_vectors=np.eye(3),
        lattice_const=1.0,
        k_points=[np.asarray([-0.5, 0.0]), np.asarray([0.0]), np.asarray([0.0])],
        neighbor=[],
        invert_longitudinal_energies=False,
    )
    state = SimpleNamespace(
        E=energies,
        E_idx=indices,
        S=overlaps,
        base_hamiltonian_at=lambda index: bases[index],
        k_shape=shape,
        k_indices=lambda: iter(np.ndindex(shape)),
        get_k_num=lambda: 2,
        bloch_sign=1,
        band_channels={
            0: BandChannelReference("H", 0),
            1: BandChannelReference("H", 1),
            2: BandChannelReference("L", 0),
        },
    )
    ctx = SimpleNamespace(
        config=config,
        state=state,
        output_state_coefficients_at=lambda i, _j, _k: gauges[i],
    )
    model = TBAModel(ctx)
    hoppings = model.collect_hoppings()
    neighbors = model._band_neighbors(hoppings)
    direct_factory = model._h_of_k_factory(
        hoppings[(0, 0, 0)],
        neighbors,
        model._hoppings_for_neighbors(hoppings, neighbors),
    )
    band_factory = model._band_hamiltonian_factory(hoppings)
    k_cart = model._kfrac_to_kcart(np.asarray([[0.25, 0.0, 0.0]]))

    assert np.allclose(band_factory(k_cart), direct_factory(k_cart))

    projector_grid = model.transverse_projectors()
    assert projector_grid.shape == shape
    assert np.allclose(
        projector_grid[0, 0, 0] @ projector_grid[0, 0, 0],
        projector_grid[0, 0, 0],
        atol=1.0e-13,
    )
    assert np.trace(projector_grid[0, 0, 0]).real == pytest.approx(2.0)
    sampled = band_factory(
        model._kfrac_to_kcart(
            np.asarray([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
        )
    )
    assert np.allclose(
        np.linalg.eigvalsh(sampled),
        np.asarray([[2.0, 5.0, 7.0], [2.0, 5.0, 7.0]]),
        atol=1.0e-12,
    )


def test_parallel_map_preserves_deterministic_input_order():
    expected = [(value, value * value) for value in range(32)]
    for threads in (1, 2, 4):
        actual = list(parallel_map(range(32), lambda value: (value, value * value), threads))
        assert actual == expected


def test_memory_limited_threads_respects_request_and_budget():
    assert memory_limited_threads(8, 128, budget_bytes=512) == 4
    assert memory_limited_threads(2, 128, budget_bytes=512) == 2
    assert memory_limited_threads(8, 1024, budget_bytes=512) == 1


def test_execution_context_disables_nested_numba_parallelism_in_workers():
    with ExecutionContext(4):
        allowed = list(parallel_map(range(16), lambda _: numba_parallel_allowed(), 4))
    assert allowed == [False] * 16


def test_execution_context_assigns_total_budget_to_main_numba_phase():
    numba = pytest.importorskip("numba")
    previous_threads = numba.get_num_threads()
    with ExecutionContext(2):
        assert not numba_parallel_allowed()
        previous_policy = set_numba_parallel_allowed(True)
        assert not previous_policy
        assert numba_parallel_allowed()
        assert numba.get_num_threads() == 2
        set_numba_parallel_allowed(False)
        assert numba.get_num_threads() == 1
    assert numba.get_num_threads() == previous_threads


def test_execution_context_cancels_tasks_not_started_after_failure():
    started = []
    lock = Lock()

    def fail_first(value):
        with lock:
            started.append(value)
        if value == 0:
            raise RuntimeError("stop")
        return value

    with pytest.raises(RuntimeError, match="stop"):
        with ExecutionContext(4):
            list(parallel_map(range(100), fail_first, 4))
    assert len(started) <= 4


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


def test_nyquist_detection_accepts_unpadded_2d_vectors():
    shape = (10, 10, 1)

    assert TBAModel.is_nyquist([5, 0], shape)
    assert TBAModel.is_nyquist([0, 5], shape)
    assert TBAModel.is_nyquist([5, 5], shape)
    assert not TBAModel.is_nyquist([1, 0], shape)


def test_collect_hoppings_is_complete_and_independent_of_band_neighbors():
    tba = object.__new__(TBAModel)
    tba.config = SimpleNamespace(
        neighbor=[[1, 0]],
        kdim=2,
        band_calc_num=1,
        real_lattice_vectors=np.eye(2),
        lattice_const=1.0,
    )
    tba.state = SimpleNamespace(k_shape=(4, 4, 1))
    tba.threads = 1
    tba._projected_k_hamiltonians = lambda: (
        np.empty((0, 2)),
        np.empty((0, 1, 1)),
    )
    tba.gen_hopping = lambda r=None: np.asarray(
        [[10 * int(r[0]) + int(r[1])]],
        dtype=np.complex128,
    )

    hoppings = tba.collect_hoppings()
    complete = TBAModel.R_half_rect(tba.state.k_shape)
    expected_residues = {
        tuple(int(row[axis]) % tba.state.k_shape[axis] for axis in range(2))
        for row in complete
    }
    actual_residues = {
        tuple(int(row[axis]) % tba.state.k_shape[axis] for axis in range(2))
        for row in hoppings
        if row != (0, 0, 0)
    }

    assert actual_residues == expected_residues
    assert tba.config.neighbor == [[1, 0]]


def test_collect_hoppings_does_not_fill_an_empty_band_neighbor_selection():
    tba = object.__new__(TBAModel)
    tba.config = SimpleNamespace(
        neighbor=[],
        kdim=2,
        band_calc_num=1,
        real_lattice_vectors=np.eye(2),
        lattice_const=1.0,
    )
    tba.state = SimpleNamespace(k_shape=(4, 4, 1))
    tba.threads = 1
    tba._projected_k_hamiltonians = lambda: (
        np.empty((0, 2)),
        np.empty((0, 1, 1)),
    )
    tba.gen_hopping = lambda r=None: np.asarray([[0.0]], dtype=np.complex128)

    hoppings = tba.collect_hoppings()

    assert tba.config.neighbor == []
    residues = {
        (key[0] % 4, key[1] % 4) for key in hoppings if key != (0, 0, 0)
    }
    assert residues == {
        (int(row[0]) % 4, int(row[1]) % 4)
        for row in TBAModel.R_half_rect(tba.state.k_shape)
    }


def test_empty_band_neighbor_selection_uses_complete_hopping_set():
    tba = object.__new__(TBAModel)
    tba.config = SimpleNamespace(neighbor=[])
    hoppings = {
        (0, 0, 0): np.asarray([[1.0]]),
        (1, 0, 0): np.asarray([[2.0]]),
        (0, 1, 0): np.asarray([[3.0]]),
    }

    actual = tba._band_neighbors(hoppings)

    assert np.array_equal(actual, [[1, 0, 0], [0, 1, 0]])


def test_band_neighbor_selection_does_not_change_complete_hopping_output():
    tba = object.__new__(TBAModel)
    tba.config = SimpleNamespace(
        neighbor=[[-1, 0]],
        kdim=2,
        band_calc_num=1,
        real_lattice_vectors=np.eye(2),
        lattice_const=1.0,
    )
    tba.state = SimpleNamespace(k_shape=(4, 4, 1))
    tba.threads = 1
    tba._projected_k_hamiltonians = lambda: (
        np.empty((0, 2)),
        np.empty((0, 1, 1)),
    )
    tba.gen_hopping = lambda r=None: np.asarray(
        [[10 * int(r[0]) + int(r[1])]],
        dtype=np.complex128,
    )
    hoppings = tba.collect_hoppings()

    selected = tba._hoppings_for_neighbors(
        hoppings,
        np.asarray(tba.config.neighbor, dtype=int),
    )

    assert (-1, 0, 0) not in hoppings
    assert selected.shape == (1, 1, 1)
    assert selected[0, 0, 0] == pytest.approx(-10.0)
    assert len(hoppings) >= 1 + len(TBAModel.R_half_rect(tba.state.k_shape))


def test_hopping_fourier_roundtrip_is_hermitian_on_rectangular_lattice():
    rng = np.random.default_rng(1234)
    shape = (4, 4, 1)
    avec = np.array([[1.0, 0.0], [0.0, np.sqrt(3.0)]])
    config = SimpleNamespace(
        band_calc_num=3,
        real_lattice_vectors=avec,
        lattice_const=1.0,
        dataset_type="synthetic-source",
    )
    tba = object.__new__(TBAModel)
    tba.config = config
    tba.state = SimpleNamespace(k_shape=shape, bloch_sign=-1)

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


def test_wigner_seitz_hopping_representatives_use_cartesian_lattice_metric():
    tba = object.__new__(TBAModel)
    tba.config = SimpleNamespace(
        kdim=3,
        real_lattice_vectors=np.asarray(
            [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
        ),
        lattice_const=1.0,
    )
    tba.state = SimpleNamespace(k_shape=(4, 4, 4))

    representatives = tba._wigner_seitz_representatives((2, 2, 0))

    assert representatives == ((-2, 2, 0), (2, -2, 0))
    lattice = np.asarray(tba.config.real_lattice_vectors)
    assert all(np.linalg.norm(np.asarray(row) @ lattice) == pytest.approx(np.sqrt(2.0)) for row in representatives)
    assert np.linalg.norm(np.asarray([2, 2, 0]) @ lattice) == pytest.approx(np.sqrt(6.0))


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
        dataset_type="synthetic-source",
    )
    tba = object.__new__(TBAModel)
    tba.config = config
    tba.state = SimpleNamespace(k_shape=(4, 4, 1), bloch_sign=-1)

    result = tba.gen_hs_bands({(0, 0, 0): np.array([[1.0]])})

    assert result.k_path.shape == (41, 2)
    assert np.allclose(result.k_path[0], [-0.5, 0.0])
    assert np.allclose(result.k_path[-1], [0.5, 0.0])
    assert result.high_sym_points == [["X", 0], ["G", 20], ["X", 40]]
    assert result.k_axis[-1] == 40


def test_band_path_still_closes_distinct_endpoints():
    assert not periodically_equivalent_kpoint(
        np.array([0.5, 0.5]), np.array([0.0, 0.0])
    )
    assert periodically_equivalent_kpoint(
        np.array([-0.5, 0.0]), np.array([0.5, 0.0])
    )


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


def test_quadratic_m0_uses_full_bloch_fields_and_unwrapped_neighbor_phase():
    config = SimpleNamespace(
        composition_of_b=[[1, 0], [-1, 0]],
        band_calc_num=1,
        M_in=False,
        use_cached_data=[],
        kdim=2,
        k_points=[np.array([-0.25, 0.25]), np.array([0.0])],
        reciprocal_lattice_vectors=np.eye(2),
        lattice_const=1.0,
    )
    band_indices = np.empty((2, 1, 1), dtype=object)
    transforms = np.empty((2, 1, 1), dtype=object)
    for index in np.ndindex(band_indices.shape):
        band_indices[index] = [0]
        transforms[index] = np.eye(1, dtype=np.complex128)
    phase_wavevectors = []
    full_block_calls = []

    def object_grid(factory):
        result = np.empty((2, 1, 1), dtype=object)
        for index in np.ndindex(result.shape):
            result[index] = factory(*index)
        return result

    def full_block(i, j, k):
        full_block_calls.append((i, j, k))
        return np.full((1, 3), i + 1.0, dtype=np.complex128)

    def overlap(left, right, *, phase_wavevector=None, **_kwargs):
        phase_wavevectors.append(np.asarray(phase_wavevector).copy())
        return np.array([[left[0, 0] + 1j * right[0, 0]]])

    state = SimpleNamespace(
        config=config,
        k_shape=(2, 1, 1),
        E_idx=band_indices,
        inner_product=SimpleNamespace(uses_full_bloch_fields=True, overlap=overlap),
        bloch_sign=-1,
        k_indices=lambda: iter(np.ndindex((2, 1, 1))),
        turn_to_bloch=lambda: None,
        gen_matrix_on_kmesh=lambda factory: object_grid(factory),
        get_full_bloch_block=full_block,
        get_block=lambda *_: (_ for _ in ()).throw(
            AssertionError("quadratic M0 must not use periodic nodal blocks")
        ),
        get_transform=lambda: transforms,
    )
    mset = MSet(state, threads=1)

    mset.init_M0()

    assert len(full_block_calls) == 4
    assert len(phase_wavevectors) == 2
    assert np.allclose(phase_wavevectors, [[np.pi, 0.0], [np.pi, 0.0]])
    reverse = mset.get_M0(0, 0, 0, 1)
    assert np.allclose(reverse, mset.mM0[1, 0, 0][0].conj().T)


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
    config = SimpleNamespace(kdim=2, integration_mode="nodal", dataset_type="synthetic-source")
    bundle = InputBundle(
        config=config,
        maxwell=MaxwellProblem.for_components("Ez"),
        bloch_convention=BlochConvention(-1),
        mesh=mesh,
        fields=fields,
        metric_material=np.ones(3),
        energies=energies,
        band_indices=indices,
        inner_band_indices=indices.copy(),
        energy_matrix=np.array([[[[1.0, 2.0]]]]),
    )
    source_fields = np.asarray(bundle.fields[0, 0, 0]).copy()
    state = StateCollection(bundle, threads=1)
    overlap_calls = 0
    original_overlap = state._overlap_matrix

    def counted_overlap(*index):
        nonlocal overlap_calls
        overlap_calls += 1
        return original_overlap(*index)

    state._overlap_matrix = counted_overlap

    _, initially_needs_orth = state.check_orthogonality()
    raw_s = state.S[0, 0, 0].copy()
    state.orthogonalize()
    strict_report, strict_needs_orth = state.check_orthogonality(apply_transform=True)
    mixed_report, mixed_needs_orth = state.check_orthogonality(apply_transform=False)

    assert initially_needs_orth
    assert not strict_needs_orth
    assert np.max(strict_report[..., 3]) < 1e-10
    assert mixed_needs_orth
    assert np.max(mixed_report[..., 1]) < 1e-12
    assert np.max(mixed_report[..., 2]) > 1e-3
    expected_normalization = np.diag(1.0 / np.sqrt(np.real(np.diag(raw_s))))
    assert np.allclose(
        state.normalization_transform[0, 0, 0], expected_normalization
    )
    assert np.allclose(
        state.normalization_transform[0, 0, 0]
        @ state.transform_correction[0, 0, 0],
        state.transform[0, 0, 0],
    )
    assert overlap_calls == 1
    assert np.array_equal(state.S[0, 0, 0], raw_s)
    assert state.fields is not bundle.fields
    assert np.array_equal(bundle.fields[0, 0, 0], source_fields)
    repeated = StateCollection(bundle, threads=1)
    assert np.array_equal(repeated.get_block(0, 0, 0), source_fields)


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


@pytest.mark.parametrize("norm", [0.0, -1.0, np.nan])
def test_projection_rejects_nonpositive_or_nonfinite_basis_norm(norm):
    initializer = object.__new__(StateInitializer)
    initializer.config = SimpleNamespace(
        band_calc_num=1,
        projections=[{"frac_position": [0.0, 0.0], "xaxis_angluar": 0.0, "states": [[1, 0, 1.0]]}],
        real_lattice_vectors=np.eye(2),
        origin=[0.0, 0.0],
        lattice_const=1.0,
        integration_mode="nodal",
    )
    initializer.state = SimpleNamespace(
        extended_mesh=SimpleNamespace(rfunc=lambda *args: np.ones(3)),
        extended_metric_material=np.ones(3),
        compute_backend="python",
        E_idx=_object_grid([0]),
        extended_inner_product=SimpleNamespace(norms=lambda *args, **kwargs: np.array([norm])),
    )

    with pytest.raises(ValueError, match="strictly positive"):
        initializer.projection()


def test_projection_rank_check_is_shared_by_direct_and_cached_a_paths():
    initializer = object.__new__(StateInitializer)
    initializer.config = SimpleNamespace(
        band_calc_num=2,
        inner_window=False,
        projection_rank_tolerance=1.0e-10,
    )

    with pytest.raises(ValueError, match="numerical rank 1"):
        initializer._projection_frame_from_a(
            np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            (0, 0, 0),
        )

    frame = initializer._projection_frame_from_a(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        (0, 0, 0),
    )
    assert np.allclose(frame.conj().T @ frame, np.eye(2))


def test_projection_overlap_is_expressed_in_strict_internal_state_basis():
    state = object.__new__(StateCollection)
    transforms = np.empty((1, 1, 1), dtype=object)
    transforms[0, 0, 0] = np.array(
        [[1.0, 0.25j], [0.0, 0.75]], dtype=np.complex128
    )
    state.get_transform = lambda _zero=False: transforms
    overlap = np.array(
        [[1.0 + 2.0j, 0.5], [-0.25j, 3.0]], dtype=np.complex128
    )

    converted = state.overlap_to_internal_basis((0, 0, 0), overlap)

    assert np.allclose(
        converted,
        transforms[0, 0, 0].conj().T @ overlap,
    )


def test_frozen_projection_requires_outer_membership_and_complement_rank():
    with pytest.raises(ValueError, match=r"Frozen bands \[2\].*not contained"):
        StateInitializer.map_inner_to_local([0, 1], [0, 2], k_index=(0, 0, 0))

    initializer = object.__new__(StateInitializer)
    initializer.config = SimpleNamespace(
        band_calc_num=2,
        projection_rank_tolerance=1.0e-10,
    )
    initializer.state = SimpleNamespace(
        E_idx=_object_grid([0, 1, 2]),
        inner_E_idx=_object_grid([0]),
        k_indices=lambda: iter(((0, 0, 0),)),
    )
    initializer.matA = _object_grid(
        np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    )
    initializer.I_idx = _object_grid(None)
    initializer.O_idx = _object_grid(None)
    initializer.matV = _object_grid(None)

    with pytest.raises(ValueError, match="outer complement"):
        initializer.inner_projection()


def test_cached_v_must_contain_frozen_projector():
    initializer = object.__new__(StateInitializer)
    initializer.config = SimpleNamespace(projection_rank_tolerance=1.0e-10)
    initializer.state = SimpleNamespace(k_indices=lambda: iter(((0, 0, 0),)))
    initializer.I_idx = _object_grid(np.array([0]))
    initializer.matV = _object_grid(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    )

    with pytest.raises(ValueError, match="does not contain the frozen projector"):
        initializer._validate_cached_frozen_containment()


@pytest.mark.parametrize(
    "cached_s",
    [
        np.eye(3),
        np.array([[1.0, 0.2], [0.0, 1.0]]),
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
    ],
)
def test_cached_s_rejects_wrong_shape_nonhermitian_and_nonfinite(tmp_path, cached_s):
    state = _two_band_overlap_state()
    path = tmp_path / "S.txt"
    data = _object_grid(np.asarray(cached_s, dtype=np.complex128))
    save_cell_matrix(path, data, data.shape)
    state.config.use_cached_data = ["S"]
    state.config.S_file = str(path)
    state.config.input_path = lambda value: value

    with pytest.raises(ValueError, match="shape|Hermitian|non-finite"):
        state.check_orthogonality()


def test_s_cache_request_requires_enabled_file():
    state = _two_band_overlap_state()
    state.config.use_cached_data = ["S"]
    state.config.S_file = False
    state.config.input_path = lambda value: None

    with pytest.raises(ValueError, match="S_file is disabled"):
        state.check_orthogonality()


def test_valid_cached_s_is_reused_without_integrating(tmp_path):
    state = _two_band_overlap_state()
    path = tmp_path / "S.txt"
    cached = _object_grid(np.array([[1.0, 0.1], [0.1, 1.0]], dtype=np.complex128))
    save_cell_matrix(path, cached, cached.shape)
    state.config.use_cached_data = ["S"]
    state.config.S_file = str(path)
    state.config.input_path = lambda value: value
    state._overlap_matrix = lambda *args: (_ for _ in ()).throw(
        AssertionError("A valid S cache must not integrate fields")
    )

    report, need_orth = state.check_orthogonality()
    state.orthogonalize()
    strict_report, strict_need = state.check_orthogonality()

    assert need_orth
    assert np.max(report[..., 2]) == pytest.approx(0.1)
    assert not strict_need
    assert np.max(strict_report[..., 3]) < 1.0e-10


def test_cached_v_uses_symmetry_tolerance_for_semiunitarity():
    initializer = object.__new__(StateInitializer)
    initializer.config = SimpleNamespace(
        band_calc_num=1,
        symmetry_constrained=True,
        symmetry_tolerance=5.0e-4,
    )
    initializer.state = SimpleNamespace(
        k_indices=lambda: iter(((0, 0, 0),)),
        E_idx=_object_grid(np.asarray([0], dtype=int)),
    )
    cached = _object_grid(np.asarray([[np.sqrt(1.0 + 2.5e-5)]], dtype=np.complex128))

    initializer._validate_cached_matrix("V", cached, require_semiunitary=True)

    initializer.config.symmetry_constrained = False
    with pytest.raises(ValueError, match="not semi-unitary"):
        initializer._validate_cached_matrix("V", cached, require_semiunitary=True)


def _synthetic_spectrum_model(raw_energies, projected_hamiltonian, basis):
    energies = np.empty((1, 1, 1), dtype=object)
    energies[0, 0, 0] = np.asarray(raw_energies, dtype=float)
    band_indices = np.empty((1, 1, 1), dtype=object)
    band_indices[0, 0, 0] = list(range(len(raw_energies)))
    state = SimpleNamespace(
        E=energies,
        E_idx=band_indices,
        k_shape=(1, 1, 1),
        k_indices=lambda: iter(((0, 0, 0),)),
    )
    config = SimpleNamespace(
        band_calc_num=len(raw_energies),
        symmetry_constrained=True,
        output_basis=basis,
        representation_degeneracy_absolute=1.0e-8,
        representation_degeneracy_relative=1.0e-10,
    )
    model = object.__new__(TBAModel)
    model.config = config
    model.state = state
    model._projected_hamiltonians = np.asarray([projected_hamiltonian], dtype=np.complex128)
    model._projected_k_cart = np.zeros((1, 2), dtype=float)
    return model


def _object_grid(value):
    grid = np.empty((1, 1, 1), dtype=object)
    grid[0, 0, 0] = value
    return grid


def _synthetic_gradient_optimizer(omega_values):
    gradient = object.__new__(Gradient)
    gradient.config = SimpleNamespace(
        use_cached_data=[],
        composition_of_b=[[1.0], [-1.0]],
        mv_diagonal_floor=1.0e-8,
        mv_line_search_max_steps=24,
    )
    gradient.state = SimpleNamespace(k_indices=lambda: iter(((0, 0, 0),)))
    gradient.mset = SimpleNamespace(
        config=gradient.config,
        state=gradient.state,
        update=lambda _u: None,
        get=lambda *_args: np.eye(1, dtype=np.complex128),
    )
    gradient.U = _object_grid(np.eye(1, dtype=np.complex128))
    gradient.G = _object_grid(np.array([[1j]], dtype=np.complex128))
    gradient.omega = np.full(3, np.nan, dtype=float)
    gradient.epsilon = 0.01
    gradient.converged = False
    gradient.last_line_search = None
    gradient._cached_u_loaded = False
    calls = {"calc": 0, "update": 0}

    def calc(is_update=False):
        calls["calc"] += 1
        gradient.G[0, 0, 0] = np.array([[1j]], dtype=np.complex128)

    def update():
        index = min(calls["update"], len(omega_values) - 1)
        gradient.omega = np.array([omega_values[index], 0.0, 0.0], dtype=float)
        calls["update"] += 1

    gradient.calc = calc
    gradient.update = update
    return gradient, calls


def _two_band_overlap_state(metric_material=None, components="Ez", integration_mode="nodal"):
    mesh = Mesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([[0, 1, 2]]),
    )
    fields = _object_grid(
        np.array([[1.0, 0.2, 0.1], [0.4, 1.0, 0.3]], dtype=np.complex128)
    )
    indices = _object_grid([0, 1])
    energies = _object_grid(np.array([1.0, 2.0]))
    config = SimpleNamespace(
        kdim=2,
        integration_mode=integration_mode,
        dataset_type="synthetic-source",
        use_cached_data=[],
        real_lattice_vectors=np.eye(2),
        lattice_const=1.0,
        extension=[1, 1],
    )
    metric = (
        np.ones(mesh.vertices.shape[0])
        if metric_material is None
        else np.asarray(metric_material, dtype=float)
    )
    return StateCollection(
        InputBundle(
            config=config,
            maxwell=MaxwellProblem.for_components(components),
            bloch_convention=BlochConvention(-1),
            mesh=mesh,
            fields=fields,
            metric_material=metric,
            energies=energies,
            band_indices=indices,
            inner_band_indices=indices.copy(),
            energy_matrix=np.array([[[[1.0, 2.0]]]]),
        ),
        threads=1,
    )


def test_state_metric_interface_controls_overlap_norms_and_extension():
    metric = np.array([1.0, 2.0, 4.0])
    state = _two_band_overlap_state(metric, components="Hz")
    block = state.get_block(0, 0, 0)

    expected_overlap = state.inner_product.overlap(block, block)
    assert np.allclose(state._overlap_matrix(0, 0, 0), expected_overlap)
    assert not np.allclose(
        expected_overlap,
        _two_band_overlap_state(np.ones(3))._overlap_matrix(0, 0, 0),
    )

    expected_norms = np.real(np.diag(expected_overlap))
    assert np.allclose(state.inner_product.norms(block.T), expected_norms)

    state.extend([2, 1])
    assert np.array_equal(
        state.extended_metric_material,
        metric[state.space_to_original_mapping],
    )


def test_quadratic_state_overlap_is_hermitian_and_orthogonalizes():
    state = _two_band_overlap_state(
        np.array([1.0, 2.0, 4.0]),
        integration_mode="quadratic",
    )

    _, initially_needs_orth = state.check_orthogonality()
    raw_overlap = np.asarray(state.S[0, 0, 0])
    state.orthogonalize()
    strict_report, still_needs_orth = state.check_orthogonality()

    assert initially_needs_orth
    assert np.allclose(raw_overlap, raw_overlap.conj().T, rtol=0.0, atol=1e-13)
    assert np.min(np.linalg.eigvalsh(raw_overlap)) > 0.0
    assert not still_needs_orth
    assert np.max(strict_report[..., 3]) < 1e-10


def test_quadratic_phase_mass_is_cached_per_wavevector(monkeypatch):
    state = _two_band_overlap_state(
        np.array([1.0, 2.0, 4.0]),
        integration_mode="quadratic",
    )
    block = state.get_block(0, 0, 0)
    wavevector = np.array([1.25, -0.75])
    import pcwannier.compute.integration as integration_module

    original = integration_module._build_phase_weighted_triangle_mass
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(integration_module, "_build_phase_weighted_triangle_mass", counted)

    first = state.inner_product.overlap(block, block, phase_wavevector=wavevector)
    second = state.inner_product.overlap(block, block, phase_wavevector=wavevector.copy())

    assert calls == 1
    assert np.allclose(first, second)
