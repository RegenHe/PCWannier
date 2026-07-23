import numpy as np
import pytest
from types import SimpleNamespace

from pcwannier import BlochConvention
from pcwannier.compute.gradient import Gradient
from pcwannier.compute.context import CalculationContext
from pcwannier.compute.initializer import StateInitializer
from pcwannier.compute.matrix import MSet
from pcwannier.compute.parallel import parallel_map
from pcwannier.compute.state import StateCollection
from pcwannier.data import InputBundle, Mesh
from pcwannier.matrix_io import save_cell_matrix
from pcwannier.maxwell import MaxwellProblem
from pcwannier.compute.tba import TBAModel


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
        symmetry_constrained=True,
        symmetry_output_basis="strict",
        disable_orth=True,
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

    config.symmetry_output_basis = "fem"
    assert np.allclose(ctx.internal_state_coefficients_at(0, 0, 0), correction @ gauge)
    assert np.allclose(ctx.output_state_coefficients_at(0, 0, 0), gauge)

    config.symmetry_constrained = False
    config.disable_orth = True
    assert np.allclose(ctx.output_state_coefficients_at(0, 0, 0), gauge)
    config.disable_orth = False
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


def test_parallel_map_preserves_deterministic_input_order():
    expected = [(value, value * value) for value in range(32)]
    for threads in (1, 2, 4):
        actual = list(parallel_map(range(32), lambda value: (value, value * value), threads))
        assert actual == expected


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
        bloch_convention=BlochConvention(-1, "synthetic"),
        mesh=mesh,
        fields=fields,
        metric_material=np.ones(3),
        energies=energies,
        band_indices=indices,
        inner_band_indices=indices.copy(),
        energy_matrix=np.array([[[[1.0, 2.0]]]]),
    )
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
    assert np.max(mixed_report[..., 2]) > 1e-3
    assert overlap_calls == 1
    assert np.array_equal(state.S[0, 0, 0], raw_s)


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
        extention_mesh=SimpleNamespace(rfunc=lambda *args: np.ones(3)),
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
        symmetry_output_basis=basis,
        disable_orth=True,
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
            bloch_convention=BlochConvention(-1, "synthetic"),
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

    state.extention([2, 1])
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
