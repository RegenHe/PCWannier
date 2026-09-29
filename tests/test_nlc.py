"""Mathematical invariants and main-program integration of NLC."""
from dataclasses import replace
from copy import copy
import json

import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from pcwannier import BlochConvention, load_config, load_input, run_calculation, write_outputs
from pcwannier.cli import RunOptions, _apply_run_options
from pcwannier.compute.state import StateCollection
from pcwannier.data import PeriodicGrid
from pcwannier.nlc import NLCSettings, prepare_longitudinal_bundle
from pcwannier.nlc.completion import NeighborLongitudinalCompletion
from pcwannier.symmetry.representation import build_symmetry_context


def _incar_text(extra=""):
    lines = [
        "dataset_type = mpb", "field_components = full_vector", "primary_field = magnetic",
        "lattice_const = 1", "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
        "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
        "k_points = -0.5:0.25:0.5, -0.5:0.25:0.5, -0.5:0.25:0.5",
        "composition_of_b = 1 0 0, 0 1 0, 0 0 1",
        "wannier_subspace = T + L", "longitudinal_source = nlc",
        "band_window = 0:2", "inner_window = false",
        "dataset_file = H.h5", "E_file = E.h5", "mesh_file = grid.h5", "metric_file = false",
        "symmetry_file = hall:1", "symmetry_constrained = true", "output_basis = strict",
        "extension = 4,4,4", "max_iter = 0", "nlc_max_iter = 4", "nlc_cutoff = 1.2",
        "wannier_file = false", "wannier_figures = false", "band_file = false", "band_figure = false",
        "projections",
        "1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,4]@[1,0,0]",
        "1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,4]@[0,1,0]",
        "1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,4]@[0,0,1]",
        "end", "wannier_targets", "x; 1a; A", "y; 1a; A", "z; 1a; A", "end",
    ]
    overrides = {line.split("=", 1)[0].strip() for line in extra.splitlines() if "=" in line}
    lines = [line for line in lines if line.split("=", 1)[0].strip() not in overrides]
    return "\n".join(lines + [extra]) + "\n"


@pytest.fixture
def bundle(tmp_path):
    shape = (12, 12, 12)
    axes = [np.arange(4)/4 - .5]*3
    kpoints = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    fields = np.empty((len(kpoints), 2) + shape + (3,), dtype=complex)
    energies = np.zeros((len(kpoints), 2))
    for i, k in enumerate(kpoints):
        if np.linalg.norm(k) == 0:
            directions = np.eye(3)[:2]
        else:
            _, _, vh = np.linalg.svd(k[None], full_matrices=True)
            directions = vh[1:]
            energies[i] = np.linalg.norm(2*np.pi*k)**2
        fields[i] = directions[:, None, None, None, :]
    with h5py.File(tmp_path / "grid.h5", "w") as handle:
        handle.attrs["dimension"] = 3
        handle["shape"] = shape
        handle["basis"] = np.eye(3)
        for axis in range(3):
            handle[f"u{axis+1}"] = np.arange(shape[axis])/shape[axis] - .5
    with h5py.File(tmp_path / "H.h5", "w") as handle:
        handle["kpoints"] = kpoints
        handle["H_periodic"] = fields
    with h5py.File(tmp_path / "E.h5", "w") as handle:
        handle["kpoints"] = kpoints
        handle["E"] = energies
    incar = tmp_path / "incar"
    incar.write_text(_incar_text(), encoding="utf-8")
    return load_input(load_config(incar))


def _state(bundle):
    state = StateCollection(bundle)
    state.check_orthogonality()
    state.ensure_identity_transform()
    return state


def _independent_cost(augmented):
    """Real-space projector distance, including all directed BZ seam phases."""
    state = _state(augmented)
    cfg = augmented.config
    score = 0.
    for index in np.ndindex(state.k_shape):
        left = state.get_internal_block(*index)
        for direction, weight in zip(cfg.composition_of_b, cfg.wb):
            raw = np.asarray(index) + np.rint(direction).astype(int)
            wrapped = tuple(raw % np.asarray(state.k_shape))
            seam = (raw - np.asarray(wrapped)) / np.asarray(state.k_shape)
            phase = np.exp(-2j*np.pi*state.bloch_sign*(state.mesh.fractional_vertices @ seam))
            right = state.get_internal_block(*wrapped)*phase[None, :, None]
            overlap = state.inner_product.overlap(left, right, chunk_size=64)
            score += weight*(cfg.band_calc_num - np.linalg.norm(overlap)**2)
    return 2*score / np.prod(state.k_shape)


@pytest.mark.parametrize("sign", [1, -1])
def test_completion_preserves_t_gamma_and_seam_objective(bundle, sign):
    convention = BlochConvention(sign)
    targets = tuple(replace(target, bloch_convention=convention) for target in bundle.symmetry.model.targets)
    context = build_symmetry_context(replace(bundle.symmetry.model, targets=targets, bloch_convention=convention), bundle.config.k_points)
    state = _state(replace(bundle, bloch_convention=convention, symmetry=context))
    snapshot = [state.get_internal_block(*index).copy() for index in np.ndindex(state.k_shape)]
    with threadpool_limits(limits=1):
        completed = prepare_longitudinal_bundle(state)
    result, augmented = completed.result, completed.augmented_bundle
    assert result.transverse_dimension == 2 and result.auxiliary_dimension == 1
    assert result.final_cost <= result.initial_cost + 1e-10
    assert all(b["cost"] <= a["cost"] + 1e-10 for a, b in zip(result.history, result.history[1:]))
    assert _independent_cost(augmented) == pytest.approx(result.final_cost, abs=1e-11)
    for index, expected in zip(np.ndindex(state.k_shape), snapshot):
        assert np.array_equal(augmented.fields[index][:2], expected)
        gram = state.inner_product.overlap(augmented.fields[index], augmented.fields[index])
        assert np.allclose(gram, np.eye(3), atol=1e-10)
        assert np.allclose(np.linalg.eigvalsh(augmented.base_hamiltonians[index])[-2:], state.E[index])
        assert augmented.energies[index][-1] <= 0
        assert np.array_equal(state.get_internal_block(*index), expected)
    gamma = (2, 2, 2)
    assert np.allclose(augmented.fields[gamma].std(axis=1), 0, atol=1e-12)
    assert np.allclose(augmented.base_hamiltonians[gamma], 0, atol=1e-12)
    assert augmented.auxiliary_zero_mode_bands["longitudinal"][gamma] == [0]
    assert result.diagnostics["L_curl_error_in_2pi_units"] < 1e-12


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("filtered", [True, False])
def test_sparse_iteration_geometry_matches_full_vector_links(bundle, sign, filtered):
    """Check all T/L cross terms and updates, including Gamma and BZ seams."""
    config = copy(bundle.config)
    config.band_calc_num = 4
    convention = BlochConvention(sign)
    model = bundle.symmetry.model
    targets = tuple(replace(target, bloch_convention=convention) for target in
                    model.targets + (replace(model.targets[0], name="extra"),))
    context = build_symmetry_context(replace(model, targets=targets, bloch_convention=convention), config.k_points)
    state = _state(replace(bundle, config=config, bloch_convention=convention, symmetry=context))
    with threadpool_limits(limits=1):
        problem = NeighborLongitudinalCompletion(state, NLCSettings(cutoff=1.2, filter_candidates=filtered))
        rng = np.random.default_rng(7253)
        ell = np.zeros((problem.count, problem.ng, problem.nl), complex)
        for i, active in enumerate(problem.active):
            count = problem.nl - int(i == problem.gamma)
            raw = rng.normal(size=(len(active), count)) + 1j*rng.normal(size=(len(active), count))
            frame, _ = np.linalg.qr(raw)
            ell[i, active, int(i == problem.gamma):] = frame
        score = problem.tt_score
        for i in range(problem.count):
            left_l = problem.l_vectors(i, ell)
            columns = []
            for b, item in enumerate(problem.neighbors[i]):
                right_t, right_l = problem.neighbor_vectors(i, item, ell)
                lt = np.einsum('gcl,gcn->ln', left_l.conj(), right_t)
                tl = np.einsum('gcn,gcl->nl', problem.tc[i].conj(), right_l)
                ll = np.einsum('gcl,gcm->lm', left_l.conj(), right_l)
                score += problem.weights[b]*(np.linalg.norm(lt)**2 + np.linalg.norm(tl)**2 + np.linalg.norm(ll)**2)
                candidate_t = problem.transverse_candidate(item)
                if filtered and item[0] == problem.gamma:
                    right_l = right_l[:, :, 1:]
                active = problem.active[i]
                for values in (candidate_t, right_l):
                    columns.append(np.sqrt(problem.weights[b])*np.einsum('gc,gcn->gn', problem.qhat[i, active], values[active]))
            candidates = np.concatenate(columns, axis=1)
            expected_covariance = candidates @ candidates.conj().T
            assert np.allclose(problem.covariance(i, ell), expected_covariance, atol=1e-12, rtol=1e-12)
        expected_cost = 2*(problem.count*np.sum(problem.weights)*(problem.nt+problem.nl)-score)/problem.count
        assert problem.cost(ell) == pytest.approx(expected_cost, abs=1e-11)


def test_nlc_uses_projectors_not_input_eigenvector_gauge(bundle):
    state = _state(bundle)
    rng = np.random.default_rng(781)
    fields = np.empty(state.k_shape, dtype=object)
    for index in np.ndindex(state.k_shape):
        u, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)))
        fields[index] = np.einsum("npc,nm->mpc", bundle.fields[index], u)
    with threadpool_limits(limits=1):
        a = prepare_longitudinal_bundle(state)
        b = prepare_longitudinal_bundle(_state(replace(bundle, fields=fields)))
    assert a.result.final_cost == pytest.approx(b.result.final_cost, abs=1e-10)
    for index in np.ndindex(state.k_shape):
        la, lb = a.augmented_bundle.fields[index][2:], b.augmented_bundle.fields[index][2:]
        overlap = state.inner_product.overlap(la, lb)
        assert np.allclose(overlap.conj().T @ overlap, np.eye(1), atol=1e-9)


def test_rank_zero_gamma_candidate_uses_gradient_fallback_and_eta_scales_only_energies(bundle):
    config = copy(bundle.config)
    config.band_calc_num = 4
    model = bundle.symmetry.model
    targets = model.targets + (replace(model.targets[0], name="extra"),)
    context = build_symmetry_context(replace(model, targets=targets), config.k_points)
    grid = PeriodicGrid(bundle.mesh.shape, np.eye(3), sample_offset=[.37, -.21, .11])
    state = _state(replace(bundle, config=config, symmetry=context, mesh=grid))
    with threadpool_limits(limits=1):
        a = prepare_longitudinal_bundle(state, settings=NLCSettings(cutoff=1.2, max_iterations=0, eta=10))
        b = prepare_longitudinal_bundle(state, settings=NLCSettings(cutoff=1.2, max_iterations=0, eta=20))
    gamma_report = next(item for item in a.result.diagnostics["initial_candidate_ranks"] if item["k"] == [0., 0., 0.])
    assert gamma_report["blocks"][0]["T_candidate_rank"] == 0
    assert gamma_report["blocks"][0]["gradient_trial_fallback"]
    assert a.result.final_cost == pytest.approx(b.result.final_cost, abs=1e-12)
    assert _independent_cost(a.augmented_bundle) == pytest.approx(a.result.final_cost, abs=1e-11)
    for index in np.ndindex(state.k_shape):
        assert np.allclose(a.augmented_bundle.fields[index], b.augmented_bundle.fields[index])
        assert np.allclose(a.augmented_bundle.energies[index][2:], 2*b.augmented_bundle.energies[index][2:])
        assert np.count_nonzero(a.augmented_bundle.zero_modes[index]) == (3 if index == (2, 2, 2) else 0)


def test_gamma_preserves_positive_t_between_the_two_zero_modes(bundle):
    config = copy(bundle.config)
    config.band_calc_num = 5
    model = bundle.symmetry.model
    targets = model.targets + tuple(replace(model.targets[0], name=f"extra{j}") for j in range(2))
    context = build_symmetry_context(replace(model, targets=targets), config.k_points)
    fields = np.empty(bundle.fields.shape, dtype=object)
    energies = np.empty(bundle.fields.shape, dtype=object)
    indices = np.empty(bundle.fields.shape, dtype=object)
    energy_matrix = np.empty(bundle.fields.shape + (3,))
    phase = np.exp(2j*np.pi*bundle.mesh.fractional_vertices[:, 0])
    for index in np.ndindex(bundle.fields.shape):
        k = np.asarray([config.k_points[axis][index[axis]] for axis in range(3)])
        q = k + [1, 0, 0]
        _, _, vh = np.linalg.svd(q[None], full_matrices=True)
        positive = phase[:, None]*vh[1]
        fields[index] = np.stack((bundle.fields[index][0], positive, bundle.fields[index][1]))
        energies[index] = np.asarray([bundle.energies[index][0], 10 + np.linalg.norm(q)**2, bundle.energies[index][1]])
        indices[index] = [0, 2, 1]
        energy_matrix[index] = [energies[index][0], energies[index][2], energies[index][1]]
    source = replace(bundle, config=config, symmetry=context, fields=fields, energies=energies,
                     band_indices=indices, energy_matrix=energy_matrix)
    state = _state(source)
    with threadpool_limits(limits=1):
        completed = prepare_longitudinal_bundle(state, settings=NLCSettings(cutoff=1.2, max_iterations=0))
    gamma = (2, 2, 2)
    actual = completed.augmented_bundle.fields[gamma]
    assert np.array_equal(actual[:3], fields[gamma])
    assert np.allclose(completed.augmented_bundle.base_hamiltonians[gamma][:3, :3], np.diag([0., 11., 0.]))
    assert np.count_nonzero(completed.augmented_bundle.zero_modes[gamma]) == 3
    assert np.linalg.norm(actual[1].std(axis=0)) > .5


def test_nlc_main_pipeline_and_json_output(bundle, tmp_path):
    bundle.config.use_cached_data = ["S", "D", "M", "V", "A"]
    with threadpool_limits(limits=1):
        result = run_calculation(bundle, threads=1)
    assert result.nlc is not None and result.etbc is None
    assert result.config.use_cached_data == []
    assert bundle.config.use_cached_data == ["S", "D", "M", "V", "A"]
    assert result.A[(0, 0, 0)].shape == (3, 3)
    assert np.allclose(result.wannier_norms, 1, atol=1e-10)
    assert result.hopping_reconstruction_diagnostics.max_eigenvalue_error < 1e-10
    # Omega_I does not need diagonal-overlap phases; this toy's localization can
    # stop safely if an initial M_nn vanishes exactly at a mesh boundary.
    invariant_spread = sum(
        2*weight*(3 - np.linalg.norm(result.M0[index][b])**2)
        for index in np.ndindex(result.M0.shape)
        for b, weight in enumerate(result.config.wb[:len(result.config.wb)//2])
    ) / np.prod(result.M0.shape)
    assert invariant_spread == pytest.approx(result.nlc.omega_i, abs=1e-10)
    write_outputs(result, out_dir=tmp_path / "outputs")
    report = json.loads((tmp_path / "outputs" / "nlc.json").read_text(encoding="utf-8"))
    assert report["method"] == "Neighbor-based Longitudinal Completion (NLC)"
    assert report["diagnostics"]["source_T_subspace_unchanged"]
    assert report["final_cost"] == pytest.approx(result.nlc.final_cost)


@pytest.mark.parametrize("extra, match", [
    ("nlc_eta = 0", "eta must be positive"),
    ("nlc_mixing = 1.1", "mixing must lie"),
    ("nlc_max_iter = 1.5", "non-negative integer"),
    ("nlc_max_iter = -1", "non-negative integer"),
    ("nlc_filter_candidates = maybe", "Boolean incar"),
    ("inner_window = 0:1", "freezes all selected"),
    ("gamma_zero_regularization = true", "must be false for NLC"),
    ("symmetry_constrained = false", "sewing constraints"),
    ("M_in = true", "M_in=false"),
])
def test_invalid_nlc_configuration(tmp_path, extra, match):
    path = tmp_path / "incar"
    path.write_text(_incar_text(extra), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_nlc_rejects_unsupported_metric_and_fourier_cutoff(bundle):
    with pytest.raises(ValueError, match="mu=1"):
        prepare_longitudinal_bundle(_state(replace(bundle, metric_material=np.full(bundle.mesh.point_count, 2.))))
    with pytest.raises(ValueError, match="reduce nlc_cutoff"):
        prepare_longitudinal_bundle(_state(bundle), settings=NLCSettings(cutoff=10))
    with pytest.raises(ValueError, match="increase nlc_cutoff"):
        prepare_longitudinal_bundle(_state(bundle), settings=NLCSettings(cutoff=.1))
    fields = bundle.fields.copy()
    fields[(0, 0, 0)] = .9*fields[(0, 0, 0)]
    with pytest.raises(ValueError, match="internally orthonormal T"):
        prepare_longitudinal_bundle(_state(replace(bundle, fields=fields)))


def test_nlc_clears_file_channels_and_cache_overrides(tmp_path):
    path = tmp_path / "incar"
    path.write_text(_incar_text("\n".join([
        "longitudinal_field_file = nonexistent.h5", "longitudinal_energy_file = nonexistent.h5",
        "longitudinal_band_window = 0:4", "invert_longitudinal_energies = true",
        "use_cached_data = S,D,M,A,V",
    ])), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.longitudinal_field_file is False and cfg.longitudinal_band_window is None
    assert cfg.invert_longitudinal_energies is False and cfg.use_cached_data == []
    overridden = _apply_run_options(cfg, RunOptions(None, True, False, None))
    assert overridden.use_cached_data == []
