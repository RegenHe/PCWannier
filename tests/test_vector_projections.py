from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import h5py
import numpy as np
import pytest

from pcwannier import BlochConvention, load_input, run_calculation
from pcwannier.compute.integration import create_metric_inner_product
from pcwannier.compute.initializer import StateBases
from pcwannier.compute.runner import _needs_streaming_projection_seed
from pcwannier.config import IncarParser, load_config
from pcwannier.data import PeriodicGrid
from pcwannier.projections import (
    LocalFrame3D,
    ProjectionRecord3D,
    TrialLinearCombination,
    VectorHydrogenicOrbital,
    real_spherical_harmonic,
)
from pcwannier.symmetry import cartesian_field_matrix
from pcwannier.symmetry.bloch import build_bloch_symmetry_action
from pcwannier.symmetry.representation import build_symmetry_context
import pcwannier.compute.vector_trials as vector_trials_module
from pcwannier.compute.wannier import _uniform_grid_wannier_sum, generate_wannier


def test_skew_lattice_minimum_image_uses_cartesian_metric():
    lattice = np.asarray(
        (
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        )
    )
    fractional = np.asarray([[0.6, 0.6, -0.4]])
    component_wrapped = fractional - np.floor(fractional + 0.5)

    reduced = vector_trials_module._minimum_image_fractional(
        fractional,
        lattice,
        np.ones(3),
    )

    assert np.allclose(np.mod(reduced - fractional, 1.0), 0.0, atol=1.0e-14)
    assert np.linalg.norm(reduced @ lattice) < np.linalg.norm(
        component_wrapped @ lattice
    )


@pytest.mark.parametrize("bloch_sign", [-1, 1])
def test_translation_bloch_fft_matches_direct_sum(bloch_sign):
    k_points = (
        np.arange(3, dtype=float) / 3.0 - 1.0 / 3.0,
        np.arange(2, dtype=float) / 2.0 - 0.25,
        np.asarray([0.125]),
    )
    k_shape = tuple(len(axis) for axis in k_points)
    rng = np.random.default_rng(9021 + bloch_sign)
    images = rng.normal(size=k_shape + (5, 3)) + 1j * rng.normal(
        size=k_shape + (5, 3)
    )

    transformed = vector_trials_module._translation_bloch_fft(
        images, k_points, bloch_sign
    )
    translation_axes = vector_trials_module._born_von_karman_translation_axes(
        k_points
    )
    for k_index in np.ndindex(k_shape):
        k_fractional = np.asarray(
            [k_points[axis][k_index[axis]] for axis in range(3)]
        )
        expected = np.zeros(images.shape[-2:], dtype=np.complex128)
        for translation_index in np.ndindex(k_shape):
            translation = np.asarray(
                [
                    translation_axes[axis][translation_index[axis]]
                    for axis in range(3)
                ]
            )
            expected += images[translation_index] * np.exp(
                bloch_sign * 2j * np.pi * np.dot(k_fractional, translation)
            )
        assert np.allclose(
            transformed[k_index], expected, atol=2.0e-13, rtol=2.0e-13
        )


def test_local_frame_and_fixed_vector_orbitals():
    frame = LocalFrame3D.from_xz([1.0, 1.0, 0.0], [0.0, 0.0, 2.0])
    assert np.allclose(frame.matrix.T @ frame.matrix, np.eye(3), atol=1.0e-14)
    assert np.linalg.det(frame.matrix) == pytest.approx(1.0)

    orbital = VectorHydrogenicOrbital(1, 0, 0, 2.0, [0.0, 0.0, 4.0])
    values = orbital.evaluate(
        np.asarray([[0.1, 0.0, 0.0], [0.2, 0.1, 0.0]]),
        frame,
        1.0,
        lambda *_: lambda radius, zeta: np.exp(-zeta * radius),
    )
    direction = frame.ez
    assert np.allclose(np.cross(values.real, direction), 0.0, atol=1.0e-14)


def test_hydrogenic_1s_radial_uses_the_same_zeta_convention_as_higher_states():
    radius = np.asarray([0.0, 0.1, 0.4])
    zeta = 3.5

    values = StateBases.Radial(1, 0)(radius, zeta)

    assert np.allclose(
        values,
        2.0 * zeta ** 1.5 * np.exp(-zeta * radius),
        atol=1.0e-14,
        rtol=1.0e-14,
    )


def test_real_spherical_harmonic_px_py_convention():
    theta = np.asarray([np.pi / 2, np.pi / 2])
    phi = np.asarray([0.0, np.pi / 2])
    px = real_spherical_harmonic(1, 1, theta, phi)
    py = real_spherical_harmonic(1, -1, theta, phi)
    assert px[0] > 0.0
    assert py[1] > 0.0
    assert abs(px[1]) < 1.0e-14
    assert abs(py[0]) < 1.0e-14


def test_parse_3d_vector_projection_and_complex_combination(tmp_path):
    parser = IncarParser(tmp_path / "incar")
    records = parser._parse_projections(
        "3c; [0.5,0.5,0.0]; (z=[0,0,1], x=[1,0,0]); "
        "{[2,1,1,10]@[1,0,0],[2,1,-1,10]@[0,1,0]}{1/sqrt(2),i/sqrt(2)}"
    )
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, ProjectionRecord3D)
    assert record.wyckoff == "3c"
    assert len(record.states) == 1
    assert isinstance(record.states[0], TrialLinearCombination)
    assert record.states[0].coefficients == pytest.approx(
        (1.0 / np.sqrt(2), 1.0j / np.sqrt(2))
    )
    assert [orbital.m for orbital in record.states[0].orbitals] == [1, -1]


@pytest.mark.parametrize(
    "text,match",
    [
        ("1a; [0,0,0]; (z=[0,0,1], x=[0,0,2]); [1,0,0,2]@[1,0,0]", "parallel"),
        ("1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [2,1,2,2]@[1,0,0]", "-l <= m <= l"),
        ("1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,2]@[0,0,0]", "non-zero"),
    ],
)
def test_invalid_3d_projection_inputs(tmp_path, text, match):
    parser = IncarParser(tmp_path / "incar")
    with pytest.raises(ValueError, match=match):
        parser._parse_projections(text)


def test_pm3m_wyckoff_projection_target_binding(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
                "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
                "k_points = 0:1:1, 0:1:1, 0:1:1",
                "composition_of_b = 1 0 0, 0 1 0, 0 0 1",
                "dataset_type = mpb",
                "field_components = full_vector",
                "primary_field = magnetic",
                "dataset_file = H.h5",
                "mesh_file = grid.h5",
                "E_file = E.h5",
                "metric_file = false",
                "band_window = 0:3",
                "extension = 1,1,1",
                "wannier_figures = false",
                "symmetry_file = Pm-3m",
                "projections",
                "3c; [0.5,0.5,0.0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]",
                "end",
                "wannier_targets",
                "center_A2g_3c; 3c; A2g",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(incar)

    assert len(config.projection_target_bindings) == 1
    assert config.band_calc_num == 3
    target = config.symmetry_context.model.target("center_A2g_3c")
    assert target.multiplicity == 3
    assert target.site_irrep.dimension == 1
    positions = {
        tuple(np.round(point.position, 12)) for point in target.orbit.points
    }
    assert positions == {
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    }


def test_etbc_config_uses_targets_without_longitudinal_files(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
                "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
                "k_points = 0:1:1, 0:1:1, 0:1:1",
                "composition_of_b = 1 0 0, 0 1 0, 0 0 1",
                "dataset_type = mpb",
                "field_components = full_vector",
                "primary_field = magnetic",
                "dataset_file = H.h5",
                "mesh_file = grid.h5",
                "E_file = E.h5",
                "metric_file = false",
                "wannier_subspace = T + L",
                "longitudinal_source = etbc",
                "etbc_auxiliary_eigenvalue = 0",
                "etbc_rank_tolerance = 1e-9",
                "band_window = 0:2",
                "inner_window = false",
                "extension = 1,1,1",
                "wannier_figures = false",
                "symmetry_file = Pm-3m",
                "symmetry_constrained = true",
                "projections",
                "3c; [0.5,0.5,0.0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]",
                "end",
                "wannier_targets",
                "center_A2g_3c; 3c; A2g",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(incar)

    assert config.wannier_subspace == "T+L"
    assert config.longitudinal_source == "etbc"
    assert config.band_calc_num == 3
    assert config.longitudinal_field_file is False
    assert config.longitudinal_band_window is None
    assert config.etbc_auxiliary_eigenvalue == pytest.approx(0.0)
    assert config.etbc_rank_tolerance == pytest.approx(1.0e-9)
    target = config.symmetry_context.model.target("center_A2g_3c")
    record = config.projection_target_bindings[0].projection
    directions = []
    for point in target.orbit.points:
        matrix = cartesian_field_matrix(
            point.representative_operation,
            config.real_lattice_vectors,
            config.maxwell_problem.symmetry_field_kind,
        )
        direction = matrix @ record.frame.ez
        directions.append(direction)
        # A2g@3c is the face-normal axial trial.  The coset representative
        # must rotate the complete representative function, so the generated
        # direction follows the zero coordinate of each face center.
        assert int(np.argmax(np.abs(direction))) == int(
            np.argmin(np.abs(point.position))
        )
    assert np.allclose(
        np.sort(np.abs(np.asarray(directions)), axis=0),
        np.sort(np.eye(3), axis=0),
        atol=1.0e-14,
    )


def test_3d_projection_target_center_is_reduced_modulo_lattice(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
                "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
                "k_points = 0:1:1, 0:1:1, 0:1:1",
                "composition_of_b = 1 0 0, 0 1 0, 0 0 1",
                "dataset_type = mpb",
                "field_components = full_vector",
                "primary_field = magnetic",
                "dataset_file = H.h5",
                "mesh_file = grid.h5",
                "E_file = E.h5",
                "metric_file = false",
                "band_window = 0:8",
                "extension = 1,1,1",
                "wannier_figures = false",
                "symmetry_file = hall:509",
                "projections",
                "4b; [-0.125,-0.125,-0.125]; (z=[1,1,1], x=[1,-1,0]); "
                "[1,0,0,5]@[1,-1,0]; [1,0,0,5]@[1,1,-2]",
                "end",
                "wannier_targets",
                "center_E_4b; 4b; E",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(incar)

    record = config.projection_target_bindings[0].projection
    target = config.symmetry_context.model.target("center_E_4b")
    assert np.allclose(record.frac_position, [-0.125, -0.125, -0.125])
    assert any(
        np.allclose(point.position, [0.875, 0.875, 0.875])
        for point in target.orbit.points
    )
    assert target.multiplicity == 4


def test_boundary_site_p_tensor_trial_covariance_uses_local_sample_cloud(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 0 0.5 0.5, 0.5 0 0.5, 0.5 0.5 0",
                "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
                "k_points = 0:1:1, 0:1:1, 0:1:1",
                "composition_of_b = 1 0 0, 0 1 0, 0 0 1, 1 1 1",
                "dataset_type = mpb",
                "field_components = full_vector",
                "primary_field = magnetic",
                "dataset_file = H.h5",
                "mesh_file = grid.h5",
                "E_file = E.h5",
                "metric_file = false",
                "band_window = 0:3",
                "extension = 1,1,1",
                "wannier_figures = false",
                "symmetry_file = hall:512",
                "projections",
                "4b; [0.5,0.5,0.5]; (z=[0,0,1], x=[1,0,0]); "
                "{[2,1,-1,10]@[0,0,1],[2,1,0,10]@[0,1,0]}{1,1}; "
                "{[2,1,0,10]@[1,0,0],[2,1,1,10]@[0,0,1]}{1,1}; "
                "{[2,1,1,10]@[0,1,0],[2,1,-1,10]@[1,0,0]}{1,1}",
                "end",
                "wannier_targets",
                "center_T1_4b; 4b; T1",
                "end",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(incar)
    lattice = np.asarray(config.real_lattice_vectors, dtype=float)
    grid = PeriodicGrid((10, 10, 10), lattice)
    state = SimpleNamespace(
        config=config,
        mesh=grid,
        maxwell=config.maxwell_problem,
    )

    _, diagnostics = vector_trials_module.prepare_vector_trial_targets(
        state, config.symmetry_context
    )

    assert len(diagnostics) == 1
    assert diagnostics[0].closure_residual < 1.0e-10
    assert diagnostics[0].max_residual < 1.0e-10


def test_sg213_vector_trials_obey_full_nonsymmorphic_bloch_covariance(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
                "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
                "k_points = -0.5:0.5:0.5, -0.5:0.5:0.5, -0.5:0.5:0.5",
                "composition_of_b = 1 0 0, 0 1 0, 0 0 1",
                "dataset_type = mpb",
                "field_components = full_vector",
                "primary_field = magnetic",
                "dataset_file = H.h5",
                "mesh_file = grid.h5",
                "E_file = E.h5",
                "metric_file = false",
                "band_window = 0:8",
                # The projection must depend on the BvK k mesh, not this
                # unrelated real-space output extension.
                "extension = 1,1,1",
                "wannier_figures = false",
                "symmetry_file = hall:509",
                "projections",
                "4b; [-0.125,-0.125,-0.125]; (z=[1,1,1], x=[1,-1,0]); "
                "[1,0,0,5]@[1,0,0]; [1,0,0,5]@[0,1,0]",
                "end",
                "wannier_targets",
                "center_E_4b; 4b; E",
                "end",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(incar)
    grid = PeriodicGrid((8, 8, 8), np.eye(3))
    inner_product = create_metric_inner_product(
        grid, np.ones(grid.point_count), mode="nodal", backend="python"
    )
    context = config.symmetry_context
    state = SimpleNamespace(
        config=config,
        mesh=grid,
        symmetry=context,
        bloch_sign=context.model.bloch_convention.sign,
        maxwell=config.maxwell_problem,
        inner_product=inner_product,
        configured_threads=1,
    )
    context, _ = vector_trials_module.prepare_vector_trial_targets(state, context)
    state.symmetry = context
    action = build_bloch_symmetry_action(
        grid,
        grid.fractional_vertices,
        config.real_lattice_vectors,
        bloch_sign=state.bloch_sign,
        tolerance=config.symmetry_tolerance,
    )
    shape = tuple(len(axis) for axis in config.k_points)
    fft_trials = vector_trials_module.build_vector_bloch_trial_grid(
        state, context=context, workspace_bytes=8 << 20
    )
    source = vector_trials_module.build_vector_bloch_trial_source(
        state, context=context, workspace_bytes=8 << 20
    )
    streamed = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        streamed[index] = np.empty_like(fft_trials[index])
    for start, stop, chunk in source.iter_raw_chunks():
        for index in np.ndindex(shape):
            streamed[index][start:stop] = chunk[index]
    for index in np.ndindex(shape):
        norms = inner_product.norms(np.swapaxes(streamed[index], 1, 2))
        streamed[index] /= np.sqrt(norms)[None, :, None]
        assert np.allclose(streamed[index], fft_trials[index], atol=2.0e-13)

    for source_index in ((0, 0, 0), (1, 1, 1)):
        source_k = np.asarray(
            [config.k_points[axis][source_index[axis]] for axis in range(3)]
        )
        source = vector_trials_module.build_vector_bloch_trials(
            state, source_index, context=context
        )
        assert np.allclose(
            fft_trials[source_index], source, atol=2.0e-13, rtol=2.0e-13
        )
        cache = {source_index: source}
        source_rows = np.swapaxes(source, 0, 1)
        flat = np.ravel_multi_index(source_index, shape)
        for operation_index, operation in enumerate(context.model.group.operations):
            mapping = context.k_mappings[operation_index][flat]
            target = cache.get(mapping.target_k_index)
            if target is None:
                target = vector_trials_module.build_vector_bloch_trials(
                    state, mapping.target_k_index, context=context
                )
                cache[mapping.target_k_index] = target
            dmat = context.target_matrix(operation_index, source_k)
            expected = np.einsum("pjc,ji->pic", target, dmat, optimize=True)
            transformed = np.swapaxes(
                action.apply_full_bloch(
                    source_rows,
                    operation,
                    source_k,
                    state.maxwell.symmetry_field_kind,
                    time_reversal=state.maxwell.apply_time_reversal,
                ),
                0,
                1,
            )
            assert np.allclose(transformed, expected, atol=2.0e-12, rtol=2.0e-12)

        config.extension = [7, 7, 7]
        extended_output_trial = vector_trials_module.build_vector_bloch_trials(
            state, source_index, context=context
        )
        assert np.allclose(extended_output_trial, source, atol=1.0e-14, rtol=1.0e-14)


@pytest.mark.parametrize("bloch_sign", [-1, 1])
@pytest.mark.parametrize("case", ["fcc_p", "hex_d", "shifted_s"])
def test_angular_vector_bloch_trials_preserve_space_group_covariance(
    tmp_path, case, bloch_sign
):
    k_mesh = "-0.5:0.25:0.5"
    if case == "fcc_p":
        lattice = "0 0.5 0.5, 0.5 0 0.5, 0.5 0.5 0"
        group, wyckoff, position, irrep = "hall:512", "4b", "[0.5,0.5,0.5]", "T1"
        states = (
            "{[2,1,-1,5]@[0,0,1],[2,1,0,5]@[0,1,0]}{1,1}; "
            "{[2,1,0,5]@[1,0,0],[2,1,1,5]@[0,0,1]}{1,1}; "
            "{[2,1,1,5]@[0,1,0],[2,1,-1,5]@[1,0,0]}{1,1}"
        )
    elif case == "hex_d":
        lattice = "1 0 0, -0.5 sqrt(3)/2 0, 0 0 1"
        # This symbol also names a bundled 2D plane group. The 3D parser must
        # select space group 183 (Hall 477) rather than that plane-group YAML.
        group, wyckoff, position, irrep = "P6mm", "1a", "[0,0,0]", "E2"
        # Unequal input amplitudes must be normalized locally, before applying
        # the common Bloch normalization of this two-dimensional irrep.
        states = "{[3,2,2,8]@[0,0,1]}{2}; [3,2,-2,8]@[0,0,1]"
    else:
        lattice = "1 0 0, 0 1 0, 0 0 1"
        group, wyckoff, position, irrep = "Pm-3m", "1a", "[0,0,0]", "T1g"
        states = "[1,0,0,5]@[1,0,0]; [1,0,0,5]@[0,1,0]; [1,0,0,5]@[0,0,1]"
        k_mesh = "-0.375:0.25:0.625"
    incar = tmp_path / "incar"
    incar.write_text("\n".join([
        "lattice_const = 1", "real_lattice_vectors = " + lattice,
        "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
        f"k_points = {k_mesh}, {k_mesh}, {k_mesh}",
        "composition_of_b = 1 0 0, 0 1 0, 0 0 1, 1 1 0, 1 1 1",
        "dataset_type = mpb", "field_components = full_vector", "primary_field = magnetic",
        "dataset_file = H.h5", "mesh_file = grid.h5", "E_file = E.h5", "metric_file = false",
        "band_window = 0:8", "extension = 1,1,1", "wannier_figures = false",
        "symmetry_file = " + group, "symmetry_constrained = true",
        "projections", f"{wyckoff}; {position}; (z=[0,0,1], x=[1,0,0]); {states}",
        "end", "wannier_targets", f"trial; {wyckoff}; {irrep}", "end",
    ]), encoding="utf-8")
    config = load_config(incar)
    convention = BlochConvention(bloch_sign)
    model = replace(config.symmetry_context.model, bloch_convention=convention,
        targets=tuple(replace(target, bloch_convention=convention)
                      for target in config.symmetry_context.model.targets))
    context = build_symmetry_context(model, config.k_points)
    grid = PeriodicGrid((8, 8, 8), np.asarray(config.real_lattice_vectors))
    inner = create_metric_inner_product(
        grid, np.ones(grid.point_count), mode="nodal", backend="python"
    )
    state = SimpleNamespace(config=config, mesh=grid, maxwell=config.maxwell_problem,
        bloch_sign=bloch_sign, inner_product=inner, configured_threads=1)
    context, reports = vector_trials_module.prepare_vector_trial_targets(state, context)
    state.symmetry = context
    assert max(report.max_residual for report in reports) < 1.0e-12
    trials = vector_trials_module.build_vector_bloch_trial_grid(state, workspace_bytes=8 << 20)
    action = build_bloch_symmetry_action(grid, grid.fractional_vertices,
        config.real_lattice_vectors, bloch_sign=bloch_sign, tolerance=config.symmetry_tolerance)
    for index in ((0, 0, 0), (2, 2, 2), (1, 2, 3)):
        direct = vector_trials_module.build_vector_bloch_trials(state, index)
        assert np.allclose(direct, trials[index], atol=2.0e-13)
        k = np.array([config.k_points[a][index[a]] for a in range(3)])
        translated_state = SimpleNamespace(**vars(state))
        translated_state.mesh = SimpleNamespace(
            vertices=grid.vertices + np.asarray(config.real_lattice_vectors[0])
        )
        translated = vector_trials_module.build_vector_bloch_trials(translated_state, index)
        assert np.allclose(translated, np.exp(bloch_sign * 2j * np.pi * k[0]) * direct,
            atol=2.0e-13)
        flat = np.ravel_multi_index(index, trials.shape)
        for gi, operation in enumerate(context.model.group.operations):
            mapping = context.k_mappings[gi][flat]
            expected = np.einsum("pjc,ji->pic", trials[mapping.target_k_index],
                context.target_matrix(gi, k))
            transformed = action.apply_full_bloch(direct.swapaxes(0, 1), operation, k,
                state.maxwell.symmetry_field_kind,
                time_reversal=state.maxwell.apply_time_reversal).swapaxes(0, 1)
            assert np.linalg.norm(transformed - expected) / np.linalg.norm(expected) < 1.0e-12

    # Streaming A must use precisely the same block normalization as direct
    # and FFT-materialized trials, including at generic k where norms differ.
    state.k_shape = trials.shape
    state.E_idx = np.empty(trials.shape, dtype=object)
    for index in np.ndindex(trials.shape):
        state.E_idx[index] = [0, 1]
    state.k_indices = lambda: np.ndindex(trials.shape)
    rows = np.zeros((2, grid.point_count, 3), dtype=np.complex128)
    rows[0, :, 0] = 1.0
    rows[1, :, 1] = 1.0
    state.get_block = lambda *_: rows
    state.get_phase = lambda *index: np.exp(bloch_sign * 2j * np.pi * (
        grid.fractional_vertices @ np.array([config.k_points[a][index[a]] for a in range(3)])))
    state.overlap_to_internal_basis = lambda _index, values: values
    seed = vector_trials_module.build_vector_bloch_trial_source(state).projection_seed()
    index = (1, 2, 3)
    expected = inner.overlap(rows * state.get_phase(*index)[None, :, None],
        trials[index].swapaxes(0, 1))
    assert np.allclose(seed.matrices[index], expected, atol=2.0e-13)


def _write_pm3m_vector_binding_incar(
    path: Path,
    projection_lines: list[str],
    target_lines: list[str] | None,
) -> None:
    lines = [
        "lattice_const = 1",
        "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
        "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
        "k_points = 0:1:1, 0:1:1, 0:1:1",
        "composition_of_b = 1 0 0, 0 1 0, 0 0 1",
        "dataset_type = mpb",
        "field_components = full_vector",
        "primary_field = magnetic",
        "dataset_file = H.h5",
        "mesh_file = grid.h5",
        "E_file = E.h5",
        "metric_file = false",
        "band_window = 0:3",
        "extension = 1,1,1",
        "wannier_figures = false",
        "symmetry_file = Pm-3m",
        "projections",
        *projection_lines,
        "end",
    ]
    if target_lines is not None:
        lines.extend(("wannier_targets", *target_lines, "end"))
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.parametrize(
    ("projection_lines", "target_lines", "match"),
    [
        (
            ["1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]"],
            None,
            "matching wannier_targets",
        ),
        (
            [
                "1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]",
                "1b; [0.5,0.5,0.5]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]",
            ],
            ["center_A2g_1a; 1a; A2g"],
            "same number of records",
        ),
        (
            ["1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]"],
            ["center_A2g_1b; 1b; A2g"],
            "does not match",
        ),
        (
            ["1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); [1,0,0,5]@[0,0,1]"],
            ["center_T1g_1a; 1a; T1g"],
            "defines 1 functions",
        ),
    ],
)
def test_invalid_3d_projection_target_bindings(
    tmp_path,
    projection_lines,
    target_lines,
    match,
):
    incar = tmp_path / "incar"
    _write_pm3m_vector_binding_incar(incar, projection_lines, target_lines)
    with pytest.raises(ValueError, match=match):
        load_config(incar)


def test_synthetic_3d_vector_projection_to_wannier_and_hopping(tmp_path):
    shape = (4, 4, 4)
    kpoints = np.zeros((1, 3), dtype=float)
    with h5py.File(tmp_path / "grid.h5", "w") as handle:
        handle.attrs["dimension"] = 3
        handle.create_dataset("shape", data=np.asarray(shape, dtype=np.int64))
        handle.create_dataset("basis", data=np.eye(3))
        for axis in range(3):
            handle.create_dataset(
                f"u{axis + 1}", data=np.arange(shape[axis]) / shape[axis] - 0.5
            )
    with h5py.File(tmp_path / "E.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("E", data=np.ones((1, 3), dtype=float))
    fields = np.zeros((1, 3) + shape + (3,), dtype=np.complex128)
    for component in range(3):
        fields[0, component, ..., component] = 1.0
    with h5py.File(tmp_path / "H.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("H_periodic", data=fields)

    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "dataset_type = mpb",
                "field_components = full_vector",
                "primary_field = magnetic",
                "lattice_const = 1",
                "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
                "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
                "k_points = 0:1:1, 0:1:1, 0:1:1",
                "composition_of_b = 1 0 0, 0 1 0, 0 0 1",
                "band_window = 0:3",
                "dataset_file = ./H.h5",
                "mesh_file = ./grid.h5",
                "metric_file = false",
                "E_file = ./E.h5",
                "extension = 1,1,1",
                "max_iter = 0",
                "wannier_figures = false",
                "symmetry_file = Pm-3m",
                "symmetry_constrained = true",
                "symmetry_validate_wannier = true",
                "symmetry_minimum_retained_norm = 0.4",
                "projections",
                "1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); "
                "[1,0,0,4]@[1,0,0]; [1,0,0,4]@[0,1,0]; [1,0,0,4]@[0,0,1]",
                "end",
                "wannier_targets",
                "center_T1g_1a; 1a; T1g",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    bundle = load_input(load_config(incar))
    field_snapshot = {
        index: np.asarray(bundle.fields[index]).copy()
        for index in np.ndindex(bundle.fields.shape)
    }
    energy_snapshot = np.asarray(bundle.energy_matrix).copy()
    metric_snapshot = np.asarray(bundle.metric_material).copy()

    result = run_calculation(bundle, threads=1)
    repeated = run_calculation(bundle, threads=1)

    wannier = result.wanniers[(0, 0, 0)]
    assert wannier.shape == (np.prod(shape), 3, 3)
    assert np.all(np.isfinite(wannier))
    assert np.all(result.wannier_norms > 0.0)
    assert result.A[0, 0, 0].shape == (3, 3)
    assert result.V[0, 0, 0].shape == (3, 3)
    assert np.allclose(
        result.hoppings[(0, 0, 0)],
        result.hoppings[(0, 0, 0)].conj().T,
        atol=1.0e-12,
    )
    assert np.allclose(repeated.wannier_norms, result.wannier_norms, atol=1.0e-12)
    assert np.array_equal(bundle.energy_matrix, energy_snapshot)
    assert np.array_equal(bundle.metric_material, metric_snapshot)
    for index, expected in field_snapshot.items():
        assert np.array_equal(bundle.fields[index], expected)


def test_synthetic_3d_etbc_completion_runs_full_wannier_pipeline(tmp_path):
    shape = (4, 4, 4)
    kpoints = np.zeros((1, 3), dtype=float)
    with h5py.File(tmp_path / "grid.h5", "w") as handle:
        handle.attrs["dimension"] = 3
        handle.create_dataset("shape", data=np.asarray(shape, dtype=np.int64))
        handle.create_dataset("basis", data=np.eye(3))
        for axis in range(3):
            handle.create_dataset(
                f"u{axis + 1}", data=np.arange(shape[axis]) / shape[axis] - 0.5
            )
    with h5py.File(tmp_path / "E.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("E", data=np.zeros((1, 2), dtype=float))
    fields = np.zeros((1, 2) + shape + (3,), dtype=np.complex128)
    fields[0, 0, ..., 0] = 1.0
    fields[0, 1, ..., 1] = 1.0
    with h5py.File(tmp_path / "H.h5", "w") as handle:
        handle.create_dataset("kpoints", data=kpoints)
        handle.create_dataset("H_periodic", data=fields)

    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "dataset_type = mpb",
                "field_components = full_vector",
                "primary_field = magnetic",
                "lattice_const = 1",
                "real_lattice_vectors = 1 0 0, 0 1 0, 0 0 1",
                "reciprocal_lattice_vectors = 0 0 0, 0 0 0, 0 0 0",
                "k_points = 0:1:1, 0:1:1, 0:1:1",
                "composition_of_b = 1 0 0, 0 1 0, 0 0 1",
                "wannier_subspace = T + L",
                "longitudinal_source = etbc",
                "etbc_auxiliary_eigenvalue = 0",
                "band_window = 0:2",
                "inner_window = false",
                "dataset_file = ./H.h5",
                "mesh_file = ./grid.h5",
                "metric_file = false",
                "E_file = ./E.h5",
                "extension = 1,1,1",
                "max_iter = 0",
                "wannier_figures = false",
                "symmetry_file = Pm-3m",
                "symmetry_constrained = true",
                "projections",
                "1a; [0,0,0]; (z=[0,0,1], x=[1,0,0]); "
                "[1,0,0,4]@[1,0,0]; [1,0,0,4]@[0,1,0]; [1,0,0,4]@[0,0,1]",
                "end",
                "wannier_targets",
                "center_T1g_1a; 1a; T1g",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    result = run_calculation(load_input(load_config(incar)), threads=1)

    assert result.etbc is not None
    assert result.etbc.transverse_dimension == 2
    assert result.etbc.auxiliary_dimension == 1
    assert result.etbc.gamma_regularized_indices == ((0, 0, 0),)
    assert result.A[0, 0, 0].shape == (3, 3)
    assert result.V[0, 0, 0].shape == (3, 3)
    assert result.wanniers[(0, 0, 0)].shape == (np.prod(shape), 3, 3)
    assert np.allclose(result.wannier_norms, 1.0, atol=1.0e-10)
    assert np.allclose(result.hoppings[(0, 0, 0)], 0.0, atol=1.0e-12)


def test_cached_v_skips_streaming_vector_projection_seed():
    state = SimpleNamespace(inner_product=SimpleNamespace(domain_kind="points"))
    config = SimpleNamespace(
        projection_target_bindings=(object(),),
        use_cached_data=[],
    )

    assert _needs_streaming_projection_seed(config, state, None)

    config.use_cached_data = ["V"]
    assert not _needs_streaming_projection_seed(config, state, None)


def test_orbit_expanded_vector_trials_normalize_both_irrep_blocks(monkeypatch):
    point_count = 5
    operation = SimpleNamespace(antiunitary=False)
    orbit = SimpleNamespace(
        points=(
            SimpleNamespace(position=np.zeros(3), representative_operation=operation),
            SimpleNamespace(position=np.full(3, 0.5), representative_operation=operation),
        )
    )
    target = SimpleNamespace(orbit=orbit, site_irrep=SimpleNamespace(dimension=3))
    context = SimpleNamespace(model=SimpleNamespace(target=lambda _name: target, targets=(target,)))
    record = SimpleNamespace(states=(1.0, 2.0, 3.0))
    binding = SimpleNamespace(target_name="T1_2a", projection=record)

    class InnerProduct:
        def norms(self, values, *, name):
            assert name == "3D projection Bloch-sum norms"
            assert values.shape == (point_count, 3, 6)
            return np.sum(np.abs(values) ** 2, axis=(0, 1), dtype=np.float64)

    state = SimpleNamespace(
        config=SimpleNamespace(
            symmetry_context=context,
            projection_target_bindings=(binding,),
            real_lattice_vectors=np.eye(3),
            lattice_const=1.0,
            k_points=(np.asarray([0.0]),) * 3,
            extension=(1, 1, 1),
        ),
        mesh=SimpleNamespace(vertices=np.zeros((point_count, 3))),
        bloch_sign=1,
        inner_product=InnerProduct(),
    )

    monkeypatch.setattr(
        vector_trials_module,
        "_evaluate_transformed_trial",
        lambda _state, _record, trial, points, _center, _operation, **_kwargs: np.full(
            (points.shape[0], 3), trial, dtype=np.complex128
        ),
    )

    values = vector_trials_module.build_vector_bloch_trials(state, (0, 0, 0))

    assert values.shape == (point_count, 6, 3)
    assert np.allclose(
        np.sum(np.abs(values) ** 2, axis=(0, 2), dtype=np.float64),
        np.tile(np.array([1.0, 4.0, 9.0]) / (14.0 / 3.0), 2),
    )


def test_vector_wannier_norms_follow_six_wannier_columns():
    point_count = 6
    base = np.zeros((6, point_count, 3), dtype=np.complex128)
    for band in range(6):
        base[band, band, band % 3] = 1.0

    class InnerProduct:
        def norms(self, values, *, chunk_size, name):
            assert name == "Wannier norms"
            assert chunk_size == 2048
            assert values.shape == (point_count, 3, 6)
            return np.sum(np.abs(values) ** 2, axis=(0, 1), dtype=np.float64)

    config = SimpleNamespace(
        real_lattice_vectors=np.eye(3),
        reciprocal_lattice_vectors=np.eye(3),
        lattice_const=1.0,
        kdim=3,
        k_points=(np.asarray([0.0]),) * 3,
        extension=(1, 1, 1),
        band_calc_num=6,
    )
    state = SimpleNamespace(
        extended_mesh=SimpleNamespace(vertices=np.zeros((point_count, 3))),
        space_to_original_mapping=np.arange(point_count),
        extended_inner_product=InnerProduct(),
        bloch_sign=1,
        k_indices=lambda: iter(((0, 0, 0),)),
        get_k_num=lambda: 1,
        get_block=lambda *_: base,
    )
    context = SimpleNamespace(
        config=config,
        state=state,
        output_state_coefficients_at=lambda *_: np.eye(6, dtype=np.complex128),
    )

    _, values, norms = generate_wannier(context)

    assert values.shape == (point_count, 6, 3)
    assert norms.shape == (6,)
    assert np.allclose(norms, np.ones(6))


def test_uniform_grid_wannier_fourier_path_matches_direct_sum():
    rng = np.random.default_rng(617)
    mesh = PeriodicGrid((2, 2), np.eye(2))
    extended = mesh.__deepcopy__()
    mapping = extended.extension((2, 2), np.eye(2), 1.0)
    k_points = (np.asarray([-0.5, 0.0]),) * 2
    k_shape = (2, 2, 1)
    blocks = {
        index: rng.normal(size=(2, mesh.point_count, 3))
        + 1j * rng.normal(size=(2, mesh.point_count, 3))
        for index in np.ndindex(k_shape)
    }
    config = SimpleNamespace(
        k_points=k_points,
        kdim=2,
        band_calc_num=2,
        extension=(2, 2),
        real_lattice_vectors=np.eye(2),
        reciprocal_lattice_vectors=np.eye(2),
        lattice_const=1.0,
    )

    def phase(index):
        k = np.asarray([k_points[axis][index[axis]] for axis in range(2)])
        return np.exp(2j * np.pi * (mesh.fractional_vertices @ k))

    state = SimpleNamespace(
        mesh=mesh,
        extended_mesh=extended,
        space_to_original_mapping=mapping,
        bloch_sign=1,
        k_shape=k_shape,
        get_block=lambda *index: blocks[tuple(index)],
        get_phase=lambda *index: phase(tuple(index)),
    )
    context = SimpleNamespace(
        config=config,
        state=state,
        output_state_coefficients_at=lambda *_: np.eye(2),
    )
    r = [1, -1, 0]

    actual = _uniform_grid_wannier_sum(context, r)
    expected = np.zeros_like(actual)
    for point, (fractional, local) in enumerate(
        zip(extended.fractional_vertices, mapping)
    ):
        for index in np.ndindex(k_shape):
            k = np.asarray([k_points[axis][index[axis]] for axis in range(2)])
            expected[point] += blocks[index][:, local, :] * np.exp(
                2j * np.pi * np.dot(k, fractional - np.asarray(r[:2]))
            )
    expected /= np.sqrt(4.0 * 4.0)

    assert np.allclose(actual, expected, atol=2.0e-13, rtol=2.0e-13)


def test_uniform_grid_scalar_wannier_fourier_path_matches_direct_sum():
    rng = np.random.default_rng(618)
    mesh = PeriodicGrid((2, 2), np.eye(2))
    extended = mesh.__deepcopy__()
    mapping = extended.extension((2, 2), np.eye(2), 1.0)
    k_points = (np.asarray([-0.5, 0.0]),) * 2
    k_shape = (2, 2, 1)
    blocks = {
        index: rng.normal(size=(2, mesh.point_count))
        + 1j * rng.normal(size=(2, mesh.point_count))
        for index in np.ndindex(k_shape)
    }
    config = SimpleNamespace(
        k_points=k_points,
        kdim=2,
        band_calc_num=2,
        extension=(2, 2),
        real_lattice_vectors=np.eye(2),
        reciprocal_lattice_vectors=np.eye(2),
        lattice_const=1.0,
    )

    def phase(index):
        k = np.asarray([k_points[axis][index[axis]] for axis in range(2)])
        return np.exp(2j * np.pi * (mesh.fractional_vertices @ k))

    state = SimpleNamespace(
        mesh=mesh,
        extended_mesh=extended,
        space_to_original_mapping=mapping,
        bloch_sign=1,
        k_shape=k_shape,
        get_block=lambda *index: blocks[tuple(index)],
        get_phase=lambda *index: phase(tuple(index)),
    )
    context = SimpleNamespace(
        config=config,
        state=state,
        output_state_coefficients_at=lambda *_: np.eye(2),
    )
    r = [-1, 1, 0]

    actual = _uniform_grid_wannier_sum(context, r)
    expected = np.zeros_like(actual)
    for point, (fractional, local) in enumerate(
        zip(extended.fractional_vertices, mapping)
    ):
        for index in np.ndindex(k_shape):
            k = np.asarray([k_points[axis][index[axis]] for axis in range(2)])
            expected[point] += blocks[index][:, local] * np.exp(
                2j * np.pi * np.dot(k, fractional - np.asarray(r[:2]))
            )
    expected /= np.sqrt(4.0 * 4.0)

    assert np.allclose(actual, expected, atol=2.0e-13, rtol=2.0e-13)
