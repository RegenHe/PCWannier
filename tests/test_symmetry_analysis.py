from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

import pcwannier.symmetry.analysis as symmetry_analysis_module
from pcwannier import BlochConvention
from pcwannier.conventions import BlochFieldRepresentation
from pcwannier.compute.integration import MetricInnerProduct
from pcwannier.compute.state import StateCollection
from pcwannier.data import (
    BlochSymmetryChannelResult,
    BlochSymmetryRunResult,
    InputBundle,
    Mesh,
    PeriodicGrid,
)
from pcwannier.maxwell import MaxwellProblem
from pcwannier.matrix_io import load_cell_matrix
from pcwannier.outputs import write_bloch_symmetry_outputs
from pcwannier.symmetry.reporting import (
    format_bloch_symmetry_report,
    format_symmetry_analysis_report,
)
from pcwannier.symmetry.definition import build_factor_system
from pcwannier.symmetry.tables import ConcreteFiniteGroup
from pcwannier.symmetry import (
    BlochSymmetryAction,
    DegeneracyTolerance,
    FieldKind,
    IrrepDecomposition,
    RepresentationAnalysisSpec,
    RepresentationPointSpec,
    SpaceGroupOperation,
    SpaceGroup,
    StateBlochSymmetryProvider,
    SymmetryModel,
    SymmetryCalculationSpec,
    WannierTargetSpec,
    analyze_bloch_symmetry,
    apply_magnetic_bias_to_model,
    build_symmetry_context,
    coefficient_metric_overlap,
    compare_representations,
    compose_symmetry_model,
    decompose_little_group_characters,
    group_degenerate_bands,
    intertwiner_residual,
    little_group,
    load_symmetry,
    run_symmetry_analysis,
    run_bloch_symmetry_analysis,
    load_sewing_matrix_cache,
    save_sewing_matrix_cache,
)

from .symmetry_models import P4MM, p4mm_model, square_2c_model


def _square_mesh(points_per_axis: int = 5) -> Mesh:
    axis = np.linspace(0.0, 1.0, points_per_axis)
    vertices = np.asarray([(x, y) for x in axis for y in axis])
    elements = []
    for i in range(points_per_axis - 1):
        for j in range(points_per_axis - 1):
            a = i * points_per_axis + j
            b = (i + 1) * points_per_axis + j
            c = i * points_per_axis + j + 1
            d = (i + 1) * points_per_axis + j + 1
            elements.extend(((a, b, d), (a, d, c)))
    return Mesh(vertices, np.asarray(elements, dtype=np.intp))


def _orthonormalize(mesh: Mesh, fields: np.ndarray) -> np.ndarray:
    epsilon = np.ones(mesh.vertices.shape[0])
    gram = MetricInnerProduct(mesh, epsilon).overlap(fields, fields)
    values, vectors = np.linalg.eigh(gram)
    correction = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.conj().T
    return (fields.T @ correction).T


def _synthetic_state(fields: np.ndarray, energies=None):
    mesh = _square_mesh()
    normalized = _orthonormalize(mesh, fields)
    count = normalized.shape[0]
    blocks = np.empty((1, 1, 1), dtype=object)
    blocks[0, 0, 0] = normalized
    band_ids = np.empty((1, 1, 1), dtype=object)
    band_ids[0, 0, 0] = list(range(count))
    transforms = np.empty((1, 1, 1), dtype=object)
    transforms[0, 0, 0] = np.eye(count, dtype=np.complex128)
    energy_matrix = np.asarray(energies if energies is not None else np.arange(count), dtype=float)
    energy_matrix = energy_matrix.reshape(1, 1, 1, count)
    config = SimpleNamespace(
        real_lattice_vectors=[[1.0, 0.0], [0.0, 1.0]],
        lattice_const=1.0,
        dataset_type="synthetic",
    )
    state = SimpleNamespace(
        is_bloch=True,
        config=config,
        maxwell=MaxwellProblem.for_components("Ez"),
        bloch_convention=BlochConvention(),
        mesh=mesh,
        fields=blocks,
        E_idx=band_ids,
        energy_matrix=energy_matrix,
        metric_material=np.ones(mesh.vertices.shape[0]),
        inner_product=MetricInnerProduct(mesh, np.ones(mesh.vertices.shape[0])),
        get_block=lambda i, j, k: blocks[i, j, k],
        get_transform=lambda: transforms,
    )
    return state


def test_bloch_action_uses_comsol_translation_phase_and_periodic_part():
    mesh = _square_mesh()
    operation = SpaceGroupOperation(np.eye(2, dtype=int), np.array([1.0, 0.0]))
    action = BlochSymmetryAction(
        mesh.vertices,
        mesh.elements,
        np.eye(2),
        bloch_sign=-1,
    )
    field = np.ones((1, mesh.vertices.shape[0]), dtype=np.complex128)
    transformed = action.apply(field, operation, np.array([0.25, 0.0]), FieldKind.SCALAR)
    assert np.allclose(transformed, 1j * field, atol=1e-12)


def test_full_bloch_action_uses_quasiperiodic_lattice_shift():
    mesh = _square_mesh()
    operation = SpaceGroupOperation(np.eye(2, dtype=int), np.array([1.0, 0.0]))
    action = BlochSymmetryAction(
        mesh.vertices,
        mesh.elements,
        np.eye(2),
        bloch_sign=-1,
    )
    field = np.ones((1, mesh.vertices.shape[0]), dtype=np.complex128)

    transformed = action.apply_full_bloch(
        field,
        operation,
        np.array([0.25, 0.0]),
        FieldKind.SCALAR,
    )

    expected = np.where(mesh.vertices[:, 0] < 1.0 - 1.0e-12, 1j, 1.0)
    assert np.allclose(transformed, expected[None, :], atol=1e-12)


def test_full_bloch_stencil_cache_keeps_exact_lattice_translation_phase():
    mesh = _square_mesh()
    action = BlochSymmetryAction(
        mesh.vertices,
        mesh.elements,
        np.eye(2),
        bloch_sign=-1,
    )
    kpoint = np.array([0.25, 0.0])
    field = np.exp(
        -2j * np.pi * (mesh.vertices @ kpoint)
    )[None, :]
    identity = SpaceGroupOperation.identity(2)
    translation = SpaceGroupOperation.lattice_translation([1, 0])

    unchanged = action.apply_full_bloch(
        field, identity, kpoint, FieldKind.SCALAR
    )
    transformed = action.apply_full_bloch(
        field, translation, kpoint, FieldKind.SCALAR
    )

    assert np.allclose(unchanged, field, atol=1.0e-12)
    assert np.allclose(transformed, 1.0j * field, atol=1.0e-12)


def test_full_bloch_antiunitary_action_conjugates_quasiperiodic_phase():
    mesh = _square_mesh()
    operation = SpaceGroupOperation(
        np.eye(2, dtype=int),
        np.array([1.0, 0.0]),
        antiunitary=True,
    )
    action = BlochSymmetryAction(
        mesh.vertices,
        mesh.elements,
        np.eye(2),
        bloch_sign=-1,
    )
    field = np.ones((1, mesh.vertices.shape[0]), dtype=np.complex128)

    transformed = action.apply_full_bloch(
        field,
        operation,
        np.array([0.25, 0.0]),
        FieldKind.ELECTRIC_Z,
        time_reversal=np.conj,
    )

    expected = np.where(mesh.vertices[:, 0] < 1.0 - 1.0e-12, -1j, 1.0)
    assert np.allclose(transformed, expected[None, :], atol=1e-12)


def test_uniform_3d_screw_sewing_matches_its_nonsymmorphic_factor_system():
    identity = SpaceGroupOperation.identity(3, "E")
    screw = SpaceGroupOperation(
        np.diag([-1, -1, 1]),
        np.array([0.0, 0.0, 0.5]),
        "screw_2_1_z",
    )
    group = SpaceGroup((identity, screw), tolerance=1.0e-10)
    convention = BlochConvention(1, "mpb-test")
    model = SymmetryModel(3, 1.0e-10, group, (), bloch_convention=convention)
    k_axes = (np.array([0.0]), np.array([0.0]), np.array([-0.5]))
    context = build_symmetry_context(model, k_axes)

    grid = PeriodicGrid((2, 2, 2), np.eye(3))
    block = np.zeros((1, grid.point_count, 3), dtype=np.complex128)
    block[0, :, 2] = 1.0
    fields = np.empty((1, 1, 1), dtype=object)
    fields[0, 0, 0] = block
    bands = np.empty((1, 1, 1), dtype=object)
    bands[0, 0, 0] = [0]
    energies = np.empty((1, 1, 1), dtype=object)
    energies[0, 0, 0] = np.array([1.0])
    config = SimpleNamespace(
        kdim=3,
        integration_mode="nodal",
        use_cached_data=[],
        real_lattice_vectors=np.eye(3),
        reciprocal_lattice_vectors=np.eye(3),
        lattice_const=1.0,
        extension=[1, 1, 1],
        k_points=k_axes,
        symmetry_context=context,
        maxwell_problem=MaxwellProblem.for_components("full_vector", "magnetic"),
    )
    state = StateCollection(
        InputBundle(
            config=config,
            maxwell=config.maxwell_problem,
            bloch_convention=convention,
            mesh=grid,
            fields=fields,
            metric_material=np.ones(grid.point_count),
            energies=energies,
            band_indices=bands,
            inner_band_indices=bands.copy(),
            energy_matrix=np.ones((1, 1, 1, 1)),
            field_representation=BlochFieldRepresentation.PERIODIC_PART,
            symmetry=context,
        )
    )
    report, need_orthogonalization = state.check_orthogonality()
    assert not need_orthogonalization
    assert report[0, 0, 0, 0] < 1.0e-14
    state.ensure_identity_transform()

    provider = StateBlochSymmetryProvider(state, context)
    mapping = provider.mapping(1, (0, 0, 0))
    sewing = provider.sewing_matrix_for_mapping(mapping, (0,))
    concrete = ConcreteFiniteGroup.from_space_group(group)
    factor = build_factor_system(
        concrete,
        [0.0, 0.0, -0.5],
        1.0e-10,
        bloch_convention=convention,
    )

    assert np.array_equal(concrete.lattice_shifts[1, 1], [0, 0, 1])
    assert sewing[0, 0] == pytest.approx(1.0j, abs=1.0e-13)
    assert (sewing @ sewing)[0, 0] == pytest.approx(-1.0, abs=1.0e-13)
    assert factor.phases[1, 1] == pytest.approx(-1.0, abs=1.0e-13)


def test_quadratic_identity_sewing_at_bz_boundary_is_unitary():
    model = square_2c_model(analysis=False)
    k_axes = [np.array([-0.5, 0.0]), np.array([-0.5, 0.0])]
    context = build_symmetry_context(model, k_axes)
    mesh = _square_mesh()
    fractional = mesh.vertices
    periodic = np.asarray(
        [
            np.ones(len(fractional)),
            np.cos(2.0 * np.pi * fractional[:, 0])
            + 0.4 * np.cos(2.0 * np.pi * fractional[:, 1]),
        ],
        dtype=np.complex128,
    )
    fields = np.empty((2, 2, 1), dtype=object)
    bands = np.empty((2, 2, 1), dtype=object)
    energies = np.empty((2, 2, 1), dtype=object)
    for i, kx in enumerate(k_axes[0]):
        for j, ky in enumerate(k_axes[1]):
            phase = np.exp(2j * np.pi * (fractional @ np.array([kx, ky])))
            fields[i, j, 0] = periodic * phase[None, :]
            bands[i, j, 0] = [0, 1]
            energies[i, j, 0] = np.array([1.0, 1.0])
    config = SimpleNamespace(
        kdim=2,
        integration_mode="quadratic",
        dataset_type="synthetic",
        use_cached_data=[],
        real_lattice_vectors=np.eye(2),
        reciprocal_lattice_vectors=np.eye(2),
        lattice_const=1.0,
        extension=[1, 1],
        k_points=k_axes,
        symmetry_context=context,
    )
    metric = 1.0 + fractional[:, 0] * (1.0 - fractional[:, 0])
    state = StateCollection(
        InputBundle(
            config=config,
            maxwell=MaxwellProblem.for_components("Ez"),
            bloch_convention=BlochConvention(),
            mesh=mesh,
            fields=fields,
            metric_material=metric,
            energies=energies,
            band_indices=bands,
            inner_band_indices=bands.copy(),
            energy_matrix=np.ones((2, 2, 1, 2)),
        ),
        threads=1,
    )
    state.orthogonalize()
    state.turn_to_bloch()
    provider = StateBlochSymmetryProvider(state, context)
    identity_index = model.group.operation_index(model.group.operation_by_name("E"))
    mapping = provider.mapping(identity_index, (0, 1))

    sewing = provider.sewing_matrix_for_mapping(mapping, (0, 1))

    assert np.allclose(sewing, np.eye(2), rtol=0.0, atol=2e-12)


def test_scalar_ez_and_hz_have_distinct_mirror_actions():
    mesh = _square_mesh()
    mirror = SpaceGroupOperation([[-1, 0], [0, 1]], [0.0, 0.0])
    action = BlochSymmetryAction(mesh.vertices, mesh.elements, np.eye(2))
    field = np.ones((1, mesh.vertices.shape[0]), dtype=np.complex128)

    electric = action.apply(field, mirror, [0.0, 0.0], FieldKind.ELECTRIC_Z)
    magnetic = action.apply(
        field, mirror, [0.0, 0.0], FieldKind.MAGNETIC_AXIAL_Z
    )

    assert np.allclose(electric, field)
    assert np.allclose(magnetic, -field)


def test_state_sewing_uses_maxwell_ez_or_hz_field_action():
    model = square_2c_model(analysis=False)
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    field = np.ones((1, _square_mesh().vertices.shape[0]))
    mirror_index = model.group.operation_index(
        model.group.operation_by_name("sigma_x")
    )

    electric_state = _synthetic_state(field, energies=[1.0])
    electric_provider = StateBlochSymmetryProvider(electric_state, context)
    electric = electric_provider.sewing_matrix_for_mapping(
        electric_provider.mapping(mirror_index, (0, 0)), (0,)
    )

    magnetic_state = _synthetic_state(field, energies=[1.0])
    magnetic_state.maxwell = MaxwellProblem.for_components("Hz")
    magnetic_provider = StateBlochSymmetryProvider(magnetic_state, context)
    magnetic = magnetic_provider.sewing_matrix_for_mapping(
        magnetic_provider.mapping(mirror_index, (0, 0)), (0,)
    )

    assert np.allclose(electric, [[1.0]], atol=1.0e-12)
    assert np.allclose(magnetic, [[-1.0]], atol=1.0e-12)


def test_full_bloch_and_periodic_part_symmetry_formulas_agree():
    mesh = _square_mesh()
    operation = SpaceGroupOperation(
        np.array([[0, -1], [1, 0]]),
        np.array([0.25, 0.25]),
    )
    action = BlochSymmetryAction(mesh.vertices, mesh.elements, np.eye(2), bloch_sign=-1)
    kpoint = np.array([0.25, 0.0])
    transformed_k = operation.act_reciprocal(kpoint)
    periodic = np.asarray([1.0 + 0.3 * np.cos(2 * np.pi * mesh.vertices[:, 0])])
    preimages = (mesh.vertices - operation.translation) @ np.linalg.inv(operation.rotation).T
    periodic_at_preimage = 1.0 + 0.3 * np.cos(2 * np.pi * preimages[:, 0])
    directly_transformed_full = (
        periodic_at_preimage * np.exp(-2j * np.pi * (preimages @ kpoint))
    )[None, :]
    transformed_periodic = action.apply(periodic, operation, kpoint, FieldKind.SCALAR)
    reconstructed_full = transformed_periodic * np.exp(
        -2j * np.pi * (mesh.vertices @ transformed_k)
    )[None, :]
    assert np.allclose(reconstructed_full, directly_transformed_full, atol=1e-12)


def test_state_provider_gives_a1_and_e_sewing_matrices():
    model = square_2c_model()
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    x = _square_mesh().vertices[:, 0]
    y = _square_mesh().vertices[:, 1]
    fields = np.asarray([np.sin(2 * np.pi * x), np.sin(2 * np.pi * y)])
    state = _synthetic_state(fields, energies=[1.0, 1.0])
    provider = StateBlochSymmetryProvider(state, context)
    c4_index = model.group.operation_index(model.group.operation_by_name("C4"))
    mapping = provider.mapping(c4_index, (0, 0))
    request = provider.request_for_mapping(mapping, (0, 1))
    sewing = provider.sewing_matrix(request)
    assert np.allclose(sewing, [[0.0, -1.0], [1.0, 0.0]], atol=1e-12)
    assert abs(np.trace(sewing)) < 1e-12
    assert np.linalg.norm(sewing.conj().T @ sewing - np.eye(2)) < 1e-12
    rectangular = provider.sewing_matrix_between_mapping(mapping, (0, 1), (0,))
    assert rectangular.shape == (1, 2)
    assert np.allclose(rectangular, [[0.0, -1.0]], atol=1e-12)

    invariant = np.asarray([np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)])
    a1_state = _synthetic_state(invariant, energies=[1.0])
    a1_provider = StateBlochSymmetryProvider(a1_state, context)
    c4_mapping = a1_provider.mapping(c4_index, (0, 0))
    c4_matrix = a1_provider.sewing_matrix_for_mapping(c4_mapping, (0,))
    assert np.allclose(c4_matrix, [[1.0]], atol=1e-12)

    gamma_only = replace(
        model,
        representation_analysis=replace(
            model.representation_analysis,
            points=(model.representation_analysis.points[0],),
        ),
    )
    gamma_context = build_symmetry_context(gamma_only, [np.array([0.0]), np.array([0.0])])
    full = run_symmetry_analysis(state, gamma_context)
    point = full.physical.point("Gamma")
    assert point.diagnostics.unitarity_error < 1e-12
    assert point.diagnostics.outer_composition_residual < 1e-12
    assert point.physical_decomposition.multiplicities["E"] == 1
    assert full.target_compatibility("Gamma").point_name == "Gamma"


def test_state_provider_roundtrips_full_outer_sewing_cache(tmp_path, monkeypatch):
    model = square_2c_model(analysis=False)
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    x = mesh.vertices[:, 0]
    y = mesh.vertices[:, 1]
    fields = np.asarray([np.sin(2 * np.pi * x), np.sin(2 * np.pi * y)])
    state = _synthetic_state(fields, energies=[1.0, 1.0])
    state.config.use_cached_data = []
    provider = StateBlochSymmetryProvider(state, context)
    c4_index = model.group.operation_index(model.group.operation_by_name("C4"))
    mapping = provider.mapping(c4_index, (0, 0))
    expected = provider.sewing_matrix_between_mapping(mapping, (1,), (0, 1))

    path = tmp_path / "D.txt"
    save_sewing_matrix_cache(
        path,
        provider.cached_sewing_matrices,
        dimension=2,
        bloch_sign=model.bloch_convention.sign,
        k_shape=(1, 1),
    )
    saved = load_sewing_matrix_cache(path)
    generic = load_cell_matrix(path, (1,))
    assert len(saved.entries) == 1
    assert saved.entries[0].matrix.shape == (2, 2)
    assert saved.entries[0].source_band_indices == (0, 1)
    assert np.allclose(generic[0], saved.entries[0].matrix, atol=1e-15)
    assert "CELL(0,)" in path.read_text(encoding="utf-8")

    cached_state = _synthetic_state(fields, energies=[1.0, 1.0])
    cached_state.config.use_cached_data = ["D"]
    cached_state.config.D_file = str(path)
    cached_state.config.input_path = lambda value: Path(value)
    cached_provider = StateBlochSymmetryProvider(cached_state, context)

    def fail_integration(*args, **kwargs):
        raise AssertionError("A cache hit must not integrate Bloch fields again.")

    cached_state.inner_product.overlap = fail_integration
    actual = cached_provider.sewing_matrix_between_mapping(mapping, (1,), (0, 1))
    assert np.allclose(actual, expected, atol=1e-12)

    identity = model.group.operation_index(model.group.operation_by_name("E"))
    with pytest.raises(ValueError, match="does not contain the requested exact Seitz action"):
        cached_provider.sewing_matrix_for_mapping(
            cached_provider.mapping(identity, (0, 0)), (0, 1)
        )


def test_state_provider_rejects_sewing_cache_with_different_outer_bands(tmp_path):
    model = square_2c_model(analysis=False)
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    fields = np.asarray(
        [
            np.sin(2 * np.pi * mesh.vertices[:, 0]),
            np.sin(2 * np.pi * mesh.vertices[:, 1]),
        ]
    )
    state = _synthetic_state(fields, energies=[1.0, 1.0])
    state.config.use_cached_data = []
    provider = StateBlochSymmetryProvider(state, context)
    c4_index = model.group.operation_index(model.group.operation_by_name("C4"))
    provider.sewing_matrix_for_mapping(provider.mapping(c4_index, (0, 0)), (0, 1))
    path = tmp_path / "D.txt"
    save_sewing_matrix_cache(
        path,
        provider.cached_sewing_matrices,
        dimension=2,
        bloch_sign=model.bloch_convention.sign,
        k_shape=(1, 1),
    )

    changed = _synthetic_state(fields, energies=[1.0, 1.0])
    changed.E_idx[0, 0, 0] = [0, 2]
    changed.config.use_cached_data = ["D"]
    changed.config.D_file = str(path)
    changed.config.input_path = lambda value: Path(value)
    with pytest.raises(ValueError, match="outer-window band ids do not match"):
        StateBlochSymmetryProvider(changed, context)


def test_state_provider_rejects_inconsistent_periodic_duplicate_nodes():
    model = square_2c_model(analysis=False)
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    x = _square_mesh().vertices[:, 0]
    field = np.asarray([np.sin(2 * np.pi * x)])

    field_state = _synthetic_state(field, energies=[1.0])
    field_state.fields[0, 0, 0][0, -1] += 0.1
    with pytest.raises(ValueError, match="inconsistent periodic Bloch fields"):
        StateBlochSymmetryProvider(field_state, context)

    metric_state = _synthetic_state(field, energies=[1.0])
    metric_state.metric_material[-1] = 2.0
    with pytest.raises(ValueError, match="inconsistent metric material"):
        StateBlochSymmetryProvider(metric_state, context)


def test_representation_analysis_rejects_noninvariant_energy_block():
    model = square_2c_model()
    gamma_only = replace(
        model,
        representation_analysis=replace(
            model.representation_analysis,
            points=(model.representation_analysis.points[0],),
        ),
    )
    context = build_symmetry_context(gamma_only, [np.array([0.0]), np.array([0.0])])
    x = _square_mesh().vertices[:, 0]
    y = _square_mesh().vertices[:, 1]
    fields = np.asarray([np.sin(2 * np.pi * x), np.sin(2 * np.pi * y)])
    state = _synthetic_state(fields, energies=[1.0, 2.0])

    with pytest.raises(ValueError, match="Degenerate block.*not a valid unitary"):
        run_symmetry_analysis(state, context)


def test_physical_bloch_analysis_needs_no_wannier_targets_and_identifies_irrep():
    model = p4mm_model(
        points=(("Gamma", [0.0, 0.0], (0, 1), None),)
    )
    assert model.targets == ()
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    fields = np.asarray(
        [
            np.sin(2 * np.pi * mesh.vertices[:, 0]),
            np.sin(2 * np.pi * mesh.vertices[:, 1]),
        ]
    )
    state = _synthetic_state(fields, energies=[1.0, 1.0])

    result = run_bloch_symmetry_analysis(state, context)
    point = result.point("Gamma")
    block = point.degenerate_blocks[0]

    assert point.little_group_name == "C4v"
    assert point.antiunitary_operation_names == ()
    assert block.decomposition is not None
    assert block.decomposition.multiplicities["E"] == 1
    assert block.irrep_unavailable_reason is None
    with pytest.raises(KeyError, match="Unknown Bloch symmetry analysis point"):
        result.point("Missing")
    combined = run_symmetry_analysis(state, context)
    assert combined.physical.point("Gamma").physical_decomposition.multiplicities["E"] == 1
    assert combined.target_compatibilities == ()
    with pytest.raises(KeyError, match="No target compatibility"):
        combined.target_compatibility("Gamma")


def test_bloch_analysis_reports_leakage_without_constructing_an_irrep():
    model = p4mm_model(
        points=(("Gamma", [0.0, 0.0], (0,), None),)
    )
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    fields = np.asarray(
        [
            np.sin(2 * np.pi * mesh.vertices[:, 0]),
            np.sin(2 * np.pi * mesh.vertices[:, 1]),
        ]
    )
    state = _synthetic_state(fields, energies=[1.0, 1.0])

    point = analyze_bloch_symmetry(
        state,
        context,
        [0.0, 0.0],
        [0],
        name="Gamma",
    )
    block = point.degenerate_blocks[0]

    assert block.leakage > 0.5
    assert block.coupled_outer_bands == (1,)
    assert block.decomposition is None
    assert "leakage" in block.irrep_unavailable_reason


def test_leaky_block_can_report_an_approximate_character_label():
    model = p4mm_model(
        points=(("Gamma", [0.0, 0.0], (0, 1), None),)
    )
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    fields = np.asarray(
        [
            np.sin(2 * np.pi * mesh.vertices[:, 0]),
            np.sin(2 * np.pi * mesh.vertices[:, 1]),
            np.ones(mesh.vertices.shape[0]),
        ]
    )
    state = _synthetic_state(fields, energies=[1.0, 1.0, 2.0])
    provider = StateBlochSymmetryProvider(state, context)
    original = provider.sewing_matrix_in_band_basis

    def leaky_band_basis(mapping, source_band_indices, target_band_indices=None, **kwargs):
        source = tuple(source_band_indices)
        target = source if target_band_indices is None else tuple(target_band_indices)
        matrix = original(
            mapping,
            source,
            target,
            **kwargs,
        )
        if source == (0, 1) and target == (2,):
            return matrix + np.array([[0.02, 0.0]])
        if source == (2,) and target == (0, 1):
            return matrix + np.array([[0.02], [0.0]])
        return matrix

    provider.sewing_matrix_in_band_basis = leaky_band_basis
    result = run_bloch_symmetry_analysis(state, context, provider=provider)
    block = result.point("Gamma").degenerate_blocks[0]

    assert block.leakage > model.representation_analysis.leakage_tolerance
    assert block.decomposition is None
    assert block.approximate_decomposition is not None
    assert block.approximate_decomposition.multiplicities["E"] == 1
    assert block.character_fit_error == pytest.approx(0.0, abs=1.0e-12)

    report = format_bloch_symmetry_report(result)
    assert "irrep=E (approximate; character_error=" in report
    assert "K1(E) character=2" in report
    assert "note=band subspace leakage" in report


def test_band_basis_leakage_does_not_use_globally_mixed_sewing_blocks():
    model = p4mm_model()
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    elements = little_group(model.group, [0.0, 0.0])
    mixed = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    full_matrices = {
        model.group.operations[element.operation_index].name: mixed
        for element in elements
    }

    class BandBasisProvider:
        def __init__(self):
            self.context = context

        def mapping(self, operation_index, source_index):
            return context.k_mappings[operation_index][0]

        def sewing_matrix_in_band_basis(
            self,
            mapping,
            source_band_indices,
            target_band_indices=None,
            **kwargs,
        ):
            target = (
                tuple(source_band_indices)
                if target_band_indices is None
                else tuple(target_band_indices)
            )
            return np.zeros(
                (len(target), len(tuple(source_band_indices))),
                dtype=np.complex128,
            )

    old_leakage, old_coupled = symmetry_analysis_module._matrix_subspace_leakage(
        full_matrices,
        (0, 1),
        (0,),
        1.0e-10,
    )
    leakage, coupled = symmetry_analysis_module._band_basis_subspace_leakage(
        BandBasisProvider(),
        elements,
        (0, 0),
        np.zeros(2),
        full_matrices,
        (0, 1),
        (0,),
        1.0e-10,
    )

    assert old_leakage > 1.0
    assert old_coupled == (1,)
    assert leakage == pytest.approx(0.0)
    assert coupled == ()


def test_nonunitary_band_representation_does_not_receive_an_irrep_label():
    model = p4mm_model(
        points=(("Gamma", [0.0, 0.0], (0, 1), None),)
    )
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    fields = np.asarray(
        [
            np.sin(2 * np.pi * mesh.vertices[:, 0]),
            np.sin(2 * np.pi * mesh.vertices[:, 1]),
        ]
    )
    state = _synthetic_state(fields, energies=[1.0, 1.0])
    provider = StateBlochSymmetryProvider(state, context)
    original = provider.sewing_matrix_in_band_basis
    basis = np.array([[1.0, 0.4], [0.0, 1.0]], dtype=np.complex128)
    basis_inverse = np.linalg.inv(basis)

    def nonunitary_band_basis(*args, **kwargs):
        matrix = original(*args, **kwargs)
        if matrix.shape == (2, 2):
            return basis @ matrix @ basis_inverse
        return matrix

    provider.sewing_matrix_in_band_basis = nonunitary_band_basis
    block = run_bloch_symmetry_analysis(
        state, context, provider=provider
    ).point("Gamma").degenerate_blocks[0]

    assert block.twisted_composition_residual < 1.0e-10
    assert block.unitarity_error > 1.0e-2
    assert block.decomposition is None
    assert "unitarity" in block.irrep_unavailable_reason


def test_bloch_analysis_can_disable_degenerate_block_splitting():
    model = p4mm_model()
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    fields = np.asarray(
        [
            np.sin(2 * np.pi * mesh.vertices[:, 0]),
            np.sin(2 * np.pi * mesh.vertices[:, 1]),
        ]
    )
    state = _synthetic_state(fields, energies=[1.0, 2.0])

    point = analyze_bloch_symmetry(
        state,
        context,
        [0.0, 0.0],
        [0, 1],
        split_degenerate_blocks=False,
    )

    assert tuple(block.band_indices for block in point.degenerate_blocks) == ((0, 1),)


def test_magnetic_bloch_analysis_only_calls_unitary_traces_characters():
    model = p4mm_model(
        points=(("Gamma", [0.0, 0.0], (0, 1), None),)
    )
    model = apply_magnetic_bias_to_model(model, np.eye(2), [0.0, 0.0, 1.0])
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    fields = np.asarray(
        [
            np.sin(2 * np.pi * mesh.vertices[:, 0]),
            np.sin(2 * np.pi * mesh.vertices[:, 1]),
        ]
    )
    state = _synthetic_state(fields, energies=[1.0, 1.0])

    point = run_bloch_symmetry_analysis(state, context).points[0]
    block = point.degenerate_blocks[0]

    assert point.unitary_subgroup_name == "C4"
    assert set(point.antiunitary_operation_names) == {
        "sigma_x",
        "sigma_y",
        "sigma_d",
        "sigma_d2",
    }
    assert not set(point.antiunitary_operation_names) & set(block.unitary_characters)
    assert block.decomposition is None
    assert "corepresentation" in block.irrep_unavailable_reason
    assert len(block.antiunitary_diagnostics) == 4


def test_magnetic_target_compatibility_excludes_antiunitary_traces():
    base = apply_magnetic_bias_to_model(
        load_symmetry(P4MM),
        np.eye(2),
        [0.0, 0.0, 1.0],
    )
    target_specs = (
        WannierTargetSpec("p_plus", [0.0, 0.0], "E_plus"),
        WannierTargetSpec("p_minus", [0.0, 0.0], "E_minus"),
        WannierTargetSpec("s", [0.0, 0.0], "A"),
    )
    tolerance = DegeneracyTolerance()
    analysis = RepresentationAnalysisSpec(
        FieldKind.SCALAR,
        tolerance,
        (
            RepresentationPointSpec(
                "Gamma",
                np.zeros(2),
                (0, 1, 2),
                tuple(target.name for target in target_specs),
                tolerance,
            ),
        ),
    )
    model = compose_symmetry_model(
        base,
        SymmetryCalculationSpec(target_specs, analysis),
    )
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    px = np.sin(2 * np.pi * mesh.vertices[:, 0])
    py = np.sin(2 * np.pi * mesh.vertices[:, 1])
    fields = np.asarray(
        [
            (px + 1j * py) / np.sqrt(2.0),
            (px - 1j * py) / np.sqrt(2.0),
            np.cos(2 * np.pi * mesh.vertices[:, 0])
            + np.cos(2 * np.pi * mesh.vertices[:, 1]),
        ]
    )
    state = _synthetic_state(fields, energies=[1.0, 1.0, 1.0])

    result = run_symmetry_analysis(state, context)
    point = result.physical.point("Gamma")
    compatibility = result.target_compatibility("Gamma")

    assert set(compatibility.target_unitary_characters) == set(
        point.unitary_operation_names
    )
    assert not set(compatibility.target_unitary_characters) & set(
        point.antiunitary_operation_names
    )
    report = format_symmetry_analysis_report(result)
    assert "[target compatibility]" in report
    assert "Gamma: targets=" in report
    assert "compatible=unavailable" in report
    assert "intertwiner_dimension=3" in report


def test_bloch_analysis_reuses_one_full_outer_sewing_per_operation():
    model = p4mm_model(
        points=(("Gamma", [0.0, 0.0], (0, 1), None),)
    )
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    fields = np.asarray(
        [
            np.sin(2 * np.pi * mesh.vertices[:, 0]),
            np.sin(2 * np.pi * mesh.vertices[:, 1]),
        ]
    )
    state = _synthetic_state(fields, energies=[1.0, 1.0])
    original_overlap = state.inner_product.overlap
    calls = 0

    def counted_overlap(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_overlap(*args, **kwargs)

    state.inner_product.overlap = counted_overlap
    provider = StateBlochSymmetryProvider(state, context)
    run_bloch_symmetry_analysis(state, context, provider=provider)

    assert calls == len(model.group.operations)
    assert len(provider.cached_sewing_matrices) == len(model.group.operations)
    assert len(provider._transform_inverse_cache) == 1
    assert len(provider._band_lowdin_cache) == 1
    assert len(provider._band_basis_sewing_cache) == len(model.group.operations)

    run_bloch_symmetry_analysis(state, context, provider=provider)
    assert calls == len(model.group.operations)
    assert len(provider._transform_inverse_cache) == 1
    assert len(provider._band_lowdin_cache) == 1
    assert len(provider._band_basis_sewing_cache) == len(model.group.operations)


def test_band_basis_sewing_cache_returns_independent_arrays():
    model = p4mm_model(
        points=(("Gamma", [0.0, 0.0], (0, 1), None),)
    )
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    fields = np.asarray(
        [
            np.sin(2 * np.pi * mesh.vertices[:, 0]),
            np.sin(2 * np.pi * mesh.vertices[:, 1]),
        ]
    )
    provider = StateBlochSymmetryProvider(
        _synthetic_state(fields, energies=[1.0, 1.0]), context
    )
    c4_index = model.group.operation_index(model.group.operation_by_name("C4"))
    mapping = provider.mapping(c4_index, (0, 0))

    first = provider.sewing_matrix_in_band_basis(mapping, (0, 1))
    expected = first.copy()
    first[0, 0] = 99.0
    second = provider.sewing_matrix_in_band_basis(mapping, (0, 1))

    assert np.allclose(second, expected)
    assert len(provider._band_basis_sewing_cache) == 1


def test_bloch_preanalysis_writes_reusable_s_and_d_text_caches(tmp_path):
    model = p4mm_model(
        points=(("Gamma", [0.0, 0.0], (0, 1), None),)
    )
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    mesh = _square_mesh()
    fields = np.asarray(
        [
            np.sin(2 * np.pi * mesh.vertices[:, 0]),
            np.sin(2 * np.pi * mesh.vertices[:, 1]),
        ]
    )
    state = _synthetic_state(fields, energies=[1.0, 1.0])
    provider = StateBlochSymmetryProvider(state, context)
    analysis = run_bloch_symmetry_analysis(state, context, provider=provider)
    smat = np.empty((1, 1, 1), dtype=object)
    smat[0, 0, 0] = np.eye(2, dtype=np.complex128)
    config = SimpleNamespace(S_file="S.txt", D_file="D.txt", base_dir=tmp_path)
    result = BlochSymmetryRunResult(
        config=config,
        symmetry=context,
        primary=BlochSymmetryChannelResult(
            field_kind=FieldKind.SCALAR,
            orthogonality_report=np.zeros((1, 1, 1, 6)),
            S=smat,
            analysis=analysis,
            sewing_matrices=provider.cached_sewing_matrices,
        ),
    )

    write_bloch_symmetry_outputs(result, config, tmp_path)

    loaded_s = load_cell_matrix(tmp_path / "S.txt", (1, 1, 1))
    loaded_d = load_sewing_matrix_cache(tmp_path / "D.txt")
    report = (tmp_path / "sym.txt").read_text(encoding="utf-8")
    assert np.allclose(loaded_s[0, 0, 0], np.eye(2))
    assert len(loaded_d.entries) == len(model.group.operations)
    assert "[physical]" in report
    assert "Gamma: k=(0, 0)" in report
    assert "bands 0,1:" in report
    assert "irrep=E" in report


def test_target_2c_a1_has_expected_gamma_x_m_content():
    model = square_2c_model()
    target = model.target("square_2c_A1")
    expected = {
        "Gamma": {"A1": 1, "B1": 1},
        "X": {"A1": 1, "B2": 1},
        "M": {"E": 1},
    }
    for point in model.representation_analysis.points:
        elements = little_group(model.group, point.k_fractional)
        indices = tuple(element.operation_index for element in elements)
        characters = {
            model.group.operations[index].name: complex(np.trace(target.matrix(index, point.k_fractional)))
            for index in indices
        }
        resolved = model.group_definition.resolve_little_group(indices, point.k_fractional)
        decomposition = decompose_little_group_characters(resolved, characters)
        nonzero = {name: value for name, value in decomposition.multiplicities.items() if value}
        assert nonzero == expected[point.name]
        assert decomposition.max_residual < 1e-12


def test_degeneracy_metric_compatibility_and_intertwiner_helpers():
    tolerance = DegeneracyTolerance(absolute=1e-5, relative=1e-8)
    assert group_degenerate_bands([0, 1, 2], [1.0, 1.0 + 1e-6, 2.0], tolerance) == ((0, 1), (2,))
    no_chain = DegeneracyTolerance(absolute=1.0, relative=0.0)
    assert group_degenerate_bands([0, 1, 2], [0.0, 0.9, 1.8], no_chain) == ((0, 1), (2,))

    left = np.array([[1.0], [1.0j]])
    right = np.array([[2.0], [1.0]])
    metric = np.diag([2.0, 3.0])
    assert np.allclose(coefficient_metric_overlap(left, right, metric), [[4.0 - 3.0j]])
    assert not np.allclose(coefficient_metric_overlap(left, right, metric), left.conj().T @ right)

    target = IrrepDecomposition({"A": 1.0}, {"A": 1}, {"A": 0.0})
    exact = IrrepDecomposition({"A": 1.0}, {"A": 1}, {"A": 0.0})
    superset = IrrepDecomposition({"A": 2.0}, {"A": 2}, {"A": 0.0})
    missing = IrrepDecomposition({"A": 0.0}, {"A": 0}, {"A": 0.0})
    assert compare_representations(1, 1, target, exact).compatible
    assert compare_representations(1, 2, target, superset).compatible
    assert not compare_representations(1, 2, target, missing).compatible
    assert not compare_representations(2, 1, target, exact).compatible

    d_matrix = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert intertwiner_residual(np.eye(2), np.eye(2), d_matrix, d_matrix) < 1e-12
    assert intertwiner_residual(np.eye(2), np.eye(2), d_matrix, d_matrix + 0.1) > 0.0
    with pytest.raises(ValueError, match="N_W"):
        intertwiner_residual(np.ones((2, 1)), np.ones((2, 1)), np.eye(2), np.eye(2))
