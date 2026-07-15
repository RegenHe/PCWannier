from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from pcwannier.compute.integration import integrate_overlap_matrix, mesh_integral_view
from pcwannier.data import Mesh
from pcwannier.symmetry import (
    BlochSymmetryAction,
    DegeneracyTolerance,
    FieldKind,
    IrrepDecomposition,
    SewingMatrixRequest,
    SpaceGroupOperation,
    StateBlochSymmetryProvider,
    analyze_little_group,
    build_symmetry_context,
    coefficient_metric_overlap,
    compare_representations,
    decompose_little_group_characters,
    group_degenerate_bands,
    intertwiner_residual,
    little_group,
    run_symmetry_analysis,
)

from .symmetry_models import p4mm_model, square_2c_model


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
    gram = integrate_overlap_matrix(mesh, fields, fields, epsilon)
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
    return SimpleNamespace(
        is_bloch=True,
        config=config,
        mesh=mesh,
        fields=blocks,
        E_idx=band_ids,
        energy_matrix=energy_matrix,
        epsilon=np.ones(mesh.vertices.shape[0]),
        integral_view=mesh_integral_view(mesh),
        compute_backend="python",
        integration_mode="nodal",
        get_block=lambda i, j, k: blocks[i, j, k],
        get_transform=lambda: transforms,
    )


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
    analysis = analyze_little_group(model.group, [0.0, 0.0], [0], a1_provider)
    c4_entry = next(
        entry for entry in analysis.entries if model.group.operations[entry.element.operation_index].name == "C4"
    )
    assert np.allclose(c4_entry.matrix, [[1.0]], atol=1e-12)

    gamma_only = replace(
        model,
        representation_analysis=replace(
            model.representation_analysis,
            points=(model.representation_analysis.points[0],),
        ),
    )
    gamma_context = build_symmetry_context(gamma_only, [np.array([0.0]), np.array([0.0])])
    full = run_symmetry_analysis(state, gamma_context)
    assert full.points[0].diagnostics.unitarity_error < 1e-12
    assert full.points[0].diagnostics.max_composition_residual < 1e-12
    assert full.points[0].physical_decomposition.multiplicities["E"] == 1


def test_state_provider_rejects_inconsistent_periodic_duplicate_nodes():
    model = square_2c_model(analysis=False)
    context = build_symmetry_context(model, [np.array([0.0]), np.array([0.0])])
    x = _square_mesh().vertices[:, 0]
    field = np.asarray([np.sin(2 * np.pi * x)])

    field_state = _synthetic_state(field, energies=[1.0])
    field_state.fields[0, 0, 0][0, -1] += 0.1
    with pytest.raises(ValueError, match="inconsistent periodic Bloch fields"):
        StateBlochSymmetryProvider(field_state, context)

    epsilon_state = _synthetic_state(field, energies=[1.0])
    epsilon_state.epsilon[-1] = 2.0
    with pytest.raises(ValueError, match="inconsistent epsilon"):
        StateBlochSymmetryProvider(epsilon_state, context)


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

    with pytest.raises(ValueError, match="Degenerate block.*not invariant"):
        run_symmetry_analysis(state, context)


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


def test_nonclosed_subspace_reports_nonunitary_projection():
    matrix = np.array([[0.5 + 0.0j]])

    class Provider:
        def sewing_matrix(self, request: SewingMatrixRequest) -> np.ndarray:
            return matrix

    model = p4mm_model()
    result = analyze_little_group(model.group, [0.0, 0.0], [0], Provider())
    c4 = next(
        entry for entry in result.entries if model.group.operations[entry.element.operation_index].name == "C4"
    )
    assert np.linalg.norm(c4.matrix.conj().T @ c4.matrix - np.eye(1)) == pytest.approx(0.75)
