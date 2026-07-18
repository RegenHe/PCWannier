from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pcwannier.symmetry import (
    BlochConvention,
    ConcreteFiniteGroup,
    DegeneracyTolerance,
    FactorSystem,
    FieldKind,
    RepresentationAnalysisSpec,
    RepresentationPointSpec,
    SpaceGroup,
    SpaceGroupOperation,
    SymmetryCalculationSpec,
    TwistedRepresentation,
    WannierTargetSpec,
    build_factor_system,
    build_symmetry_context,
    compose_symmetry_model,
    load_space_group,
    load_symmetry,
    resolve_little_group,
    run_symmetry_analysis,
    solve_intertwiner_space,
)

from .symmetry_models import P4GM, P4MM


def _glide_group() -> SpaceGroup:
    return SpaceGroup(
        (
            SpaceGroupOperation.identity(2),
            SpaceGroupOperation(
                np.diag([1, -1]),
                np.array([0.5, 0.0]),
                "glide_x",
            ),
        )
    )


def _projective_regular_representation(concrete, factor) -> TwistedRepresentation:
    matrices = []
    for left in range(concrete.order):
        matrix = np.zeros((concrete.order, concrete.order), dtype=np.complex128)
        for right in range(concrete.order):
            result = int(concrete.table.multiplication[left, right])
            matrix[result, right] = factor.phases[left, right]
        matrices.append(matrix)
    return TwistedRepresentation(
        tuple(matrices),
        concrete.table.multiplication,
        factor,
    )


def test_seitz_product_exposes_glide_lattice_translation():
    group = _glide_group()
    product = group.multiply_mod_lattice(1, 1)
    concrete = ConcreteFiniteGroup.from_space_group(group)

    assert product.result_index == group.identity_index
    assert product.lattice_shift == (1, 0)
    assert concrete.seitz_product(1, 1) == product
    assert np.array_equal(concrete.lattice_shifts[1, 1], [1, 0])


def test_glide_factor_and_twisted_square_follow_bloch_convention():
    concrete = ConcreteFiniteGroup.from_space_group(_glide_group())
    identity = concrete.table.identity_index
    glide = concrete.local_index(1)

    for sign in (1, -1):
        factor = build_factor_system(
            concrete,
            [0.5, 0.0],
            1.0e-10,
            bloch_convention=BlochConvention(sign),
        )
        expected = np.exp(-sign * 2j * np.pi * 0.5)
        root = np.exp(-sign * 0.5j * np.pi)
        representation = TwistedRepresentation(
            (np.eye(2, dtype=np.complex128), root * np.eye(2, dtype=np.complex128)),
            concrete.table.multiplication,
            factor,
        )

        assert factor.phases[glide, glide] == pytest.approx(expected)
        assert np.allclose(
            representation.matrices[glide] @ representation.matrices[glide],
            -np.eye(2),
            atol=1.0e-12,
        )
        assert int(concrete.table.multiplication[glide, glide]) == identity
        representation.require_valid(tolerance=1.0e-12)


def test_nonreal_factor_changes_to_complex_conjugate_with_bloch_sign():
    concrete = ConcreteFiniteGroup.from_space_group(_glide_group())
    positive = build_factor_system(
        concrete,
        [0.25, 0.0],
        1.0e-10,
        bloch_convention=BlochConvention(1),
    )
    negative = build_factor_system(
        concrete,
        [0.25, 0.0],
        1.0e-10,
        bloch_convention=BlochConvention(-1),
    )

    assert positive.phases[1, 1] == pytest.approx(-1j)
    assert negative.phases[1, 1] == pytest.approx(1j)
    assert np.allclose(negative.phases, positive.phases.conj())


def test_factor_cocycle_is_checked_independently_of_stored_diagnostic():
    concrete = ConcreteFiniteGroup.from_space_group(_glide_group())
    factor = build_factor_system(concrete, [0.25, 0.0], 1.0e-10)
    assert factor.cocycle_residual_for(concrete.table.multiplication) < 1.0e-12

    corrupted_phases = factor.phases.copy()
    corrupted_phases[1, 0] *= np.exp(0.3j)
    corrupted = FactorSystem(
        factor.lattice_shifts,
        corrupted_phases,
        0.0,
        None,
        factor.bloch_sign,
        factor.tolerance,
    )
    representation = _projective_regular_representation(concrete, corrupted)

    assert representation.cocycle_residual > 1.0e-2
    with pytest.raises(ValueError, match="cocycle condition"):
        representation.require_valid()


def test_symmorphic_space_group_has_unit_factor_system():
    definition = load_space_group(P4MM)
    concrete = ConcreteFiniteGroup.from_space_group(definition.group)
    factor = build_factor_system(concrete, [0.37, 0.19], definition.tolerance)

    assert np.count_nonzero(concrete.lattice_shifts) == 0
    assert np.allclose(factor.phases, 1.0)
    assert factor.raw_trivial


def test_twisted_intertwiner_requires_the_same_factor_system():
    concrete = ConcreteFiniteGroup.from_space_group(_glide_group())
    positive_factor = build_factor_system(
        concrete, [0.25, 0.0], 1.0e-10, bloch_convention=BlochConvention(1)
    )
    negative_factor = build_factor_system(
        concrete, [0.25, 0.0], 1.0e-10, bloch_convention=BlochConvention(-1)
    )
    positive = _projective_regular_representation(concrete, positive_factor)
    negative = _projective_regular_representation(concrete, negative_factor)

    assert solve_intertwiner_space(positive, positive).dimension > 0
    with pytest.raises(ValueError, match="different Bloch signs"):
        solve_intertwiner_space(positive, negative)


def test_coboundary_factor_trivializes_but_projective_factor_does_not():
    definition = load_space_group(P4GM)
    coboundary = resolve_little_group(definition, [0.5, 0.5])
    projective = resolve_little_group(definition, [0.5, 0.0])
    coboundary_representation = _projective_regular_representation(
        coboundary.concrete, coboundary.factor_system
    )
    projective_representation = _projective_regular_representation(
        projective.concrete, projective.factor_system
    )

    coboundary_representation.require_valid(tolerance=1.0e-10)
    ordinary = coboundary_representation.trivialized_matrices(tolerance=1.0e-10)
    assert len(ordinary) == coboundary.concrete.order
    assert coboundary.factor_system.cohomologically_trivial

    projective_representation.require_valid(tolerance=1.0e-10)
    assert not projective.factor_system.cohomologically_trivial
    assert solve_intertwiner_space(
        projective_representation, projective_representation
    ).dimension > 0
    with pytest.raises(NotImplementedError, match="ordinary irreps"):
        projective_representation.trivialized_matrices()


def test_antiunitary_twisted_product_uses_semilinear_conjugation():
    group = SpaceGroup(
        (
            SpaceGroupOperation.identity(2),
            SpaceGroupOperation(
                np.diag([1, -1]),
                np.array([0.5, 0.0]),
                "anti_glide_x",
                True,
            ),
        )
    )
    concrete = ConcreteFiniteGroup.from_space_group(group)
    factor = build_factor_system(concrete, [0.5, 0.0], 1.0e-10)
    glide_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])
    representation = TwistedRepresentation(
        (np.eye(2), glide_matrix),
        concrete.table.multiplication,
        factor,
        (False, True),
    )

    assert factor.phases[1, 1] == pytest.approx(-1.0)
    assert np.allclose(glide_matrix @ glide_matrix.conj(), -np.eye(2))
    representation.require_valid(tolerance=1.0e-12)


def test_projective_high_symmetry_analysis_keeps_direct_intertwiner(monkeypatch):
    base = load_symmetry(P4GM)
    tolerance = DegeneracyTolerance()
    model = compose_symmetry_model(
        base,
        SymmetryCalculationSpec(
            target_specs=(WannierTargetSpec("origin_A", [0.0, 0.0], "A"),),
            representation_analysis=RepresentationAnalysisSpec(
                FieldKind.SCALAR,
                tolerance,
                (
                    RepresentationPointSpec(
                        "X",
                        np.array([0.5, 0.0]),
                        None,
                        ("origin_A",),
                        tolerance,
                    ),
                ),
            ),
        ),
    )
    axes = (np.array([-0.5, 0.0]), np.array([-0.5, 0.0]))
    context = build_symmetry_context(model, axes)
    target = model.target("origin_A")
    bands = tuple(range(target.wannier_dimension))
    band_grid = np.empty((2, 2, 1), dtype=object)
    for index in np.ndindex(band_grid.shape):
        band_grid[index] = list(bands)
    energies = np.ones((2, 2, 1, len(bands)), dtype=float)
    state = SimpleNamespace(E_idx=band_grid, energy_matrix=energies)

    class TargetProvider:
        def find_k_index(self, k_fractional):
            point = np.asarray(k_fractional, dtype=float)
            return tuple(
                int(np.argmin(np.abs((axis - point[dimension] + 0.5) % 1.0 - 0.5)))
                for dimension, axis in enumerate(axes)
            )

        def mapping(self, operation_index, source_index):
            flat = int(np.ravel_multi_index(tuple(source_index), (2, 2)))
            return context.k_mappings[operation_index][flat]

        def request_for_mapping(
            self,
            mapping,
            band_indices,
            *,
            operation=None,
            source_k_fractional=None,
            target_band_indices=None,
        ):
            source = (
                np.asarray(
                    [axes[dimension][mapping.source_k_index[dimension]] for dimension in range(2)]
                )
                if source_k_fractional is None
                else np.asarray(source_k_fractional, dtype=float)
            )
            return SimpleNamespace(
                operation_index=mapping.operation_index,
                operation=operation or model.group.operations[mapping.operation_index],
                source_k_fractional=source,
                band_indices=tuple(band_indices),
                target_band_indices=target_band_indices,
            )

        def sewing_matrix(self, request):
            return target.matrix(request.operation_index, request.source_k_fractional)

    monkeypatch.setattr(
        "pcwannier.symmetry.analysis._composition_residual",
        lambda *args, **kwargs: 0.0,
    )
    result = run_symmetry_analysis(state, context, provider=TargetProvider())
    point = result.points[0]

    assert point.factor_system is not None
    assert not point.factor_system.cohomologically_trivial
    assert point.physical_decomposition is None
    assert point.target_decomposition is None
    assert point.compatibility is None
    assert point.intertwiner_dimension is not None
    assert point.intertwiner_dimension > 0
    assert point.diagnostics.max_twisted_composition_residual < 1.0e-12
