from pathlib import Path

import numpy as np
import pytest

from pcwannier.symmetry import (
    BlochConvention,
    CrystallographicEmbedding,
    decompose_little_group_characters,
    load_space_group,
    resolve_little_group,
    symmetry_engine_versions,
)


SPACE_GROUPS = Path("pcwannier/symmetry/space_groups")


def test_crystallographic_embedding_preserves_fractional_operation_order():
    embedding = CrystallographicEmbedding(2)
    rotations = embedding.rotations(
        (
            np.eye(2, dtype=int),
            np.array([[0, -1], [1, 0]], dtype=int),
        )
    )

    assert rotations.shape == (2, 3, 3)
    assert np.array_equal(rotations[0], np.eye(3, dtype=int))
    assert np.array_equal(
        rotations[1],
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int),
    )
    assert np.array_equal(embedding.vector([0.25, -0.5]), [0.25, -0.5, 0.0])


def test_p4mm_point_group_and_ordinary_little_group_labels_use_external_engines():
    definition = load_space_group(SPACE_GROUPS / "p4mm.yaml")

    assert definition.point_group.symbol == "4mm"
    assert {
        irrep.name for irrep in resolve_little_group(definition, [0.0, 0.0]).require_irreps()
    } == {"A1", "A2", "B1", "B2", "E"}
    assert {
        irrep.name for irrep in resolve_little_group(definition, [0.5, 0.0]).require_irreps()
    } == {"A1", "A2", "B1", "B2"}
    spglib_version, spgrep_version = symmetry_engine_versions()
    assert spglib_version
    assert spgrep_version


@pytest.mark.parametrize("sign", [1, -1])
def test_p4gm_projective_small_representation_uses_project_factor(sign):
    definition = load_space_group(SPACE_GROUPS / "p4gm.yaml")
    resolved = resolve_little_group(
        definition,
        [0.5, 0.0],
        bloch_convention=BlochConvention(sign),
    )
    (irrep,) = resolved.require_irreps()
    factor = resolved.factor_system
    product = resolved.table.multiplication

    assert not factor.cohomologically_trivial
    assert irrep.name == "P1"
    assert irrep.dimension == 2
    assert irrep.label_source == "projective"
    assert max(
        np.linalg.norm(
            irrep.matrices[left] @ irrep.matrices[right]
            - factor.phases[left, right]
            * irrep.matrices[int(product[left, right])],
            ord="fro",
        )
        for left in range(resolved.table.order)
        for right in range(resolved.table.order)
    ) < 1.0e-10

    decomposition = decompose_little_group_characters(
        resolved,
        dict(zip(resolved.table.operation_names, irrep.characters, strict=True)),
    )
    assert decomposition.multiplicities == {"P1": 1}
    assert decomposition.max_residual < 1.0e-10


def test_p4gm_projective_factors_follow_bloch_sign():
    definition = load_space_group(SPACE_GROUPS / "p4gm.yaml")
    positive = resolve_little_group(
        definition,
        [0.5, 0.0],
        bloch_convention=BlochConvention(1),
    )
    negative = resolve_little_group(
        definition,
        [0.5, 0.0],
        bloch_convention=BlochConvention(-1),
    )

    assert np.allclose(
        positive.factor_system.phases,
        negative.factor_system.phases.conj(),
        rtol=0.0,
        atol=1.0e-12,
    )
