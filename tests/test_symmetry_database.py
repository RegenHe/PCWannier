from itertools import combinations
from pathlib import Path

import numpy as np

from pcwannier.symmetry import (
    ConcreteFiniteGroup,
    load_builtin_finite_groups,
    load_finite_group,
    load_space_group,
    resolve_symmetry_file,
)


FINITE_GROUPS = {
    "C1.yaml": ("C1", 1),
    "C2.yaml": ("C2", 2),
    "C3.yaml": ("C3", 3),
    "C4.yaml": ("C4", 4),
    "C6.yaml": ("C6", 6),
    "Cs.yaml": ("Cs", 2),
    "C2v.yaml": ("C2v", 4),
    "C3v.yaml": ("C3v", 6),
    "C4v.yaml": ("C4v", 8),
    "C6v.yaml": ("C6v", 12),
}

SPACE_GROUPS = {
    "p1.yaml": "C1",
    "p2.yaml": "C2",
    "pm.yaml": "Cs",
    "pg.yaml": "Cs",
    "c1m1.yaml": "Cs",
    "p2mm.yaml": "C2v",
    "p2mg.yaml": "C2v",
    "p2gg.yaml": "C2v",
    "c2mm.yaml": "C2v",
    "p4.yaml": "C4",
    "p4mm.yaml": "C4v",
    "p4gm.yaml": "C4v",
    "p3.yaml": "C3",
    "p3m1.yaml": "C3v",
    "p31m.yaml": "C3v",
    "p6.yaml": "C6",
    "p6mm.yaml": "C6v",
}

CENTERED_RECTANGULAR = {"c1m1.yaml", "c2mm.yaml"}
RECTANGULAR = {
    "pm.yaml",
    "pg.yaml",
    "p2mm.yaml",
    "p2mg.yaml",
    "p2gg.yaml",
}
SQUARE = {"p4.yaml", "p4mm.yaml", "p4gm.yaml"}
HEXAGONAL = {"p3.yaml", "p3m1.yaml", "p31m.yaml", "p6.yaml", "p6mm.yaml"}


def test_finite_group_database_is_complete_and_valid():
    root = Path("pcwannier/symmetry/finite_groups")
    assert {path.name for path in root.glob("*.yaml")} == set(FINITE_GROUPS)

    loaded = load_builtin_finite_groups()
    assert {definition.name for definition in loaded.definitions} == {
        name for name, _ in FINITE_GROUPS.values()
    }
    for filename, (name, order) in FINITE_GROUPS.items():
        definition = load_finite_group(root / filename)
        assert definition.name == name
        assert definition.table.order == order
        assert sum(irrep.dimension**2 for irrep in definition.irreps) == order


def test_space_group_database_contains_all_wallpaper_groups():
    root = Path("pcwannier/symmetry/space_groups")
    assert {path.name for path in root.glob("*.yaml")} == set(SPACE_GROUPS)

    for filename, point_group in SPACE_GROUPS.items():
        definition = load_space_group(root / filename)
        identification = definition.identify_operations(
            range(len(definition.group.operations))
        )
        assert identification.canonical.name == point_group


def test_space_group_rotations_preserve_their_crystal_family_metric():
    root = Path("pcwannier/symmetry/space_groups")
    centered_metric = np.array([[1.25, -0.75], [-0.75, 1.25]])
    rectangular_metric = np.diag([2.0, 1.0])
    square_metric = np.eye(2)
    hexagonal_metric = np.array([[1.0, -0.5], [-0.5, 1.0]])

    for filename in SPACE_GROUPS:
        definition = load_space_group(root / filename)
        if filename in CENTERED_RECTANGULAR:
            metric = centered_metric
        elif filename in RECTANGULAR:
            metric = rectangular_metric
        elif filename in SQUARE:
            metric = square_metric
        elif filename in HEXAGONAL:
            metric = hexagonal_metric
        else:
            # p1 and p2 preserve every two-dimensional lattice metric.
            metric = np.array([[1.7, 0.2], [0.2, 0.9]])
        for operation in definition.group.operations:
            assert np.allclose(
                operation.rotation.T @ metric @ operation.rotation,
                metric,
                rtol=0.0,
                atol=1.0e-12,
            ), (filename, operation.name)


def test_all_closed_space_group_subsets_have_a_finite_group_model():
    root = Path("pcwannier/symmetry/space_groups")
    library = load_builtin_finite_groups()
    for filename in SPACE_GROUPS:
        definition = load_space_group(root / filename)
        group = definition.group
        remaining = [
            index for index in range(len(group.operations)) if index != group.identity_index
        ]
        for size in range(len(remaining) + 1):
            for selected in combinations(remaining, size):
                indices = (group.identity_index, *selected)
                try:
                    concrete = ConcreteFiniteGroup.from_space_group(group, indices)
                except ValueError:
                    continue
                identification = library.identify(concrete)
                assert identification.canonical.name in {
                    name for name, _ in FINITE_GROUPS.values()
                }


def test_common_short_space_group_symbols_resolve_to_modern_resources(tmp_path):
    aliases = {
        "cm.yaml": "c1m1.yaml",
        "cmm.yaml": "c2mm.yaml",
        "pmm.yaml": "p2mm.yaml",
        "pmg.yaml": "p2mg.yaml",
        "pgg.yaml": "p2gg.yaml",
        "p4m.yaml": "p4mm.yaml",
        "p4g.yaml": "p4gm.yaml",
        "p6m.yaml": "p6mm.yaml",
    }
    for alias, modern in aliases.items():
        assert resolve_symmetry_file(alias, tmp_path).name == modern
