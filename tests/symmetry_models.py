from __future__ import annotations

from pathlib import Path

import numpy as np

from pcwannier.symmetry import (
    DegeneracyTolerance,
    FieldKind,
    RepresentationAnalysisSpec,
    RepresentationPointSpec,
    SpaceGroup,
    SpaceGroupDefinition,
    SymmetryCalculationSpec,
    SymmetryModel,
    WannierTargetSpec,
    compose_symmetry_model,
    load_builtin_finite_groups,
    load_symmetry,
)


P4MM = Path("pcwannier/symmetry/space_groups/p4mm.yaml")
P4G = Path("pcwannier/symmetry/space_groups/p4g.yaml")


def p4mm_model(*, targets=(), points=(), gauge=None) -> SymmetryModel:
    base = load_symmetry(P4MM)
    analysis = None
    if points:
        tolerance = DegeneracyTolerance()
        specs = tuple(
            RepresentationPointSpec(
                name,
                np.asarray(kpoint, dtype=float),
                bands,
                target_names,
                tolerance,
            )
            for name, kpoint, bands, target_names in points
        )
        analysis = RepresentationAnalysisSpec(FieldKind.SCALAR, tolerance, specs)
    return compose_symmetry_model(
        base,
        SymmetryCalculationSpec(tuple(targets), analysis, gauge),
    )


def square_2c_model(*, analysis: bool = True) -> SymmetryModel:
    target = WannierTargetSpec("square_2c_A1", [0.5, 0.0], "A1")
    points = ()
    if analysis:
        selected = ("square_2c_A1",)
        points = (
            ("Gamma", [0.0, 0.0], None, selected),
            ("X", [0.5, 0.0], None, selected),
            ("M", [0.5, 0.5], None, selected),
        )
    return p4mm_model(targets=(target,), points=points)


def model_from_space_group(name: str, group: SpaceGroup, *, targets=()) -> SymmetryModel:
    definition = SpaceGroupDefinition(
        name,
        group.dimension,
        group.tolerance,
        group,
        load_builtin_finite_groups(),
    )
    base = SymmetryModel(
        group.dimension,
        group.tolerance,
        group,
        (),
        None,
        None,
        definition,
    )
    return compose_symmetry_model(base, SymmetryCalculationSpec(tuple(targets)))
