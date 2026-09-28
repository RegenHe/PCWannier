from importlib import resources

import numpy as np
import pytest

from pcwannier.formal_tb import (
    build_formal_tight_binding_from_ebr,
    fit_formal_tight_binding,
)
from pcwannier.symmetry import load_symmetry
from pcwannier.symmetry.io import load_symmetry_from_spglib
from pcwannier.symmetry.representation import build_symmetry_context


def _p1_context():
    path = resources.files("pcwannier.symmetry").joinpath(
        "space_groups", "p1.yaml"
    )
    model = load_symmetry(path)
    return build_symmetry_context(
        model,
        (np.asarray([-0.5, 0.0]), np.asarray([-0.5, 0.0])),
    )


def test_sg221_a2g_3d_first_range_reproduces_four_parameter_form():
    model = load_symmetry_from_spglib("hall:517")
    context = build_symmetry_context(model, [np.asarray([-0.5, 0.0])] * 3)

    formal = build_formal_tight_binding_from_ebr(
        context,
        "sg221",
        {"A2g@3d": 1},
        np.eye(3),
        hopping_order=1,
    )

    assert formal.dimension == 3
    assert formal.parameter_count == 4
    assert formal.bond_distances == pytest.approx((np.sqrt(0.5), 1.0))
    assert max(seed.distance for seed in formal.seeds) <= 1.0 + 1.0e-10
    parameters = np.asarray([0.7, -0.2, 0.11, 0.31])
    assert formal.covariance_residual(parameters) < 1.0e-12
    assert formal.time_reversal_residual(parameters) < 1.0e-12


def test_formal_tba_fit_recovers_a_synthetic_periodic_spectrum():
    context = _p1_context()
    formal = build_formal_tight_binding_from_ebr(
        context,
        "p1",
        {"A@1a": 1},
        np.eye(2),
        hopping_order=1,
    )
    assert formal.parameter_count == 3
    points = np.asarray(
        [
            [0.0, 0.0],
            [0.125, 0.0],
            [0.0, 0.25],
            [0.2, -0.3],
            [-0.4, 0.1],
            [0.35, 0.4],
        ]
    )
    expected_parameters = np.asarray([1.25, -0.3, 0.17])
    target = formal.eigenvalues(points, expected_parameters)

    fitted = fit_formal_tight_binding(formal, points, target)

    assert fitted.success
    assert fitted.rms_error < 1.0e-11
    assert fitted.max_error < 1.0e-10
    assert fitted.fitted_eigenvalues == pytest.approx(target, abs=1.0e-10)


def test_formal_tba_hopping_order_changes_the_allowed_parameter_space():
    context = _p1_context()
    onsite = build_formal_tight_binding_from_ebr(
        context,
        "p1",
        {"A@1a": 1},
        np.eye(2),
        hopping_order=0,
    )
    nearest = build_formal_tight_binding_from_ebr(
        context,
        "p1",
        {"A@1a": 1},
        np.eye(2),
        hopping_order=1,
    )

    assert onsite.parameter_count == 1
    assert nearest.parameter_count > onsite.parameter_count
    assert onsite.bond_distances == ()


def test_formal_tba_transfers_parameters_between_shell_bases():
    context = _p1_context()
    first = build_formal_tight_binding_from_ebr(
        context, "p1", {"A@1a": 1}, np.eye(2), hopping_order=1
    )
    second = build_formal_tight_binding_from_ebr(
        context, "p1", {"A@1a": 1}, np.eye(2), hopping_order=2
    )
    parameters = np.asarray([1.25, -0.3, 0.17])

    transferred = second.transfer_parameters_from(first, parameters)
    points = np.asarray([[0.0, 0.0], [0.17, -0.23], [-0.41, 0.32]])

    assert second.eigenvalues(points, transferred) == pytest.approx(
        first.eigenvalues(points, parameters), abs=1.0e-11
    )


def test_formal_tba_uses_standard_band_path_and_plotter(tmp_path):
    context = _p1_context()
    formal = build_formal_tight_binding_from_ebr(
        context,
        "p1",
        {"A@1a": 1},
        np.eye(2),
        hopping_order=1,
    )
    path = [
        {"name": "G", "point": [0.0, 0.0], "num": 4},
        {"name": "X", "point": [0.5, 0.0], "num": 4},
        {"name": "G", "point": [0.0, 0.0], "num": 4},
    ]
    output = tmp_path / "formal-band.png"

    result = formal.plot_bands(
        output,
        np.asarray([1.0, -0.2, 0.1]),
        path,
    )

    assert result.k_path.shape == (9, 2)
    assert result.energies.shape == (9, 1)
    assert result.high_sym_points == [["G", 0], ["X", 4], ["G", 8]]
    assert output.is_file()
    assert output.stat().st_size > 0


def test_nonsymmorphic_formal_model_reuses_target_translation_phases():
    path = resources.files("pcwannier.symmetry").joinpath(
        "space_groups", "p4gm.yaml"
    )
    context = build_symmetry_context(
        load_symmetry(path),
        (np.asarray([-0.5, 0.0]), np.asarray([-0.5, 0.0])),
    )
    formal = build_formal_tight_binding_from_ebr(
        context,
        "p4gm",
        {"A1@2b": 1},
        np.eye(2),
        hopping_order=1,
    )

    parameters = np.arange(1, formal.parameter_count + 1, dtype=float)
    assert formal.covariance_residual(parameters) < 1.0e-12


def test_formal_tba_rejects_unknown_or_negative_ebr_multiplicity():
    context = _p1_context()
    with pytest.raises(ValueError, match="Unknown EBR"):
        build_formal_tight_binding_from_ebr(
            context, "p1", {"missing": 1}, np.eye(2), hopping_order=0
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        build_formal_tight_binding_from_ebr(
            context, "p1", {"A@1a": -1}, np.eye(2), hopping_order=0
        )
