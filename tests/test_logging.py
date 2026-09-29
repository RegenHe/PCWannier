from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

import pcwannier.timing as timing_module
from pcwannier.compute.gradient import Gradient
from pcwannier.logging_utils import should_log_progress
from pcwannier.symmetry.reporting import (
    log_bloch_symmetry_analysis,
    log_gamma_zero_regularization,
    log_target_compatibilities,
)

from .test_compute import _synthetic_gradient_optimizer


def test_progress_logging_keeps_every_iteration_by_default():
    assert all(should_log_progress(iteration, total=100) for iteration in range(1, 101))


def test_explicit_sparse_progress_interval_keeps_boundaries():
    assert should_log_progress(1, total=100, interval=25)
    assert not should_log_progress(2, total=100, interval=25)
    assert should_log_progress(25, total=100, interval=25)
    assert should_log_progress(73, total=100, finished=True, interval=25)
    assert should_log_progress(100, total=100, interval=25)


def test_gradient_logs_every_iteration_and_keeps_line_search_at_debug(caplog):
    gradient, _ = _synthetic_gradient_optimizer([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    with caplog.at_level("DEBUG", logger="pcwannier.compute.gradient"):
        Gradient.iter(gradient, err_diff=1.0e-12, max_iter=6, epsilon=0.01)

    iterations = [record for record in caplog.records if record.getMessage().startswith("gradient iter ")]
    diagnostics = [record for record in caplog.records if record.getMessage().startswith("gradient line search iter ")]
    assert len(iterations) == len(diagnostics) == 6
    for iteration, record in enumerate(iterations, start=1):
        message = record.getMessage()
        assert record.levelno == logging.INFO
        assert message.startswith(f"gradient iter {iteration} omega=")
        assert "omega_I=" in message and "omega_OD=" in message and "omega_D=" in message
        assert "max_gradient_norm=" in message and "epsilon=" in message
        assert "accepted_step=" not in message and "backtracks=" not in message
    assert all(record.levelno == logging.DEBUG for record in diagnostics)
    assert all("accepted_step=" in record.getMessage() for record in diagnostics)


def test_timed_step_logs_start_and_end_in_original_style(caplog, monkeypatch):
    timestamps = iter((10.0, 10.125))
    snapshots = iter((
        SimpleNamespace(rss_mb=10.0, peak_rss_mb=20.0),
        SimpleNamespace(rss_mb=11.5, peak_rss_mb=25.0),
    ))
    monkeypatch.setattr(timing_module, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(timing_module, "memory_snapshot", lambda: next(snapshots))

    with caplog.at_level("INFO"):
        with timing_module.timed_step("load input data", dataset_type="mpb", disabled=None):
            pass

    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.INFO, "START load input data (dataset_type=mpb)"),
        (logging.INFO, "END load input data in 0.125s (dataset_type=mpb) "
         "(rss_delta=+1.5 MB, rss=11.5 MB)"),
    ]


@pytest.mark.parametrize("irrep", ["exact", "approximate", "unavailable"])
def test_symmetry_info_log_keeps_full_details_and_one_based_bands(caplog, irrep):
    decomposition = SimpleNamespace(multiplicities={"E": 1})
    antiunitary = SimpleNamespace(
        operation_name="T", square_operation_name="E",
        square_eigenvalues=(1.0, 1.0), square_residual=1.0e-14,
    )
    block = SimpleNamespace(
        band_indices=(0, 8),
        energies=(0.25, 0.25),
        decomposition=decomposition if irrep == "exact" else None,
        approximate_decomposition=decomposition if irrep == "approximate" else None,
        character_fit_error=1.0e-5 if irrep == "approximate" else None,
        irrep_unavailable_reason="test representation unavailable",
        unitary_characters={"E": 2.0},
        coupled_outer_bands=(2,),
        candidate_excluded_bands=(3,),
        unitarity_error=1.0e-14,
        leakage=2.0e-14,
        twisted_composition_residual=3.0e-14,
        antiunitary_diagnostics=(antiunitary,),
    )
    point = SimpleNamespace(
        name="Gamma",
        sampled_k_fractional=np.zeros(2),
        little_group_name="C4v",
        unitary_subgroup_name="C4v",
        unitary_operation_names=("E",),
        antiunitary_operation_names=("T",),
        conjugacy_classes=(("E",),),
        finite_group_mapping={"E": "E"},
        outer_band_indices=(0, 2, 8),
        band_indices=(0, 8),
        outer_unitarity_error=4.0e-14,
        unitary_characters={"E": 2.0},
        resolved_little_group=None,
        factor_system=None,
        degenerate_blocks=(block,),
        diagnostics=SimpleNamespace(
            unitarity_error=1.0e-14,
            leakage=2.0e-14,
            outer_composition_residual=5.0e-14,
            selected_twisted_composition_residual=3.0e-14,
        ),
    )

    with caplog.at_level("INFO"):
        log_bloch_symmetry_analysis(SimpleNamespace(points=(point,)))

    assert "Bloch symmetry point Gamma: little_co_group=C4v unitary_subgroup=C4v" in caplog.text
    assert "outer_bands(1-based)=(1, 3, 9) analyzed_bands(1-based)=(1, 9)" in caplog.text
    assert "blocks=((1, 9),)" in caplog.text
    assert "unitary_operations=('E',) antiunitary_operations=('T',)" in caplog.text
    assert "unitary_characters={'E': (2+0j)}" in caplog.text
    assert "Bloch symmetry block Gamma bands(1-based)=(1, 9)" in caplog.text
    assert "eigenvalues=((0.25+0j), (0.25+0j)) degeneracy=2" in caplog.text
    assert "coupled_outer_bands(1-based)=(3,) candidate_excluded_bands(1-based)=(4,)" in caplog.text
    assert "Bloch antiunitary block Gamma bands(1-based)=(1, 9) operation=T square=E" in caplog.text
    expected_label = {
        "exact": "irrep=E class_character_summary=",
        "approximate": "irrep=E (approximate; character_error=1e-05)",
        "unavailable": "irrep=unavailable (test representation unavailable)",
    }[irrep]
    assert expected_label in caplog.text
    assert all(record.levelno == logging.INFO for record in caplog.records)
    assert point.band_indices == (0, 8) and block.band_indices == (0, 8)


def test_gamma_and_target_compatibility_logs_keep_full_details(caplog):
    decomposition = SimpleNamespace(multiplicities={"T1g": 1})
    gamma = SimpleNamespace(
        point_name="Gamma", transverse_band_indices=(0, 1),
        longitudinal_zero_band_indices=(8,), decomposition=decomposition,
        unitarity_error=1.0e-14, twisted_composition_residual=2.0e-14,
        note="constant magnetic fields",
    )
    target = SimpleNamespace(
        point_name="Gamma", target_names=("magnetic",),
        target_unitary_characters={"E": 3.0}, target_decomposition=decomposition,
        compatibility=SimpleNamespace(compatible=True), intertwiner_dimension=1,
    )
    with caplog.at_level("INFO"):
        log_gamma_zero_regularization(gamma)
        log_target_compatibilities((target,))

    assert "physical_T_bands(1-based)=(1, 2) longitudinal_zero_bands(1-based)=(9,)" in caplog.text
    assert "note=constant magnetic fields" in caplog.text
    assert "Target compatibility Gamma: targets=('magnetic',)" in caplog.text
    assert "target_unitary_characters={'E': (3+0j)} target_irreps={'T1g': 1}" in caplog.text
    assert "compatible=True direct_intertwiner_dimension=1" in caplog.text
