from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pcwannier.logging_utils import should_log_progress
from pcwannier.symmetry.reporting import log_bloch_symmetry_analysis


def test_progress_logging_is_sparse_and_keeps_boundaries():
    assert should_log_progress(1, total=100)
    assert not should_log_progress(2, total=100)
    assert should_log_progress(25, total=100)
    assert should_log_progress(73, total=100, finished=True)
    assert should_log_progress(100, total=100)


def test_symmetry_info_log_is_compact(caplog):
    block = SimpleNamespace(
        band_indices=(0, 1),
        decomposition=SimpleNamespace(multiplicities={"E": 1}),
        approximate_decomposition=None,
        character_fit_error=None,
    )
    point = SimpleNamespace(
        name="Gamma",
        sampled_k_fractional=np.zeros(2),
        little_group_name="C4v",
        factor_system=None,
        degenerate_blocks=(block,),
        diagnostics=SimpleNamespace(
            unitarity_error=1.0e-14,
            leakage=2.0e-14,
            selected_twisted_composition_residual=3.0e-14,
        ),
    )

    with caplog.at_level("INFO"):
        log_bloch_symmetry_analysis(SimpleNamespace(points=(point,)))

    assert "Bloch symmetry: points=1 blocks=1 labeled=1" in caplog.text
    assert "Symmetry Gamma:" in caplog.text
    assert "blocks=[0,1:E]" in caplog.text
    assert "unitary_operations" not in caplog.text
    assert "unitary_characters" not in caplog.text
    assert "eigenvalues" not in caplog.text
