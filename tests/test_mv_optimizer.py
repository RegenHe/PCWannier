from types import SimpleNamespace

import numpy as np
import pytest

from pcwannier.compute.mv_optimizer import (
    MVCandidate,
    protected_mv_line_search,
)


def _object_grid(shape, factory):
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        result[index] = factory(index)
    return result


def test_protected_line_search_rejects_zero_diagonal_without_consuming_the_step():
    config = SimpleNamespace(composition_of_b=[[1], [-1]])
    state = SimpleNamespace(k_indices=lambda: iter(((0, 0, 0),)))
    holder = {"gauge": _object_grid((1, 1, 1), lambda _idx: np.eye(2, dtype=np.complex128))}

    def update(gauge):
        holder["gauge"] = gauge

    def get(*_args):
        return np.asarray(holder["gauge"][0, 0, 0])

    gradient = SimpleNamespace(
        U=holder["gauge"],
        omega=np.array([1.0, 0.0, 0.0]),
        mset=SimpleNamespace(config=config, state=state, update=update, get=get),
        update=lambda: setattr(gradient, "omega", np.array([0.9, 0.0, 0.0])),
    )

    def candidate(step):
        rotation = np.array(
            [[np.cos(step), -np.sin(step)], [np.sin(step), np.cos(step)]],
            dtype=np.complex128,
        )
        return MVCandidate(_object_grid((1, 1, 1), lambda _idx: rotation.copy()))

    result = protected_mv_line_search(
        gradient,
        candidate,
        initial_step=np.pi / 2.0,
        diagonal_floor=1.0e-8,
        max_steps=4,
    )

    assert result.accepted
    assert result.backtracking_steps == 1
    assert result.trial_step == pytest.approx(np.pi / 4.0)
    assert result.diagnostics.min_abs_diagonal == pytest.approx(2.0**-0.5)

