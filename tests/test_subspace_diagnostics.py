from types import SimpleNamespace

import numpy as np

from pcwannier.compute.subspace_diagnostics import _evaluate_neighbor_subspace


class _FakeState:
    def __init__(self):
        self.config = SimpleNamespace(
            kdim=1,
            k_points=[np.asarray([-0.5, 0.0, 0.5])],
            composition_of_b=np.asarray([[1], [-1]], dtype=int),
        )

    def k_indices(self):
        yield from ((0, 0, 0), (1, 0, 0), (2, 0, 0))


def test_neighbor_subspace_smoothness_excludes_zero_mode_links():
    state = _FakeState()
    overlaps = {
        (0, 0, 0): np.diag([1.0, 0.1]),
        (1, 0, 0): np.diag([1.0, 0.2]),
        (2, 0, 0): np.diag([1.0, 0.8]),
    }

    result = _evaluate_neighbor_subspace(
        state,
        "fixture",
        lambda index, _direction: overlaps[index],
        lambda _index: np.eye(2),
        zero_mode_at=lambda index: index == (1, 0, 0),
    )

    assert result.link_count == 3
    assert result.minimum_singular_value == 0.1
    assert result.minimum_regular_singular_value == 0.8
    assert result.worst_source_index == (0, 0, 0)
    assert result.worst_regular_source_index == (2, 0, 0)
