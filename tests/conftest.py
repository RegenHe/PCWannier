from pathlib import Path

import pytest


C4V_DATASET = Path("datasets/c4v")


def pytest_runtest_setup(item):
    if "requires_dataset" not in item.keywords:
        return
    required = ("incar", "mesh.mphtxt", "Ez.txt", "eps.txt", "E.txt")
    if any(not (C4V_DATASET / name).is_file() for name in required):
        pytest.skip("local datasets/c4v COMSOL data is unavailable")
