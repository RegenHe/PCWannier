from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec


BACKEND_PYTHON = "python"
BACKEND_NUMBA = "numba"
BACKEND_AUTO = "auto"
VALID_BACKENDS = {BACKEND_PYTHON, BACKEND_NUMBA, BACKEND_AUTO}


def normalize_backend(value: str | None) -> str:
    if value is None:
        return BACKEND_PYTHON
    name = str(value).strip().lower()
    aliases = {
        "numpy": BACKEND_PYTHON,
        "np": BACKEND_PYTHON,
        "py": BACKEND_PYTHON,
        "jit": BACKEND_NUMBA,
    }
    name = aliases.get(name, name)
    if name not in VALID_BACKENDS:
        raise ValueError(f"Invalid compute backend {value!r}; expected one of {sorted(VALID_BACKENDS)}.")
    return name


@lru_cache(maxsize=1)
def is_numba_available() -> bool:
    return find_spec("numba") is not None


def resolve_backend(value: str | None) -> str:
    name = normalize_backend(value)
    if name == BACKEND_AUTO:
        return BACKEND_NUMBA if is_numba_available() else BACKEND_PYTHON
    if name == BACKEND_NUMBA and not is_numba_available():
        raise RuntimeError(
            "compute_backend='numba' was requested, but numba is not installed. "
            "Install the optional numba extra or use compute_backend='python'/'auto'."
        )
    return name
