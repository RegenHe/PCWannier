from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


_FALLBACK_VERSION = "1.0.3"


def get_version() -> str:
    try:
        return version("pcwannier")
    except PackageNotFoundError:
        return _FALLBACK_VERSION


__version__ = get_version()
