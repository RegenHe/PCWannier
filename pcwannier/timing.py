from __future__ import annotations

from contextlib import contextmanager
import logging
from time import perf_counter


@contextmanager
def timed_step(name: str, logger: logging.Logger | None = None, **details):
    log = logger or logging.getLogger(__name__)
    suffix = _format_details(details)
    log.info("START %s%s", name, suffix)
    start = perf_counter()
    try:
        yield
    except Exception:
        elapsed = perf_counter() - start
        log.exception("FAILED %s after %.3fs%s", name, elapsed, suffix)
        raise
    elapsed = perf_counter() - start
    log.info("END %s in %.3fs%s", name, elapsed, suffix)


def _format_details(details: dict) -> str:
    clean = {key: value for key, value in details.items() if value is not None}
    if not clean:
        return ""
    body = ", ".join(f"{key}={value}" for key, value in clean.items())
    return f" ({body})"
