from __future__ import annotations

from contextlib import contextmanager
import logging
from time import perf_counter

from .runtime_info import memory_snapshot


@contextmanager
def timed_step(name: str, logger: logging.Logger | None = None, **details):
    log = logger or logging.getLogger(__name__)
    suffix = _format_details(details)
    log.debug("START %s%s", name, suffix)
    start = perf_counter()
    start_memory = memory_snapshot()
    try:
        yield
    except Exception:
        elapsed = perf_counter() - start
        memory_suffix = _format_memory_delta(start_memory, memory_snapshot())
        log.exception("FAILED %s after %.3fs%s%s", name, elapsed, suffix, memory_suffix)
        raise
    elapsed = perf_counter() - start
    memory_suffix = _format_memory_delta(start_memory, memory_snapshot())
    log.info("END %s in %.3fs%s%s", name, elapsed, suffix, memory_suffix)


def _format_details(details: dict) -> str:
    clean = {key: value for key, value in details.items() if value is not None}
    if not clean:
        return ""
    body = ", ".join(f"{key}={value}" for key, value in clean.items())
    return f" ({body})"


def _format_memory_delta(start, end) -> str:
    parts = []
    if start.rss_mb is not None and end.rss_mb is not None:
        parts.extend(
            (
                f"rss_delta={end.rss_mb - start.rss_mb:+.1f} MB",
                f"rss={end.rss_mb:.1f} MB",
            )
        )
    if start.peak_rss_mb is not None and end.peak_rss_mb is not None:
        parts.extend(
            (
                f"peak_delta={end.peak_rss_mb - start.peak_rss_mb:+.1f} MB",
                f"peak_rss={end.peak_rss_mb:.1f} MB",
            )
        )
    if not parts:
        return ""
    return " (" + ", ".join(parts) + ")"
