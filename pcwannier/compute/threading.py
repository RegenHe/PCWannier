from __future__ import annotations

from contextlib import nullcontext


def blas_thread_limit(worker_threads: int):
    if int(worker_threads) <= 1:
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()
    return threadpool_limits(limits=1)


def threadpool_summary() -> str:
    try:
        from threadpoolctl import threadpool_info
    except Exception:
        return "unavailable"
    info = threadpool_info()
    if not info:
        return "none"
    parts = []
    for item in info:
        api = item.get("user_api") or item.get("internal_api") or "unknown"
        prefix = item.get("prefix") or "library"
        threads = item.get("num_threads", "?")
        parts.append(f"{prefix}:{api}:{threads}")
    return ", ".join(parts)
