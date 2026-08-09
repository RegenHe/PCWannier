from __future__ import annotations

def threadpool_summary() -> str:
    from threadpoolctl import threadpool_info
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
