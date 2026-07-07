from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySnapshot:
    rss_mb: float | None = None
    peak_rss_mb: float | None = None
    traced_current_mb: float | None = None
    traced_peak_mb: float | None = None


def now() -> float:
    return time.perf_counter()


def start_memory_tracking() -> None:
    return None


def memory_snapshot() -> MemorySnapshot:
    rss_mb, peak_rss_mb = _process_memory_mb()
    return MemorySnapshot(
        rss_mb=rss_mb,
        peak_rss_mb=peak_rss_mb,
    )


def format_elapsed(start: float, end: float | None = None) -> str:
    elapsed = (now() if end is None else end) - start
    return f"{elapsed:.3f} s"


def format_memory(snapshot: MemorySnapshot) -> str:
    parts = []
    if snapshot.rss_mb is not None:
        parts.append(f"rss={snapshot.rss_mb:.1f} MB")
    if snapshot.peak_rss_mb is not None:
        parts.append(f"peak_rss={snapshot.peak_rss_mb:.1f} MB")
    if snapshot.traced_peak_mb is not None:
        parts.append(f"python_peak={snapshot.traced_peak_mb:.1f} MB")
    return ", ".join(parts) if parts else "unavailable"


def _bytes_to_mb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def _process_memory_mb() -> tuple[float | None, float | None]:
    if os.name == "nt":
        return _windows_process_memory_mb()
    return _posix_process_memory_mb()


def _windows_process_memory_mb() -> tuple[float | None, float | None]:
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)

        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = kernel32.GetCurrentProcess()

        ok = False
        for dll_name, func_name in (("kernel32", "K32GetProcessMemoryInfo"), ("psapi", "GetProcessMemoryInfo")):
            try:
                dll = ctypes.WinDLL(dll_name, use_last_error=True)
                func = getattr(dll, func_name)
                func.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
                func.restype = wintypes.BOOL
                ok = bool(func(handle, ctypes.byref(counters), counters.cb))
                if ok:
                    break
            except (AttributeError, OSError):
                continue

        if not ok:
            return None, None
        return _bytes_to_mb(counters.WorkingSetSize), _bytes_to_mb(counters.PeakWorkingSetSize)
    except Exception:
        return None, None


def _posix_process_memory_mb() -> tuple[float | None, float | None]:
    try:
        import resource

        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            peak = _bytes_to_mb(peak)
        else:
            peak = peak / 1024.0
        return None, peak
    except Exception:
        return None, None
