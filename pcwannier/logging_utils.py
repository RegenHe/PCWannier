from __future__ import annotations

from pathlib import Path
import logging
import sys


DEFAULT_PROGRESS_INTERVAL = 1


def configure_logging(log_file: str | Path | None = "log.txt", level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None and str(log_file).lower() != "false":
        path = Path(log_file)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def should_log_progress(
    iteration: int,
    *,
    total: int | None = None,
    finished: bool = False,
    interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> bool:
    """Log every iteration by default; callers may request a sparser interval."""

    if iteration <= 1 or finished:
        return True
    if total is not None and iteration >= total:
        return True
    return interval > 0 and iteration % interval == 0
