from __future__ import annotations

from pathlib import Path
import re

import numpy as np


_CELL_HEADER_RE = re.compile(
    r"^CELL\s*[\(\[]\s*([^\)\]]*)\s*[\)\]]\s*(?:shape\s*=\s*\(([^)]*)\))?\s*:\s*$",
    re.IGNORECASE,
)


def load_cell_matrix(filename: str | Path, shape: tuple[int, ...] | None = None) -> np.ndarray:
    path = Path(filename)
    blocks: list[tuple[tuple[int, ...], np.ndarray]] = []
    current: tuple[int, ...] | None = None
    rows: list[list[complex]] = []

    def flush() -> None:
        nonlocal current, rows
        if current is None:
            return
        if rows:
            widths = {len(row) for row in rows}
            if len(widths) != 1:
                raise ValueError(f"Ragged rows in CELL{current}: row lengths = {sorted(widths)}")
            matrix = np.asarray(rows, dtype=np.complex128)
        else:
            matrix = np.empty((0, 0), dtype=np.complex128)
        blocks.append((current, matrix))
        current = None
        rows = []

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = _CELL_HEADER_RE.match(line)
            if match:
                flush()
                idx_text = (match.group(1) or "").strip()
                if idx_text == "" or idx_text.lower() == "root":
                    current = ()
                else:
                    current = tuple(int(part.strip()) for part in idx_text.split(",") if part.strip())
                continue
            if current is None:
                continue
            rows.append([_parse_complex(part) for part in line.split(",") if part.strip()])
    flush()

    if not blocks:
        raise ValueError(f"No CELL blocks found in {path}.")
    if all(idx == () for idx, _ in blocks):
        if len(blocks) != 1:
            raise ValueError(f"Multiple root CELL blocks found in {path}.")
        return blocks[0][1]

    ndims = max(len(idx) for idx, _ in blocks)
    if shape is None:
        maxima = [0] * ndims
        for idx, _ in blocks:
            for axis, value in enumerate(idx):
                maxima[axis] = max(maxima[axis], value)
        shape = tuple(value + 1 for value in maxima)
    if len(shape) < ndims:
        raise ValueError(f"Provided shape {shape} has fewer dims than cached CELL indices ({ndims}).")

    data = np.empty(shape, dtype=object)
    data.flat[:] = None
    for idx, matrix in blocks:
        full_idx = idx + (0,) * (len(shape) - len(idx))
        for axis, value in enumerate(full_idx):
            if value < 0 or value >= shape[axis]:
                raise ValueError(f"CELL index {full_idx} is out of bounds for shape {shape}.")
        data[full_idx] = matrix
    missing = [idx for idx in np.ndindex(shape) if data[idx] is None]
    if missing:
        preview = ", ".join(str(idx) for idx in missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise ValueError(f"Cached matrix {path} is missing CELL entries: {preview}{suffix}.")
    return data


def _parse_complex(token: str) -> complex:
    text = token.strip().strip("'\"")
    text = re.sub(r"\s+", "", text).replace("−", "-").replace("i", "j")
    if len(text) >= 2 and text[0] == "(" and text[-1] == ")":
        text = text[1:-1]
    return complex(text)
