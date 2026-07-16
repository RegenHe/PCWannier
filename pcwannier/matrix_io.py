from __future__ import annotations

from pathlib import Path
import math
import re
from collections.abc import Iterable, Mapping

import numpy as np


_CELL_HEADER_RE = re.compile(
    r"^CELL\s*[\(\[]\s*([^\)\]]*)\s*[\)\]]\s*(?:shape\s*=\s*\(([^)]*)\))?\s*:\s*$",
    re.IGNORECASE,
)


def save_cell_matrix(
    filename: str | Path,
    data,
    shape: tuple | None = None,
    *,
    precision: int = 8,
    header_comments: Iterable[str] = (),
    cell_comments: Mapping[tuple[int, ...], Iterable[str]] | None = None,
) -> None:
    """Write nested numeric matrices in the shared human-readable CELL format."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    comments = {} if cell_comments is None else cell_comments

    def as_matrix(obj):
        arr = np.asarray(obj)
        if arr.dtype == object:
            return None
        if arr.ndim == 2:
            return arr
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 0 and np.issubdtype(arr.dtype, np.number):
            return arr.reshape(1, 1)
        return None

    def iter_cells(obj, prefix=()):
        matrix = as_matrix(obj)
        if matrix is not None:
            yield prefix, matrix
            return
        if isinstance(obj, (list, tuple, np.ndarray)):
            for index, child in enumerate(obj):
                yield from iter_cells(child, prefix + (index,))
            return
        raise TypeError(f"Unsupported cell data at {prefix}: {type(obj)}")

    with path.open("w", encoding="utf-8") as handle:
        for comment in header_comments:
            handle.write(f"# {str(comment).strip()}\n")
        if shape is not None:
            handle.write(f"# Declared grid shape (top-level): {shape}\n")
        handle.write("# Each CELL may have its own matrix shape (ragged supported).\n")
        for index, matrix in iter_cells(data):
            for comment in comments.get(index, ()):
                handle.write(f"# {str(comment).strip()}\n")
            matrix = np.asarray(matrix)
            handle.write(f"CELL{index if index else '(root)'} shape={tuple(matrix.shape)}:\n")
            if matrix.size:
                for row in matrix:
                    handle.write(
                        ", ".join(_format_complex(value, precision) for value in np.asarray(row).reshape(-1))
                        + "\n"
                    )
            handle.write("\n")


def load_cell_matrix(filename: str | Path, shape: tuple[int, ...] | None = None) -> np.ndarray:
    path = Path(filename)
    blocks: list[tuple[tuple[int, ...], np.ndarray]] = []
    current: tuple[int, ...] | None = None
    declared_shape: tuple[int, int] | None = None
    rows: list[list[complex]] = []
    seen_indices: set[tuple[int, ...]] = set()
    saw_root = False
    saw_indexed = False

    def flush() -> None:
        nonlocal current, declared_shape, rows
        if current is None:
            return
        if rows:
            widths = {len(row) for row in rows}
            if len(widths) != 1:
                raise ValueError(f"Ragged rows in CELL{current}: row lengths = {sorted(widths)}")
            matrix = np.asarray(rows, dtype=np.complex128)
        else:
            empty_shape = declared_shape if declared_shape is not None and 0 in declared_shape else (0, 0)
            matrix = np.empty(empty_shape, dtype=np.complex128)
        if declared_shape is not None and matrix.shape != declared_shape:
            raise ValueError(
                f"CELL{current} declares shape {declared_shape}, but contains matrix shape {matrix.shape}."
            )
        blocks.append((current, matrix))
        current = None
        declared_shape = None
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
                    saw_root = True
                else:
                    parts = [part.strip() for part in idx_text.split(",")]
                    if parts and parts[-1] == "":
                        parts.pop()
                    if any(not part for part in parts):
                        raise ValueError(f"Invalid CELL index in {line!r}.")
                    current = tuple(int(part) for part in parts)
                    if any(value < 0 for value in current):
                        raise ValueError(f"CELL index {current} must not contain negative values.")
                    saw_indexed = True
                if current in seen_indices:
                    raise ValueError(f"Duplicate CELL index {current} in {path}.")
                seen_indices.add(current)
                shape_text = (match.group(2) or "").strip()
                if shape_text:
                    shape_parts = [part.strip() for part in shape_text.split(",") if part.strip()]
                    if len(shape_parts) != 2:
                        raise ValueError(f"CELL{current} must declare a two-dimensional matrix shape.")
                    declared_shape = tuple(int(part) for part in shape_parts)
                    if any(value < 0 for value in declared_shape):
                        raise ValueError(f"CELL{current} declares a negative matrix shape {declared_shape}.")
                continue
            if current is None:
                continue
            rows.append([_parse_complex(part) for part in line.split(",") if part.strip()])
    flush()

    if not blocks:
        raise ValueError(f"No CELL blocks found in {path}.")
    if saw_root and saw_indexed:
        raise ValueError(f"Root and indexed CELL blocks must not be mixed in {path}.")
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
    assigned_full_indices: set[tuple[int, ...]] = set()
    for idx, matrix in blocks:
        full_idx = idx + (0,) * (len(shape) - len(idx))
        for axis, value in enumerate(full_idx):
            if value < 0 or value >= shape[axis]:
                raise ValueError(f"CELL index {full_idx} is out of bounds for shape {shape}.")
        if full_idx in assigned_full_indices:
            raise ValueError(f"Duplicate effective CELL index {full_idx} in {path}.")
        assigned_full_indices.add(full_idx)
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


def _format_complex(value, precision: int) -> str:
    real = float(np.real(value))
    imag = float(np.imag(value))
    if abs(imag) < 1.0e-12:
        imag = 0.0
    sign = " - " if math.copysign(1.0, imag) < 0 else " + "
    return f"{real:.{precision}f}{sign}{abs(imag):.{precision}f}j"
