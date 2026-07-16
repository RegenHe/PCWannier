from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ..matrix_io import load_cell_matrix, save_cell_matrix


_FORMAT = "pcwannier-sewing-text-v1"
_CACHE_PREFIX = "PCWANNIER_D_CACHE "
_ENTRY_PREFIX = "PCWANNIER_D_ENTRY "


@dataclass(frozen=True)
class SewingMatrixCacheEntry:
    """One full outer-window sewing matrix for an exact Seitz action at k."""

    operation_rotation: np.ndarray
    operation_translation: np.ndarray
    source_k_fractional: np.ndarray
    target_k_fractional: np.ndarray
    reciprocal_lattice_shift: tuple[int, ...]
    source_band_indices: tuple[int, ...]
    target_band_indices: tuple[int, ...]
    field_kind: str
    matrix: np.ndarray


@dataclass(frozen=True)
class SewingMatrixCache:
    dimension: int
    bloch_sign: int
    k_shape: tuple[int, ...]
    calculation_fingerprint: str
    entries: tuple[SewingMatrixCacheEntry, ...]


def save_sewing_matrix_cache(
    filename: str | Path,
    entries: Iterable[SewingMatrixCacheEntry],
    *,
    dimension: int,
    bloch_sign: int,
    k_shape: tuple[int, ...],
    calculation_fingerprint: str,
) -> None:
    """Write physical sewing matrices as ordinary text CELL blocks."""

    cached_entries = tuple(entries)
    if not cached_entries:
        raise ValueError("Cannot write an empty D matrix cache.")
    matrices = np.empty((len(cached_entries),), dtype=object)
    cell_comments: dict[tuple[int, ...], tuple[str, ...]] = {}
    for index, entry in enumerate(cached_entries):
        matrix = _validated_matrix(entry.matrix, f"CELL({index})")
        expected_shape = (len(entry.target_band_indices), len(entry.source_band_indices))
        if matrix.shape != expected_shape:
            raise ValueError(
                f"D cache entry {index} has shape {matrix.shape}; expected {expected_shape}."
            )
        matrices[index] = matrix
        record = {
            "index": index,
            "operation_rotation": np.asarray(entry.operation_rotation, dtype=np.int64).tolist(),
            "operation_translation": np.asarray(entry.operation_translation, dtype=float).tolist(),
            "source_k_fractional": np.asarray(entry.source_k_fractional, dtype=float).tolist(),
            "target_k_fractional": np.asarray(entry.target_k_fractional, dtype=float).tolist(),
            "reciprocal_lattice_shift": [int(value) for value in entry.reciprocal_lattice_shift],
            "source_band_indices": [int(value) for value in entry.source_band_indices],
            "target_band_indices": [int(value) for value in entry.target_band_indices],
            "field_kind": str(entry.field_kind),
        }
        cell_comments[(index,)] = (_ENTRY_PREFIX + _compact_json(record),)

    metadata = {
        "format": _FORMAT,
        "dimension": int(dimension),
        "bloch_sign": int(bloch_sign),
        "k_shape": [int(value) for value in k_shape],
        "calculation_fingerprint": _validated_fingerprint(calculation_fingerprint),
        "entry_count": len(cached_entries),
    }
    save_cell_matrix(
        filename,
        matrices,
        matrices.shape,
        precision=17,
        header_comments=(
            "Physical Bloch sewing matrices d_tilde(g,k).",
            _CACHE_PREFIX + _compact_json(metadata),
        ),
        cell_comments=cell_comments,
    )


def load_sewing_matrix_cache(filename: str | Path) -> SewingMatrixCache:
    path = Path(filename)
    metadata = None
    records: dict[int, dict] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if line.startswith("#"):
                    line = line[1:].strip()
                if line.startswith(_CACHE_PREFIX):
                    if metadata is not None:
                        raise ValueError("multiple D cache metadata records")
                    metadata = json.loads(line[len(_CACHE_PREFIX) :])
                elif line.startswith(_ENTRY_PREFIX):
                    record = json.loads(line[len(_ENTRY_PREFIX) :])
                    index = int(record.get("index", -1))
                    if index in records:
                        raise ValueError(f"duplicate D cache entry metadata for CELL({index})")
                    records[index] = record
    except Exception as exc:
        raise ValueError(f"Cannot read D matrix cache {path}: {exc}") from exc

    if metadata is None or metadata.get("format") != _FORMAT:
        raise ValueError(f"D matrix cache {path} has missing or unsupported metadata.")
    dimension = int(metadata.get("dimension", 0))
    bloch_sign = int(metadata.get("bloch_sign", 0))
    k_shape = tuple(int(value) for value in metadata.get("k_shape", ()))
    fingerprint = _validated_fingerprint(metadata.get("calculation_fingerprint", ""))
    entry_count = int(metadata.get("entry_count", -1))
    if dimension <= 0 or bloch_sign not in (-1, 1) or len(k_shape) != dimension:
        raise ValueError(f"D matrix cache {path} has invalid global metadata.")
    if entry_count < 0 or set(records) != set(range(entry_count)):
        raise ValueError(f"D matrix cache {path} has incomplete CELL metadata.")

    matrices = load_cell_matrix(path, (entry_count,))
    entries = []
    for index in range(entry_count):
        record = records[index]
        source_bands = tuple(int(value) for value in record.get("source_band_indices", ()))
        target_bands = tuple(int(value) for value in record.get("target_band_indices", ()))
        if len(set(source_bands)) != len(source_bands) or len(set(target_bands)) != len(target_bands):
            raise ValueError(f"D cache entry {index} contains duplicate band ids.")
        matrix = _validated_matrix(matrices[index], f"CELL({index})")
        expected_shape = (len(target_bands), len(source_bands))
        if matrix.shape != expected_shape:
            raise ValueError(
                f"D cache entry {index} has shape {matrix.shape}; expected {expected_shape}."
            )
        rotation = np.asarray(record.get("operation_rotation"), dtype=np.int64)
        translation = np.asarray(record.get("operation_translation"), dtype=float)
        source_k = np.asarray(record.get("source_k_fractional"), dtype=float)
        target_k = np.asarray(record.get("target_k_fractional"), dtype=float)
        shift = tuple(int(value) for value in record.get("reciprocal_lattice_shift", ()))
        if rotation.shape != (dimension, dimension):
            raise ValueError(f"D cache entry {index} has invalid rotation shape {rotation.shape}.")
        if translation.shape != (dimension,) or source_k.shape != (dimension,) or target_k.shape != (dimension,):
            raise ValueError(f"D cache entry {index} has invalid fractional-vector dimensions.")
        if len(shift) != dimension or not all(
            np.all(np.isfinite(value)) for value in (translation, source_k, target_k)
        ):
            raise ValueError(f"D cache entry {index} contains invalid coordinates.")
        entries.append(
            SewingMatrixCacheEntry(
                operation_rotation=rotation,
                operation_translation=translation,
                source_k_fractional=source_k,
                target_k_fractional=target_k,
                reciprocal_lattice_shift=shift,
                source_band_indices=source_bands,
                target_band_indices=target_bands,
                field_kind=str(record.get("field_kind", "")),
                matrix=matrix.copy(),
            )
        )
    return SewingMatrixCache(dimension, bloch_sign, k_shape, fingerprint, tuple(entries))


def _validated_matrix(matrix, label: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError(f"D cache matrix {label} must be two-dimensional, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"D cache matrix {label} contains non-finite values.")
    return array


def _validated_fingerprint(value) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("D matrix cache calculation fingerprint must be a SHA-256 hex digest.")
    return text


def _compact_json(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
