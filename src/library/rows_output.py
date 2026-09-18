"""Writes a batch run's result rows, replacing the target only once the write succeeds."""

import csv
import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

#: Largest single Parquet file written, under the 100 MB limit GitHub enforces on a push.
MAX_PART_BYTES = 64 * 1024 * 1024


def _remove(path: Path) -> None:
    """Removes a file or a directory tree, silently when there is nothing there."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _replace_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Writes through a sibling temp path and renames, so an interrupted run keeps the old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        write(temp)
        #: A rename cannot replace a directory, and a file cannot replace a directory either.
        if path.is_dir() or temp.is_dir():
            _remove(path)
        temp.replace(path)
    finally:
        _remove(temp)


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Writes rows to CSV, the header being every key in first-seen order across all rows."""
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))

    def write(target: Path) -> None:
        """Serialises every row into the temp file."""
        with target.open("w", newline="") as handle:
            if not fieldnames:
                return
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _replace_atomically(path, write)


def write_dataframe_parquet(
    path: Path, frame: "pd.DataFrame", *, max_part_bytes: int = MAX_PART_BYTES, **options: object
) -> None:
    """Writes a frame as one Parquet file, or as a directory of part files each under the cap."""
    #: One codec across the dataset, defaulted here so no call site can fall back to snappy.
    settings: dict[str, object] = {"compression": "zstd", **options}

    def write(target: Path) -> None:
        """Writes the frame once, then re-cuts it into equal row slices only if it is too big."""
        frame.to_parquet(target, index=False, **settings)
        parts = math.ceil(target.stat().st_size / max_part_bytes)
        if parts <= 1:
            return
        target.unlink()
        target.mkdir()
        rows = math.ceil(len(frame) / parts)
        for index in range(parts):
            slice_ = frame.iloc[index * rows : (index + 1) * rows]
            slice_.to_parquet(target / f"part-{index}.parquet", index=False, **settings)

    _replace_atomically(path, write)


def write_text(path: Path, text: str) -> None:
    """Writes text, replacing the target only once the write succeeds."""

    def write(target: Path) -> None:
        """Discards the character count write_text returns, which the writer contract forbids."""
        target.write_text(text)

    _replace_atomically(path, write)


def json_safe(value: Any) -> Any:
    """Replaces non-finite floats with None, which JSON can express and NaN cannot."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: object) -> None:
    """Writes JSON, replacing the target only once serialisation succeeds, and never with NaN."""
    #: allow_nan=False catches anything non-finite json_safe missed, rather than emitting NaN.
    write_text(path, json.dumps(json_safe(payload), allow_nan=False))
