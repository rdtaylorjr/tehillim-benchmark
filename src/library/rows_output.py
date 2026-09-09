"""Writes a batch run's result rows, replacing the target only once the write succeeds."""

import csv
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def _replace_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Writes through a sibling temp file and renames, so an interrupted run keeps the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        write(temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


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


def write_dataframe_parquet(path: Path, frame: "pd.DataFrame", **options: object) -> None:
    """Writes a result frame to Parquet, replacing the target only once the write succeeds."""
    #: One codec across the dataset, defaulted here so no call site can fall back to snappy.
    settings: dict[str, object] = {"compression": "zstd", **options}
    _replace_atomically(path, lambda target: frame.to_parquet(target, index=False, **settings))


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
