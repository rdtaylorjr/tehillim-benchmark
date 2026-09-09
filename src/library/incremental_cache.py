"""Reads a prior scoring run's output CSV so already-scored models can be skipped on rerun."""

from pathlib import Path
from typing import Any

import pandas as pd


def load_cached_rows(
    cache_path: Path, columns: tuple[str, ...] | None = None
) -> tuple[list[dict[str, Any]], set[str]]:
    """Reads a prior output CSV's rows (optionally a column subset) and its covered model set."""
    if not cache_path.exists():
        return [], set()
    # round_trip so a cached row is rewritten with the exact bits the scoring run produced.
    cached_df = pd.read_csv(cache_path, float_precision="round_trip")
    models = set(cached_df["model"].unique())
    if columns is not None:
        cached_df = cached_df[list(columns)]
    return cached_df.to_dict("records"), models


def load_cached_parquet_set(
    output_dir: Path, filenames: tuple[str, ...]
) -> tuple[list[list[dict[str, Any]]], set[str]]:
    """Reads several prior output parquet files' rows and the model set covered by all of them."""
    rows_by_file: list[list[dict[str, Any]]] = []
    models: set[str] | None = None
    for name in filenames:
        path = output_dir / name
        if not path.exists():
            return [[] for _ in filenames], set()
        df = pd.read_parquet(path)
        rows_by_file.append(df.to_dict("records"))
        file_models = set(df["model"].unique())
        models = file_models if models is None else models & file_models
    return rows_by_file, models or set()
