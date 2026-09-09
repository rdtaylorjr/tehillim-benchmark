"""Result writers must not leave a partial file behind when a run is interrupted."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from library.rows_output import write_dataframe_parquet, write_rows_csv


def test_csv_write_replaces_an_existing_file_atomically(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    write_rows_csv(path, [{"model": "a", "score": 1.0}])

    write_rows_csv(path, [{"model": "b", "score": 2.0}])

    assert "b" in path.read_text()
    assert not list(tmp_path.glob("*.tmp*")), "temp file left behind"


def test_a_failed_csv_write_leaves_the_previous_file_intact(tmp_path: Path) -> None:
    """A serialisation error must not destroy the last good result file."""
    path = tmp_path / "out.csv"
    write_rows_csv(path, [{"model": "good", "score": 1.0}])

    class Unwritable:
        """A value whose serialisation fails partway through the write."""

        def __str__(self) -> str:
            raise RuntimeError("serialisation failed")

    with pytest.raises(RuntimeError):
        write_rows_csv(path, [{"model": "new", "score": Unwritable()}])

    assert "good" in path.read_text()


def test_parquet_write_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    frame = pd.DataFrame({"model": ["a", "b"], "score": [1.0, 2.0]})

    write_dataframe_parquet(path, frame)

    assert pd.read_parquet(path).equals(frame)
    assert not list(tmp_path.glob("*.tmp*"))


def test_a_failed_parquet_write_leaves_the_previous_file_intact(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    write_dataframe_parquet(path, pd.DataFrame({"model": ["good"], "score": [1.0]}))

    class Unserialisable:
        """A column type pyarrow cannot convert, so the write fails partway."""

    with pytest.raises((pa.ArrowException, ValueError, TypeError)):
        write_dataframe_parquet(path, pd.DataFrame({"bad": [Unserialisable()]}))

    assert pd.read_parquet(path)["model"].tolist() == ["good"]


def test_parquet_write_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "out.parquet"

    write_dataframe_parquet(path, pd.DataFrame({"a": [1]}))

    assert path.exists()


def test_parquet_writes_default_to_zstd(tmp_path: Path) -> None:
    """One codec across the dataset, set here so no call site can forget to name it."""
    path = tmp_path / "out.parquet"

    write_dataframe_parquet(path, pd.DataFrame({"a": [1, 2, 3]}))

    codec = pq.ParquetFile(path).metadata.row_group(0).column(0).compression
    assert codec == "ZSTD"


def test_an_explicit_codec_still_overrides_the_default(tmp_path: Path) -> None:
    """The trajectory shards set level 19 deliberately, so callers must keep the last word."""
    path = tmp_path / "out.parquet"

    write_dataframe_parquet(path, pd.DataFrame({"a": [1, 2, 3]}), compression="snappy")

    codec = pq.ParquetFile(path).metadata.row_group(0).column(0).compression
    assert codec == "SNAPPY"
