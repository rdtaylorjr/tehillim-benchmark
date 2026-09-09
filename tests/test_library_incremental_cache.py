from __future__ import annotations

from pathlib import Path

import pandas as pd

from library.incremental_cache import load_cached_parquet_set, load_cached_rows


class TestLoadCachedRows:
    def test_returns_empty_when_the_cache_path_does_not_exist(self, tmp_path: Path) -> None:
        rows, models = load_cached_rows(tmp_path / "missing.csv")

        assert rows == []
        assert models == set()

    def test_reads_every_row_and_the_covered_model_set_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.csv"
        pd.DataFrame({"model": ["a", "b"], "score": [1, 2], "extra": ["x", "y"]}).to_csv(
            path, index=False
        )

        rows, models = load_cached_rows(path)

        assert models == {"a", "b"}
        assert rows == [
            {"model": "a", "score": 1, "extra": "x"},
            {"model": "b", "score": 2, "extra": "y"},
        ]

    def test_restricts_to_the_given_columns_when_provided(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.csv"
        pd.DataFrame({"model": ["a"], "score": [1], "stale_q": [0.5]}).to_csv(path, index=False)

        rows, models = load_cached_rows(path, columns=("model", "score"))

        assert models == {"a"}
        assert rows == [{"model": "a", "score": 1}]


class TestLoadCachedParquetSet:
    def test_returns_empty_when_no_prior_output_exists(self, tmp_path: Path) -> None:
        rows_by_file, models = load_cached_parquet_set(tmp_path, ("a.parquet", "b.parquet"))

        assert rows_by_file == [[], []]
        assert models == set()

    def test_reads_rows_and_the_model_set_shared_by_every_file(self, tmp_path: Path) -> None:
        pd.DataFrame({"model": ["a", "b"], "x": [1, 2]}).to_parquet(tmp_path / "one.parquet")
        pd.DataFrame({"model": ["a", "b"], "y": [3, 4]}).to_parquet(tmp_path / "two.parquet")

        rows_by_file, models = load_cached_parquet_set(tmp_path, ("one.parquet", "two.parquet"))

        assert models == {"a", "b"}
        assert rows_by_file[0] == [{"model": "a", "x": 1}, {"model": "b", "x": 2}]

    def test_returns_empty_when_only_some_files_exist(self, tmp_path: Path) -> None:
        pd.DataFrame({"model": ["a"], "x": [1]}).to_parquet(tmp_path / "one.parquet")

        rows_by_file, models = load_cached_parquet_set(tmp_path, ("one.parquet", "two.parquet"))

        assert rows_by_file == [[], []]
        assert models == set()


def test_cached_floats_round_trip_to_the_exact_same_bits(tmp_path: Path) -> None:
    """A cached row is rewritten verbatim, so a lossy parse would alter published numbers."""
    value = 1.5547970191810556e-09
    cache = tmp_path / "cache.csv"
    cache.write_text(f"model,separation_p\nm,{value!r}\n")

    rows, _models = load_cached_rows(cache)

    assert rows[0]["separation_p"] == value


def test_cached_parquet_floats_round_trip_exactly(tmp_path: Path) -> None:
    value = 4.1377175583932606e-42
    pd.DataFrame({"model": ["m"], "p": [value]}).to_parquet(tmp_path / "a.parquet")

    (rows,), _models = load_cached_parquet_set(tmp_path, ("a.parquet",))

    assert rows[0]["p"] == value
