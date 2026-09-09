from __future__ import annotations

from pathlib import Path

import pandas as pd

from genre.scripts.export_detail import _OUTPUT_FILES
from library.incremental_cache import load_cached_parquet_set


class TestLoadCachedDetail:
    def test_returns_empty_when_no_prior_output_exists(self, tmp_path: Path) -> None:
        rows_by_file, models = load_cached_parquet_set(tmp_path, _OUTPUT_FILES)

        assert rows_by_file == [[], []]
        assert models == set()

    def test_reads_rows_and_the_model_set_shared_by_both_files(self, tmp_path: Path) -> None:
        pd.DataFrame({"model": ["a", "b"], "x": [1, 2]}).to_parquet(
            tmp_path / "genre_pair_detail.parquet"
        )
        pd.DataFrame({"model": ["a", "b"], "y": [3, 4]}).to_parquet(
            tmp_path / "genre_summary.parquet"
        )

        rows_by_file, models = load_cached_parquet_set(tmp_path, _OUTPUT_FILES)

        assert models == {"a", "b"}
        assert len(rows_by_file) == 2
        assert rows_by_file[0] == [{"model": "a", "x": 1}, {"model": "b", "x": 2}]

    def test_returns_empty_when_only_some_output_files_exist(self, tmp_path: Path) -> None:
        pd.DataFrame({"model": ["a"], "x": [1]}).to_parquet(tmp_path / "genre_pair_detail.parquet")

        rows_by_file, models = load_cached_parquet_set(tmp_path, _OUTPUT_FILES)

        assert rows_by_file == [[], []]
        assert models == set()
