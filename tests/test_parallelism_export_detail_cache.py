from __future__ import annotations

from pathlib import Path

import pandas as pd

from library.incremental_cache import load_cached_parquet_set
from parallelism.scripts.export_detail import _OUTPUT_FILES


class TestLoadCachedDetail:
    def test_returns_empty_when_no_prior_output_exists(self, tmp_path: Path) -> None:
        rows_by_file, models = load_cached_parquet_set(tmp_path, _OUTPUT_FILES)

        assert rows_by_file == [[], [], []]
        assert models == set()

    def test_reads_rows_and_the_model_set_shared_by_all_three_files(self, tmp_path: Path) -> None:
        pd.DataFrame({"model": ["a", "b"], "x": [1, 2]}).to_parquet(
            tmp_path / "pair_detail.parquet"
        )
        pd.DataFrame({"model": ["a", "b"], "y": [3, 4]}).to_parquet(
            tmp_path / "baseline_detail.parquet"
        )
        pd.DataFrame({"model": ["a", "b"], "z": [5, 6]}).to_parquet(
            tmp_path / "type_vs_baseline.parquet"
        )

        rows_by_file, models = load_cached_parquet_set(tmp_path, _OUTPUT_FILES)

        assert models == {"a", "b"}
        assert len(rows_by_file) == 3
        assert rows_by_file[0] == [{"model": "a", "x": 1}, {"model": "b", "x": 2}]

    def test_returns_empty_when_only_some_output_files_exist(self, tmp_path: Path) -> None:
        pd.DataFrame({"model": ["a"], "x": [1]}).to_parquet(tmp_path / "pair_detail.parquet")

        rows_by_file, models = load_cached_parquet_set(tmp_path, _OUTPUT_FILES)

        assert rows_by_file == [[], [], []]
        assert models == set()
