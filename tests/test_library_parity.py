import json
from pathlib import Path

import pandas as pd
import pytest
from core.skips import skipped_in_log

from library.parity import ParityError, check_parity, coverage_tables, master_models
from library.stages import Roots


def _roots(tmp_path: Path) -> Roots:
    embeddings = tmp_path / "emb"
    for name in ("a", "b", "c"):
        part = (
            embeddings
            / "corpus=bhsa/unit=half_verse/domain=lexical"
            / f"type={name}"
            / "construction=icf/part-0.parquet"
        )
        part.parent.mkdir(parents=True)
        part.write_bytes(b"x")
    return Roots(
        data_root=tmp_path / "data",
        embeddings_root=embeddings,
        config_root=tmp_path / "config",
        genre_csv=tmp_path / "g.csv",
        gunkel_csv=tmp_path / "gunkel.csv",
        ui_root=tmp_path / "ui",
        workers=1,
    )


def _write_masters(roots: Roots, domain: str, models: list[str]) -> None:
    for _, path, _ in coverage_tables(roots, domain):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"model": models, "value": range(len(models))}).to_parquet(path)


class TestCoverageTables:
    def test_one_chain_per_benchmark_and_one_per_genre_register(self, tmp_path: Path) -> None:
        keys = [key for key, _, _ in coverage_tables(_roots(tmp_path), "lexical")]
        assert keys[0] == "parallelism.lexical"
        assert keys[-1] == "trajectory.lexical"
        assert keys[1:-1] == [
            "genre.lexical.logos",
            "genre.lexical.gunkel.song",
            "genre.lexical.gunkel.song_component",
            "genre.lexical.gunkel.song_component_motif",
        ]

    def test_a_genre_table_sits_under_its_taxonomy_and_unit(self, tmp_path: Path) -> None:
        roots = _roots(tmp_path)
        tables = {key: path for key, path, _ in coverage_tables(roots, "lexical")}
        assert tables["genre.lexical.gunkel.song"] == roots.data_root / (
            "analysis=benchmark/benchmark=genre/taxonomy=gunkel/unit=song/domain=lexical/"
            "stage=master/genre_metrics_wide.parquet"
        )


class TestMasterModels:
    def test_reads_the_model_column_of_each_benchmark_table(self, tmp_path: Path) -> None:
        roots = _roots(tmp_path)
        _write_masters(roots, "lexical", ["a_icf", "b_icf"])
        for _, path, _ in coverage_tables(roots, "lexical"):
            assert master_models(path) == {"a_icf", "b_icf"}


class TestSkippedInLog:
    def test_collects_the_names_a_stage_reported_skipping(self, tmp_path: Path) -> None:
        log = tmp_path / "x.log"
        log.write_text("scoring...\nskipping c_icf: retrieval scoring needs at least two pairs\n")
        assert skipped_in_log(log) == {"c_icf"}

    def test_a_missing_log_reports_nothing_skipped(self, tmp_path: Path) -> None:
        assert skipped_in_log(tmp_path / "none.log") == set()


class TestCheckParity:
    def test_passes_when_every_dataset_is_scored(self, tmp_path: Path) -> None:
        roots = _roots(tmp_path)
        _write_masters(roots, "lexical", ["a_icf", "b_icf", "c_icf"])
        report = check_parity(roots, tmp_path / "logs")
        assert report["parallelism.lexical"]["unexplained"] == []
        assert report["parallelism.lexical"]["stale"] == []

    def test_a_skip_reported_in_the_raw_stage_log_is_explained(self, tmp_path: Path) -> None:
        roots = _roots(tmp_path)
        _write_masters(roots, "lexical", ["a_icf", "b_icf"])
        logs = tmp_path / "logs"
        logs.mkdir()
        for key, _, raw_stage in coverage_tables(roots, "lexical"):
            (logs / f"{key}.{raw_stage}.log").write_text("skipping c_icf: degenerate\n")
        report = check_parity(roots, logs)
        assert report["genre.lexical.gunkel.song"]["skipped"] == ["c_icf"]
        assert report["genre.lexical.gunkel.song"]["unexplained"] == []

    def test_an_unexplained_gap_or_a_stale_model_fails(self, tmp_path: Path) -> None:
        roots = _roots(tmp_path)
        _write_masters(roots, "lexical", ["a_icf", "b_icf", "retired_icf"])
        with pytest.raises(ParityError, match="c_icf") as error:
            check_parity(roots, tmp_path / "logs")
        assert "retired_icf" in str(error.value)

    def test_a_missing_master_table_is_reported_as_the_gap(self, tmp_path: Path) -> None:
        roots = _roots(tmp_path)
        with pytest.raises(ParityError, match=r"parallelism\.lexical"):
            check_parity(roots, tmp_path / "logs")

    def test_report_is_json_serialisable(self, tmp_path: Path) -> None:
        roots = _roots(tmp_path)
        _write_masters(roots, "lexical", ["a_icf", "b_icf", "c_icf"])
        json.dumps(check_parity(roots, tmp_path / "logs"))
