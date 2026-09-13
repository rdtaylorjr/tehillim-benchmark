import json
from pathlib import Path

import pandas as pd
import pytest

from library.parity import ParityError, check_parity, master_models, skipped_in_log
from library.stages import Roots


def _roots(tmp_path: Path) -> Roots:
    embeddings = tmp_path / "emb"
    for name in ("a", "b", "c"):
        part = (
            embeddings / "domain=lexical" / f"unit={name}" / "construction=icf" / "part-0.parquet"
        )
        part.parent.mkdir(parents=True)
        part.write_bytes(b"x")
    return Roots(
        data_root=tmp_path / "data",
        embeddings_root=embeddings,
        config_root=tmp_path / "config",
        genre_csv=tmp_path / "g.csv",
        ui_root=tmp_path / "ui",
        workers=1,
    )


def _write_masters(roots: Roots, domain: str, models: list[str]) -> None:
    base = roots.data_root / "analysis=benchmark"
    tables = (
        f"benchmark=parallelism/domain={domain}/stage=master/model_metrics_overall.parquet",
        f"benchmark=genre/domain={domain}/stage=master/genre_metrics_wide.parquet",
        f"benchmark=trajectory/domain={domain}/stage=raw/trajectory_distances.parquet",
    )
    for rel in tables:
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"model": models, "value": range(len(models))}).to_parquet(path)


class TestMasterModels:
    def test_reads_the_model_column_of_each_benchmark_table(self, tmp_path: Path) -> None:
        roots = _roots(tmp_path)
        _write_masters(roots, "lexical", ["a_icf", "b_icf"])
        assert master_models(roots, "parallelism", "lexical") == {"a_icf", "b_icf"}
        assert master_models(roots, "trajectory", "lexical") == {"a_icf", "b_icf"}


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
        for stage in (
            "parallelism.lexical.retrieval",
            "genre.lexical.summary",
            "trajectory.lexical.distances",
        ):
            (logs / f"{stage}.log").write_text("skipping c_icf: degenerate\n")
        report = check_parity(roots, logs)
        assert report["genre.lexical"]["skipped"] == ["c_icf"]
        assert report["genre.lexical"]["unexplained"] == []

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
