"""Runs each entry point end to end over a small fake corpus, the way a batch script invokes it."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genre.scripts import build_master_report as genre_build_master_report
from genre.scripts import compare_by_genre as genre_compare_by_genre
from genre.scripts import compare_calibrated
from genre.scripts import compare_models as genre_compare_models
from genre.scripts import compute_bootstrap_cis as genre_compute_bootstrap_cis
from genre.scripts import export_detail as genre_export_detail
from genre.scripts import shuffle_order_control as genre_shuffle_order_control
from parallelism.scripts import build_master_report as parallelism_build_master_report
from parallelism.scripts import compare_baseline as parallelism_compare_baseline
from parallelism.scripts import compare_models as parallelism_compare_models
from parallelism.scripts import compare_true_similarity as parallelism_compare_true_similarity
from parallelism.scripts import compute_bootstrap_cis as parallelism_compute_bootstrap_cis
from parallelism.scripts import export_detail as parallelism_export_detail
from parallelism.scripts import shuffle_order_control as parallelism_shuffle_order_control
from trajectory.scripts import compute_profiles as trajectory_compute_profiles
from trajectory.scripts import export_ui_rows as trajectory_export_ui_rows
from trajectory.scripts import validate_against_genre as trajectory_validate_against_genre
from ui_export import export as ui_export
from ui_export.scripts import build_ui_page

N_SHUFFLE_DRAWS = 40

#: Three psalms a genre, so leaving one out still leaves a genre a within-genre pair to score.
GENRES = {1: "lament", 2: "lament", 3: "lament", 4: "praise", 5: "praise", 6: "praise"}
#: Four half-verses a psalm: the first two are an annotated couplet, the rest are background.
HALF_VERSES = {psalm: [psalm * 100 + i for i in range(1, 5)] for psalm in GENRES}


@pytest.fixture
def genre_csv(tmp_path: Path) -> Path:
    """A psalms-browser.csv export carrying only the columns the loader reads."""
    path = tmp_path / "genres.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Psalm", "Genre"])
        writer.writeheader()
        for psalm, genre in GENRES.items():
            writer.writerow({"Psalm": f"Ps {psalm}", "Genre": genre})
    return path


@pytest.fixture
def embeddings_dir(tmp_path: Path, write_embeddings_parquet) -> Path:
    """Two models over the same nodes, so a batch has more than one row to rank."""
    directory = tmp_path / "embeddings" / "domain=semantic"
    for offset, model in enumerate(("model_a", "model_b")):
        vectors = {
            node: [float(psalm), float(node % 10), float(offset)]
            for psalm, nodes in HALF_VERSES.items()
            for node in nodes
        }
        write_embeddings_parquet(directory / f"model={model}" / "vectors.parquet", vectors)
    return directory


class TestGenreCompareModels:
    def test_writes_one_row_per_model_scored(
        self, genre_csv: Path, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        output = tmp_path / "summary.csv"

        genre_compare_models.main(
            [str(genre_csv), str(embeddings_dir), "--output", str(output), "--workers", "1"],
            api_factory=lambda _checkout: bhsa_api_over(HALF_VERSES),
        )

        assert set(pd.read_csv(output)["model"]) == {"model_a", "model_b"}

    def test_ranks_the_stronger_model_first(
        self, genre_csv: Path, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        output = tmp_path / "summary.csv"

        genre_compare_models.main(
            [str(genre_csv), str(embeddings_dir), "--output", str(output), "--workers", "1"],
            api_factory=lambda _checkout: bhsa_api_over(HALF_VERSES),
        )

        scores = pd.read_csv(output)["average_precision"]
        assert scores.is_monotonic_decreasing

    def test_reuses_an_existing_output_rather_than_rescoring_it(
        self, genre_csv: Path, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        output = tmp_path / "summary.csv"
        argv = [str(genre_csv), str(embeddings_dir), "--output", str(output), "--workers", "1"]
        api = lambda _checkout: bhsa_api_over(HALF_VERSES)  # noqa: E731

        genre_compare_models.main(argv, api_factory=api)
        first = output.read_text()
        genre_compare_models.main(argv, api_factory=api)

        assert output.read_text() == first


class TestGenreCompareCalibrated:
    def test_adds_the_calibrated_effect_size_column(
        self, genre_csv: Path, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        output = tmp_path / "calibrated.csv"

        compare_calibrated.main(
            [str(genre_csv), str(embeddings_dir), "--output", str(output), "--workers", "1"],
            api_factory=lambda _checkout: bhsa_api_over(HALF_VERSES),
        )

        assert "gap" in pd.read_csv(output).columns


#: One synonymous couplet per psalm, the smallest annotation a retrieval pair can be built from.
def _couplet(group_id: str, group: str, nodes: tuple[int, int]) -> dict[int, dict[str, str]]:
    return {
        node: {
            "parallel_group_id": group_id,
            "parallel_member_id": str(member),
            "parallel_type": "Synonymous",
            "parallel_group": group,
            "parallel_member": "AB"[member],
            "parallel_signature": "AB",
            "parallel_ambiguous": "0",
        }
        for member, node in enumerate(nodes)
    }


PARALLEL_ANNOTATIONS = {
    node: values
    for index, psalm in enumerate(GENRES)
    for node, values in _couplet(
        str(index), f"g{psalm}", (psalm * 100 + 1, psalm * 100 + 2)
    ).items()
}


class TestParallelismCompareModels:
    def test_writes_one_row_per_model_scored(
        self, embeddings_dir: Path, tmp_path: Path, parallel_bhsa_api_over
    ) -> None:
        output = tmp_path / "retrieval.csv"

        parallelism_compare_models.main(
            [str(embeddings_dir), "--output", str(output), "--workers", "1"],
            api_factory=lambda _checkout: parallel_bhsa_api_over(HALF_VERSES, PARALLEL_ANNOTATIONS),
        )

        assert set(pd.read_csv(output)["model"]) == {"model_a", "model_b"}

    def test_ranks_by_separation_auc(
        self, embeddings_dir: Path, tmp_path: Path, parallel_bhsa_api_over
    ) -> None:
        output = tmp_path / "retrieval.csv"

        parallelism_compare_models.main(
            [str(embeddings_dir), "--output", str(output), "--workers", "1"],
            api_factory=lambda _checkout: parallel_bhsa_api_over(HALF_VERSES, PARALLEL_ANNOTATIONS),
        )

        assert pd.read_csv(output)["separation_auc"].is_monotonic_decreasing


class TestGenreComputeBootstrapCis:
    def test_writes_a_confidence_interval_per_model(
        self, genre_csv: Path, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        output = tmp_path / "bootstrap.csv"

        genre_compute_bootstrap_cis.main(
            [
                str(genre_csv),
                str(embeddings_dir),
                "--output",
                str(output),
                "--workers",
                "1",
                "--n-resamples",
                "20",
                "--seed",
                "0",
            ],
            api_factory=lambda _checkout: bhsa_api_over(HALF_VERSES),
        )

        assert set(pd.read_csv(output)["model"]) == {"model_a", "model_b"}


class TestGenreCompareByGenre:
    def test_writes_a_row_per_model_and_genre(
        self, genre_csv: Path, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        output = tmp_path / "by_genre.csv"

        genre_compare_by_genre.main(
            [
                str(genre_csv),
                str(embeddings_dir),
                "--output",
                str(output),
                "--workers",
                "1",
                "--n-resamples",
                "20",
                "--n-permutations",
                "20",
                "--seed",
                "0",
            ],
            api_factory=lambda _checkout: bhsa_api_over(HALF_VERSES),
        )

        written = pd.read_csv(output)
        assert set(written["genre"]) == {"lament", "praise"}


class TestGenreExportDetail:
    def test_writes_both_the_pair_detail_and_the_model_summary(
        self, genre_csv: Path, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        output_dir = tmp_path / "detail"

        genre_export_detail.main(
            [
                str(genre_csv),
                str(embeddings_dir),
                "--output-dir",
                str(output_dir),
                "--workers",
                "1",
            ],
            api_factory=lambda _checkout: bhsa_api_over(HALF_VERSES),
        )

        assert list(output_dir.rglob("*.parquet"))


class TestParallelismCompareBaseline:
    def test_writes_one_row_per_model(
        self, embeddings_dir: Path, tmp_path: Path, parallel_bhsa_api_over
    ) -> None:
        output = tmp_path / "baseline.csv"

        parallelism_compare_baseline.main(
            [str(embeddings_dir), "--output", str(output), "--workers", "1"],
            api_factory=lambda _checkout: parallel_bhsa_api_over(HALF_VERSES, PARALLEL_ANNOTATIONS),
        )

        assert set(pd.read_csv(output)["model"]) == {"model_a", "model_b"}


class TestParallelismCompareTrueSimilarity:
    def test_writes_one_row_per_model(
        self, embeddings_dir: Path, tmp_path: Path, parallel_bhsa_api_over
    ) -> None:
        output = tmp_path / "true_similarity.csv"

        parallelism_compare_true_similarity.main(
            [str(embeddings_dir), "--output", str(output), "--workers", "1"],
            api_factory=lambda _checkout: parallel_bhsa_api_over(HALF_VERSES, PARALLEL_ANNOTATIONS),
        )

        assert set(pd.read_csv(output)["model"]) == {"model_a", "model_b"}


class TestParallelismComputeBootstrapCis:
    def test_writes_intervals_for_every_model(
        self, embeddings_dir: Path, tmp_path: Path, parallel_bhsa_api_over
    ) -> None:
        output = tmp_path / "bootstrap.csv"

        parallelism_compute_bootstrap_cis.main(
            [
                str(embeddings_dir),
                "--output",
                str(output),
                "--workers",
                "1",
                "--n-resamples",
                "20",
                "--seed",
                "0",
            ],
            api_factory=lambda _checkout: parallel_bhsa_api_over(HALF_VERSES, PARALLEL_ANNOTATIONS),
        )

        assert set(pd.read_csv(output)["model"]) == {"model_a", "model_b"}


class TestParallelismExportDetail:
    def test_writes_the_detail_tree(
        self, embeddings_dir: Path, tmp_path: Path, parallel_bhsa_api_over
    ) -> None:
        output_dir = tmp_path / "detail"

        parallelism_export_detail.main(
            [str(embeddings_dir), "--output-dir", str(output_dir), "--workers", "1"],
            api_factory=lambda _checkout: parallel_bhsa_api_over(HALF_VERSES, PARALLEL_ANNOTATIONS),
        )

        assert list(output_dir.rglob("*.parquet"))


class TestTrajectoryComputeProfiles:
    def test_writes_the_distances_parquet_and_a_shard_per_model(
        self, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        output_dir = tmp_path / "trajectory"

        trajectory_compute_profiles.main(
            [str(embeddings_dir), "--output-dir", str(output_dir), "--workers", "1"],
            api_factory=lambda _checkout: bhsa_api_over(HALF_VERSES),
        )

        assert (output_dir / "trajectory_distances.parquet").exists()

    def test_a_second_run_reuses_the_shards_the_first_wrote(
        self, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        output_dir = tmp_path / "trajectory"
        argv = [str(embeddings_dir), "--output-dir", str(output_dir), "--workers", "1"]
        api = lambda _checkout: bhsa_api_over(HALF_VERSES)  # noqa: E731

        trajectory_compute_profiles.main(argv, api_factory=api)
        first = (output_dir / "trajectory_distances.parquet").read_bytes()
        trajectory_compute_profiles.main(argv, api_factory=api)

        assert (output_dir / "trajectory_distances.parquet").read_bytes() == first


class TestGenreBuildMasterReport:
    def test_joins_the_summary_and_bootstrap_outputs_into_both_tables(
        self, genre_csv: Path, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        api = lambda _checkout: bhsa_api_over(HALF_VERSES)  # noqa: E731
        summary, bootstrap = tmp_path / "summary.csv", tmp_path / "bootstrap.csv"
        compare_calibrated.main(
            [str(genre_csv), str(embeddings_dir), "--output", str(summary), "--workers", "1"],
            api_factory=api,
        )
        genre_compute_bootstrap_cis.main(
            [
                str(genre_csv),
                str(embeddings_dir),
                "--output",
                str(bootstrap),
                "--workers",
                "1",
                "--n-resamples",
                "20",
                "--seed",
                "0",
            ],
            api_factory=api,
        )
        detail_dir = tmp_path / "detail"
        genre_export_detail.main(
            [
                str(genre_csv),
                str(embeddings_dir),
                "--output-dir",
                str(detail_dir),
                "--workers",
                "1",
            ],
            api_factory=api,
        )
        output_dir = tmp_path / "report"

        genre_build_master_report.main(
            [
                "--summary-csv",
                str(summary),
                "--bootstrap-csv",
                str(bootstrap),
                "--detail-dir",
                str(detail_dir),
                "--output-dir",
                str(output_dir),
            ]
        )

        assert (output_dir / "genre_metrics_long.parquet").exists()
        assert (output_dir / "genre_metrics_wide.parquet").exists()


@pytest.fixture
def shuffled_embeddings_dir(tmp_path: Path, write_embeddings_parquet) -> Path:
    """Enough order-shuffle draws for BH q <= 0.05 to be reachable across the two genres."""
    directory = tmp_path / "shuffled" / "domain=semantic"
    rng = np.random.default_rng(0)
    for draw in range(1, N_SHUFFLE_DRAWS + 1):
        vectors = {
            node: rng.normal(size=3).tolist() for nodes in HALF_VERSES.values() for node in nodes
        }
        write_embeddings_parquet(
            directory / f"construction=shuffle{draw:04d}" / "v.parquet", vectors
        )
    return directory


class TestGenreShuffleOrderControl:
    def test_writes_one_row_per_genre_comparing_real_against_the_shuffled_null(
        self,
        genre_csv: Path,
        embeddings_dir: Path,
        shuffled_embeddings_dir: Path,
        tmp_path: Path,
        bhsa_api_over,
    ) -> None:
        real = next(embeddings_dir.rglob("*.parquet"))
        output = tmp_path / "shuffle_control.csv"

        genre_shuffle_order_control.main(
            [
                str(genre_csv),
                str(real),
                str(shuffled_embeddings_dir),
                "--output",
                str(output),
                "--workers",
                "1",
                "--n-shuffles",
                str(N_SHUFFLE_DRAWS),
            ],
            api_factory=lambda _checkout: bhsa_api_over(HALF_VERSES),
        )

        assert set(pd.read_csv(output)["genre"]) == {"lament", "praise"}


class TestParallelismShuffleOrderControl:
    def test_writes_the_real_against_shuffled_comparison(
        self,
        embeddings_dir: Path,
        shuffled_embeddings_dir: Path,
        tmp_path: Path,
        parallel_bhsa_api_over,
    ) -> None:
        real = next(embeddings_dir.rglob("*.parquet"))
        output = tmp_path / "shuffle_control.csv"

        parallelism_shuffle_order_control.main(
            [
                str(real),
                str(shuffled_embeddings_dir),
                "--output",
                str(output),
                "--workers",
                "1",
                "--n-shuffles",
                str(N_SHUFFLE_DRAWS),
            ],
            api_factory=lambda _checkout: parallel_bhsa_api_over(HALF_VERSES, PARALLEL_ANNOTATIONS),
        )

        assert "delta_order" in pd.read_csv(output).columns


class TestParallelismBuildMasterReport:
    def test_joins_retrieval_calibration_and_detail_into_the_three_tables(
        self, embeddings_dir: Path, tmp_path: Path, parallel_bhsa_api_over
    ) -> None:
        api = lambda _checkout: parallel_bhsa_api_over(  # noqa: E731
            HALF_VERSES, PARALLEL_ANNOTATIONS
        )
        retrieval, calibration = tmp_path / "retrieval.csv", tmp_path / "calibration.csv"
        detail_dir = tmp_path / "detail"
        parallelism_compare_models.main(
            [str(embeddings_dir), "--output", str(retrieval), "--workers", "1"], api_factory=api
        )
        parallelism_compare_true_similarity.main(
            [str(embeddings_dir), "--output", str(calibration), "--workers", "1"], api_factory=api
        )
        parallelism_export_detail.main(
            [str(embeddings_dir), "--output-dir", str(detail_dir), "--workers", "1"],
            api_factory=api,
        )
        output_dir = tmp_path / "report"

        parallelism_build_master_report.main(
            [
                "--retrieval-csv",
                str(retrieval),
                "--calibration-csv",
                str(calibration),
                "--detail-dir",
                str(detail_dir),
                "--output-dir",
                str(output_dir),
            ]
        )

        assert (output_dir / "model_metrics_long.parquet").exists()
        assert (output_dir / "model_metrics_overall.parquet").exists()
        assert (output_dir / "model_metrics_by_type.parquet").exists()


class TestTrajectoryValidateAgainstGenre:
    def test_writes_the_validation_csv_and_its_by_genre_breakdown(
        self, genre_csv: Path, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        api = lambda _checkout: bhsa_api_over(HALF_VERSES)  # noqa: E731
        profiles_dir = tmp_path / "trajectory"
        trajectory_compute_profiles.main(
            [str(embeddings_dir), "--output-dir", str(profiles_dir), "--workers", "1"],
            api_factory=api,
        )
        output = tmp_path / "validation.csv"
        breakdown = tmp_path / "by_genre.csv"

        trajectory_validate_against_genre.main(
            [
                str(genre_csv),
                str(profiles_dir / "trajectory_distances.parquet"),
                "--output",
                str(output),
                "--breakdown-output",
                str(breakdown),
                "--workers",
                "1",
                "--n-permutations",
                "50",
                "--seed",
                "0",
            ],
            api_factory=api,
        )

        assert set(pd.read_csv(output)["model"]) == {"model_a", "model_b"}


class TestTrajectoryExportUiRows:
    def test_converts_the_validation_csv_into_the_ui_row_json(
        self, genre_csv: Path, embeddings_dir: Path, tmp_path: Path, bhsa_api_over
    ) -> None:
        api = lambda _checkout: bhsa_api_over(HALF_VERSES)  # noqa: E731
        profiles_dir = tmp_path / "trajectory"
        trajectory_compute_profiles.main(
            [str(embeddings_dir), "--output-dir", str(profiles_dir), "--workers", "1"],
            api_factory=api,
        )
        validation = tmp_path / "validation.csv"
        breakdown = tmp_path / "by_genre.csv"
        trajectory_validate_against_genre.main(
            [
                str(genre_csv),
                str(profiles_dir / "trajectory_distances.parquet"),
                "--output",
                str(validation),
                "--breakdown-output",
                str(breakdown),
                "--workers",
                "1",
                "--n-permutations",
                "50",
                "--seed",
                "0",
            ],
            api_factory=api,
        )
        output = tmp_path / "rows.json"
        breakdown_output = tmp_path / "by_genre.json"

        trajectory_export_ui_rows.main(
            [
                str(validation),
                "--breakdown-csv",
                str(breakdown),
                "--output",
                str(output),
                "--breakdown-output",
                str(breakdown_output),
            ]
        )

        assert json.loads(output.read_text())
        assert json.loads(breakdown_output.read_text())


class TestUiExportPipeline:
    """Runs the whole chain a domain's site payload is built from, in a release's order."""

    def _domain_json(
        self,
        genre_csv: Path,
        embeddings_dir: Path,
        shuffled_embeddings_dir: Path,
        tmp_path: Path,
        bhsa_api_over,
        parallel_bhsa_api_over,
    ) -> Path:
        genre_api = lambda _checkout: bhsa_api_over(HALF_VERSES)  # noqa: E731
        parallel_api = lambda _checkout: parallel_bhsa_api_over(  # noqa: E731
            HALF_VERSES, PARALLEL_ANNOTATIONS
        )

        parallelism_dir = tmp_path / "parallelism"
        retrieval = parallelism_dir / "stage=raw" / "retrieval.csv"
        calibration = parallelism_dir / "stage=raw" / "calibration.csv"
        detail_dir = parallelism_dir / "stage=detail"
        retrieval.parent.mkdir(parents=True)
        parallelism_compare_models.main(
            [str(embeddings_dir), "--output", str(retrieval), "--workers", "1"],
            api_factory=parallel_api,
        )
        parallelism_compare_true_similarity.main(
            [str(embeddings_dir), "--output", str(calibration), "--workers", "1"],
            api_factory=parallel_api,
        )
        parallelism_export_detail.main(
            [str(embeddings_dir), "--output-dir", str(detail_dir), "--workers", "1"],
            api_factory=parallel_api,
        )
        parallelism_build_master_report.main(
            [
                "--retrieval-csv",
                str(retrieval),
                "--calibration-csv",
                str(calibration),
                "--detail-dir",
                str(detail_dir),
                "--output-dir",
                str(parallelism_dir / "stage=master"),
            ]
        )

        genre_dir = tmp_path / "genre"
        summary = genre_dir / "stage=raw" / "summary.csv"
        bootstrap = genre_dir / "stage=raw" / "bootstrap.csv"
        by_genre = genre_dir / "stage=raw" / "by_genre.csv"
        summary.parent.mkdir(parents=True)
        compare_calibrated.main(
            [str(genre_csv), str(embeddings_dir), "--output", str(summary), "--workers", "1"],
            api_factory=genre_api,
        )
        genre_compute_bootstrap_cis.main(
            [
                str(genre_csv),
                str(embeddings_dir),
                "--output",
                str(bootstrap),
                "--workers",
                "1",
                "--n-resamples",
                "20",
                "--seed",
                "0",
            ],
            api_factory=genre_api,
        )
        genre_compare_by_genre.main(
            [
                str(genre_csv),
                str(embeddings_dir),
                "--output",
                str(by_genre),
                "--workers",
                "1",
                "--n-resamples",
                "20",
                "--n-permutations",
                "20",
                "--seed",
                "0",
            ],
            api_factory=genre_api,
        )
        genre_detail_dir = genre_dir / "stage=detail"
        genre_export_detail.main(
            [
                str(genre_csv),
                str(embeddings_dir),
                "--output-dir",
                str(genre_detail_dir),
                "--workers",
                "1",
            ],
            api_factory=genre_api,
        )
        genre_build_master_report.main(
            [
                "--summary-csv",
                str(summary),
                "--bootstrap-csv",
                str(bootstrap),
                "--detail-dir",
                str(genre_detail_dir),
                "--output-dir",
                str(genre_dir / "stage=master"),
            ]
        )

        profiles_dir = tmp_path / "trajectory"
        trajectory_compute_profiles.main(
            [str(embeddings_dir), "--output-dir", str(profiles_dir), "--workers", "1"],
            api_factory=genre_api,
        )
        validation = profiles_dir / "validation.csv"
        breakdown = profiles_dir / "by_genre.csv"
        trajectory_validate_against_genre.main(
            [
                str(genre_csv),
                str(profiles_dir / "trajectory_distances.parquet"),
                "--output",
                str(validation),
                "--breakdown-output",
                str(breakdown),
                "--workers",
                "1",
                "--n-permutations",
                "50",
                "--seed",
                "0",
            ],
            api_factory=genre_api,
        )
        ui_rows = profiles_dir / "rows.json"
        by_genre_rows = profiles_dir / "by_genre.json"
        trajectory_export_ui_rows.main(
            [
                str(validation),
                "--breakdown-csv",
                str(breakdown),
                "--output",
                str(ui_rows),
                "--breakdown-output",
                str(by_genre_rows),
            ]
        )

        domain_json = tmp_path / "semantic.json"
        ui_export.main(
            [
                "semantic",
                "--parallelism-dir",
                str(parallelism_dir),
                "--genre-dir",
                str(genre_dir),
                "--trajectory-ui-rows",
                str(ui_rows),
                "--trajectory-by-genre-rows",
                str(by_genre_rows),
                "--output",
                str(domain_json),
            ]
        )
        return domain_json

    def test_writes_a_domain_payload_naming_the_domain_it_was_built_for(
        self,
        genre_csv: Path,
        embeddings_dir: Path,
        shuffled_embeddings_dir: Path,
        tmp_path: Path,
        bhsa_api_over,
        parallel_bhsa_api_over,
    ) -> None:
        domain_json = self._domain_json(
            genre_csv,
            embeddings_dir,
            shuffled_embeddings_dir,
            tmp_path,
            bhsa_api_over,
            parallel_bhsa_api_over,
        )

        assert set(json.loads(domain_json.read_text())) == {"semantic"}

    def test_build_ui_page_assembles_the_payload_into_the_template(
        self,
        genre_csv: Path,
        embeddings_dir: Path,
        shuffled_embeddings_dir: Path,
        tmp_path: Path,
        bhsa_api_over,
        parallel_bhsa_api_over,
    ) -> None:
        domain_json = self._domain_json(
            genre_csv,
            embeddings_dir,
            shuffled_embeddings_dir,
            tmp_path,
            bhsa_api_over,
            parallel_bhsa_api_over,
        )
        template = tmp_path / "template.html"
        template.write_text(
            "<html><script>/*UI_DATA_JSON*/{}/*END_UI_DATA_JSON*/</script>"
            "<script>/*UI_BUNDLE_JS*/</script></html>"
        )
        bundle = tmp_path / "bundle.js"
        bundle.write_text("console.log('ui')")
        output = tmp_path / "index.html"

        build_ui_page.main(
            [
                str(domain_json),
                "--template",
                str(template),
                "--bundle",
                str(bundle),
                "--output",
                str(output),
            ]
        )

        assert "semantic" in output.read_text()
