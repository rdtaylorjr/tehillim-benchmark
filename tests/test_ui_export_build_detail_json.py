import json
from pathlib import Path

import numpy as np
import pandas as pd

from ui_export.scripts.build_detail_json import (
    ModelDetailInputs,
    attach_genre_columns,
    build_domain,
    build_one_model,
    choose_primary_metric,
    residualize_trajectory_metric,
    split_sections,
    table_model_sets,
)


def _domain_json() -> dict:
    return {
        "parallelism_overall": [{"model": "a"}, {"model": "b"}],
        "genre_overall": [{"model": "b"}, {"model": "b_psalm"}],
        "trajectory": [{"model": "a"}, {"model": "c"}],
    }


def test_table_model_sets_keeps_each_table_s_own_model_set_separate() -> None:
    """A psalm-level model excluded from parallelism_overall must not gain a parallelism section."""
    sets = table_model_sets(_domain_json())
    assert sets["parallelism"] == {"a", "b"}
    assert sets["genre"] == {"b", "b_psalm"}
    assert sets["trajectory"] == {"a", "c"}
    assert "b_psalm" not in sets["parallelism"]


def test_choose_primary_metric_picks_the_smallest_length_controlled_p() -> None:
    df = pd.DataFrame(
        [
            {"model": "m1", "metric": "content_distance", "length_controlled_p": 0.2},
            {"model": "m1", "metric": "structural_distance", "length_controlled_p": 0.01},
            {"model": "other", "metric": "content_distance", "length_controlled_p": 0.001},
        ]
    )
    assert choose_primary_metric(df, "m1") == "structural_distance"


def test_choose_primary_metric_returns_none_when_the_model_is_absent() -> None:
    df = pd.DataFrame(
        [{"model": "other", "metric": "content_distance", "length_controlled_p": 0.001}]
    )
    assert choose_primary_metric(df, "m1") is None


def test_choose_primary_metric_skips_metrics_with_a_nan_p_value() -> None:
    df = pd.DataFrame(
        [
            {"model": "m1", "metric": "content_distance", "length_controlled_p": float("nan")},
            {"model": "m1", "metric": "structural_distance", "length_controlled_p": 0.02},
        ]
    )
    assert choose_primary_metric(df, "m1") == "structural_distance"


def test_choose_primary_metric_returns_none_when_every_row_is_nan() -> None:
    df = pd.DataFrame(
        [{"model": "m1", "metric": "content_distance", "length_controlled_p": float("nan")}]
    )
    assert choose_primary_metric(df, "m1") is None


def test_choose_primary_metric_never_picks_content_distance() -> None:
    """content_distance's own content-controlled source is self-referential, always NaN."""
    df = pd.DataFrame(
        [
            {"model": "m1", "metric": "content_distance", "length_controlled_p": 0.0001},
            {"model": "m1", "metric": "structural_distance", "length_controlled_p": 0.2},
        ]
    )
    assert choose_primary_metric(df, "m1") == "structural_distance"


def test_choose_primary_metric_returns_none_when_only_content_distance_is_available() -> None:
    df = pd.DataFrame(
        [{"model": "m1", "metric": "content_distance", "length_controlled_p": 0.0001}]
    )
    assert choose_primary_metric(df, "m1") is None


def test_attach_genre_columns_derives_same_genre_from_the_two_labels() -> None:
    df = pd.DataFrame([{"psalm_a": 1, "psalm_b": 2}, {"psalm_a": 1, "psalm_b": 3}])
    genre_by_psalm = {1: "Hymn", 2: "Hymn", 3: "Lament"}
    out = attach_genre_columns(df, genre_by_psalm)
    assert out["genre_a"].tolist() == ["Hymn", "Hymn"]
    assert out["genre_b"].tolist() == ["Hymn", "Lament"]
    assert out["same_genre"].tolist() == [True, False]


def test_residualize_trajectory_metric_adds_both_controlled_columns() -> None:
    df = pd.DataFrame(
        [
            {"psalm_a": 1, "psalm_b": 2, "structural_distance": 0.5, "content_distance": 0.2},
            {"psalm_a": 1, "psalm_b": 3, "structural_distance": 0.9, "content_distance": 0.4},
            {"psalm_a": 2, "psalm_b": 3, "structural_distance": 0.3, "content_distance": 0.1},
        ]
    )
    n_half_verses = {1: 10, 2: 12, 3: 20}
    out = residualize_trajectory_metric(df, "structural_distance", n_half_verses)
    assert "length_controlled" in out.columns
    assert "length_and_content_controlled" in out.columns
    assert np.isfinite(out["length_controlled"]).all()
    assert np.isfinite(out["length_and_content_controlled"]).all()


def test_residualize_trajectory_metric_drops_rows_missing_the_metric_or_content_value() -> None:
    """A single NaN in the OLS response corrupts every row's fit, not just the missing one."""
    df = pd.DataFrame(
        [
            {"psalm_a": 1, "psalm_b": 2, "structural_distance": 0.5, "content_distance": 0.2},
            {
                "psalm_a": 1,
                "psalm_b": 3,
                "structural_distance": float("nan"),
                "content_distance": 0.4,
            },
            {
                "psalm_a": 2,
                "psalm_b": 3,
                "structural_distance": 0.3,
                "content_distance": float("nan"),
            },
        ]
    )
    n_half_verses = {1: 10, 2: 12, 3: 20}
    out = residualize_trajectory_metric(df, "structural_distance", n_half_verses)
    assert len(out) == 1
    assert np.isfinite(out["length_controlled"]).all()
    assert np.isfinite(out["length_and_content_controlled"]).all()


def test_split_sections_writes_one_file_per_section(tmp_path) -> None:
    payload = {
        "model": "berel",
        "domain": "semantic",
        "parallelism": {"series": [1]},
        "genre": {"heatmap": [2]},
    }
    written = split_sections(payload, tmp_path)
    assert sorted(p.name for p in written) == [
        "detail_semantic_berel_genre.json",
        "detail_semantic_berel_parallelism.json",
    ]


def test_split_sections_keeps_each_file_self_describing(tmp_path) -> None:
    payload = {"model": "berel", "domain": "semantic", "genre": {"heatmap": [2]}}
    written = split_sections(payload, tmp_path)
    body = json.loads(written[0].read_text())
    assert body["model"] == "berel"
    assert body["domain"] == "semantic"
    assert body["genre"] == {"heatmap": [2]}


def test_split_sections_carries_no_section_the_file_is_not_for(tmp_path) -> None:
    payload = {
        "model": "berel",
        "domain": "semantic",
        "parallelism": {"series": [1]},
        "genre": {"heatmap": [2]},
    }
    written = split_sections(payload, tmp_path)
    par = json.loads((tmp_path / "detail_semantic_berel_parallelism.json").read_text())
    assert "genre" not in par
    assert len(written) == 2


def test_split_sections_writes_nothing_when_no_section_has_data(tmp_path) -> None:
    assert split_sections({"model": "m", "domain": "semantic"}, tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def _pair_detail_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pair_id": "p1", "parallelism_type": "Synonymous", "calibrated_z": 2.0},
            {"pair_id": "p2", "parallelism_type": "Synonymous", "calibrated_z": 1.5},
            {"pair_id": "p3", "parallelism_type": "Antithetic", "calibrated_z": 1.0},
        ]
    )


def _baseline_detail_df() -> pd.DataFrame:
    return pd.DataFrame([{"calibrated_z": 0.1}, {"calibrated_z": -0.2}, {"calibrated_z": 0.0}])


def _genre_pair_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "psalm_a": 1,
                "psalm_b": 2,
                "genre_a": "Hymn",
                "genre_b": "Hymn",
                "same_genre": True,
                "calibrated_z": 1.2,
            },
            {
                "psalm_a": 3,
                "psalm_b": 4,
                "genre_a": "Lament",
                "genre_b": "Lament",
                "same_genre": True,
                "calibrated_z": 0.8,
            },
            {
                "psalm_a": 1,
                "psalm_b": 3,
                "genre_a": "Hymn",
                "genre_b": "Lament",
                "same_genre": False,
                "calibrated_z": -0.3,
            },
        ]
    )


def _build(output_dir: Path, **overrides: object) -> list[str]:
    """Runs build_one_model with every section absent unless the test supplies it."""
    build_one_model(
        "m", "syntactic", output_dir, ["Hymn", "Lament"], ModelDetailInputs(**overrides)
    )
    return sorted(p.name for p in output_dir.glob("*.json"))


class TestBuildOneModel:
    def test_writes_one_file_per_section_that_has_data(self, tmp_path: Path) -> None:
        written = _build(
            tmp_path,
            pair_detail=_pair_detail_df(),
            baseline_detail=_baseline_detail_df(),
            genre_pair=_genre_pair_df(),
        )

        assert written == ["detail_syntactic_m_genre.json", "detail_syntactic_m_parallelism.json"]

    def test_omits_a_section_whose_frame_is_absent(self, tmp_path: Path) -> None:
        written = _build(tmp_path, genre_pair=_genre_pair_df())

        assert written == ["detail_syntactic_m_genre.json"]

    def test_omits_a_section_whose_frame_is_empty(self, tmp_path: Path) -> None:
        """An empty frame reaches the builders as a real model with nothing to plot."""
        written = _build(
            tmp_path,
            genre_pair=_genre_pair_df(),
            pair_detail=_pair_detail_df().iloc[0:0],
            baseline_detail=_baseline_detail_df(),
        )

        assert written == ["detail_syntactic_m_genre.json"]

    def test_writes_nothing_when_no_section_has_data(self, tmp_path: Path) -> None:
        """A model in the tables but absent from every detail parquet is skipped silently."""
        assert _build(tmp_path) == []

    def test_parallelism_needs_its_baseline_frame_as_well_as_its_pairs(
        self, tmp_path: Path
    ) -> None:
        written = _build(tmp_path, pair_detail=_pair_detail_df())

        assert written == []

    def test_each_file_carries_only_its_own_section(self, tmp_path: Path) -> None:
        _build(
            tmp_path,
            pair_detail=_pair_detail_df(),
            baseline_detail=_baseline_detail_df(),
            genre_pair=_genre_pair_df(),
        )

        body = json.loads((tmp_path / "detail_syntactic_m_genre.json").read_text())
        assert set(body) == {"model", "domain", "genre"}
        assert body["model"] == "m"
        assert body["domain"] == "syntactic"


def _trajectory_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "psalm_a": 1,
                "psalm_b": 2,
                "genre_a": "Hymn",
                "genre_b": "Hymn",
                "same_genre": True,
                "length_controlled": 0.4,
                "length_and_content_controlled": 0.3,
            },
            {
                "psalm_a": 1,
                "psalm_b": 3,
                "genre_a": "Hymn",
                "genre_b": "Lament",
                "same_genre": False,
                "length_controlled": 0.9,
                "length_and_content_controlled": 0.7,
            },
        ]
    )


class TestBuildOneModelTrajectory:
    def test_writes_the_trajectory_section_when_a_metric_was_chosen(self, tmp_path: Path) -> None:
        written = _build(
            tmp_path, trajectory=_trajectory_df(), trajectory_metric="structural_distance"
        )

        assert written == ["detail_syntactic_m_trajectory.json"]

    def test_omits_trajectory_when_no_primary_metric_could_be_chosen(self, tmp_path: Path) -> None:
        """choose_primary_metric returns None when every source p-value is NaN for that model."""
        written = _build(tmp_path, trajectory=_trajectory_df(), trajectory_metric=None)

        assert written == []


def _write_domain_tree(root: Path, domain: str) -> None:
    """A minimal tehillim-data tree: two models in the parallelism tables, one also in genre."""
    par = root / f"analysis=benchmark/benchmark=parallelism/domain={domain}/stage=detail"
    gen = root / f"analysis=benchmark/benchmark=genre/domain={domain}/stage=detail"
    traj = root / f"analysis=benchmark/benchmark=trajectory/domain={domain}/stage=profiles"
    raw = root / f"analysis=benchmark/benchmark=trajectory/domain={domain}/stage=raw"
    for d in (par, gen, traj, raw):
        d.mkdir(parents=True, exist_ok=True)

    pair_rows = [
        {"model": m, "pair_id": f"p{i}", "parallelism_type": "Synonymous", "calibrated_z": z}
        for m in ("a", "b")
        for i, z in enumerate((2.0, 1.5, 1.0))
    ]
    pd.DataFrame(pair_rows).to_parquet(par / "pair_detail.parquet", index=False)
    pd.DataFrame(
        [{"model": m, "calibrated_z": z} for m in ("a", "b") for z in (0.1, -0.2, 0.0)]
    ).to_parquet(par / "baseline_detail.parquet", index=False)

    # Only model "a" has genre detail, so only "a" can get a genre section.
    pd.DataFrame(
        [
            {"model": "a", "psalm_a": 1, "psalm_b": 2, "same_genre": True, "calibrated_z": 1.2},
            {"model": "a", "psalm_a": 3, "psalm_b": 4, "same_genre": True, "calibrated_z": 0.8},
            {"model": "a", "psalm_a": 1, "psalm_b": 3, "same_genre": False, "calibrated_z": -0.3},
        ]
    ).to_parquet(gen / "genre_pair_detail.parquet", index=False)

    # All six pairs over four psalms, so both the same-genre and cross-genre sides are populated.
    pd.DataFrame(
        [
            {
                "model": "a",
                "psalm_a": a,
                "psalm_b": b,
                "structural_distance": d,
                "content_distance": d / 2,
            }
            for (a, b), d in zip(
                [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
                [0.4, 0.9, 0.8, 0.85, 0.95, 0.5],
                strict=True,
            )
        ]
    ).to_parquet(traj / "trajectory_distances.parquet", index=False)
    pd.DataFrame(
        [
            {
                "model": "a",
                "metric": "structural_distance",
                "length_controlled_gap": 0.1,
                "length_controlled_p": 0.01,
                "length_controlled_effect_size": 0.5,
                "length_and_content_controlled_gap": 0.05,
                "length_and_content_controlled_p": 0.10,
                "length_and_content_controlled_effect_size": 0.2,
            }
        ]
    ).to_csv(raw / "validate_against_genre.csv", index=False)


def _domain_payload() -> dict[str, object]:
    return {
        "parallelism_overall": [{"model": "a"}, {"model": "b"}],
        "genre_overall": [{"model": "a"}],
        "trajectory": [{"model": "a"}],
    }


class TestBuildDomain:
    def test_writes_a_file_for_every_model_section_that_has_data(self, tmp_path: Path) -> None:
        data_dir, output_dir = tmp_path / "data", tmp_path / "out"
        _write_domain_tree(data_dir, "syntactic")

        written = build_domain(
            "syntactic",
            data_dir,
            _domain_payload(),
            {1: "Hymn", 2: "Hymn", 3: "Lament", 4: "Lament"},
            {1: 10, 2: 12, 3: 8, 4: 9},
            output_dir,
            max_workers=1,
        )

        names = sorted(p.name for p in output_dir.glob("*.json"))
        assert names == [
            "detail_syntactic_a_genre.json",
            "detail_syntactic_a_parallelism.json",
            "detail_syntactic_a_trajectory.json",
            "detail_syntactic_b_parallelism.json",
        ]
        assert written == len(names)

    def test_a_model_absent_from_a_table_gets_no_section_for_it(self, tmp_path: Path) -> None:
        """Model b is only in the parallelism table, so it must never gain a genre file."""
        data_dir, output_dir = tmp_path / "data", tmp_path / "out"
        _write_domain_tree(data_dir, "syntactic")

        build_domain(
            "syntactic",
            data_dir,
            _domain_payload(),
            {1: "Hymn", 2: "Hymn", 3: "Lament", 4: "Lament"},
            {1: 10, 2: 12, 3: 8, 4: 9},
            output_dir,
            max_workers=1,
        )

        assert not (output_dir / "detail_syntactic_b_genre.json").exists()

    def test_a_missing_optional_genre_parquet_leaves_every_other_section_intact(
        self, tmp_path: Path
    ) -> None:
        data_dir, output_dir = tmp_path / "data", tmp_path / "out"
        _write_domain_tree(data_dir, "syntactic")
        (
            data_dir
            / "analysis=benchmark/benchmark=genre/domain=syntactic/stage=detail"
            / "genre_pair_detail.parquet"
        ).unlink()

        build_domain(
            "syntactic",
            data_dir,
            _domain_payload(),
            {1: "Hymn", 2: "Hymn", 3: "Lament", 4: "Lament"},
            {1: 10, 2: 12, 3: 8, 4: 9},
            output_dir,
            max_workers=1,
        )

        names = sorted(p.name for p in output_dir.glob("*.json"))
        assert "detail_syntactic_a_genre.json" not in names
        assert "detail_syntactic_a_parallelism.json" in names


def test_a_task_names_itself_by_its_model_for_the_skip_report() -> None:
    """The batch reports a skipped item by name, so a task has to say which model it is."""
    from ui_export.scripts.build_detail_json import _ModelTask, _task_model

    task = _ModelTask(
        model="alephbert_consonantal",
        domain="semantic",
        output_dir=Path("out"),
        genres=["Hymn"],
        inputs=None,
    )

    assert _task_model(task) == "alephbert_consonantal"
