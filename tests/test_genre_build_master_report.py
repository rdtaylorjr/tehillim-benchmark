import pandas as pd

from genre.scripts.build_master_report import build_long_metrics


def _summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "semantic_foo_consonantal",
                "n_same_genre": 2791,
                "n_different_genre": 8384,
                "prevalence": 0.25,
                "average_precision": 0.32,
                "same_genre_effect_size": 1.2,
                "different_genre_effect_size": 0.4,
                "gap": 0.8,
                "separation_auc": 0.58,
                "separation_p": 0.001,
            }
        ]
    )


def _bootstrap_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "semantic_foo_consonantal",
                "prevalence": 0.25,
                "point_ap": 0.32,
                "ap_ci_low": 0.28,
                "ap_ci_high": 0.36,
                "ap_ci_low_pct": 0.29,
                "ap_ci_high_pct": 0.35,
                "point_gap": 0.8,
                "gap_ci_low": 0.6,
                "gap_ci_high": 1.0,
                "gap_ci_low_pct": 0.65,
                "gap_ci_high_pct": 0.95,
                "point_auc": 0.58,
                "auc_ci_low": 0.54,
                "auc_ci_high": 0.62,
                "auc_ci_low_pct": 0.55,
                "auc_ci_high_pct": 0.61,
                "n_valid_resamples": 1000,
                "n_valid_jackknife": 150,
            }
        ]
    )


def test_build_long_metrics_covers_both_sources_with_a_single_overall_scope() -> None:
    long_df = build_long_metrics(_summary_df(), _bootstrap_df())

    assert set(long_df["scope"]) == {"overall"}
    assert set(long_df["scope_kind"]) == {"overall"}
    assert set(long_df["source"]) == {"genre_discrimination", "bootstrap_ci"}
    assert (long_df["model_base"] == "foo").all()
    assert (long_df["text_variant"] == "consonantal").all()


def test_build_long_metrics_preserves_specific_values() -> None:
    long_df = build_long_metrics(_summary_df(), _bootstrap_df())

    def value(source: str, metric: str) -> float:
        row = long_df[(long_df["source"] == source) & (long_df["metric"] == metric)]
        assert len(row) == 1
        return float(row["value"].iloc[0])

    assert value("genre_discrimination", "average_precision") == 0.32
    assert value("genre_discrimination", "gap") == 0.8
    assert value("bootstrap_ci", "ap_ci_low") == 0.28


def test_build_long_metrics_adds_fdr_q_values_only_for_separation_p() -> None:
    long_df = build_long_metrics(_summary_df(), _bootstrap_df())

    p_rows = long_df[long_df["metric"] == "separation_p"]
    assert p_rows["q_value"].notna().all()
    assert p_rows["q_value_by"].notna().all()

    non_p_rows = long_df[long_df["metric"].isin(["average_precision", "separation_auc"])]
    assert non_p_rows["q_value"].isna().all()
    assert non_p_rows["q_value_by"].isna().all()


def test_the_summary_contract_matches_what_the_producer_writes() -> None:
    """A mismatch here failed every master report at run time, after hours of scoring."""
    from genre.calibrated import GenreCalibratedComparison, genre_calibrated_row
    from genre.scripts.build_master_report import _SUMMARY_METRICS

    result = GenreCalibratedComparison(
        n_same_genre=1,
        n_different_genre=1,
        prevalence=0.5,
        mean_same_genre_similarity=0.0,
        mean_different_genre_similarity=0.0,
        same_genre_effect_size=0.0,
        different_genre_effect_size=0.0,
        average_precision=0.0,
        separation_auc=0.0,
        separation_p=1.0,
    )

    produced = set(genre_calibrated_row("m", result))

    assert set(_SUMMARY_METRICS) <= produced


def test_the_bootstrap_contract_matches_what_the_producer_writes() -> None:
    from genre.scripts.build_master_report import _BOOTSTRAP_METRICS
    from genre.scripts.compute_bootstrap_cis import ci_row
    from library.ap_gap_auc_bootstrap import ApGapAucCI

    ci = ApGapAucCI(
        **{k: (0 if k.startswith("n_") else 0.0) for k in ApGapAucCI.__dataclass_fields__}
    )

    assert set(_BOOTSTRAP_METRICS) <= set(ci_row("m", ci))
