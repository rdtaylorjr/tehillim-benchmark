import numpy as np
import pandas as pd

from ui_export.detail import (
    auc_ap_ci_for,
    build_genre_detail,
    build_parallelism_detail,
    build_trajectory_detail,
    genre_mean_matrix,
    heatmap_cells,
    order_psalms_by_own_stat,
    raincloud_group,
    roc_pr_series,
    validated_gap_stats_for,
)


def test_raincloud_group_reports_full_values_n_and_mean() -> None:
    group = raincloud_group(pd.Series([1.0, 2.0, 3.0]))
    assert group == {"values": [1.0, 2.0, 3.0], "n": 3, "mean": 2.0}


def test_raincloud_group_rounds_values_and_mean_to_four_places() -> None:
    group = raincloud_group(pd.Series([1.0 / 3.0]))
    assert group["values"] == [0.3333]
    assert group["mean"] == 0.3333


def test_roc_pr_series_carries_name_n_and_monotonic_curve_points() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    series = roc_pr_series("Combined", labels, scores, n=2)
    assert series["name"] == "Combined"
    assert series["n"] == 2
    assert series["roc"][0] == {"fpr": 0.0, "tpr": 0.0}
    assert series["roc"][-1] == {"fpr": 1.0, "tpr": 1.0}
    assert all(0.0 <= p["precision"] <= 1.0 for p in series["pr"])


def test_genre_mean_matrix_averages_both_pair_orderings() -> None:
    df = pd.DataFrame(
        [
            {"genre_a": "Hymn", "genre_b": "Lament", "value": 0.2},
            {"genre_a": "Lament", "genre_b": "Hymn", "value": 0.6},
        ]
    )
    cells = genre_mean_matrix(df, "value", ["Hymn", "Lament"])
    by_pair = {(c["genre_a"], c["genre_b"]): c["value"] for c in cells}
    assert by_pair[("Hymn", "Lament")] == 0.4
    assert by_pair[("Lament", "Hymn")] == 0.4


def test_genre_mean_matrix_omits_cells_with_no_observed_pairs() -> None:
    df = pd.DataFrame([{"genre_a": "Hymn", "genre_b": "Hymn", "value": 0.5}])
    cells = genre_mean_matrix(df, "value", ["Hymn", "Lament"])
    pairs = {(c["genre_a"], c["genre_b"]) for c in cells}
    assert ("Hymn", "Lament") not in pairs
    assert ("Hymn", "Hymn") in pairs


def test_heatmap_cells_carries_psalm_ids_and_rounded_value() -> None:
    df = pd.DataFrame([{"psalm_a": 1, "psalm_b": 2, "value": 1.0 / 3.0}])
    cells = heatmap_cells(df, "value")
    assert cells == [{"psalm_a": 1, "psalm_b": 2, "value": 0.3333}]


def test_order_psalms_by_own_stat_groups_by_genre_then_descending_value() -> None:
    genre_by_psalm = {1: "Hymn", 2: "Hymn", 3: "Lament"}
    same_genre_df = pd.DataFrame(
        [
            {"psalm_a": 1, "psalm_b": 2, "value": 0.1},
            {"psalm_a": 3, "psalm_b": 3, "value": 0.9},
        ]
    )
    order = order_psalms_by_own_stat(same_genre_df, "value", genre_by_psalm)
    assert order == [
        {"psalm": 1, "genre": "Hymn"},
        {"psalm": 2, "genre": "Hymn"},
        {"psalm": 3, "genre": "Lament"},
    ]


def test_load_auc_ap_ci_reads_the_matching_model_and_scope_row() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "m1",
                "scope": "overall",
                "point_auc": 0.7,
                "auc_ci_low": 0.6,
                "auc_ci_high": 0.8,
                "point_ap": 0.4,
                "ap_ci_low": 0.3,
                "ap_ci_high": 0.5,
            },
            {
                "model": "m1",
                "scope": "Synonymous",
                "point_auc": 0.9,
                "auc_ci_low": 0.85,
                "auc_ci_high": 0.95,
                "point_ap": 0.6,
                "ap_ci_low": 0.5,
                "ap_ci_high": 0.7,
            },
        ]
    )
    stats = auc_ap_ci_for(df, "m1", scope="overall")
    assert stats == {
        "auc": 0.7,
        "auc_ci_low": 0.6,
        "auc_ci_high": 0.8,
        "ap": 0.4,
        "ap_ci_low": 0.3,
        "ap_ci_high": 0.5,
    }


def test_load_auc_ap_ci_returns_none_when_the_model_has_no_row() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "other",
                "point_auc": 0.5,
                "auc_ci_low": 0.4,
                "auc_ci_high": 0.6,
                "point_ap": 0.2,
                "ap_ci_low": 0.1,
                "ap_ci_high": 0.3,
            }
        ]
    )
    assert auc_ap_ci_for(df, "m1", scope=None) is None


def test_load_validated_gap_stats_reads_both_controlled_sources() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "m1",
                "metric": "structural_distance",
                "length_controlled_gap": 0.1,
                "length_controlled_p": 0.02,
                "length_controlled_effect_size": 0.5,
                "length_and_content_controlled_gap": 0.05,
                "length_and_content_controlled_p": 0.1,
                "length_and_content_controlled_effect_size": 0.2,
            }
        ]
    )
    stats = validated_gap_stats_for(df, "m1", "structural_distance")
    assert stats == {
        "length_controlled": {"gap": 0.1, "p": 0.02, "effect_size": 0.5},
        "length_and_content_controlled": {"gap": 0.05, "p": 0.1, "effect_size": 0.2},
    }


def test_load_validated_gap_stats_returns_none_when_the_model_metric_is_missing() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "other",
                "metric": "structural_distance",
                "length_controlled_gap": 0.1,
                "length_controlled_p": 0.02,
                "length_controlled_effect_size": 0.5,
                "length_and_content_controlled_gap": 0.05,
                "length_and_content_controlled_p": 0.1,
                "length_and_content_controlled_effect_size": 0.2,
            }
        ]
    )
    assert validated_gap_stats_for(df, "m1", "structural_distance") is None


def _pair_detail_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pair_id": "p1", "parallelism_type": "Synonymous", "calibrated_z": 2.0},
            {"pair_id": "p2", "parallelism_type": "Synonymous", "calibrated_z": 1.5},
            {"pair_id": "p3", "parallelism_type": "Antithetic", "calibrated_z": 1.0},
        ]
    )


def _baseline_detail_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"calibrated_z": 0.1},
            {"calibrated_z": -0.2},
            {"calibrated_z": 0.0},
        ]
    )


def test_build_parallelism_detail_has_a_combined_group_and_one_group_per_observed_type() -> None:
    detail = build_parallelism_detail(_pair_detail_df(), _baseline_detail_df(), auc_ap_stats=None)
    keys = {g["key"] for g in detail["raincloud_groups"]}
    assert keys == {"baseline", "combined", "Synonymous", "Antithetic"}


def test_build_parallelism_detail_orders_types_in_canonical_scholarly_order() -> None:
    detail = build_parallelism_detail(_pair_detail_df(), _baseline_detail_df(), auc_ap_stats=None)
    series_names = [s["name"] for s in detail["series"]]
    assert series_names == ["Combined", "Synonymous", "Antithetic"]


def test_build_parallelism_detail_passes_through_auc_ap_stats_when_given() -> None:
    stats = {
        "auc": 0.8,
        "auc_ci_low": 0.7,
        "auc_ci_high": 0.9,
        "ap": 0.5,
        "ap_ci_low": 0.4,
        "ap_ci_high": 0.6,
    }
    detail = build_parallelism_detail(_pair_detail_df(), _baseline_detail_df(), auc_ap_stats=stats)
    assert detail["auc_ap_stats"] == stats


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


def test_build_genre_detail_has_a_combined_group_and_one_group_per_observed_genre() -> None:
    detail = build_genre_detail(_genre_pair_df(), genres=["Hymn", "Lament"], auc_ap_stats=None)
    keys = {g["key"] for g in detail["raincloud_groups"]}
    assert keys == {"different", "combined", "Hymn", "Lament"}


def test_build_genre_detail_heatmap_covers_every_pair_row() -> None:
    detail = build_genre_detail(_genre_pair_df(), genres=["Hymn", "Lament"], auc_ap_stats=None)
    assert len(detail["heatmap"]) == 3


def test_build_genre_detail_genre_order_covers_every_psalm_seen_in_the_pairs() -> None:
    detail = build_genre_detail(_genre_pair_df(), genres=["Hymn", "Lament"], auc_ap_stats=None)
    psalms = {e["psalm"] for e in detail["genre_order"]}
    assert psalms == {1, 2, 3, 4}


def test_build_trajectory_detail_reports_both_controlled_sources() -> None:
    traj_df = pd.DataFrame(
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
    gap_stats = {
        "length_controlled": {"gap": 0.1, "p": 0.02, "effect_size": 0.5},
        "length_and_content_controlled": {"gap": 0.05, "p": 0.1, "effect_size": 0.2},
    }
    detail = build_trajectory_detail(
        traj_df, metric="structural_distance", genres=["Hymn", "Lament"], gap_stats=gap_stats
    )
    assert detail["metric"] == "structural_distance"
    assert set(detail["sources"]) == {"length_controlled", "length_and_content_controlled"}
    assert detail["sources"]["length_controlled"]["gap_stats"] == gap_stats["length_controlled"]
    assert detail["sources"]["length_controlled"]["raincloud"]["same"]["n"] == 1
    assert detail["sources"]["length_controlled"]["raincloud"]["different"]["n"] == 1
