from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from library.multiple_comparisons import benjamini_hochberg, benjamini_yekutieli
from trajectory.scripts.validate_against_genre import (
    _METRICS,
    add_fdr_columns,
    add_genre_breakdown_fdr_columns,
    breakdown_path_for,
    build_genre_breakdown_rows,
    build_validation_row,
    residualize_by_length,
    residualize_on_covariates,
    validate_models,
    validate_one_model,
)
from trajectory.validation import (
    _null_gaps,
    observed_gap,
    permutation_test,
    same_genre_matrix,
)


def test_observed_gap_is_positive_when_same_genre_pairs_are_closer() -> None:
    distances = np.array([0.1, 0.1, 0.9, 0.9])
    same_genre = np.array([True, True, False, False])

    assert observed_gap(distances, same_genre) == pytest.approx(0.8)


def test_observed_gap_is_zero_when_within_and_between_means_match() -> None:
    distances = np.array([0.5, 0.5, 0.5, 0.5])
    same_genre = np.array([True, False, True, False])

    assert observed_gap(distances, same_genre) == pytest.approx(0.0)


def test_permutation_test_finds_a_strong_signal_significant() -> None:
    """8 psalms, 2 genres of 4; within-genre pairs are all close, between-genre all far."""
    pairs = list(combinations(range(8), 2))
    idx_a = np.array([a for a, _ in pairs])
    idx_b = np.array([b for _, b in pairs])
    genre_labels = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    same = genre_labels[idx_a] == genre_labels[idx_b]
    distances = np.where(same, 0.1, 0.9)
    rng = np.random.default_rng(0)

    observed, p_value, effect_size = permutation_test(
        idx_a, idx_b, distances, genre_labels, n_permutations=2000, rng=rng
    )

    assert observed == pytest.approx(0.8)
    assert p_value < 0.05
    assert effect_size > 2.0


def test_permutation_test_reports_a_large_p_value_when_there_is_no_signal() -> None:
    idx_a = np.array([0, 0, 0, 1, 1, 2])
    idx_b = np.array([1, 2, 3, 2, 3, 3])
    genre_labels = np.array(["A", "B", "A", "B"])
    distances = np.full(6, 0.5)
    rng = np.random.default_rng(0)

    observed, p_value, effect_size = permutation_test(
        idx_a, idx_b, distances, genre_labels, n_permutations=500, rng=rng
    )

    assert observed == pytest.approx(0.0)
    assert p_value > 0.05
    # A constant array gives a zero gap under every permutation, so the z-score is undefined.
    assert np.isnan(effect_size)


def test_permutation_test_effect_size_matches_a_manual_z_score_against_the_null() -> None:
    """Effect size is (observed - null mean) / null SD, with the ddof=1 sample SD throughout."""
    idx_a = np.array([0, 0, 0, 1, 1, 2])
    idx_b = np.array([1, 2, 3, 2, 3, 3])
    genre_labels = np.array(["A", "B", "A", "B"])
    distances = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9])

    _, codes = np.unique(genre_labels, return_inverse=True)
    tiled_codes = np.tile(codes, (500, 1))
    shuffled_codes = np.random.default_rng(3).permuted(tiled_codes, axis=1)
    same_matrix = shuffled_codes[:, idx_a] == shuffled_codes[:, idx_b]
    expected_null = _null_gaps(same_matrix, distances)
    same_genre = genre_labels[idx_a] == genre_labels[idx_b]
    expected_observed = observed_gap(distances, same_genre)
    expected_effect_size = (expected_observed - expected_null.mean()) / expected_null.std(ddof=1)

    observed, _, effect_size = permutation_test(
        idx_a, idx_b, distances, genre_labels, n_permutations=500, rng=np.random.default_rng(3)
    )

    assert observed == pytest.approx(expected_observed)
    assert effect_size == pytest.approx(expected_effect_size)


def test_residualize_by_length_removes_a_perfect_linear_relationship() -> None:
    length_diff = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    distances = 2.0 * length_diff + 1.0

    residual = residualize_by_length(distances, length_diff)

    assert residual == pytest.approx(np.zeros(5), abs=1e-9)


def test_residualize_by_length_leaves_a_constant_curve_centered_on_zero() -> None:
    length_diff = np.array([0.0, 5.0, 10.0, 15.0])
    distances = np.full(4, 0.7)

    residual = residualize_by_length(distances, length_diff)

    assert residual == pytest.approx(np.zeros(4), abs=1e-9)


def test_residualize_by_length_preserves_a_genuine_non_length_effect() -> None:
    """A length-driven trend plus an offset orthogonal to length: residual isolates the offset."""
    length_diff = np.array([0.0, 1.0, 2.0, 3.0])
    offset = np.array([0.2, -0.2, -0.2, 0.2])  # mean zero, uncorrelated with length_diff
    distances = 0.5 * length_diff + offset

    residual = residualize_by_length(distances, length_diff)

    assert residual == pytest.approx(offset, abs=1e-9)


def test_residualize_on_covariates_matches_residualize_by_length_for_one_covariate() -> None:
    length_diff = np.array([0.0, 1.0, 2.0, 3.0])
    offset = np.array([0.2, -0.2, -0.2, 0.2])
    distances = 0.5 * length_diff + offset

    single = residualize_by_length(distances, length_diff)
    general = residualize_on_covariates(distances, length_diff.reshape(-1, 1))

    assert general == pytest.approx(single, abs=1e-9)


def test_residualize_on_covariates_isolates_a_signal_orthogonal_to_two_covariates() -> None:
    """A 2^2 orthogonal design: the interaction column is orthogonal to both main effects."""
    cov1 = np.array([-1.0, -1.0, 1.0, 1.0])
    cov2 = np.array([-1.0, 1.0, -1.0, 1.0])
    offset = cov1 * cov2  # [1, -1, -1, 1], orthogonal to intercept, cov1, and cov2
    distances = 2.0 * cov1 + 3.0 * cov2 + 5.0 + offset

    residual = residualize_on_covariates(distances, np.column_stack([cov1, cov2]))

    assert residual == pytest.approx(offset, abs=1e-9)


def test_null_gaps_matches_looping_observed_gap_row_by_row() -> None:
    """Proves the vectorized null-distribution computation is lossless against the naive loop."""
    rng = np.random.default_rng(1)
    distances = rng.random(20)
    same_matrix = rng.random((15, 20)) < 0.5

    vectorized = _null_gaps(same_matrix, distances)
    looped = np.array([observed_gap(distances, row) for row in same_matrix])

    assert vectorized == pytest.approx(looped, abs=1e-12)


def test_permutation_test_null_gaps_are_reproducible_from_the_same_seed() -> None:
    """Same seed must give the same p-value: the vectorized RNG usage is deterministic."""
    idx_a = np.array([0, 0, 0, 1, 1, 2])
    idx_b = np.array([1, 2, 3, 2, 3, 3])
    genre_labels = np.array(["A", "B", "A", "B"])
    distances = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9])

    _, p_first, _ = permutation_test(
        idx_a, idx_b, distances, genre_labels, n_permutations=500, rng=np.random.default_rng(7)
    )
    _, p_second, _ = permutation_test(
        idx_a, idx_b, distances, genre_labels, n_permutations=500, rng=np.random.default_rng(7)
    )

    assert p_first == p_second


def _base_subset() -> tuple[pd.DataFrame, dict[int, int], np.ndarray, dict[int, int]]:
    base_subset = pd.DataFrame(
        {
            "psalm_a": [1, 1, 1, 2, 2, 3],
            "psalm_b": [2, 3, 4, 3, 4, 4],
            "metric_x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "content_distance": [0.15, 0.25, 0.35, 0.05, 0.45, 0.55],
        }
    )
    index_of = {1: 0, 2: 1, 3: 2, 4: 3}
    genre_labels = np.array(["A", "A", "B", "B"])
    n_half_verses = {1: 10, 2: 12, 3: 8, 4: 20}
    return base_subset, index_of, genre_labels, n_half_verses


def test_build_validation_row_reports_full_pair_counts_when_nothing_is_nan() -> None:
    base_subset, index_of, genre_labels, n_half_verses = _base_subset()

    row = build_validation_row(
        "model_a", "metric_x", base_subset, index_of, genre_labels, n_half_verses, 500, 0
    )

    assert row["n_pairs_total"] == 6
    assert row["n_pairs_valid"] == 6
    assert not np.isnan(row["raw_gap"])
    assert not np.isnan(row["raw_p"])
    # Every 2-2 bipartition of this arithmetic sequence gives zero gap, so effect size is undefined.
    assert np.isnan(row["raw_effect_size"])


def test_build_validation_row_excludes_nan_valued_pairs() -> None:
    base_subset, index_of, genre_labels, n_half_verses = _base_subset()
    base_subset.loc[1, "metric_x"] = np.nan  # drops pair (1,3), a different-genre pair

    row = build_validation_row(
        "model_a", "metric_x", base_subset, index_of, genre_labels, n_half_verses, 500, 0
    )

    assert row["n_pairs_total"] == 6
    assert row["n_pairs_valid"] == 5
    assert not np.isnan(row["raw_p"])


def test_build_validation_row_returns_nan_when_too_few_valid_pairs_remain() -> None:
    base_subset, index_of, genre_labels, n_half_verses = _base_subset()
    base_subset.loc[[1, 2, 3, 4, 5], "metric_x"] = np.nan  # only pair (1,2), same genre, survives

    row = build_validation_row(
        "model_a", "metric_x", base_subset, index_of, genre_labels, n_half_verses, 500, 0
    )

    assert row["n_pairs_valid"] == 1
    assert np.isnan(row["raw_gap"])
    assert np.isnan(row["raw_p"])
    assert np.isnan(row["raw_effect_size"])
    assert np.isnan(row["length_controlled_gap"])
    assert np.isnan(row["length_controlled_p"])
    assert np.isnan(row["length_controlled_effect_size"])
    assert np.isnan(row["length_and_content_controlled_p"])
    assert np.isnan(row["length_and_content_controlled_effect_size"])


def test_build_validation_row_computes_length_and_content_controlled_for_non_content_metrics() -> (
    None
):
    base_subset, index_of, genre_labels, n_half_verses = _base_subset()

    row = build_validation_row(
        "model_a", "metric_x", base_subset, index_of, genre_labels, n_half_verses, 500, 0
    )

    assert not np.isnan(row["length_and_content_controlled_gap"])
    assert not np.isnan(row["length_and_content_controlled_p"])
    assert not np.isnan(row["length_and_content_controlled_effect_size"])


def test_build_validation_row_content_distance_has_no_third_source() -> None:
    """Controlling content_distance for content_distance itself is not a meaningful comparison."""
    base_subset, index_of, genre_labels, n_half_verses = _base_subset()

    row = build_validation_row(
        "model_a", "content_distance", base_subset, index_of, genre_labels, n_half_verses, 500, 0
    )

    assert np.isnan(row["length_and_content_controlled_gap"])
    assert np.isnan(row["length_and_content_controlled_p"])
    assert np.isnan(row["length_and_content_controlled_effect_size"])
    assert not np.isnan(row["raw_p"])
    assert not np.isnan(row["raw_effect_size"])


def test_add_fdr_columns_ignores_nan_p_values_in_the_correction_family() -> None:
    rows = _rows_for_two_metrics()
    rows.append(
        {
            "model": "m4",
            "metric": "content_distance",
            "raw_p": float("nan"),
            "length_controlled_p": float("nan"),
            "length_and_content_controlled_p": float("nan"),
        }
    )

    result = add_fdr_columns(rows)

    content_rows = result[result["metric"] == "content_distance"]
    without_nan = content_rows[content_rows["model"] != "m4"].sort_values("model")
    expected_q = benjamini_hochberg(without_nan["raw_p"].to_numpy())
    assert without_nan["raw_q"].to_numpy() == pytest.approx(expected_q)
    m4_row = content_rows[content_rows["model"] == "m4"].iloc[0]
    assert np.isnan(m4_row["raw_q"])


def _rows_for_two_metrics() -> list[dict]:
    nan = float("nan")
    return [
        {
            "model": "m1",
            "metric": "content_distance",
            "raw_p": 0.001,
            "length_controlled_p": 0.01,
            "length_and_content_controlled_p": nan,
        },
        {
            "model": "m2",
            "metric": "content_distance",
            "raw_p": 0.04,
            "length_controlled_p": 0.2,
            "length_and_content_controlled_p": nan,
        },
        {
            "model": "m3",
            "metric": "content_distance",
            "raw_p": 0.5,
            "length_controlled_p": 0.6,
            "length_and_content_controlled_p": nan,
        },
        {
            "model": "m1",
            "metric": "structural_distance",
            "raw_p": 0.02,
            "length_controlled_p": 0.03,
            "length_and_content_controlled_p": 0.05,
        },
        {
            "model": "m2",
            "metric": "structural_distance",
            "raw_p": 0.3,
            "length_controlled_p": 0.4,
            "length_and_content_controlled_p": 0.45,
        },
        {
            "model": "m3",
            "metric": "structural_distance",
            "raw_p": 0.7,
            "length_controlled_p": 0.8,
            "length_and_content_controlled_p": 0.85,
        },
    ]


def test_add_fdr_columns_matches_direct_benjamini_hochberg_within_one_metric_and_source() -> None:
    rows = _rows_for_two_metrics()

    result = add_fdr_columns(rows)

    content_rows = result[result["metric"] == "content_distance"].sort_values("model")
    expected_q = benjamini_hochberg(content_rows["raw_p"].to_numpy())
    assert content_rows["raw_q"].to_numpy() == pytest.approx(expected_q)
    expected_q_by = benjamini_yekutieli(content_rows["raw_p"].to_numpy())
    assert content_rows["raw_q_by"].to_numpy() == pytest.approx(expected_q_by)


def test_add_fdr_columns_scopes_correction_separately_per_metric() -> None:
    """content_distance's q-values must not be influenced by structural_distance's p-values."""
    rows = _rows_for_two_metrics()

    result = add_fdr_columns(rows)

    content_only = add_fdr_columns([r for r in rows if r["metric"] == "content_distance"])
    merged = result[result["metric"] == "content_distance"].sort_values("model")
    isolated = content_only.sort_values("model")
    assert merged["raw_q"].to_numpy() == pytest.approx(isolated["raw_q"].to_numpy())


def test_add_fdr_columns_scopes_correction_separately_per_source() -> None:
    rows = _rows_for_two_metrics()

    result = add_fdr_columns(rows)

    content_rows = result[result["metric"] == "content_distance"].sort_values("model")
    expected_controlled_q = benjamini_hochberg(content_rows["length_controlled_p"].to_numpy())
    assert content_rows["length_controlled_q"].to_numpy() == pytest.approx(expected_controlled_q)


def test_build_genre_breakdown_rows_has_one_row_per_genre_and_available_source() -> None:
    base_subset, index_of, genre_labels, n_half_verses = _base_subset()

    rows = build_genre_breakdown_rows(
        "model_a", "metric_x", base_subset, index_of, genre_labels, n_half_verses, 200, 0
    )

    genres = {r["genre"] for r in rows}
    sources = {r["source"] for r in rows}
    assert genres == {"A", "B"}
    assert sources == {"raw", "length_controlled", "length_and_content_controlled"}
    assert all(r["model"] == "model_a" and r["metric"] == "metric_x" for r in rows)
    assert all(not np.isnan(r["gap"]) for r in rows)
    assert all(0.0 < r["p_perm"] <= 1.0 for r in rows)
    assert all(0.0 < r["p_maxT"] <= 1.0 for r in rows)


def test_build_genre_breakdown_rows_content_distance_has_no_third_source() -> None:
    base_subset, index_of, genre_labels, n_half_verses = _base_subset()

    rows = build_genre_breakdown_rows(
        "model_a", "content_distance", base_subset, index_of, genre_labels, n_half_verses, 200, 0
    )

    sources = {r["source"] for r in rows}
    assert sources == {"raw", "length_controlled"}


def test_build_genre_breakdown_rows_empty_when_no_valid_pairs_remain() -> None:
    base_subset, index_of, genre_labels, n_half_verses = _base_subset()
    base_subset["metric_x"] = np.nan

    rows = build_genre_breakdown_rows(
        "model_a", "metric_x", base_subset, index_of, genre_labels, n_half_verses, 200, 0
    )

    assert rows == []


def test_add_genre_breakdown_fdr_columns_matches_direct_benjamini_hochberg() -> None:
    rows = [
        {
            "model": "m1",
            "metric": "content_distance",
            "source": "raw",
            "genre": "A",
            "p_perm": 0.001,
            "p_maxT": 0.01,
        },
        {
            "model": "m2",
            "metric": "content_distance",
            "source": "raw",
            "genre": "A",
            "p_perm": 0.04,
            "p_maxT": 0.2,
        },
        {
            "model": "m3",
            "metric": "content_distance",
            "source": "raw",
            "genre": "A",
            "p_perm": 0.5,
            "p_maxT": 0.6,
        },
        {
            "model": "m1",
            "metric": "content_distance",
            "source": "raw",
            "genre": "B",
            "p_perm": 0.3,
            "p_maxT": 0.4,
        },
        {
            "model": "m2",
            "metric": "content_distance",
            "source": "raw",
            "genre": "B",
            "p_perm": 0.6,
            "p_maxT": 0.7,
        },
        {
            "model": "m3",
            "metric": "content_distance",
            "source": "raw",
            "genre": "B",
            "p_perm": 0.9,
            "p_maxT": 0.95,
        },
    ]

    result = add_genre_breakdown_fdr_columns(rows)

    genre_a = result[result["genre"] == "A"].sort_values("model")
    expected_q = benjamini_hochberg(genre_a["p_perm"].to_numpy())
    assert genre_a["perm_q"].to_numpy() == pytest.approx(expected_q)
    expected_q_by = benjamini_yekutieli(genre_a["p_perm"].to_numpy())
    assert genre_a["perm_q_by"].to_numpy() == pytest.approx(expected_q_by)
    expected_maxt_q = benjamini_hochberg(genre_a["p_maxT"].to_numpy())
    assert genre_a["maxT_q"].to_numpy() == pytest.approx(expected_maxt_q)


def test_add_genre_breakdown_fdr_columns_scopes_correction_separately_per_genre() -> None:
    rows = [
        {
            "model": "m1",
            "metric": "content_distance",
            "source": "raw",
            "genre": "A",
            "p_perm": 0.001,
            "p_maxT": 0.01,
        },
        {
            "model": "m2",
            "metric": "content_distance",
            "source": "raw",
            "genre": "A",
            "p_perm": 0.04,
            "p_maxT": 0.2,
        },
        {
            "model": "m1",
            "metric": "content_distance",
            "source": "raw",
            "genre": "B",
            "p_perm": 0.3,
            "p_maxT": 0.4,
        },
        {
            "model": "m2",
            "metric": "content_distance",
            "source": "raw",
            "genre": "B",
            "p_perm": 0.6,
            "p_maxT": 0.7,
        },
    ]

    result = add_genre_breakdown_fdr_columns(rows)

    genre_a_only = add_genre_breakdown_fdr_columns([r for r in rows if r["genre"] == "A"])
    merged = result[result["genre"] == "A"].sort_values("model")
    isolated = genre_a_only.sort_values("model")
    assert merged["perm_q"].to_numpy() == pytest.approx(isolated["perm_q"].to_numpy())


def _distances_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for model in ("phrase_a", "phrase_b", "phrase_c"):
        for a, b in combinations(range(1, 9), 2):
            rows.append(
                {
                    "model": model,
                    "psalm_a": a,
                    "psalm_b": b,
                    "content_distance": float(rng.random()),
                    "structural_distance": float(rng.random()),
                    "adjacent_similarity_distance": float(rng.random()),
                    "step_magnitude_distance": float(rng.random()),
                    "turning_angle_distance": float(rng.random()),
                }
            )
    return pd.DataFrame(rows)


_GENRES = {
    1: "Hymn",
    2: "Hymn",
    3: "Hymn",
    4: "Lament",
    5: "Lament",
    6: "Lament",
    7: "Wisdom",
    8: "Wisdom",
}
_N_HALF_VERSES = {p: 10 + p for p in range(1, 9)}


def test_validate_one_model_returns_a_row_per_metric() -> None:
    df = _distances_frame()
    group = df[df.model == "phrase_a"]

    rows, breakdown = validate_one_model("phrase_a", group, _GENRES, _N_HALF_VERSES, 200, 0)

    assert {r["metric"] for r in rows} == set(_METRICS)
    assert all(r["model"] == "phrase_a" for r in rows)
    assert breakdown


def test_validate_models_matches_running_each_model_alone() -> None:
    df = _distances_frame()

    together, _ = validate_models(df, _GENRES, _N_HALF_VERSES, 200, 0, max_workers=2)
    apart = [
        row
        for model in sorted(df["model"].unique())
        for row in validate_one_model(
            model, df[df.model == model], _GENRES, _N_HALF_VERSES, 200, 0
        )[0]
    ]

    def key(row: dict) -> tuple[str, str]:
        return (row["model"], row["metric"])

    assert [key(r) for r in sorted(together, key=key)] == [key(r) for r in sorted(apart, key=key)]
    for a, b in zip(sorted(together, key=key), sorted(apart, key=key), strict=True):
        assert a["raw_gap"] == pytest.approx(b["raw_gap"])
        assert a["length_controlled_p"] == pytest.approx(b["length_controlled_p"])


def test_validate_models_is_deterministic_regardless_of_worker_count() -> None:
    df = _distances_frame()

    one, _ = validate_models(df, _GENRES, _N_HALF_VERSES, 200, 0, max_workers=1)
    four, _ = validate_models(df, _GENRES, _N_HALF_VERSES, 200, 0, max_workers=4)

    def key(row: dict) -> tuple[str, str]:
        return (row["model"], row["metric"])

    for a, b in zip(sorted(one, key=key), sorted(four, key=key), strict=True):
        assert a["length_controlled_p"] == pytest.approx(b["length_controlled_p"])
        assert a["raw_effect_size"] == pytest.approx(b["raw_effect_size"])


def test_same_genre_matrix_is_identical_for_a_repeated_seed() -> None:
    labels = np.array(["Hymn", "Hymn", "Lament", "Lament", "Wisdom"])
    idx_a = np.array([0, 1, 2, 3])
    idx_b = np.array([1, 2, 3, 4])

    first = same_genre_matrix(idx_a, idx_b, labels, 50, np.random.default_rng(0))
    second = same_genre_matrix(idx_a, idx_b, labels, 50, np.random.default_rng(0))

    assert np.array_equal(first, second)


def test_permutation_test_with_a_supplied_matrix_matches_building_it_inline() -> None:
    rng_values = np.random.default_rng(11)
    labels = np.array(["Hymn", "Hymn", "Hymn", "Lament", "Lament", "Lament"])
    idx_a = np.array([0, 1, 2, 3, 4, 0, 1])
    idx_b = np.array([1, 2, 3, 4, 5, 4, 5])
    distances = rng_values.random(len(idx_a))

    inline = permutation_test(
        idx_a, idx_b, distances, labels, n_permutations=200, rng=np.random.default_rng(7)
    )
    shared = same_genre_matrix(idx_a, idx_b, labels, 200, np.random.default_rng(7))
    reused = permutation_test(
        idx_a,
        idx_b,
        distances,
        labels,
        n_permutations=200,
        rng=np.random.default_rng(7),
        same_matrix=shared,
    )

    assert inline == pytest.approx(reused)


def test_breakdown_path_sits_beside_the_validation_csv() -> None:
    from pathlib import Path

    assert breakdown_path_for(Path("/data/validate_against_genre.csv")) == Path(
        "/data/validate_against_genre_by_genre.csv"
    )


def test_breakdown_path_keeps_a_non_standard_stem() -> None:
    from pathlib import Path

    assert breakdown_path_for(Path("out/run7.csv")) == Path("out/run7_by_genre.csv")


def test_null_gaps_chunks_permutations_without_changing_a_single_bit() -> None:
    """Each permutation row is independent, so chunking must be exact, not merely close."""
    from library.blocking import rows_per_block
    from trajectory.validation import _null_gaps

    rng = np.random.default_rng(11)
    n_pairs = 400
    n_rows = rows_per_block(n_pairs) * 2 + 5
    same = rng.random((n_rows, n_pairs)) < 0.3
    same[:, 0] = True
    distances = rng.random(n_pairs)

    chunked = _null_gaps(same, distances)
    row_by_row = np.concatenate([_null_gaps(same[i : i + 1], distances) for i in range(n_rows)])

    assert np.array_equal(chunked, row_by_row)


def test_null_gaps_peak_memory_does_not_grow_with_the_permutation_count() -> None:
    """Casting a 10000 by 11175 boolean to float64 costs 894 MB in one allocation."""
    import tracemalloc

    from trajectory.validation import _null_gaps

    rng = np.random.default_rng(12)
    n_pairs = 2000
    distances = rng.random(n_pairs)

    def peak_for(n_rows: int) -> int:
        same = rng.random((n_rows, n_pairs)) < 0.3
        same[:, 0] = True
        tracemalloc.start()
        _null_gaps(same, distances)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    small, large = peak_for(500), peak_for(5000)

    assert large < small * 3, f"peak grew from {small} to {large} with ten times the permutations"
