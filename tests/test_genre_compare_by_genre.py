from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from genre.across_genres import (
    GenreRunConfig,
    compare_model_across_genres,
    compare_model_across_genres_sparse,
)
from genre.pairs import build_genre_pairs
from genre.scripts.compare_by_genre import add_fdr_columns, load_cached_genre_rows, score_model
from library.errors import BenchmarkDataError
from library.scoring import skipping_unscorable


def _config(genre_by_psalm, genres, pairs) -> GenreRunConfig:
    """The fixture's run configuration, at the small resampling budget the tests use."""
    return GenreRunConfig(
        genre_by_psalm=genre_by_psalm,
        genres=genres,
        pairs=pairs,
        n_permutations=50,
        n_resamples=50,
        seed=0,
    )


def _nan_safe(row: dict) -> dict:
    """NaN never equals itself, so an undefined CI needs a sentinel to compare rows for equality."""
    return {
        key: "nan" if isinstance(value, float) and np.isnan(value) else value
        for key, value in row.items()
    }


def _fixture():
    genre_by_psalm = {
        1: "Lament",
        2: "Lament",
        3: "Lament",
        4: "Praise",
        5: "Praise",
        6: "Wisdom",
        7: "Wisdom",
    }
    psalm_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.95, 0.05]),
        3: np.array([0.9, 0.1]),
        4: np.array([0.0, 1.0]),
        5: np.array([0.05, 0.95]),
        6: np.array([0.5, 0.5]),
        7: np.array([0.55, 0.45]),
    }
    genres = ("Lament", "Praise", "Wisdom")
    pairs = build_genre_pairs(genre_by_psalm)
    psalm_ids = sorted(genre_by_psalm)
    return psalm_ids, psalm_vectors, genre_by_psalm, genres, pairs


class TestCompareModelAcrossGenres:
    def test_returns_one_row_per_genre(self) -> None:
        psalm_ids, psalm_vectors, genre_by_psalm, genres, pairs = _fixture()

        rows = compare_model_across_genres(
            "model_a",
            psalm_ids,
            psalm_vectors,
            _config(genre_by_psalm, genres, pairs),
        )

        assert {row["genre"] for row in rows} == set(genres)
        assert all(row["model"] == "model_a" for row in rows)

    def test_reports_naive_perm_and_maxt_p_values(self) -> None:
        psalm_ids, psalm_vectors, genre_by_psalm, genres, pairs = _fixture()

        rows = compare_model_across_genres(
            "model_a",
            psalm_ids,
            psalm_vectors,
            _config(genre_by_psalm, genres, pairs),
        )

        for row in rows:
            assert "separation_p_naive" in row
            assert "separation_p_perm" in row
            assert "separation_p_maxT" in row
            assert 0.0 < row["separation_p_perm"] <= 1.0
            assert 0.0 < row["separation_p_maxT"] <= 1.0

    def test_reports_ap_and_auc_jackknife_bootstrap_cis(self) -> None:
        psalm_ids, psalm_vectors, genre_by_psalm, genres, pairs = _fixture()

        rows = compare_model_across_genres(
            "model_a",
            psalm_ids,
            psalm_vectors,
            _config(genre_by_psalm, genres, pairs),
        )

        by_genre = {row["genre"]: row for row in rows}
        lament = by_genre["Lament"]
        assert lament["ap_ci_low"] <= lament["average_precision"] <= lament["ap_ci_high"]
        assert lament["auc_ci_low"] <= lament["separation_auc"] <= lament["auc_ci_high"]

    def test_reports_nan_cis_for_a_genre_with_too_few_psalms_to_resample(self) -> None:
        """Praise and Wisdom hold 2 psalms each, so 1 same-genre pair, too few for AP or AUC."""
        psalm_ids, psalm_vectors, genre_by_psalm, genres, pairs = _fixture()

        rows = compare_model_across_genres(
            "model_a",
            psalm_ids,
            psalm_vectors,
            _config(genre_by_psalm, genres, pairs),
        )

        by_genre = {row["genre"]: row for row in rows}
        for genre in ("Praise", "Wisdom"):
            assert np.isnan(by_genre[genre]["ap_ci_low"])
            assert np.isnan(by_genre[genre]["ap_ci_high"])
            assert np.isnan(by_genre[genre]["auc_ci_low"])
            assert np.isnan(by_genre[genre]["auc_ci_high"])
            assert not np.isnan(by_genre[genre]["average_precision"])

    def test_average_precision_matches_the_existing_pair_based_evaluator(self) -> None:
        """The AP point estimate is unchanged: still evaluate_genre_discrimination on the pairs."""
        from genre.evaluate import evaluate_genre_discrimination
        from genre.pairs import filter_pairs_by_genre

        psalm_ids, psalm_vectors, genre_by_psalm, genres, pairs = _fixture()

        rows = compare_model_across_genres(
            "model_a",
            psalm_ids,
            psalm_vectors,
            _config(genre_by_psalm, genres, pairs),
        )

        for row in rows:
            restricted = filter_pairs_by_genre(pairs, row["genre"])
            expected = evaluate_genre_discrimination(restricted, psalm_vectors)
            assert row["average_precision"] == pytest.approx(expected.average_precision)


class TestCompareModelAcrossGenresSparse:
    def test_matches_the_dense_function_exactly(self) -> None:
        """Proves the sparse psalm-similarity path reports identically to the dense one."""
        psalm_ids, psalm_vectors, genre_by_psalm, genres, pairs = _fixture()
        sparse_matrix = sp.csr_matrix(np.stack([psalm_vectors[p] for p in psalm_ids]))

        dense_rows = compare_model_across_genres(
            "model_a",
            psalm_ids,
            psalm_vectors,
            _config(genre_by_psalm, genres, pairs),
        )
        sparse_rows = compare_model_across_genres_sparse(
            "model_a",
            psalm_ids,
            sparse_matrix,
            _config(genre_by_psalm, genres, pairs),
        )

        assert [_nan_safe(row) for row in sparse_rows] == [_nan_safe(row) for row in dense_rows]


def _rows_for_two_genres() -> list[dict]:
    return [
        {
            "model": "m1",
            "genre": "Lament",
            "separation_p_naive": 0.001,
            "separation_p_perm": 0.01,
            "separation_p_maxT": 0.03,
        },
        {
            "model": "m2",
            "genre": "Lament",
            "separation_p_naive": 0.04,
            "separation_p_perm": 0.06,
            "separation_p_maxT": 0.1,
        },
        {
            "model": "m3",
            "genre": "Lament",
            "separation_p_naive": 0.5,
            "separation_p_perm": 0.5,
            "separation_p_maxT": 0.6,
        },
        {
            "model": "m1",
            "genre": "Trust",
            "separation_p_naive": 0.02,
            "separation_p_perm": 0.03,
            "separation_p_maxT": 0.05,
        },
        {
            "model": "m2",
            "genre": "Trust",
            "separation_p_naive": 0.3,
            "separation_p_perm": 0.3,
            "separation_p_maxT": 0.4,
        },
        {
            "model": "m3",
            "genre": "Trust",
            "separation_p_naive": 0.7,
            "separation_p_perm": 0.7,
            "separation_p_maxT": 0.8,
        },
    ]


def _cache_row() -> dict:
    return {
        "model": "m1",
        "genre": "Lament",
        "n_same_genre": 3,
        "n_different_genre": 10,
        "prevalence": 0.2,
        "average_precision": 0.5,
        "ap_ci_low": 0.4,
        "ap_ci_high": 0.6,
        "separation_auc": 0.7,
        "auc_ci_low": 0.6,
        "auc_ci_high": 0.8,
        "separation_p_naive": 0.01,
        "separation_p_perm": 0.02,
        "separation_p_maxT": 0.05,
        "n_permutations": 2000,
        # Stale q-values: a rerun adding models changes the FDR family and invalidates them.
        "naive_q": 0.03,
        "naive_q_by": 0.04,
        "perm_q": 0.03,
        "perm_q_by": 0.04,
        "maxT_q": 0.03,
        "maxT_q_by": 0.04,
    }


class TestLoadCachedRows:
    def test_returns_empty_when_the_cache_path_does_not_exist(self, tmp_path) -> None:
        rows, models = load_cached_genre_rows(tmp_path / "missing.csv")

        assert rows == []
        assert models == set()

    def test_reads_back_the_raw_columns_and_the_model_set(self, tmp_path) -> None:
        cache_path = tmp_path / "cache.csv"
        pd.DataFrame([_cache_row()]).to_csv(cache_path, index=False)

        rows, models = load_cached_genre_rows(cache_path)

        assert models == {"m1"}
        assert len(rows) == 1
        assert rows[0]["model"] == "m1"
        assert rows[0]["separation_p_perm"] == pytest.approx(0.02)

    def test_drops_the_stale_q_value_columns(self, tmp_path) -> None:
        cache_path = tmp_path / "cache.csv"
        pd.DataFrame([_cache_row()]).to_csv(cache_path, index=False)

        rows, _ = load_cached_genre_rows(cache_path)

        for stale_column in ("naive_q", "naive_q_by", "perm_q", "perm_q_by", "maxT_q", "maxT_q_by"):
            assert stale_column not in rows[0]

    def test_covers_every_model_present_in_the_cache(self, tmp_path) -> None:
        cache_path = tmp_path / "cache.csv"
        rows = [_cache_row(), {**_cache_row(), "model": "m2", "genre": "Praise"}]
        pd.DataFrame(rows).to_csv(cache_path, index=False)

        _, models = load_cached_genre_rows(cache_path)

        assert models == {"m1", "m2"}


class TestAddFdrColumns:
    def test_scopes_correction_separately_per_genre(self) -> None:
        rows = _rows_for_two_genres()

        result = add_fdr_columns(rows)

        lament_only = add_fdr_columns([r for r in rows if r["genre"] == "Lament"])
        merged = result[result["genre"] == "Lament"].sort_values("model")
        isolated = lament_only.sort_values("model")
        assert merged["naive_q"].to_numpy() == pytest.approx(isolated["naive_q"].to_numpy())

    def test_adds_q_values_for_all_three_sources(self) -> None:
        rows = _rows_for_two_genres()

        result = add_fdr_columns(rows)

        for source in ("naive", "perm", "maxT"):
            assert f"{source}_q" in result.columns
            assert f"{source}_q_by" in result.columns
            assert result[f"{source}_q"].notna().all()


class TestScoreModel:
    def test_names_every_row_after_the_files_dataset_identifier(
        self, tmp_path: Path, write_embeddings_parquet
    ) -> None:
        _psalm_ids, psalm_vectors, genre_by_psalm, genres, pairs = _fixture()
        path = write_embeddings_parquet(
            tmp_path / "domain=d" / "model=mine" / "v.parquet",
            {psalm: vector.tolist() for psalm, vector in psalm_vectors.items()},
        )

        rows = score_model(
            path,
            {psalm: [psalm] for psalm in psalm_vectors},
            _config(genre_by_psalm, genres, pairs),
        )

        assert {row["model"] for row in rows} == {"mine"}
        assert {row["genre"] for row in rows} == set(genres)

    def test_raises_for_a_model_whose_psalm_vectors_cannot_be_scored(
        self, tmp_path: Path, write_embeddings_parquet
    ) -> None:
        """One usable psalm leaves no pair, which the shared policy turns into a skip."""
        path = write_embeddings_parquet(
            tmp_path / "domain=d" / "model=tiny" / "v.parquet", {1: [1.0, 0.0]}
        )
        score = partial(
            score_model,
            half_verses_by_psalm={1: [1]},
            config=GenreRunConfig(
                genre_by_psalm={1: "Lament"},
                genres=("Lament",),
                pairs=build_genre_pairs({1: "Lament"}),
                n_permutations=10,
                n_resamples=10,
                seed=0,
            ),
        )

        with pytest.raises(BenchmarkDataError):
            score(path)
        assert skipping_unscorable(score)(path) is None
