from pathlib import Path

import numpy as np
import pytest
from conftest import _write_embeddings_parquet as _write_parquet
from families.shuffle import Draws

from genre.bootstrap import psalm_similarity_matrix
from genre.evaluate import evaluate_genre_discrimination
from genre.pairs import build_genre_pairs, filter_pairs_by_genre
from genre.scripts.shuffle_order_control import (
    genre_ap,
    require_scoreable,
    score_built_seed,
    score_genre_ap,
    shuffled_scores_by_genre,
)
from library.errors import BenchmarkDataError


class TestScoreGenreAp:
    def test_perfect_genre_clustering_scores_ap_one(self, tmp_path: Path) -> None:
        # Psalms 1-2 share Lament, psalms 3-4 share Praise: two tight clusters, well separated.
        genre_by_psalm = {1: "Lament", 2: "Lament", 3: "Praise", 4: "Praise"}
        pairs = build_genre_pairs(genre_by_psalm)
        half_verses_by_psalm = {1: [10], 2: [11], 3: [12], 4: [13]}
        path = tmp_path / "embeddings.parquet"
        _write_parquet(
            path,
            {10: [1.0, 0.0], 11: [1.0, 0.0], 12: [0.0, 1.0], 13: [0.0, 1.0]},
        )

        scores = score_genre_ap(path, half_verses_by_psalm, pairs, ["Lament", "Praise"])

        assert scores == {"Lament": 1.0, "Praise": 1.0}

    def test_returns_one_score_per_requested_genre(self, tmp_path: Path) -> None:
        genre_by_psalm = {1: "Lament", 2: "Lament", 3: "Praise", 4: "Praise"}
        pairs = build_genre_pairs(genre_by_psalm)
        half_verses_by_psalm = {1: [10], 2: [11], 3: [12], 4: [13]}
        path = tmp_path / "embeddings.parquet"
        _write_parquet(
            path,
            {10: [1.0, 0.0], 11: [1.0, 0.0], 12: [0.0, 1.0], 13: [0.0, 1.0]},
        )

        scores = score_genre_ap(path, half_verses_by_psalm, pairs, ["Lament", "Praise"])

        assert set(scores) == {"Lament", "Praise"}

    def test_pools_a_psalm_s_half_verse_vectors_into_one_centroid(self, tmp_path: Path) -> None:
        # Psalm 1's two half-verses average to [1,0], matching psalm 2 exactly: still perfect AP.
        genre_by_psalm = {1: "Lament", 2: "Lament", 3: "Praise", 4: "Praise"}
        pairs = build_genre_pairs(genre_by_psalm)
        half_verses_by_psalm = {1: [10, 11], 2: [12], 3: [13], 4: [14]}
        path = tmp_path / "embeddings.parquet"
        _write_parquet(
            path,
            {
                10: [2.0, 0.0],
                11: [0.0, 0.1],
                12: [1.0, 0.05],
                13: [0.0, 1.0],
                14: [0.0, 1.0],
            },
        )

        scores = score_genre_ap(path, half_verses_by_psalm, pairs, ["Lament", "Praise"])

        assert scores["Lament"] == 1.0


class TestShuffledScoresByGenre:
    def test_transposes_per_seed_scores_into_one_array_per_genre(self) -> None:
        scores_by_seed = [{"A": 0.1, "B": 0.9}, {"A": 0.2, "B": 0.8}]

        result = shuffled_scores_by_genre(scores_by_seed, ["A", "B"])

        assert result["A"].tolist() == [0.1, 0.2]
        assert result["B"].tolist() == [0.9, 0.8]

    def test_preserves_seed_order_so_a_parallel_run_matches_a_sequential_one(self) -> None:
        """Shuffle scores form a null distribution, so their order must not depend on scheduling."""
        scores_by_seed = [{"A": float(i)} for i in range(50)]

        result = shuffled_scores_by_genre(scores_by_seed, ["A"])

        assert result["A"].tolist() == [float(i) for i in range(50)]

    def test_returns_an_empty_array_for_a_genre_with_no_shuffles(self) -> None:
        result = shuffled_scores_by_genre([], ["A"])

        assert result["A"].tolist() == []


#: Two psalms a genre, tight within genre and orthogonal across it, so every AP is defined.
BUILT_VECTORS = {10: [1.0, 0.0], 11: [1.0, 0.0], 12: [0.0, 1.0], 13: [0.0, 1.0]}
GENRE_BY_PSALM = {1: "Lament", 2: "Lament", 3: "Praise", 4: "Praise"}
HALF_VERSES_BY_PSALM = {1: [10], 2: [11], 3: [12], 4: [13]}


def _draws(vectors: dict[int, object], sparse_width: int | None) -> Draws:
    """A registry entry whose every seed builds the same vectors, so a score is comparable."""
    return Draws(
        key="test/family/construction",
        psalms=(),
        permute=lambda psalms, seed: {},
        build=lambda psalms, order: vectors,
        sparse_width=sparse_width,
    )


def _sparse(vectors: dict[int, list[float]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        node: (
            np.array([i for i, value in enumerate(values) if value], dtype=np.int32),
            np.array([value for value in values if value], dtype="<f4"),
        )
        for node, values in vectors.items()
    }


class TestScoreBuiltSeed:
    """The fused control scores a draw it built, which must match the file it replaces."""

    def _arguments(self) -> tuple[list[object], list[str]]:
        return build_genre_pairs(GENRE_BY_PSALM), ["Lament", "Praise"]

    def test_scores_a_dense_draw_as_it_scores_the_written_file(self, tmp_path: Path) -> None:
        pairs, genres = self._arguments()
        path = tmp_path / "embeddings.parquet"
        _write_parquet(path, BUILT_VECTORS)
        dense = {node: np.array(values, dtype="<f4") for node, values in BUILT_VECTORS.items()}

        built = score_built_seed(_draws(dense, None), HALF_VERSES_BY_PSALM, pairs, genres, seed=1)

        assert built == score_genre_ap(path, HALF_VERSES_BY_PSALM, pairs, genres)

    def test_scores_a_sparse_draw_as_it_scores_the_dense_draw_it_matches(self) -> None:
        pairs, genres = self._arguments()
        dense = {node: np.array(values, dtype="<f4") for node, values in BUILT_VECTORS.items()}

        sparse = score_built_seed(
            _draws(_sparse(BUILT_VECTORS), 2), HALF_VERSES_BY_PSALM, pairs, genres, seed=1
        )

        assert sparse == score_built_seed(
            _draws(dense, None), HALF_VERSES_BY_PSALM, pairs, genres, seed=1
        )


class TestGenreApSharesOneSimilarityMatrix:
    """Every genre reads one psalm matrix, so a seed builds it once rather than once per genre."""

    def _vectors(self) -> dict[int, np.ndarray]:
        rng = np.random.default_rng(11)
        return {psalm: rng.normal(size=48).astype("<f4") for psalm in range(1, 13)}

    def _labels(self) -> dict[int, str]:
        names = ["Lament", "Praise", "Hymn", "Royal"]
        return {psalm: names[psalm % len(names)] for psalm in range(1, 13)}

    def test_it_matches_scoring_each_genre_against_its_own_matrix(self) -> None:
        """A shared matrix must not move a single value, since the arithmetic is unchanged."""
        vectors = self._vectors()
        pairs = build_genre_pairs(self._labels())
        genres = sorted(set(self._labels().values()))

        shared = genre_ap(vectors, pairs, genres)
        per_genre = {
            genre: evaluate_genre_discrimination(
                filter_pairs_by_genre(pairs, genre), vectors
            ).average_precision
            for genre in genres
        }

        assert shared == per_genre

    def test_the_matrix_is_built_once_no_matter_how_many_genres_are_scored(self) -> None:
        builds: list[int] = []

        def _counting_matrix(psalm_ids, psalm_vectors):  # type: ignore[no-untyped-def]
            builds.append(len(psalm_ids))
            return psalm_similarity_matrix(psalm_ids, psalm_vectors)

        genres = sorted(set(self._labels().values()))
        genre_ap(
            self._vectors(),
            build_genre_pairs(self._labels()),
            genres,
            similarity_matrix=_counting_matrix,
        )

        assert len(builds) == 1


class TestUnscoreableFamilyIsRefused:
    """A family no genre can be scored on must fail, since a NaN row reads as a finding."""

    def test_all_nan_real_scores_raise_instead_of_being_written(self) -> None:
        with pytest.raises(BenchmarkDataError, match="no genre could be scored"):
            require_scoreable({"Lament": float("nan"), "Praise": float("nan")}, "syn/x/1_2gram")

    def test_a_partly_scoreable_family_is_left_alone(self) -> None:
        """One untestable genre is left to the multiple-comparison correction, the run goes on."""
        scores = {"Lament": 0.3, "Praise": float("nan")}

        assert require_scoreable(scores, "syn/x/1_2gram") == scores

    def test_the_message_names_the_family_so_a_sweep_says_which_one_failed(self) -> None:
        with pytest.raises(BenchmarkDataError, match="syntactic/phrase/subphrase_rela/1_2gram"):
            require_scoreable({"Lament": float("nan")}, "syntactic/phrase/subphrase_rela/1_2gram")
