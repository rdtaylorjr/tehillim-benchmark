from pathlib import Path

from conftest import _write_embeddings_parquet as _write_parquet

from genre.pairs import build_genre_pairs
from genre.scripts.shuffle_order_control import score_genre_ap, shuffled_scores_by_genre


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
    def test_transposes_per_file_scores_into_one_array_per_genre(self) -> None:
        scores_by_file = [{"A": 0.1, "B": 0.9}, {"A": 0.2, "B": 0.8}]

        result = shuffled_scores_by_genre(scores_by_file, ["A", "B"])

        assert result["A"].tolist() == [0.1, 0.2]
        assert result["B"].tolist() == [0.9, 0.8]

    def test_preserves_file_order_so_a_parallel_run_matches_a_sequential_one(self) -> None:
        """Shuffle scores form a null distribution, so their order must not depend on scheduling."""
        scores_by_file = [{"A": float(i)} for i in range(50)]

        result = shuffled_scores_by_genre(scores_by_file, ["A"])

        assert result["A"].tolist() == [float(i) for i in range(50)]

    def test_returns_an_empty_array_for_a_genre_with_no_shuffles(self) -> None:
        result = shuffled_scores_by_genre([], ["A"])

        assert result["A"].tolist() == []
