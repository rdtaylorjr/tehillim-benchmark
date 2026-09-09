from pathlib import Path

import numpy as np
from conftest import _write_embeddings_parquet as _write_parquet
from families.shuffle import Draws

from genre.pairs import build_genre_pairs
from genre.scripts.shuffle_order_control import (
    score_built_seed,
    score_genre_ap,
    shuffled_scores_by_genre,
)


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
