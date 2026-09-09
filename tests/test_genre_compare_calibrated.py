from functools import partial
from pathlib import Path

import numpy as np
import pytest

from genre.calibrated import compare_genre_calibrated
from genre.pairs import GenrePair, build_genre_pairs
from genre.scripts.compare_calibrated import score_model
from library.calibration import BackgroundStats
from library.errors import BenchmarkDataError
from library.scoring import skipping_unscorable


def test_reports_higher_effect_size_for_same_genre_when_psalms_are_closer() -> None:
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(3, 4, "Praise", "Praise", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(1, 4, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
        GenrePair(2, 4, "Lament", "Praise", same_genre=False),
    ]
    psalm_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.99, 0.14107]),
        3: np.array([0.0, 1.0]),
        4: np.array([0.14107, 0.99]),
    }
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=4)

    result = compare_genre_calibrated(pairs, psalm_vectors, background)

    assert result.n_same_genre == 2
    assert result.n_different_genre == 4
    assert result.mean_same_genre_similarity > result.mean_different_genre_similarity
    assert result.same_genre_effect_size > result.different_genre_effect_size
    assert result.separation_auc == 1.0
    assert result.average_precision == 1.0
    assert result.prevalence == pytest.approx(1 / 3)


def test_skips_a_pair_whose_psalm_has_no_vector() -> None:
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(1, 4, "Lament", "Praise", same_genre=False),
    ]
    psalm_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.9, 0.1]),
        3: np.array([0.0, 1.0]),
        # psalm 4 has no vector: the pair touching it must be excluded.
    }
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=3)

    result = compare_genre_calibrated(pairs, psalm_vectors, background)

    assert result.n_same_genre == 1
    assert result.n_different_genre == 1


def test_average_precision_at_chance_level_equals_prevalence() -> None:
    """With zero discrimination, AP equals the same-genre prevalence, not 0.5 (MTEB convention)."""
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(1, 4, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
    ]
    psalm_vectors = {n: np.array([1.0, 0.0]) for n in range(1, 5)}
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=4)

    result = compare_genre_calibrated(pairs, psalm_vectors, background)

    assert result.prevalence == pytest.approx(1 / 4)
    assert result.average_precision == pytest.approx(1 / 4)


def test_score_model_names_the_row_after_the_files_dataset_identifier(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=mine" / "v.parquet",
        {1: [1.0, 0.0], 2: [0.9, 0.1], 3: [0.0, 1.0], 4: [0.1, 0.9]},
    )
    pairs = build_genre_pairs({1: "A", 2: "A", 3: "B", 4: "B"})

    row = score_model(path, {1: [1], 2: [2], 3: [3], 4: [4]}, pairs)

    assert row is not None
    assert row["model"] == "mine"
    assert row["gap"] == row["same_genre_effect_size"] - row["different_genre_effect_size"]


def test_score_model_raises_when_the_background_has_no_variance(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Identical psalm vectors give a zero-variance background, which cannot be calibrated."""
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=flat" / "v.parquet",
        {1: [1.0, 0.0], 2: [1.0, 0.0], 3: [1.0, 0.0], 4: [1.0, 0.0]},
    )
    pairs = build_genre_pairs({1: "A", 2: "A", 3: "B", 4: "B"})

    score = partial(score_model, half_verses_by_psalm={1: [1], 2: [2], 3: [3], 4: [4]}, pairs=pairs)

    with pytest.raises(BenchmarkDataError):
        score(path)
    assert skipping_unscorable(score)(path) is None


def test_the_calibrated_row_is_built_in_one_place() -> None:
    """Two producers wrote this ten-field row, including the derived gap, independently."""
    from genre.calibrated import GenreCalibratedComparison, genre_calibrated_row

    result = GenreCalibratedComparison(
        n_same_genre=3,
        n_different_genre=5,
        prevalence=0.375,
        mean_same_genre_similarity=0.8,
        mean_different_genre_similarity=0.4,
        same_genre_effect_size=1.5,
        different_genre_effect_size=0.5,
        average_precision=0.6,
        separation_auc=0.7,
        separation_p=0.02,
    )

    row = genre_calibrated_row("m", result)

    assert row["model"] == "m"
    assert row["gap"] == pytest.approx(1.0)
    assert set(row) == {
        "model",
        "n_same_genre",
        "n_different_genre",
        "prevalence",
        "average_precision",
        "same_genre_effect_size",
        "different_genre_effect_size",
        "gap",
        "separation_auc",
        "separation_p",
    }
