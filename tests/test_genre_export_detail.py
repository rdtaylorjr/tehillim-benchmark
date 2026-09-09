from functools import partial
from pathlib import Path

import numpy as np
import pytest

from genre.pairs import GenrePair, build_genre_pairs
from genre.scripts.export_detail import build_pair_detail_rows, build_summary_rows, score_model
from library.calibration import BackgroundStats
from library.errors import BenchmarkDataError
from library.scoring import skipping_unscorable


def _pairs_and_vectors() -> tuple[list[GenrePair], dict[int, np.ndarray]]:
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
    ]
    psalm_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.9, 0.1]),
        3: np.array([0.0, 1.0]),
    }
    return pairs, psalm_vectors


def test_build_pair_detail_rows_has_one_row_per_usable_pair() -> None:
    pairs, psalm_vectors = _pairs_and_vectors()
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=3)

    rows = build_pair_detail_rows("model_a", pairs, psalm_vectors, background)

    assert len(rows) == 3
    first = next(r for r in rows if r["psalm_a"] == 1 and r["psalm_b"] == 2)
    assert first["model"] == "model_a"
    assert first["same_genre"] is True
    assert first["raw_similarity"] == pytest.approx(0.99388373, abs=1e-6)
    assert "genre_a" not in first
    assert "genre_b" not in first


def test_build_pair_detail_rows_skips_a_pair_with_a_missing_vector() -> None:
    pairs, psalm_vectors = _pairs_and_vectors()
    del psalm_vectors[3]
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=2)

    rows = build_pair_detail_rows("model_a", pairs, psalm_vectors, background)

    assert len(rows) == 1
    assert rows[0]["psalm_a"] == 1
    assert rows[0]["psalm_b"] == 2


def test_build_summary_rows_has_one_row_summarizing_the_model() -> None:
    pairs, psalm_vectors = _pairs_and_vectors()
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=3)

    rows = build_summary_rows("model_a", pairs, psalm_vectors, background)

    assert len(rows) == 1
    assert rows[0]["model"] == "model_a"
    assert rows[0]["n_same_genre"] == 1
    assert rows[0]["n_different_genre"] == 2
    assert rows[0]["gap"] == pytest.approx(
        rows[0]["same_genre_effect_size"] - rows[0]["different_genre_effect_size"]
    )


def test_score_model_returns_pair_and_summary_rows_for_one_file(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=mine" / "v.parquet",
        {1: [1.0, 0.0], 2: [0.9, 0.1], 3: [0.0, 1.0], 4: [0.1, 0.9]},
    )
    pairs = build_genre_pairs({1: "A", 2: "A", 3: "B", 4: "B"})

    pair_rows, summary_rows = score_model(path, {p: [p] for p in (1, 2, 3, 4)}, pairs)

    assert len(pair_rows) == len(pairs)
    assert {row["model"] for row in pair_rows} == {"mine"}
    assert len(summary_rows) == 1
    assert summary_rows[0]["model"] == "mine"


def test_score_model_raises_and_the_shared_policy_skips_it(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Identical psalm vectors give a zero-variance background, which cannot be calibrated."""
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=flat" / "v.parquet",
        {1: [1.0, 0.0], 2: [1.0, 0.0], 3: [1.0, 0.0], 4: [1.0, 0.0]},
    )
    pairs = build_genre_pairs({1: "A", 2: "A", 3: "B", 4: "B"})

    score = partial(score_model, half_verses_by_psalm={p: [p] for p in (1, 2, 3, 4)}, pairs=pairs)

    with pytest.raises(BenchmarkDataError):
        score(path)
    assert skipping_unscorable(score)(path) is None
