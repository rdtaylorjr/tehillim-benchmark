from pathlib import Path

from conftest import semantic_file

from genre.pairs import GenrePair
from genre.scripts.compare_models import compare_genre_models, score_model
from library.centroid import uniform_weights

# One half-verse node per psalm, so each psalm centroid is exactly that node's vector.
_HALF_VERSES_BY_PSALM = uniform_weights({"1": [1], "2": [2], "3": [3]})
_PAIRS = [
    GenrePair("1", "2", 1, 2, "Lament", "Lament", same_genre=True),
    GenrePair("1", "3", 1, 3, "Lament", "Praise", same_genre=False),
    GenrePair("2", "3", 2, 3, "Lament", "Praise", same_genre=False),
]
_BETTER = {1: [1.0, 0.0], 2: [0.9, 0.1], 3: [0.0, 1.0]}
_WORSE = {1: [1.0, 0.0], 2: [0.0, 1.0], 3: [1.0, 0.0]}


def test_reports_one_row_per_model(tmp_path: Path, write_embeddings_parquet) -> None:
    a = write_embeddings_parquet(semantic_file(tmp_path, "model_a", "v.parquet"), _BETTER)
    b = write_embeddings_parquet(semantic_file(tmp_path, "model_b", "v.parquet"), _WORSE)

    rows = compare_genre_models(_PAIRS, [a, b], _HALF_VERSES_BY_PSALM, max_workers=1)

    assert {row["model"] for row in rows} == {"model_a_consonantal", "model_b_consonantal"}


def test_sorts_rows_by_average_precision_descending(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    worse = write_embeddings_parquet(semantic_file(tmp_path, "worse", "v.parquet"), _WORSE)
    better = write_embeddings_parquet(semantic_file(tmp_path, "better", "v.parquet"), _BETTER)

    rows = compare_genre_models(_PAIRS, [worse, better], _HALF_VERSES_BY_PSALM, max_workers=1)

    assert [row["model"] for row in rows] == ["better_consonantal", "worse_consonantal"]


def test_score_model_pools_a_psalms_half_verse_vectors_into_one_centroid(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Psalm 1 spans two opposite half-verses pooling to [1, 0], matching psalm 2 exactly."""
    path = write_embeddings_parquet(
        semantic_file(tmp_path, "m", "v.parquet"),
        {1: [1.0, 1.0], 2: [1.0, -1.0], 3: [1.0, 0.0], 4: [0.0, 1.0], 5: [0.0, 1.0]},
    )
    half_verses_by_psalm = uniform_weights({"1": [1, 2], "2": [3], "3": [4], "4": [5]})
    pairs = [
        GenrePair("1", "2", 1, 2, "Lament", "Lament", same_genre=True),
        GenrePair("3", "4", 3, 4, "Praise", "Praise", same_genre=True),
        GenrePair("1", "3", 1, 3, "Lament", "Praise", same_genre=False),
        GenrePair("1", "4", 1, 4, "Lament", "Praise", same_genre=False),
        GenrePair("2", "3", 2, 3, "Lament", "Praise", same_genre=False),
        GenrePair("2", "4", 2, 4, "Lament", "Praise", same_genre=False),
    ]

    row = score_model(path, half_verses_by_psalm, pairs)

    assert row["model"] == "m_consonantal"
    assert row["n_same_genre"] == 2
    assert row["n_different_genre"] == 4
    # Pooling to [1, 0] makes psalm 1 identical to psalm 2 and orthogonal to Praise.
    assert row["separation_auc"] == 1.0
    assert row["average_precision"] == 1.0


def test_compare_genre_models_is_unchanged_by_running_across_worker_processes(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Reruns are byte-compared, so worker count must never move a single reported number."""
    a = write_embeddings_parquet(semantic_file(tmp_path, "a", "v.parquet"), _BETTER)
    b = write_embeddings_parquet(semantic_file(tmp_path, "b", "v.parquet"), _WORSE)

    sequential = compare_genre_models(_PAIRS, [a, b], _HALF_VERSES_BY_PSALM, max_workers=1)
    parallel = compare_genre_models(_PAIRS, [a, b], _HALF_VERSES_BY_PSALM, max_workers=2)

    assert sequential == parallel
