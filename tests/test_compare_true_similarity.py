from pathlib import Path

from parallelism.pairs import RetrievalPair
from parallelism.scripts.compare_true_similarity import compare_true_similarity, score_model


def _pair(pair_id: str, source_nodes: tuple, target_nodes: tuple, ptype: str) -> RetrievalPair:
    return RetrievalPair(
        pair_id=pair_id,
        group_range="g",
        parallelism_type=ptype,
        signature="AB",
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        source_indicator="A",
        target_indicator="B",
    )


def test_compare_true_similarity_ranks_by_calibrated_effect_size_descending(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Equal raw similarity (1.0), but 'spread' has a more diverse background, so it ranks first."""
    pairs = [_pair("p1", (1,), (2,), "Synonymous")]
    background_ids = [1, 2, 5, 6, 7, 8]

    peaky = write_embeddings_parquet(
        tmp_path / "domain=x" / "model=peaky" / "v.parquet",
        {
            1: [1.0, 0.0],
            2: [1.0, 0.0],
            5: [0.999, 0.045],
            6: [0.998, -0.063],
            7: [0.999, 0.032],
            8: [0.997, 0.077],
        },
    )
    spread = write_embeddings_parquet(
        tmp_path / "domain=x" / "model=spread" / "v.parquet",
        {
            1: [1.0, 0.0],
            2: [1.0, 0.0],
            5: [0.0, 1.0],
            6: [-1.0, 0.0],
            7: [0.0, -1.0],
            8: [0.7071, 0.7071],
        },
    )

    rows = compare_true_similarity(pairs, [peaky, spread], background_ids, max_workers=1)

    assert [r["model"] for r in rows] == ["spread", "peaky"]
    assert rows[0]["mean_true_similarity"] == rows[1]["mean_true_similarity"] == 1.0
    assert rows[0]["calibrated_effect_size"] > rows[1]["calibrated_effect_size"]


def test_compare_true_similarity_flattens_per_type_effect_sizes(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    pairs = [
        _pair("p1", (1,), (2,), "Synonymous"),
        _pair("p2", (3,), (4,), "Antithetic"),
    ]
    background_ids = [1, 2, 3, 4, 5, 6]
    path = write_embeddings_parquet(
        tmp_path / "domain=x" / "model=m" / "v.parquet",
        {
            1: [1.0, 0.0],
            2: [1.0, 0.0],
            3: [1.0, 0.0],
            4: [1.0, 0.0],
            5: [0.0, 1.0],
            6: [-1.0, 0.0],
        },
    )

    rows = compare_true_similarity(pairs, [path], background_ids, max_workers=1)

    assert "calibrated_effect_size_Synonymous" in rows[0]
    assert "calibrated_effect_size_Antithetic" in rows[0]


def test_score_model_matches_a_parallel_run_exactly(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Reruns are byte-compared, so worker count must never move a single reported number."""
    pairs = [_pair("p1", (1,), (2,), "Synonymous")]
    background_ids = [1, 2, 5, 6]
    vectors = {1: [1.0, 0.0], 2: [1.0, 0.0], 5: [0.0, 1.0], 6: [-1.0, 0.0]}
    first = write_embeddings_parquet(tmp_path / "domain=x" / "model=a" / "v.parquet", vectors)
    second = write_embeddings_parquet(tmp_path / "domain=x" / "model=b" / "v.parquet", vectors)

    sequential = compare_true_similarity(pairs, [first, second], background_ids, max_workers=1)
    parallel = compare_true_similarity(pairs, [first, second], background_ids, max_workers=2)

    assert sequential == parallel


def test_score_model_row_is_named_after_the_files_dataset_identifier(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    pairs = [_pair("p1", (1,), (2,), "Synonymous")]
    vectors = {1: [1.0, 0.0], 2: [1.0, 0.0], 5: [0.0, 1.0], 6: [-1.0, 0.0]}
    path = write_embeddings_parquet(tmp_path / "domain=x" / "model=peaky" / "v.parquet", vectors)

    assert score_model(path, pairs, [1, 2, 5, 6])["model"] == "peaky"
