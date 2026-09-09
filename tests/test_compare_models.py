from pathlib import Path

from parallelism.pairs import RetrievalPair
from parallelism.scripts.compare_models import compare_models, score_model


def _pair(
    pair_id: str,
    source_nodes: tuple[int, ...],
    target_nodes: tuple[int, ...],
    ptype: str = "Synonymous",
) -> RetrievalPair:
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


_ALIGNED = {
    1: [1.0, 0.0],
    2: [1.0, 0.0],
    3: [0.0, 1.0],
    4: [0.0, 1.0],
}
_CROSSED = {
    1: [1.0, 0.0],
    2: [0.0, 1.0],
    3: [0.0, 1.0],
    4: [1.0, 0.0],
}


def test_score_model_names_the_row_after_the_files_dataset_identifier(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    path = write_embeddings_parquet(tmp_path / "domain=x" / "model=good" / "v.parquet", _ALIGNED)

    pairs = [_pair("p1", (1,), (2,)), _pair("p2", (3,), (4,))]

    row = score_model(path, pairs, n_permutations=50, seed=0)

    assert row["model"] == "good"


def test_score_model_scores_aligned_pairs_at_separation_auc_one(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    path = write_embeddings_parquet(tmp_path / "domain=x" / "model=good" / "v.parquet", _ALIGNED)
    pairs = [_pair("p1", (1,), (2,)), _pair("p2", (3,), (4,))]

    row = score_model(path, pairs, n_permutations=50, seed=0)

    assert row["separation_auc"] == 1.0


def test_compare_models_ranks_by_separation_auc_descending(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    pairs = [_pair("p1", (1,), (2,)), _pair("p2", (3,), (4,))]
    bad = write_embeddings_parquet(tmp_path / "domain=x" / "model=bad" / "v.parquet", _CROSSED)
    good = write_embeddings_parquet(tmp_path / "domain=x" / "model=good" / "v.parquet", _ALIGNED)

    rows = compare_models(pairs, [bad, good], n_permutations=50, seed=0, max_workers=1)

    assert [r["model"] for r in rows] == ["good", "bad"]
    assert rows[0]["separation_auc"] == 1.0
    assert rows[1]["separation_auc"] == 0.0


def test_compare_models_is_unchanged_by_running_across_worker_processes(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Reruns are byte-compared, so worker count must never move a single reported number."""
    pairs = [_pair("p1", (1,), (2,)), _pair("p2", (3,), (4,))]
    bad = write_embeddings_parquet(tmp_path / "domain=x" / "model=bad" / "v.parquet", _CROSSED)
    good = write_embeddings_parquet(tmp_path / "domain=x" / "model=good" / "v.parquet", _ALIGNED)

    sequential = compare_models(pairs, [bad, good], n_permutations=50, seed=0, max_workers=1)
    parallel = compare_models(pairs, [bad, good], n_permutations=50, seed=0, max_workers=2)

    assert sequential == parallel


def test_compare_models_flattens_per_type_metrics(tmp_path: Path, write_embeddings_parquet) -> None:
    pairs = [_pair("p1", (1,), (2,)), _pair("p2", (3,), (4,), ptype="Antithetic")]
    path = write_embeddings_parquet(tmp_path / "domain=x" / "model=m" / "v.parquet", _ALIGNED)

    rows = compare_models(pairs, [path], n_permutations=50, seed=0, max_workers=1)

    assert rows[0]["n_pairs_Synonymous"] == 1
    assert rows[0]["n_pairs_Antithetic"] == 1
    assert rows[0]["mrr_forward_Synonymous"] == 1.0
    assert rows[0]["mrr_forward_Antithetic"] == 1.0
    assert "discrimination_rank_biserial_Synonymous" in rows[0]
    assert "discrimination_rank_biserial_Antithetic" in rows[0]
