from pathlib import Path

from parallelism.node_pairs import as_node_pairs
from parallelism.scripts.compute_bootstrap_cis import score_model

# Four psalms, one true and one baseline pair each, so leave-one-psalm-out still leaves three.
_NODE_TO_PSALM_SPREAD = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    13: 3,
    14: 3,
    15: 4,
    16: 4,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    17: 3,
    18: 3,
    19: 4,
    20: 4,
    9: 1,
    10: 2,
    11: 3,
    12: 4,
}
_VECTORS_SPREAD = {
    1: [1.0, 0.0],
    2: [1.0, 0.0],
    3: [1.0, 0.0],
    4: [0.98, 0.2],
    13: [1.0, 0.0],
    14: [0.99, 0.1],
    15: [1.0, 0.0],
    16: [0.97, 0.24],
    5: [1.0, 0.0],
    6: [0.0, 1.0],
    7: [1.0, 0.0],
    8: [0.1, 0.99],
    17: [1.0, 0.0],
    18: [0.2, 0.98],
    19: [1.0, 0.0],
    20: [0.3, 0.95],
    9: [0.7, 0.71],
    10: [-1.0, 0.0],
    11: [0.0, 1.0],
    12: [0.3, 0.95],
}


def test_score_model_returns_one_row_per_scope(tmp_path: Path, write_embeddings_parquet) -> None:
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=mine" / "v.parquet", _VECTORS_SPREAD
    )
    true_pairs = as_node_pairs([(1, 2), (3, 4), (13, 14), (15, 16)])
    scopes = {"overall": true_pairs, "Synonymous": true_pairs}

    rows = score_model(
        path,
        scopes,
        as_node_pairs([(5, 6), (7, 8), (17, 18), (19, 20)]),
        [9, 10, 11, 12],
        _NODE_TO_PSALM_SPREAD,
        n_resamples=20,
        seed=0,
    )

    assert [row["scope"] for row in rows] == ["overall", "Synonymous"]
    assert {row["model"] for row in rows} == {"mine"}


def test_score_model_scopes_share_one_seeded_generator_so_reruns_repeat(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Every scope of a model draws from one seeded stream, so a rerun reproduces it exactly."""
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=mine" / "v.parquet", _VECTORS_SPREAD
    )
    scopes = {"overall": as_node_pairs([(1, 2), (3, 4), (13, 14), (15, 16)])}
    args = (
        path,
        scopes,
        as_node_pairs([(5, 6), (7, 8), (17, 18), (19, 20)]),
        [9, 10, 11, 12],
        _NODE_TO_PSALM_SPREAD,
    )

    first = score_model(*args, n_resamples=20, seed=0)
    second = score_model(*args, n_resamples=20, seed=0)

    assert first == second


def test_score_model_passes_over_a_scope_the_corpus_has_no_pairs_for(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """A parallelism type nobody annotated is an absent scope, not a model that failed to score."""
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=mine" / "v.parquet", _VECTORS_SPREAD
    )
    true_pairs = as_node_pairs([(1, 2), (3, 4), (13, 14), (15, 16)])
    scopes = {"overall": true_pairs, "Antithetic": as_node_pairs([])}

    rows = score_model(
        path,
        scopes,
        as_node_pairs([(5, 6), (7, 8), (17, 18), (19, 20)]),
        [9, 10, 11, 12],
        _NODE_TO_PSALM_SPREAD,
        n_resamples=20,
        seed=0,
    )

    assert [row["scope"] for row in rows] == ["overall"]
