"""Every script loading embeddings must score a sparse file as it scores a dense one."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import _write_sparse_embeddings_parquet

from parallelism.node_pairs import as_node_pairs, retrieval_pairs_as_node_pairs
from parallelism.pairs import RetrievalPair

DIM = 12
N_NODES = 24


def _vectors() -> dict[int, np.ndarray]:
    """A reproducible dense vector per node, wide enough that pooling is non-trivial."""
    rng = np.random.default_rng(17)
    return {n: rng.standard_normal(DIM) for n in range(1, N_NODES + 1)}


def write_dense(path: Path, vectors: dict[int, np.ndarray]) -> None:
    """Writes the dense embeddings layout."""
    nodes = sorted(vectors)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array([vectors[n] for n in nodes], dtype="<f4")
    table = pa.table(
        {
            "node_id": pa.array(nodes, type=pa.int32()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.ravel(), type=pa.float32()), DIM
            ),
        }
    )
    pq.write_table(table, path)


@pytest.fixture
def both_files(tmp_path: Path) -> tuple[Path, Path]:
    """The same vectors written once densely and once sparsely."""
    vectors = _vectors()
    #: Both layouts sit under a Hive partition, the only shape a batch run ever names a model from.
    dense = tmp_path / "domain=semantic/model=dense/vectors.parquet"
    sparse = tmp_path / "domain=semantic/model=sparse/vectors.parquet"
    write_dense(dense, vectors)
    _write_sparse_embeddings_parquet(sparse, vectors)
    return dense, sparse


def _retrieval_pairs() -> list[RetrievalPair]:
    """Pairs spanning both parallelism types, some with multi-node sides."""
    types = ["Synonymous", "Antithetic", "Synthetic", "Emblematic", "Climactic"]
    return [
        RetrievalPair(
            pair_id=f"p{i}",
            group_range=f"g{i}",
            parallelism_type=types[i % len(types)],
            signature="AB",
            source_nodes=(2 * i + 1,) if i % 3 else (2 * i + 1, 2 * i + 2),
            target_nodes=(2 * i + 2,),
            source_indicator="A",
            target_indicator="B",
        )
        for i in range(10)
    ]


def _close(a: float, b: float) -> bool:
    """Sparse and dense agree to float32 precision, not bit for bit."""
    return a == pytest.approx(b, abs=1e-6)


def test_compare_baseline_score_model_agrees_between_layouts(both_files) -> None:
    from parallelism.scripts.compare_baseline import score_model

    dense, sparse = both_files
    true_pairs = as_node_pairs([(1, 2), (3, 4), (5, 6)])
    baseline_pairs = as_node_pairs([(7, 8), (9, 10), (11, 12), (13, 14)])
    background_nodes = list(range(1, N_NODES + 1))

    d = score_model(dense, true_pairs, baseline_pairs, background_nodes)
    s = score_model(sparse, true_pairs, baseline_pairs, background_nodes)

    assert set(d) == set(s)
    assert all(_close(s[k], d[k]) for k in d if isinstance(d[k], float))


def test_compare_true_similarity_score_model_agrees_between_layouts(both_files) -> None:
    from parallelism.scripts.compare_true_similarity import score_model

    dense, sparse = both_files
    pairs = _retrieval_pairs()
    background_nodes = list(range(1, N_NODES + 1))

    d = score_model(dense, pairs, background_nodes)
    s = score_model(sparse, pairs, background_nodes)

    assert set(d) == set(s)
    assert all(_close(s[k], d[k]) for k in d if isinstance(d[k], float))


def test_shuffle_order_control_auc_agrees_between_layouts(both_files) -> None:
    from parallelism.scripts.shuffle_order_control import score_separation_auc

    dense, sparse = both_files
    pairs = _retrieval_pairs()

    assert _close(score_separation_auc(sparse, pairs), score_separation_auc(dense, pairs))


def test_compute_bootstrap_cis_score_model_agrees_between_layouts(both_files) -> None:
    from parallelism.scripts.compute_bootstrap_cis import score_model

    dense, sparse = both_files
    scopes = {"overall": as_node_pairs([(1, 2), (3, 4), (5, 6)])}
    baseline_pairs = as_node_pairs([(7, 8), (9, 10), (11, 12), (13, 14)])
    node_to_psalm = {n: (n - 1) // 4 + 1 for n in range(1, N_NODES + 1)}
    background_nodes = list(range(1, N_NODES + 1))

    d = score_model(dense, scopes, baseline_pairs, background_nodes, node_to_psalm, 25, 0)
    s = score_model(sparse, scopes, baseline_pairs, background_nodes, node_to_psalm, 25, 0)

    assert len(d) == len(s) == 1
    assert all(_close(s[0][k], d[0][k]) for k in d[0] if isinstance(d[0][k], float))


def test_export_detail_score_model_agrees_between_layouts(both_files) -> None:
    from parallelism.scripts.export_detail import score_model

    dense, sparse = both_files
    pairs = _retrieval_pairs()
    baseline_raw = [(15, 16), (17, 18), (19, 20)]
    baseline_pairs = as_node_pairs(baseline_raw)
    background_nodes = list(range(1, N_NODES + 1))

    d_pair, d_base, d_scope = score_model(
        dense, pairs, baseline_pairs, baseline_raw, background_nodes
    )
    s_pair, s_base, s_scope = score_model(
        sparse, pairs, baseline_pairs, baseline_raw, background_nodes
    )

    assert (len(d_pair), len(d_base), len(d_scope)) == (len(s_pair), len(s_base), len(s_scope))
    for dense_rows, sparse_rows in ((d_pair, s_pair), (d_base, s_base), (d_scope, s_scope)):
        for dr, sr in zip(dense_rows, sparse_rows, strict=True):
            assert all(_close(sr[k], dr[k]) for k in dr if isinstance(dr[k], float))


def test_compute_profiles_score_model_agrees_between_layouts(both_files, tmp_path) -> None:
    from trajectory.scripts.compute_profiles import score_model

    dense, sparse = both_files
    by_psalm = {1: list(range(1, 13)), 2: list(range(13, 25))}
    dense_dir, sparse_dir = tmp_path / "d", tmp_path / "s"
    dense_dir.mkdir()
    sparse_dir.mkdir()

    d_count, d_rows = score_model(dense, by_psalm, dense_dir)
    s_count, s_rows = score_model(sparse, by_psalm, sparse_dir)

    assert d_count == s_count
    for dr, sr in zip(d_rows, s_rows, strict=True):
        assert all(_close(sr[k], dr[k]) for k in dr if isinstance(dr[k], float))


def test_the_retrieval_pair_helper_covers_both_side_shapes() -> None:
    """The fixture must exercise single-node and multi-node sides, or pooling goes untested."""
    node_pairs = retrieval_pairs_as_node_pairs(_retrieval_pairs())

    assert any(len(source) == 1 for source, _ in node_pairs)
    assert any(len(source) > 1 for source, _ in node_pairs)
