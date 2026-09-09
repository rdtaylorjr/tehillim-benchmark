import numpy as np
import pytest
import scipy.sparse as sp

from library.embeddings import load_embeddings
from library.errors import InsufficientDataError
from parallelism.evaluate import (
    build_side_vectors,
    build_side_vectors_sparse,
    run_evaluation,
    run_evaluation_sparse,
    score_embedding_file,
)
from parallelism.pairs import RetrievalPair, filter_pairs_with_vectors


def _pair(
    pair_id: str, source_nodes: tuple[int, ...], target_nodes: tuple[int, ...], ptype: str
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


def test_build_side_vectors_mean_pools_a_spanning_member() -> None:
    pairs = [_pair("p1", (1, 2), (3,), "Synonymous")]
    node_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([3.0, 0.0]),
        3: np.array([0.0, 1.0]),
    }

    vectors = build_side_vectors(pairs, "source", node_vectors)

    assert np.allclose(vectors[0], [2.0, 0.0])


def test_build_side_vectors_matches_a_naive_per_pair_loop_with_ragged_group_sizes() -> None:
    rng = np.random.default_rng(41)
    dim = 4
    node_vectors = {n: rng.normal(size=dim) for n in range(1, 21)}
    pairs = [
        _pair("p1", (1, 2), (3,), "Synonymous"),
        _pair("p2", (4,), (5, 6, 7), "Antithetic"),
        _pair("p3", (8, 9, 10), (11,), "Synonymous"),
    ]

    naive = np.stack([np.mean([node_vectors[n] for n in p.source_nodes], axis=0) for p in pairs])
    vectorized = build_side_vectors(pairs, "source", node_vectors)

    assert np.allclose(vectorized, naive)


def test_build_side_vectors_raises_on_a_missing_node() -> None:
    pairs = [_pair("p1", (1,), (2,), "Synonymous")]
    node_vectors = {1: np.array([1.0, 0.0])}

    try:
        build_side_vectors(pairs, "target", node_vectors)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_run_evaluation_reports_perfect_retrieval_for_identical_vectors() -> None:
    pairs = [
        _pair("p1", (1,), (2,), "Synonymous"),
        _pair("p2", (3,), (4,), "Synonymous"),
        _pair("p3", (5,), (6,), "Antithetic"),
    ]
    node_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.0, 1.0]),
        4: np.array([0.0, 1.0]),
        5: np.array([1.0, 1.0]),
        6: np.array([1.0, 1.0]),
    }

    report = run_evaluation(pairs, node_vectors, n_permutations=100, rng=np.random.default_rng(0))

    assert report.n_pairs == 3
    assert report.separation.auc == 1.0
    assert report.mrr_forward == 1.0
    assert report.mrr_backward == 1.0
    assert report.recall_at_1_forward == 1.0
    assert {t.parallelism_type for t in report.by_type} == {"Synonymous", "Antithetic"}
    synonymous = next(t for t in report.by_type if t.parallelism_type == "Synonymous")
    assert synonymous.n_pairs == 2
    assert synonymous.separation.auc == 1.0
    assert synonymous.separation.n_positive == 2
    assert synonymous.mrr_forward == 1.0


def test_build_side_vectors_sparse_matches_the_dense_function_to_float_tolerance() -> None:
    """Sparse pooling via matmul reproduces the dense mean pool within float rounding."""
    rng = np.random.default_rng(7)
    dim = 200
    node_ids = list(range(1, 21))
    dense_vectors = {}
    for n in node_ids:
        row = np.zeros(dim)
        n_nonzero = rng.integers(1, 6)
        idx = rng.choice(dim, size=n_nonzero, replace=False)
        row[idx] = rng.uniform(0.1, 5.0, size=n_nonzero)
        dense_vectors[n] = row
    pairs = [
        _pair("p1", (1, 2), (3,), "Synonymous"),
        _pair("p2", (4,), (5, 6, 7), "Antithetic"),
        _pair("p3", (8, 9, 10), (11,), "Synonymous"),
    ]
    sparse_matrix = sp.csr_matrix(np.stack([dense_vectors[n] for n in node_ids]))

    dense_result = build_side_vectors(pairs, "source", dense_vectors)
    sparse_result = build_side_vectors_sparse(pairs, "source", node_ids, sparse_matrix)

    np.testing.assert_allclose(sparse_result.toarray(), dense_result, rtol=0, atol=1e-6)


def test_build_side_vectors_sparse_raises_on_a_missing_node() -> None:
    pairs = [_pair("p1", (1,), (2,), "Synonymous")]
    node_ids = [1]
    matrix = sp.csr_matrix(np.array([[1.0, 0.0]]))

    with pytest.raises(ValueError, match="missing"):
        build_side_vectors_sparse(pairs, "target", node_ids, matrix)


def test_run_evaluation_sparse_matches_run_evaluation_exactly() -> None:
    """Proves the sparse evaluation path reports identically to the dense one on the same data."""
    pairs = [
        _pair("p1", (1,), (2,), "Synonymous"),
        _pair("p2", (3,), (4,), "Synonymous"),
        _pair("p3", (5,), (6,), "Antithetic"),
    ]
    dense_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.0, 1.0]),
        4: np.array([0.0, 1.0]),
        5: np.array([1.0, 1.0]),
        6: np.array([1.0, 1.0]),
    }
    node_ids = sorted(dense_vectors)
    sparse_matrix = sp.csr_matrix(np.stack([dense_vectors[n] for n in node_ids]))

    dense_report = run_evaluation(
        pairs, dense_vectors, n_permutations=100, rng=np.random.default_rng(0)
    )
    sparse_report = run_evaluation_sparse(
        pairs, node_ids, sparse_matrix, n_permutations=100, rng=np.random.default_rng(0)
    )

    assert sparse_report == dense_report


def _pair_with(nodes: tuple[int, ...]) -> RetrievalPair:
    return RetrievalPair(
        pair_id="p",
        group_range="g",
        parallelism_type="Synonymous",
        signature="AB",
        source_nodes=nodes,
        target_nodes=(99,),
        source_indicator="A",
        target_indicator="B",
    )


def test_build_side_vectors_matches_an_unbuffered_scatter_add_bit_for_bit() -> None:
    """The segmented reduction replaced np.add.at, so it must agree to the last bit."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 4, 60)
    node_lists = []
    next_node = 0
    for length in lengths:
        node_lists.append(tuple(range(next_node, next_node + length)))
        next_node += length
    pairs = [_pair_with(nodes) for nodes in node_lists]
    node_vectors = {n: rng.normal(size=16).astype(np.float32) for n in range(next_node)}

    group_ids = np.repeat(np.arange(len(node_lists)), lengths)
    flat = np.stack([node_vectors[n] for nodes in node_lists for n in nodes])
    expected_sums = np.zeros((len(node_lists), flat.shape[1]))
    np.add.at(expected_sums, group_ids, flat)
    expected = expected_sums / np.bincount(group_ids)[:, None]

    result = build_side_vectors(pairs, "source", node_vectors)

    assert np.array_equal(result, expected)


def test_build_side_vectors_pools_a_multi_node_span_to_its_mean() -> None:
    pairs = [_pair_with((1, 2))]
    node_vectors = {
        1: np.array([1.0, 1.0], dtype=np.float32),
        2: np.array([3.0, -1.0], dtype=np.float32),
        99: np.array([1.0, 0.0], dtype=np.float32),
    }

    assert build_side_vectors(pairs, "source", node_vectors).tolist() == [[2.0, 0.0]]


def test_build_side_vectors_rejects_a_pair_with_an_empty_node_span() -> None:
    """Reduceat needs non-empty runs; an empty span would silently borrow the next span's row."""
    pairs = [_pair_with(()), _pair_with((1,))]
    node_vectors = {1: np.array([1.0, 0.0], dtype=np.float32), 99: np.array([1.0, 0.0], np.float32)}

    with pytest.raises(InsufficientDataError, match="at least one node"):
        build_side_vectors(pairs, "source", node_vectors)


def test_run_evaluation_sparse_agrees_with_the_dense_path_at_realistic_density() -> None:
    """The toy exact-equality case has 2 dimensions; at real width the paths agree to float32."""
    rng = np.random.default_rng(7)
    dim, n_pairs = 512, 40
    nodes = list(range(1, 2 * n_pairs + 1))
    dense_vectors = {n: rng.standard_normal(dim) for n in nodes}
    pairs = [
        _pair(
            f"p{i}",
            (nodes[2 * i],),
            (nodes[2 * i + 1],),
            "Synonymous" if i % 2 else "Antithetic",
        )
        for i in range(n_pairs)
    ]
    matrix = sp.csr_matrix(np.stack([dense_vectors[n] for n in nodes]))

    dense_report = run_evaluation(
        pairs, dense_vectors, n_permutations=100, rng=np.random.default_rng(0)
    )
    sparse_report = run_evaluation_sparse(
        pairs, nodes, matrix, n_permutations=100, rng=np.random.default_rng(0)
    )

    assert sparse_report.n_pairs == dense_report.n_pairs
    assert sparse_report.mrr_forward == pytest.approx(dense_report.mrr_forward, abs=1e-6)
    assert sparse_report.mrr_backward == pytest.approx(dense_report.mrr_backward, abs=1e-6)
    assert sparse_report.type_gap.observed_gap == pytest.approx(
        dense_report.type_gap.observed_gap, abs=1e-6
    )
    assert sparse_report.discrimination.statistic == pytest.approx(
        dense_report.discrimination.statistic, abs=1e-6
    )


def _write_dense_parquet(path, vectors):
    """Writes a dense embeddings parquet the loader accepts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    nodes = sorted(vectors)
    dim = len(vectors[nodes[0]])
    matrix = np.array([vectors[n] for n in nodes], dtype="<f4")
    table = pa.table(
        {
            "node_id": pa.array(nodes, type=pa.int32()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.ravel(), type=pa.float32()), dim
            ),
        }
    )
    pq.write_table(table, path)


def _write_sparse_parquet(path, vectors, dim):
    """Writes the same vectors in the sparse layout, with the dim metadata the loader reads."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    nodes = sorted(vectors)
    table = pa.table(
        {
            "node_id": pa.array(nodes, type=pa.int32()),
            "indices": pa.array(
                [np.flatnonzero(vectors[n]).astype("<i4").tolist() for n in nodes],
                type=pa.list_(pa.int32()),
            ),
            "values": pa.array(
                [vectors[n][np.flatnonzero(vectors[n])].astype("<f4").tolist() for n in nodes],
                type=pa.list_(pa.float32()),
            ),
        }
    )
    pq.write_table(table.replace_schema_metadata({"dim": str(dim), "sparse": "true"}), path)


def test_score_embedding_file_reads_a_sparse_file_without_densifying(tmp_path) -> None:
    """A sparse file must score through the sparse path rather than a dense materialization."""
    dim = 6
    vectors = {
        1: np.array([1.0, 0, 0, 0, 0, 0]),
        2: np.array([1.0, 0, 0, 0, 0, 0]),
        3: np.array([0, 1.0, 0, 0, 0, 0]),
        4: np.array([0, 1.0, 0, 0, 0, 0]),
    }
    pairs = [_pair("p1", (1,), (2,), "Synonymous"), _pair("p2", (3,), (4,), "Antithetic")]
    sparse_path = tmp_path / "s.parquet"
    _write_sparse_parquet(sparse_path, vectors, dim)

    used, report = score_embedding_file(
        sparse_path, pairs, n_permutations=50, rng=np.random.default_rng(0)
    )

    assert [p.pair_id for p in used] == ["p1", "p2"]
    assert report.n_pairs == 2


def test_score_embedding_file_gives_the_dense_path_the_untouched_dense_result(tmp_path) -> None:
    """A dense file must produce exactly what run_evaluation produced before the dispatch."""
    rng = np.random.default_rng(3)
    vectors = {n: rng.standard_normal(8) for n in range(1, 9)}
    pairs = [
        _pair(f"p{i}", (2 * i + 1,), (2 * i + 2,), "Synonymous" if i % 2 else "Antithetic")
        for i in range(4)
    ]
    dense_path = tmp_path / "d.parquet"
    _write_dense_parquet(dense_path, vectors)

    loaded = load_embeddings(dense_path)
    expected = run_evaluation(
        filter_pairs_with_vectors(pairs, loaded),
        loaded,
        n_permutations=50,
        rng=np.random.default_rng(0),
    )
    _, actual = score_embedding_file(
        dense_path, pairs, n_permutations=50, rng=np.random.default_rng(0)
    )

    assert actual == expected


def test_score_embedding_file_skips_pairs_whose_nodes_have_no_vector(tmp_path) -> None:
    """A pair referencing a node absent from the file is dropped, not raised on."""
    rng = np.random.default_rng(4)
    vectors = {n: rng.standard_normal(8) for n in range(1, 9)}
    pairs = [
        _pair(f"kept{i}", (2 * i + 1,), (2 * i + 2,), "Synonymous" if i % 2 else "Antithetic")
        for i in range(4)
    ]
    pairs.append(_pair("dropped", (1,), (99,), "Synonymous"))
    dense_path = tmp_path / "d.parquet"
    _write_dense_parquet(dense_path, vectors)

    used, _ = score_embedding_file(
        dense_path, pairs, n_permutations=50, rng=np.random.default_rng(0)
    )

    assert [p.pair_id for p in used] == ["kept0", "kept1", "kept2", "kept3"]


def test_a_model_missing_node_ids_is_skippable_rather_than_fatal() -> None:
    """A model whose file lacks nodes is degenerate for this benchmark, not a programming error."""
    from library.errors import BenchmarkDataError
    from library.scoring import skipping_unscorable
    from parallelism.evaluate import build_side_vectors

    pair = RetrievalPair(
        pair_id="p1",
        group_range="1",
        parallelism_type="Synonymous",
        signature="sig",
        source_nodes=(1, 2),
        target_nodes=(3,),
        source_indicator="a",
        target_indicator="b",
    )

    with pytest.raises(BenchmarkDataError):
        build_side_vectors([pair], "source", {1: np.ones(4, dtype=np.float32)})

    def score(_item: object) -> None:
        build_side_vectors([pair], "source", {1: np.ones(4, dtype=np.float32)})

    assert skipping_unscorable(score, label=lambda _item: "model")(object()) is None


def test_run_evaluation_rejects_a_corpus_with_a_single_retrieval_pair() -> None:
    """One pair leaves no other pair to rank against, so the background mean is undefined."""
    pairs = [_pair("p1", (1,), (2,), "Synonymous")]
    node_vectors = {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])}

    with pytest.raises(InsufficientDataError, match="at least two retrieval pairs"):
        run_evaluation(pairs, node_vectors, n_permutations=10, rng=np.random.default_rng(0))
