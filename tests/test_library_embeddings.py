from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import _write_embeddings_parquet as _write_parquet

from library.embeddings import (
    dataset_identifier,
    is_sparse_embeddings,
    load_embeddings,
    load_sparse_embeddings,
    split_model_name,
)
from library.errors import BenchmarkDataError


def test_load_embeddings_reads_a_real_parquet_file(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.parquet"
    _write_parquet(path, {7: [4.0, 5.0], 8: [1.0, -2.0]})

    vectors = load_embeddings(path)

    assert set(vectors) == {7, 8}
    np.testing.assert_array_equal(vectors[7], np.array([4.0, 5.0], dtype="<f4"))
    np.testing.assert_array_equal(vectors[8], np.array([1.0, -2.0], dtype="<f4"))


def test_load_embeddings_returns_float32_vectors(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.parquet"
    _write_parquet(path, {1: [1.0, 2.0, 3.0]})

    vectors = load_embeddings(path)

    assert vectors[1].dtype == np.dtype("<f4")


def test_load_embeddings_excludes_zero_norm_vectors(tmp_path: Path) -> None:
    """A zero vector has no cosine direction, undefined for every downstream similarity metric."""
    path = tmp_path / "embeddings.parquet"
    _write_parquet(path, {7: [4.0, 5.0], 8: [0.0, 0.0]})

    vectors = load_embeddings(path)

    assert set(vectors) == {7}


def test_load_embeddings_matches_a_naive_per_row_to_pylist_conversion(tmp_path: Path) -> None:
    """Proves the batched Arrow-column read is lossless against the original per-row path."""
    rng = np.random.default_rng(0)
    n_rows, dim = 200, 5354
    node_ids = list(range(1000, 1000 + n_rows))
    matrix = rng.normal(size=(n_rows, dim)).astype("<f4")
    matrix[3] = 0.0  # one zero-norm row, must be excluded by both paths
    path = tmp_path / "embeddings.parquet"
    _write_parquet(path, {node: matrix[i].tolist() for i, node in enumerate(node_ids)})

    table = pq.read_table(path, columns=["node_id", "vector"])
    naive_node_ids = table["node_id"].to_pylist()
    naive_vectors_raw = table["vector"].to_pylist()
    naive = {
        node: np.asarray(vector, dtype="<f4")
        for node, vector in zip(naive_node_ids, naive_vectors_raw, strict=True)
        if np.any(vector)
    }

    vectorized = load_embeddings(path)

    assert set(vectorized) == set(naive)
    for node, expected in naive.items():
        np.testing.assert_array_equal(vectorized[node], expected)


def _write_sparse_parquet(
    path: Path, sparse_vectors: dict[int, tuple[list[int], list[float]]], dim: int
) -> None:
    node_ids = sorted(sparse_vectors)
    table = pa.table(
        {
            "node_id": pa.array(node_ids, type=pa.int32()),
            "indices": pa.array(
                [sparse_vectors[n][0] for n in node_ids], type=pa.list_(pa.int32())
            ),
            "values": pa.array(
                [sparse_vectors[n][1] for n in node_ids], type=pa.list_(pa.float32())
            ),
        }
    )
    table = table.replace_schema_metadata({"dim": str(dim), "sparse": "true"})
    pq.write_table(table, path)


def test_load_sparse_embeddings_reads_a_real_sparse_parquet_file(tmp_path: Path) -> None:
    path = tmp_path / "sparse.parquet"
    _write_sparse_parquet(path, {7: ([2, 5000], [1.0, 0.5]), 8: ([9], [2.0])}, dim=74088)

    node_ids, matrix = load_sparse_embeddings(path)

    assert node_ids == [7, 8]
    assert matrix.shape == (2, 74088)
    dense = matrix.toarray()
    assert dense[0, 2] == 1.0
    assert dense[0, 5000] == 0.5
    assert dense[1, 9] == 2.0
    assert np.count_nonzero(dense) == 3


def test_load_sparse_embeddings_returns_float32_values(tmp_path: Path) -> None:
    path = tmp_path / "sparse.parquet"
    _write_sparse_parquet(path, {1: ([0], [1.0])}, dim=10)

    _, matrix = load_sparse_embeddings(path)

    assert matrix.dtype == np.dtype("<f4")


def test_load_sparse_embeddings_excludes_a_row_with_no_nonzero_entries(tmp_path: Path) -> None:
    path = tmp_path / "sparse.parquet"
    _write_sparse_parquet(path, {7: ([2], [1.0]), 8: ([], [])}, dim=10)

    node_ids, matrix = load_sparse_embeddings(path)

    assert node_ids == [7]
    assert matrix.shape == (1, 10)


def test_load_sparse_embeddings_matches_a_reconstructed_dense_matrix(tmp_path: Path) -> None:
    """Proves the CSR construction is lossless against manual dense reconstruction."""
    rng = np.random.default_rng(0)
    dim = 500
    sparse_vectors = {}
    dense_expected = {}
    for node in range(100, 110):
        n_nonzero = rng.integers(1, 6)
        indices = sorted(rng.choice(dim, size=n_nonzero, replace=False).tolist())
        values = rng.uniform(0.1, 5.0, size=n_nonzero).tolist()
        sparse_vectors[node] = (indices, values)
        dense_row = np.zeros(dim, dtype="<f4")
        dense_row[indices] = values
        dense_expected[node] = dense_row

    path = tmp_path / "sparse.parquet"
    _write_sparse_parquet(path, sparse_vectors, dim=dim)

    node_ids, matrix = load_sparse_embeddings(path)

    for i, node in enumerate(node_ids):
        np.testing.assert_allclose(
            matrix[i].toarray().ravel(), dense_expected[node], rtol=0, atol=1e-6
        )


def test_dataset_identifier_reads_model_and_variation_from_the_hive_path() -> None:
    path = Path("data/domain=semantic/model=bge_m3/text=vocalized/part-0.parquet")

    assert dataset_identifier(path) == "bge_m3_vocalized"


def test_dataset_identifier_handles_a_two_level_lexical_path() -> None:
    path = Path("data/domain=lexical/unit=homograph/construction=binary/part-0.parquet")

    assert dataset_identifier(path) == "homograph_binary"


def test_dataset_identifier_handles_a_three_level_path_with_an_extra_text_tier() -> None:
    path = Path("data/domain=lexical/unit=word/text=consonantal/construction=binary/part-0.parquet")

    assert dataset_identifier(path) == "word_consonantal_binary"


def test_split_model_name_extracts_base_and_variant() -> None:
    assert split_model_name("semantic_gemini_embedding_2_cantillation") == (
        "gemini_embedding_2",
        "cantillation",
    )
    assert split_model_name("semantic_bge_m3_vocalized") == ("bge_m3", "vocalized")


def test_split_model_name_falls_back_to_unknown_variant() -> None:
    assert split_model_name("semantic_something_odd") == ("something_odd", "unknown")


def test_split_model_name_extracts_a_variant_embedded_in_the_middle() -> None:
    # word_consonantal_binary: unit=word, text=consonantal, construction=binary, not a suffix.
    assert split_model_name("word_consonantal_binary") == ("word_binary", "consonantal")


def test_split_model_name_still_prefers_a_trailing_suffix_variant() -> None:
    assert split_model_name("bge_m3_vocalized") == ("bge_m3", "vocalized")


def test_is_sparse_embeddings_detects_a_sparse_file(tmp_path: Path) -> None:
    path = tmp_path / "sparse.parquet"
    _write_sparse_parquet(path, {7: ([2], [1.0])}, dim=100)

    assert is_sparse_embeddings(path) is True


def test_is_sparse_embeddings_detects_a_dense_file(tmp_path: Path) -> None:
    path = tmp_path / "dense.parquet"
    _write_parquet(path, {7: [1.0, 2.0]})

    assert is_sparse_embeddings(path) is False


def test_load_embeddings_reads_a_sparse_file_transparently(tmp_path: Path) -> None:
    path = tmp_path / "sparse.parquet"
    _write_sparse_parquet(path, {7: ([1, 3], [1.0, 0.5]), 8: ([0], [2.0])}, dim=5)

    vectors = load_embeddings(path)

    assert sorted(vectors) == [7, 8]
    assert np.array_equal(vectors[7], np.array([0.0, 1.0, 0.0, 0.5, 0.0], dtype="<f4"))
    assert np.array_equal(vectors[8], np.array([2.0, 0.0, 0.0, 0.0, 0.0], dtype="<f4"))


def test_load_embeddings_gives_the_same_result_for_sparse_and_dense_forms(tmp_path: Path) -> None:
    dense_path, sparse_path = tmp_path / "d.parquet", tmp_path / "s.parquet"
    _write_parquet(dense_path, {7: [0.0, 1.0, 0.0, 0.5, 0.0], 8: [2.0, 0.0, 0.0, 0.0, 0.0]})
    _write_sparse_parquet(sparse_path, {7: ([1, 3], [1.0, 0.5]), 8: ([0], [2.0])}, dim=5)

    dense, sparse = load_embeddings(dense_path), load_embeddings(sparse_path)

    assert sorted(dense) == sorted(sparse)
    assert all(np.array_equal(dense[n], sparse[n]) for n in dense)


def test_load_embeddings_excludes_zero_norm_rows_of_a_sparse_file(tmp_path: Path) -> None:
    path = tmp_path / "sparse.parquet"
    _write_sparse_parquet(path, {7: ([1], [1.0]), 8: ([], [])}, dim=4)

    assert sorted(load_embeddings(path)) == [7]


def test_load_embeddings_returns_float32_for_a_sparse_file(tmp_path: Path) -> None:
    path = tmp_path / "sparse.parquet"
    _write_sparse_parquet(path, {7: ([1], [1.0])}, dim=4)

    assert load_embeddings(path)[7].dtype == np.dtype("<f4")


def test_dataset_identifier_rejects_a_file_that_carries_no_hive_partition() -> None:
    with pytest.raises(BenchmarkDataError, match="no Hive partition"):
        dataset_identifier(Path("/data/model_a.parquet"))


def test_dataset_identifier_rejects_a_file_sitting_directly_in_the_domain_root() -> None:
    with pytest.raises(BenchmarkDataError, match="no Hive partition"):
        dataset_identifier(Path("/data/domain=semantic/model_a.parquet"))
