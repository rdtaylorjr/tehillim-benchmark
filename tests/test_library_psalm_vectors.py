from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from conftest import _write_embeddings_parquet

from library.psalm_vectors import is_sparse_embeddings, load_psalm_vectors


def _sparse(path: Path, vectors: dict[int, list[float]], dim: int) -> None:
    node_ids = sorted(vectors)
    table = pa.table(
        {
            "node_id": pa.array(node_ids, type=pa.int32()),
            "indices": pa.array(
                [[i for i, v in enumerate(vectors[n]) if v != 0] for n in node_ids],
                type=pa.list_(pa.int32()),
            ),
            "values": pa.array(
                [[v for v in vectors[n] if v != 0] for n in node_ids],
                type=pa.list_(pa.float32()),
            ),
        }
    )
    pq.write_table(table.replace_schema_metadata({"dim": str(dim), "sparse": "true"}), path)


VECTORS = {10: [1.0, 0.0, 2.0], 11: [3.0, 0.0, 0.0], 20: [0.0, 4.0, 0.0]}
HALF_VERSES = {1: [10, 11], 2: [20]}


def test_is_sparse_embeddings_reads_the_schema(tmp_path: Path) -> None:
    dense, sparse = tmp_path / "d.parquet", tmp_path / "s.parquet"
    _write_embeddings_parquet(dense, VECTORS)
    _sparse(sparse, VECTORS, dim=3)

    assert is_sparse_embeddings(dense) is False
    assert is_sparse_embeddings(sparse) is True


def test_load_psalm_vectors_pools_a_dense_file(tmp_path: Path) -> None:
    path = tmp_path / "d.parquet"
    _write_embeddings_parquet(path, VECTORS)

    centroids = load_psalm_vectors(path, HALF_VERSES)

    assert sorted(centroids) == [1, 2]
    assert np.allclose(centroids[1], [2.0, 0.0, 1.0])
    assert np.allclose(centroids[2], [0.0, 4.0, 0.0])


def test_load_psalm_vectors_pools_a_sparse_file_to_the_same_centroids(tmp_path: Path) -> None:
    dense, sparse = tmp_path / "d.parquet", tmp_path / "s.parquet"
    _write_embeddings_parquet(dense, VECTORS)
    _sparse(sparse, VECTORS, dim=3)

    from_dense = load_psalm_vectors(dense, HALF_VERSES)
    from_sparse = load_psalm_vectors(sparse, HALF_VERSES)

    assert sorted(from_dense) == sorted(from_sparse)
    for psalm in from_dense:
        assert np.allclose(from_dense[psalm], from_sparse[psalm])


def test_load_psalm_vectors_skips_a_psalm_missing_one_of_its_half_verses(tmp_path: Path) -> None:
    path = tmp_path / "s.parquet"
    _sparse(path, VECTORS, dim=3)

    centroids = load_psalm_vectors(path, {1: [10, 11], 3: [10, 999]})

    assert sorted(centroids) == [1]
