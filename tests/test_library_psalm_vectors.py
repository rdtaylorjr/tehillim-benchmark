from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from conftest import _write_embeddings_parquet

from library.psalm_vectors import draw_psalm_vectors, is_sparse_embeddings, load_psalm_vectors


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


def _dense_draw(vectors: dict[int, list[float]]) -> dict[int, np.ndarray]:
    return {node: np.array(values, dtype="<f4") for node, values in vectors.items()}


def _sparse_draw(vectors: dict[int, list[float]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        node: (
            np.array([i for i, value in enumerate(values) if value], dtype=np.int32),
            np.array([value for value in values if value], dtype="<f4"),
        )
        for node, values in vectors.items()
    }


def test_draw_psalm_vectors_pools_a_dense_draw_as_its_written_file_pools(tmp_path: Path) -> None:
    path = tmp_path / "d.parquet"
    _write_embeddings_parquet(path, VECTORS)

    built = draw_psalm_vectors(_dense_draw(VECTORS), None, HALF_VERSES)
    from_file = load_psalm_vectors(path, HALF_VERSES)

    assert sorted(built) == sorted(from_file)
    assert all(np.array_equal(built[psalm], from_file[psalm]) for psalm in from_file)


def test_draw_psalm_vectors_pools_a_sparse_draw_as_its_written_file_pools(tmp_path: Path) -> None:
    path = tmp_path / "s.parquet"
    _sparse(path, VECTORS, dim=3)

    built = draw_psalm_vectors(_sparse_draw(VECTORS), 3, HALF_VERSES)
    from_file = load_psalm_vectors(path, HALF_VERSES)

    assert sorted(built) == sorted(from_file)
    assert all(np.array_equal(built[psalm], from_file[psalm]) for psalm in from_file)


def test_draw_psalm_vectors_keeps_a_zero_norm_vector_the_reader_keeps(tmp_path: Path) -> None:
    """A written zero row comes back as an empty half-verse, so a built one pools the same way."""
    vectors = {**VECTORS, 12: [0.0, 0.0, 0.0]}
    path = tmp_path / "d.parquet"
    _write_embeddings_parquet(path, vectors)

    built = draw_psalm_vectors(_dense_draw(vectors), None, {1: [10, 11, 12], 2: [20]})
    from_file = load_psalm_vectors(path, {1: [10, 11, 12], 2: [20]})

    assert 1 in from_file
    assert sorted(built) == sorted(from_file)
    assert all(np.array_equal(built[psalm], from_file[psalm]) for psalm in from_file)


class TestAnEmptyHalfVerseIsAValueNotMissingData:
    """A half-verse with none of a feature is an observation, so its psalm keeps its place."""

    #: Psalm 1 has an empty second half-verse, psalm 2 has none, psalm 3 is empty throughout.
    NODE_VECTORS: ClassVar = {
        10: [2.0, 0.0],
        11: [0.0, 0.0],
        20: [1.0, 1.0],
        30: [0.0, 0.0],
        31: [0.0, 0.0],
    }
    BY_PSALM: ClassVar = {1: [10, 11], 2: [20], 3: [30, 31]}

    def test_a_psalm_with_an_empty_half_verse_is_still_scored(self, tmp_path: Path) -> None:
        """Dropping it deletes psalms on a length-related criterion, which biases the sample."""
        path = tmp_path / "e.parquet"
        _write_embeddings_parquet(path, self.NODE_VECTORS)

        centroids = load_psalm_vectors(path, self.BY_PSALM)

        assert 1 in centroids

    def test_the_empty_half_verse_counts_in_the_mean(self, tmp_path: Path) -> None:
        """Pooling over nonempty half-verses only would rescale the centroid by the empty count."""
        path = tmp_path / "e.parquet"
        _write_embeddings_parquet(path, self.NODE_VECTORS)

        centroids = load_psalm_vectors(path, self.BY_PSALM)

        np.testing.assert_allclose(centroids[1], [1.0, 0.0])

    def test_a_psalm_with_no_direction_at_all_is_excluded(self, tmp_path: Path) -> None:
        """Cosine is undefined against a zero centroid, the one exclusion the statistic forces."""
        path = tmp_path / "e.parquet"
        _write_embeddings_parquet(path, self.NODE_VECTORS)

        centroids = load_psalm_vectors(path, self.BY_PSALM)

        assert 3 not in centroids

    def test_a_psalm_with_no_empty_half_verse_is_unchanged(self, tmp_path: Path) -> None:
        """The estimator does not move, so only the psalms it is computed over change."""
        path = tmp_path / "e.parquet"
        _write_embeddings_parquet(path, self.NODE_VECTORS)

        centroids = load_psalm_vectors(path, self.BY_PSALM)

        np.testing.assert_allclose(centroids[2], [1.0, 1.0])

    def test_a_sparse_file_gives_the_same_centroids_as_a_dense_one(self, tmp_path: Path) -> None:
        """Storage format must not decide which psalms a genre score is computed over."""
        dense, sparse = tmp_path / "d.parquet", tmp_path / "s.parquet"
        _write_embeddings_parquet(dense, self.NODE_VECTORS)
        _sparse(sparse, self.NODE_VECTORS, dim=2)

        from_dense = load_psalm_vectors(dense, self.BY_PSALM)
        from_sparse = load_psalm_vectors(sparse, self.BY_PSALM)

        assert sorted(from_dense) == sorted(from_sparse)
        for psalm in from_dense:
            np.testing.assert_allclose(from_dense[psalm], from_sparse[psalm])

    def test_a_draw_is_pooled_the_same_way_as_the_real_embeddings(self, tmp_path: Path) -> None:
        """A null computed over a different psalm set than the real score controls for nothing."""
        path = tmp_path / "e.parquet"
        _write_embeddings_parquet(path, self.NODE_VECTORS)
        vectors = {node: np.array(v, dtype="<f4") for node, v in self.NODE_VECTORS.items()}

        real = load_psalm_vectors(path, self.BY_PSALM)
        drawn = draw_psalm_vectors(vectors, None, self.BY_PSALM)

        assert sorted(real) == sorted(drawn)
        for psalm in real:
            np.testing.assert_allclose(real[psalm], drawn[psalm])
