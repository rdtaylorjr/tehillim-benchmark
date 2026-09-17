import numpy as np
import scipy.sparse as sp

from library.centroid import psalm_centroids, sparse_psalm_centroids, uniform_weights


def test_mean_pools_a_psalms_half_verse_vectors() -> None:
    half_verses_by_psalm = {1: [100, 101]}
    node_vectors = {100: np.array([1.0, 0.0]), 101: np.array([3.0, 2.0])}

    result = psalm_centroids(uniform_weights(half_verses_by_psalm), node_vectors)

    assert np.allclose(result[1], [2.0, 1.0])


def test_keeps_one_centroid_per_psalm() -> None:
    half_verses_by_psalm = {1: [100], 2: [200, 201]}
    node_vectors = {
        100: np.array([1.0, 0.0]),
        200: np.array([0.0, 1.0]),
        201: np.array([0.0, 3.0]),
    }

    result = psalm_centroids(uniform_weights(half_verses_by_psalm), node_vectors)

    assert set(result) == {1, 2}
    assert np.allclose(result[1], [1.0, 0.0])
    assert np.allclose(result[2], [0.0, 2.0])


def test_skips_a_psalm_missing_entirely_from_node_vectors() -> None:
    half_verses_by_psalm = {1: [100], 2: [200]}
    node_vectors = {100: np.array([1.0, 0.0])}

    result = psalm_centroids(uniform_weights(half_verses_by_psalm), node_vectors)

    assert set(result) == {1}


def test_skips_a_psalm_with_a_partially_missing_node() -> None:
    """A psalm whose half-verse nodes are only partly covered is dropped, not partially pooled."""
    half_verses_by_psalm = {1: [100, 101]}
    node_vectors = {100: np.array([1.0, 0.0])}

    result = psalm_centroids(uniform_weights(half_verses_by_psalm), node_vectors)

    assert result == {}


def test_sparse_psalm_centroids_matches_the_dense_function_to_float_tolerance() -> None:
    """Proves sparse pooling via matmul gives the identical dense centroid for every psalm."""
    rng = np.random.default_rng(3)
    dim = 300
    node_ids = list(range(100, 130))
    dense_vectors = {}
    for n in node_ids:
        row = np.zeros(dim)
        n_nonzero = rng.integers(1, 6)
        idx = rng.choice(dim, size=n_nonzero, replace=False)
        row[idx] = rng.uniform(0.1, 5.0, size=n_nonzero)
        dense_vectors[n] = row
    half_verses_by_psalm = {
        1: [100, 101],
        2: [102],
        3: [103, 104, 105],
        4: [999],  # missing node: psalm 4 must be dropped by both paths
    }
    sparse_matrix = sp.csr_matrix(np.stack([dense_vectors[n] for n in node_ids]))

    dense_result = psalm_centroids(uniform_weights(half_verses_by_psalm), dense_vectors)
    psalm_ids, sparse_result = sparse_psalm_centroids(
        uniform_weights(half_verses_by_psalm), node_ids, sparse_matrix
    )

    assert set(psalm_ids) == set(dense_result)
    dense_arr = sparse_result.toarray()
    for i, psalm in enumerate(psalm_ids):
        np.testing.assert_allclose(dense_arr[i], dense_result[psalm], rtol=0, atol=1e-6)


def test_sparse_psalm_centroids_skips_a_psalm_with_a_partially_missing_node() -> None:
    node_ids = [100]
    matrix = sp.csr_matrix(np.array([[1.0, 0.0]]))
    half_verses_by_psalm = {1: [100, 101]}

    psalm_ids, result = sparse_psalm_centroids(
        uniform_weights(half_verses_by_psalm), node_ids, matrix
    )

    assert psalm_ids == []
    assert result.shape == (0, 2)


def test_unit_weights_give_exactly_the_plain_mean_in_the_rows_dtype() -> None:
    rng = np.random.default_rng(1)
    node_vectors = {n: rng.normal(size=8).astype("<f4") for n in range(100, 107)}
    weights = uniform_weights({1: list(range(100, 107))})

    result = psalm_centroids(weights, node_vectors)

    plain = np.mean([node_vectors[n] for n in range(100, 107)], axis=0)
    assert result[1].dtype == plain.dtype
    assert np.array_equal(result[1], plain)


def test_fractional_weights_pool_a_half_verse_by_its_share() -> None:
    node_vectors = {100: np.array([4.0, 0.0]), 101: np.array([0.0, 4.0])}

    result = psalm_centroids({1: {100: 1.0, 101: 0.25}}, node_vectors)

    assert np.allclose(result[1], [3.2, 0.8])


def test_sparse_fractional_weights_match_the_dense_pooling() -> None:
    node_ids = [100, 101, 102]
    matrix = sp.csr_matrix(np.array([[4.0, 0.0], [0.0, 4.0], [2.0, 2.0]]))
    weights = {"a": {100: 1.0, 101: 0.25}, "b": {101: 0.5, 102: 1.0}}

    items, centroids = sparse_psalm_centroids(weights, node_ids, matrix)

    dense = psalm_centroids(
        weights, {n: matrix[i].toarray().ravel() for i, n in enumerate(node_ids)}
    )
    assert items == ["a", "b"]
    assert np.allclose(centroids.toarray(), np.stack([dense["a"], dense["b"]]))
