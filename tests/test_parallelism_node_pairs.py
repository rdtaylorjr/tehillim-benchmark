import numpy as np
import pytest
import scipy.sparse as sp

from library.errors import InsufficientDataError
from parallelism.node_pairs import (
    as_node_pairs,
    filter_node_pairs_with_vectors,
    pair_similarities,
    pair_similarities_sparse,
)


def test_filter_node_pairs_with_vectors_keeps_a_pair_whose_nodes_are_all_present() -> None:
    pairs = as_node_pairs([(1, 2), (3, 4)])
    node_vectors = {1: object(), 2: object(), 3: object(), 4: object()}

    assert filter_node_pairs_with_vectors(pairs, node_vectors) == pairs


def test_filter_node_pairs_with_vectors_drops_a_pair_missing_either_side() -> None:
    pairs = as_node_pairs([(1, 2), (3, 4)])
    node_vectors = {1: object(), 2: object(), 3: object()}  # node 4 excluded

    kept = filter_node_pairs_with_vectors(pairs, node_vectors)

    assert kept == as_node_pairs([(1, 2)])


def test_filter_node_pairs_with_vectors_handles_multi_node_spans() -> None:
    pairs = [((1, 2), (3,)), ((4,), (5, 6))]
    node_vectors = {1: object(), 2: object(), 3: object(), 4: object(), 5: object()}  # 6 excluded

    kept = filter_node_pairs_with_vectors(pairs, node_vectors)

    assert kept == [((1, 2), (3,))]


def test_as_node_pairs_wraps_each_side_as_a_one_tuple() -> None:
    assert as_node_pairs([(1, 2), (3, 4)]) == [((1,), (2,)), ((3,), (4,))]


def test_pair_similarities_scores_each_pair_against_its_own_target_only() -> None:
    pairs = as_node_pairs([(1, 2), (3, 4)])
    node_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([1.0, 0.0]),
        4: np.array([0.0, 1.0]),
    }

    assert pair_similarities(pairs, node_vectors).tolist() == [1.0, 0.0]


def test_pair_similarities_mean_pools_a_multi_node_side_before_comparing() -> None:
    """A two-half-verse span pools to a centroid that points exactly at the target vector."""
    pairs = [((1, 2), (3,))]
    node_vectors = {
        1: np.array([1.0, 1.0]),
        2: np.array([1.0, -1.0]),
        3: np.array([1.0, 0.0]),
    }

    assert pair_similarities(pairs, node_vectors).tolist() == [1.0]


class TestPairSimilaritiesSparse:
    def test_matches_the_dense_function_on_single_node_pairs(self) -> None:
        import scipy.sparse as sp

        from parallelism.node_pairs import pair_similarities_sparse

        node_ids = [1, 2, 3, 4]
        dense = {
            1: np.array([1.0, 0.0]),
            2: np.array([1.0, 1.0]),
            3: np.array([0.0, 2.0]),
            4: np.array([3.0, 1.0]),
        }
        matrix = sp.csr_matrix(np.stack([dense[n] for n in node_ids]))
        pairs = [((1,), (2,)), ((3,), (4,))]

        result = pair_similarities_sparse(pairs, node_ids, matrix)

        assert np.allclose(result, pair_similarities(pairs, dense), rtol=0, atol=1e-9)

    def test_mean_pools_a_multi_node_side_like_the_dense_function(self) -> None:
        import scipy.sparse as sp

        from parallelism.node_pairs import pair_similarities_sparse

        node_ids = [1, 2, 3, 4]
        dense = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
            3: np.array([1.0, 1.0]),
            4: np.array([2.0, 0.0]),
        }
        matrix = sp.csr_matrix(np.stack([dense[n] for n in node_ids]))
        pairs = [((1, 2), (3,)), ((3, 4), (1,))]

        result = pair_similarities_sparse(pairs, node_ids, matrix)

        assert np.allclose(result, pair_similarities(pairs, dense), rtol=0, atol=1e-9)

    def test_filter_accepts_a_plain_set_of_node_ids(self) -> None:
        pairs = [((1,), (2,)), ((1,), (99,))]

        assert filter_node_pairs_with_vectors(pairs, {1, 2}) == [((1,), (2,))]


def test_pair_similarities_rejects_an_empty_pair_list() -> None:
    """No pairs means no similarity to summarise, which a batch must report and skip over."""
    with pytest.raises(InsufficientDataError, match="at least one pair"):
        pair_similarities([], {1: np.array([1.0, 0.0])})


def test_pair_similarities_sparse_rejects_an_empty_pair_list() -> None:
    with pytest.raises(InsufficientDataError, match="at least one pair"):
        pair_similarities_sparse([], [1], sp.csr_matrix(np.array([[1.0, 0.0]])))
