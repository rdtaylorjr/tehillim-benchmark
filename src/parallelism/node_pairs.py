"""The node-span pair type shared by the parallelism benchmark, and its pooled similarity."""

from collections.abc import Container

import numpy as np
import scipy.sparse as sp

from library.errors import InsufficientDataError
from library.retrieval_metrics import paired_cosine_similarity, sparse_paired_cosine_similarity
from parallelism.pairs import RetrievalPair

NodePairs = list[tuple[tuple[int, ...], tuple[int, ...]]]


def as_node_pairs(single_node_pairs: list[tuple[int, int]]) -> NodePairs:
    """Wraps plain (int, int) pairs, such as adjacent half-verses, into the NodePairs shape."""
    return [((a,), (b,)) for a, b in single_node_pairs]


def retrieval_pairs_as_node_pairs(pairs: list[RetrievalPair]) -> NodePairs:
    """Drops a retrieval pair's annotation down to just its two node spans."""
    return [(pair.source_nodes, pair.target_nodes) for pair in pairs]


def filter_node_pairs_with_vectors(pairs: NodePairs, node_vectors: Container[int]) -> NodePairs:
    """NodePairs whose source and target nodes are all present in node_vectors."""
    return [
        (source, target)
        for source, target in pairs
        if all(n in node_vectors for n in source + target)
    ]


def _pool_side(
    node_tuples: list[tuple[int, ...]], node_vectors: dict[int, np.ndarray]
) -> np.ndarray:
    """Stacks one vector per tuple: direct lookup when single-node, mean pool otherwise."""
    if all(len(nodes) == 1 for nodes in node_tuples):
        return np.stack([node_vectors[nodes[0]] for nodes in node_tuples])
    return np.stack([np.mean([node_vectors[n] for n in nodes], axis=0) for nodes in node_tuples])


def pair_similarities(pairs: NodePairs, node_vectors: dict[int, np.ndarray]) -> np.ndarray:
    """Row-wise cosine similarity, mean-pooling any side that spans more than one node."""
    if not pairs:
        raise InsufficientDataError("pair similarity needs at least one pair")
    source_vecs = _pool_side([source for source, _ in pairs], node_vectors)
    target_vecs = _pool_side([target for _, target in pairs], node_vectors)
    return paired_cosine_similarity(source_vecs, target_vecs)


def _pool_side_sparse(
    node_tuples: list[tuple[int, ...]], node_ids: list[int], node_vectors: sp.csr_matrix
) -> sp.csr_matrix:
    """Sparse analogue of _pool_side: one mean-pooled row per tuple, built by a pooling matmul."""
    node_index = {n: i for i, n in enumerate(node_ids)}
    lengths = np.fromiter(
        (len(nodes) for nodes in node_tuples), dtype=np.int64, count=len(node_tuples)
    )
    rows = np.repeat(np.arange(len(node_tuples)), lengths)
    cols = np.fromiter(
        (node_index[n] for nodes in node_tuples for n in nodes),
        dtype=np.int64,
        count=int(lengths.sum()),
    )
    weights = 1.0 / np.repeat(lengths, lengths)
    pooling = sp.csr_matrix((weights, (rows, cols)), shape=(len(node_tuples), len(node_ids)))
    return sp.csr_matrix(pooling @ node_vectors)


def pair_similarities_sparse(
    pairs: NodePairs, node_ids: list[int], node_vectors: sp.csr_matrix
) -> np.ndarray:
    """Sparse analogue of pair_similarities, mean-pooling and comparing without densifying."""
    if not pairs:
        raise InsufficientDataError("pair similarity needs at least one pair")
    source_vecs = _pool_side_sparse([source for source, _ in pairs], node_ids, node_vectors)
    target_vecs = _pool_side_sparse([target for _, target in pairs], node_ids, node_vectors)
    return sparse_paired_cosine_similarity(source_vecs, target_vecs)
