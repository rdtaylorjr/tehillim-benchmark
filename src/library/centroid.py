"""Pools an item's half-verse embedding vectors into one centroid, each weighted by its share."""

from collections.abc import Hashable, Mapping

import numpy as np
import scipy.sparse as sp

Weights = Mapping[int, float]


def uniform_weights[K: Hashable](nodes_by_item: Mapping[K, list[int]]) -> dict[K, dict[int, float]]:
    """Every half-verse of an item at weight one, the whole-unit case of the weighted mean."""
    return {item: dict.fromkeys(nodes, 1.0) for item, nodes in nodes_by_item.items()}


def psalm_centroids[K: Hashable](
    weights_by_item: Mapping[K, Weights], node_vectors: Mapping[int, np.ndarray]
) -> dict[K, np.ndarray]:
    """One weighted mean per item whose half-verse nodes are all present in node_vectors."""
    centroids: dict[K, np.ndarray] = {}
    for item, weights in weights_by_item.items():
        if not all(node in node_vectors for node in weights):
            continue
        rows = np.stack([node_vectors[node] for node in weights])
        #: In the rows' own dtype, so unit weights give exactly the plain mean of the rows.
        share = np.fromiter(weights.values(), dtype=rows.dtype, count=len(weights))
        centroids[item] = (rows * share[:, None]).sum(axis=0) / share.sum()
    return centroids


def sparse_psalm_centroids[K: Hashable](
    weights_by_item: Mapping[K, Weights], node_ids: list[int], node_vectors: sp.csr_matrix
) -> tuple[list[K], sp.csr_matrix]:
    """Sparse analogue of psalm_centroids: pools via one matmul, never densifies."""
    node_index = {n: i for i, n in enumerate(node_ids)}
    usable_items = [
        item
        for item, weights in weights_by_item.items()
        if all(node in node_index for node in weights)
    ]
    row_lengths = np.array([len(weights_by_item[item]) for item in usable_items], dtype=np.int64)
    group_ids = np.repeat(np.arange(len(usable_items)), row_lengths)
    flat_cols = np.array(
        [node_index[n] for item in usable_items for n in weights_by_item[item]], dtype=np.int64
    )
    share = np.array(
        [w for item in usable_items for w in weights_by_item[item].values()], dtype=np.float64
    )
    totals = np.repeat(
        np.array([sum(weights_by_item[item].values()) for item in usable_items]), row_lengths
    )
    pooling = sp.csr_matrix(
        (share / totals, (group_ids, flat_cols)), shape=(len(usable_items), len(node_ids))
    )
    return usable_items, sp.csr_matrix(pooling @ node_vectors)
