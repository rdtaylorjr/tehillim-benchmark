"""Mean-pools a psalm's half-verse embedding vectors into one centroid per psalm."""

import numpy as np
import scipy.sparse as sp


def psalm_centroids(
    half_verses_by_psalm: dict[int, list[int]], node_vectors: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """One centroid per psalm whose half-verse nodes are all present in node_vectors."""
    centroids: dict[int, np.ndarray] = {}
    for psalm, nodes in half_verses_by_psalm.items():
        if not all(node in node_vectors for node in nodes):
            continue
        centroids[psalm] = np.mean([node_vectors[node] for node in nodes], axis=0)
    return centroids


def sparse_psalm_centroids(
    half_verses_by_psalm: dict[int, list[int]], node_ids: list[int], node_vectors: sp.csr_matrix
) -> tuple[list[int], sp.csr_matrix]:
    """Sparse mean-pool analogue of psalm_centroids: pools via matmul, never densifies."""
    node_index = {n: i for i, n in enumerate(node_ids)}
    usable_psalms = [
        psalm
        for psalm, nodes in half_verses_by_psalm.items()
        if all(node in node_index for node in nodes)
    ]
    row_lengths = np.array(
        [len(half_verses_by_psalm[psalm]) for psalm in usable_psalms], dtype=np.int64
    )
    group_ids = np.repeat(np.arange(len(usable_psalms)), row_lengths)
    flat_cols = np.array(
        [node_index[n] for psalm in usable_psalms for n in half_verses_by_psalm[psalm]],
        dtype=np.int64,
    )
    weights = 1.0 / np.repeat(row_lengths, row_lengths)
    pooling = sp.csr_matrix(
        (weights, (group_ids, flat_cols)), shape=(len(usable_psalms), len(node_ids))
    )
    return usable_psalms, sp.csr_matrix(pooling @ node_vectors)
