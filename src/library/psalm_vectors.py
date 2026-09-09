"""Loads one model's psalm centroids, whether its embeddings are stored dense or sparse."""

from pathlib import Path

import numpy as np

from library.centroid import psalm_centroids, sparse_psalm_centroids
from library.embeddings import is_sparse_embeddings, load_embeddings, load_sparse_embeddings


def load_psalm_vectors(
    path: Path, half_verses_by_psalm: dict[int, list[int]]
) -> dict[int, np.ndarray]:
    """One centroid per psalm; a sparse file is pooled while still sparse and densified after."""
    if not is_sparse_embeddings(path):
        return psalm_centroids(half_verses_by_psalm, load_embeddings(path))

    node_ids, matrix = load_sparse_embeddings(path)
    psalms, centroids = sparse_psalm_centroids(half_verses_by_psalm, node_ids, matrix)
    # Only the centroids are densified: 150 rows, against tens of thousands of half-verses.
    dense = centroids.toarray().astype("<f4", copy=False)
    return {psalm: dense[i] for i, psalm in enumerate(psalms)}
