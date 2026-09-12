"""Loads one model's psalm centroids, whether its embeddings are stored dense or sparse."""

from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from library.centroid import psalm_centroids, sparse_psalm_centroids
from library.embeddings import (
    is_sparse_embeddings,
    read_dense_rows,
    read_sparse_rows,
    sparse_rows_of,
)


def with_direction(centroids: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Drops a psalm whose half-verses are all empty, since cosine is undefined on a zero vector."""
    return {psalm: v for psalm, v in centroids.items() if np.any(v)}


def load_psalm_vectors(
    path: Path, half_verses_by_psalm: dict[int, list[int]]
) -> dict[int, np.ndarray]:
    """One centroid per psalm; a sparse file is pooled while still sparse and densified after."""
    if not is_sparse_embeddings(path):
        return with_direction(psalm_centroids(half_verses_by_psalm, read_dense_rows(path)))
    return _sparse_centroids(*read_sparse_rows(path), half_verses_by_psalm)


def draw_psalm_vectors(
    vectors: dict[int, Any],
    sparse_width: int | None,
    half_verses_by_psalm: dict[int, list[int]],
) -> dict[int, np.ndarray]:
    """The centroids load_psalm_vectors returns, for a draw that never went through Parquet."""
    if sparse_width is None:
        return with_direction(psalm_centroids(half_verses_by_psalm, vectors))
    return _sparse_centroids(*sparse_rows_of(vectors, sparse_width), half_verses_by_psalm)


def _sparse_centroids(
    node_ids: list[int], matrix: sp.csr_matrix, half_verses_by_psalm: dict[int, list[int]]
) -> dict[int, np.ndarray]:
    """Pools while still sparse, then densifies only the centroids."""
    psalms, centroids = sparse_psalm_centroids(half_verses_by_psalm, node_ids, matrix)
    # Only the centroids are densified: 150 rows, against tens of thousands of half-verses.
    dense = centroids.toarray().astype("<f4", copy=False)
    return with_direction({psalm: dense[i] for i, psalm in enumerate(psalms)})
