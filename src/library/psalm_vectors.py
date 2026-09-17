"""Loads one model's item centroids, whether its embeddings are stored dense or sparse."""

from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from core.datasets import is_sparse_embeddings, read_dense_rows, read_sparse_rows

from library.centroid import Weights, psalm_centroids, sparse_psalm_centroids
from library.embeddings import sparse_rows_of


def with_direction[K: Hashable](centroids: dict[K, np.ndarray]) -> dict[K, np.ndarray]:
    """Drops an item whose half-verses are all empty, since cosine is undefined on a zero vector."""
    return {psalm: v for psalm, v in centroids.items() if np.any(v)}


def load_psalm_vectors[K: Hashable](
    path: Path, half_verses_by_psalm: Mapping[K, Weights]
) -> dict[K, np.ndarray]:
    """One centroid per item; a sparse file is pooled while still sparse and densified after."""
    if not is_sparse_embeddings(path):
        return with_direction(psalm_centroids(half_verses_by_psalm, read_dense_rows(path)))
    return _sparse_centroids(*read_sparse_rows(path), half_verses_by_psalm)


def draw_psalm_vectors[K: Hashable](
    vectors: dict[int, Any],
    sparse_width: int | None,
    half_verses_by_psalm: Mapping[K, Weights],
) -> dict[K, np.ndarray]:
    """The centroids load_psalm_vectors returns, for a draw that never went through Parquet."""
    if sparse_width is None:
        return with_direction(psalm_centroids(half_verses_by_psalm, vectors))
    return _sparse_centroids(*sparse_rows_of(vectors, sparse_width), half_verses_by_psalm)


def _sparse_centroids[K: Hashable](
    node_ids: list[int], matrix: sp.csr_matrix, half_verses_by_psalm: Mapping[K, Weights]
) -> dict[K, np.ndarray]:
    """Pools while still sparse, then densifies only the centroids."""
    psalms, centroids = sparse_psalm_centroids(half_verses_by_psalm, node_ids, matrix)
    # Only the centroids are densified: 150 rows, against tens of thousands of half-verses.
    dense = centroids.toarray().astype("<f4", copy=False)
    return with_direction({psalm: dense[i] for i, psalm in enumerate(psalms)})


def sparse_item_vectors(
    node_ids: list[int], matrix: sp.csr_matrix, half_verses_by_psalm: Mapping[str, Weights]
) -> tuple[list[str], sp.csr_matrix]:
    """Sparse centroids in sorted item order with zero rows dropped, never densified."""
    psalms, centroids = sparse_psalm_centroids(half_verses_by_psalm, node_ids, matrix)
    nonzero = set(np.flatnonzero(np.diff(centroids.indptr) > 0).tolist())
    kept = {psalm: i for i, psalm in enumerate(psalms) if i in nonzero}
    order = [kept[psalm] for psalm in sorted(kept)]
    return [psalms[i] for i in order], centroids[order]


def load_item_vectors(
    path: Path, half_verses_by_psalm: Mapping[str, Weights]
) -> dict[str, np.ndarray] | tuple[list[str], sp.csr_matrix]:
    """load_psalm_vectors, but a sparse file yields sparse centroids rather than dense rows."""
    if not is_sparse_embeddings(path):
        return with_direction(psalm_centroids(half_verses_by_psalm, read_dense_rows(path)))
    return sparse_item_vectors(*read_sparse_rows(path), half_verses_by_psalm)


def draw_item_vectors(
    vectors: dict[int, Any],
    sparse_width: int | None,
    half_verses_by_psalm: Mapping[str, Weights],
) -> dict[str, np.ndarray] | tuple[list[str], sp.csr_matrix]:
    """draw_psalm_vectors, but a sparse draw yields sparse centroids rather than dense rows."""
    if sparse_width is None:
        return with_direction(psalm_centroids(half_verses_by_psalm, vectors))
    return sparse_item_vectors(*sparse_rows_of(vectors, sparse_width), half_verses_by_psalm)
