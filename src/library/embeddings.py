"""Loads BHSA-node-keyed embedding vectors the way each benchmark scores them."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from core.datasets import (
    is_sparse_embeddings,
    read_dense_rows,
    read_sparse_rows,
    sparse_rows_to_csr,
)


def load_embeddings(path: Path) -> dict[int, np.ndarray]:
    """Reads a Parquet embeddings file into a {node: vector} map, excluding zero-norm vectors."""
    if is_sparse_embeddings(path):
        #: `load_psalm_vectors` pools while still sparse, so prefer it for centroids.
        node_ids, matrix = load_sparse_embeddings(path)
        dense = matrix.toarray().astype("<f4", copy=False)
        return {node: dense[i] for i, node in enumerate(node_ids)}
    return drop_zero_norm_vectors(read_dense_rows(path))


def drop_zero_norm_vectors(node_vectors: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """The rows load_embeddings keeps, for dense vectors that never went through Parquet."""
    return {node: vector for node, vector in node_vectors.items() if np.any(vector)}


def load_sparse_embeddings(path: Path) -> tuple[list[int], sp.csr_matrix]:
    """Reads a sparse Parquet embeddings file: node ids in row order, and one CSR matrix."""
    return drop_empty_rows(*read_sparse_rows(path))


def sparse_rows_of(
    sparse_vectors: dict[int, tuple[np.ndarray, np.ndarray]], dim: int
) -> tuple[list[int], sp.csr_matrix]:
    """Every row a built draw carries, without the Parquet round trip and without dropping."""
    #: The writer sorts node ids, so a fused reader sorts too or it sees a different row order.
    node_ids = sorted(sparse_vectors)
    return sparse_rows_to_csr(
        node_ids,
        [sparse_vectors[node][0] for node in node_ids],
        [sparse_vectors[node][1] for node in node_ids],
        dim,
    )


def sparse_vectors_to_csr(
    sparse_vectors: dict[int, tuple[np.ndarray, np.ndarray]], dim: int
) -> tuple[list[int], sp.csr_matrix]:
    """The rows a written-then-read sparse dataset yields, without the Parquet round trip."""
    return drop_empty_rows(*sparse_rows_of(sparse_vectors, dim))


def drop_empty_rows(node_ids: list[int], matrix: sp.csr_matrix) -> tuple[list[int], sp.csr_matrix]:
    """The rows a node-level cosine can use, since a row with no nonzeros has no direction."""
    kept = np.flatnonzero(np.diff(matrix.indptr) > 0)
    return [node_ids[i] for i in kept], matrix[kept]
