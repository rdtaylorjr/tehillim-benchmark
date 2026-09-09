"""Builds each psalm's half-verses embeddings as one ordered sequence, in canonical verse order."""

import numpy as np
import scipy.sparse as sp

from library.errors import DegenerateVectorError


def psalm_half_verse_sequences(
    half_verses_by_psalm: dict[int, list[int]], node_vectors: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """One (n_half_verses, dim) array per psalm whose half-verse nodes are all in node_vectors."""
    sequences = {}
    for psalm, nodes in half_verses_by_psalm.items():
        if not all(node in node_vectors for node in nodes):
            continue
        sequences[psalm] = np.stack([node_vectors[node] for node in nodes])
    return sequences


def psalm_half_verse_sequences_sparse(
    half_verses_by_psalm: dict[int, list[int]], node_ids: list[int], node_vectors: sp.csr_matrix
) -> dict[int, np.ndarray]:
    """Sparse analogue of psalm_half_verse_sequences, densifying one psalm at a time."""
    node_index = {n: i for i, n in enumerate(node_ids)}
    sequences = {}
    for psalm, nodes in half_verses_by_psalm.items():
        if not all(node in node_index for node in nodes):
            continue
        rows = [node_index[node] for node in nodes]
        sequences[psalm] = node_vectors[rows].toarray()
    return sequences


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """L2-normalizes each row so later magnitude-sensitive features aren't skewed by anisotropy."""
    norms = np.linalg.norm(sequence, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise DegenerateVectorError("cannot normalize a zero vector")
    return np.asarray(sequence / norms)
