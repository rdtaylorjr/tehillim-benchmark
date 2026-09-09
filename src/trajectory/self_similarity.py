"""Computes a psalm's self-similarity matrix from its ordered half-verses sequence."""

import numpy as np

from library.retrieval_metrics import cosine_similarity_matrix


def self_similarity_matrix(sequence: np.ndarray) -> np.ndarray:
    """S[i, j] = cosine similarity between half-verses i and j of one psalm's ordered sequence."""
    return cosine_similarity_matrix(sequence, sequence)
