"""Calibrates similarity against a model's own background; group-mean form is an effect size."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from library.errors import DegenerateVectorError, InsufficientDataError
from library.retrieval_metrics import sparse_cosine_similarity_matrix

_MIN_VECTORS_FOR_BACKGROUND = 2


@dataclass(frozen=True, slots=True)
class BackgroundStats:
    """Mean and spread of a model's background similarity, the scale effect sizes divide by."""

    mean: float
    std: float
    n_vectors: int


def background_stats_from_matrix(similarity_matrix: np.ndarray) -> BackgroundStats:
    """Same statistic as background_similarity_stats, from an already-computed matrix."""
    n = similarity_matrix.shape[0]
    if n < _MIN_VECTORS_FOR_BACKGROUND:
        raise InsufficientDataError(
            f"a background needs at least {_MIN_VECTORS_FOR_BACKGROUND} vectors, got {n}"
        )
    off_diagonal = similarity_matrix[~np.eye(n, dtype=bool)]
    return BackgroundStats(
        mean=float(off_diagonal.mean()), std=float(off_diagonal.std()), n_vectors=n
    )


def background_similarity_stats(vectors: np.ndarray) -> BackgroundStats:
    """Mean/std cosine similarity across every distinct pair among vectors, self-pairs excluded."""
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarities = norm @ norm.T
    return background_stats_from_matrix(similarities)


def background_similarity_stats_sparse(vectors: sp.csr_matrix) -> BackgroundStats:
    """Same statistic as background_similarity_stats, over sparse vectors never densified."""
    return background_stats_from_matrix(sparse_cosine_similarity_matrix(vectors, vectors))


def calibrated_z_score(value: float, background: BackgroundStats) -> float:
    """How many background standard deviations above typical a single observation sits."""
    if background.std == 0:
        raise DegenerateVectorError("background has zero variance; this model cannot be calibrated")
    return (value - background.mean) / background.std


def calibrated_effect_size(group_mean: float, background: BackgroundStats) -> float:
    """Cohen's-d-style standardized distance of a group's mean similarity from the background."""
    return calibrated_z_score(group_mean, background)
