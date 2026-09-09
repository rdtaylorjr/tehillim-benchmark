"""Permutation test of within-genre against between-genre trajectory distance."""

import numpy as np

from library.blocking import row_blocks
from library.protocol import DEFAULT_N_PERMUTATIONS


def observed_gap(distances: np.ndarray, same_genre: np.ndarray) -> float:
    """mean(between-genre distance) - mean(within-genre distance); positive means closer within."""
    return float(distances[~same_genre].mean() - distances[same_genre].mean())


def _null_gaps(same_matrix: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """observed_gap(distances, row) for every row of a (n_permutations, n_pairs) same/diff array."""
    distances = distances.astype(np.float64, copy=False)
    #: All 10000 permutations at once is a 10000 by 11175 float64 array, 853 MiB, so it chunks.
    same_sum = np.empty(len(same_matrix), dtype=np.float64)
    for span in row_blocks(len(same_matrix), same_matrix.shape[1]):
        #: Row-wise product then sum, not a matmul, whose blocking would depend on the chunk size.
        same_sum[span] = np.sum(same_matrix[span].astype(np.float64) * distances, axis=1)
    same_count = same_matrix.sum(axis=1).astype(np.float64)
    total_sum = distances.sum()
    total_count = float(len(distances))
    diff_sum = total_sum - same_sum
    diff_count = total_count - same_count
    return np.asarray((diff_sum / diff_count) - (same_sum / same_count))


def same_genre_matrix(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    genre_labels: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Whether each shuffled pair shares a genre; identical for every source of one metric."""
    _, codes = np.unique(genre_labels, return_inverse=True)
    shuffled_codes = rng.permuted(np.tile(codes, (n_permutations, 1)), axis=1)
    return np.asarray(shuffled_codes[:, idx_a] == shuffled_codes[:, idx_b])


def permutation_test(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    distances: np.ndarray,
    genre_labels: np.ndarray,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    *,
    rng: np.random.Generator,
    same_matrix: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Observed gap, p-value, and a null-calibrated effect size (NaN if null has no variance)."""
    same_genre = genre_labels[idx_a] == genre_labels[idx_b]
    observed = observed_gap(distances, same_genre)

    if same_matrix is None:
        same_matrix = same_genre_matrix(idx_a, idx_b, genre_labels, n_permutations, rng)
    null_gaps = _null_gaps(same_matrix, distances)

    p_value = (np.sum(null_gaps >= observed) + 1) / (len(null_gaps) + 1)
    null_std = null_gaps.std(ddof=1)
    effect_size = (observed - null_gaps.mean()) / null_std if null_std > 0 else float("nan")
    return observed, float(p_value), float(effect_size)
