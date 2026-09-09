"""Vertex-resampling BCa bootstrap CI (Efron 1987) for genre AP, gap, and AUC."""

from collections.abc import Iterator

import numpy as np

from library.ap_gap_auc_bootstrap import (
    ApGapAucCI,
    Split,
    bootstrap_ap_gap_and_auc,
)
from library.calibration import BackgroundStats
from library.errors import InsufficientDataError
from library.protocol import DEFAULT_N_RESAMPLES
from library.retrieval_metrics import cosine_similarity_matrix


def psalm_similarity_matrix(
    psalm_ids: list[int], psalm_vectors: dict[int, np.ndarray]
) -> np.ndarray:
    """N x N cosine similarity between psalm centroids, ordered to match psalm_ids."""
    vectors = np.stack([psalm_vectors[p] for p in psalm_ids])
    return cosine_similarity_matrix(vectors, vectors)


def build_similarity_and_genre_matrices(
    psalm_ids: list[int], psalm_vectors: dict[int, np.ndarray], genre_by_psalm: dict[int, str]
) -> tuple[np.ndarray, np.ndarray]:
    """N x N cosine similarity and genre-match matrices, ordered to match psalm_ids."""
    genres = np.array([genre_by_psalm[p] for p in psalm_ids])
    return psalm_similarity_matrix(psalm_ids, psalm_vectors), genres[:, None] == genres[None, :]


def _split_by_pair_indices(
    psalm_a: np.ndarray,
    psalm_b: np.ndarray,
    similarity_matrix: np.ndarray,
    genre_match_matrix: np.ndarray,
    population_mask: np.ndarray | None,
) -> Split:
    """Same/different similarities for the given psalm-index pairs, within population_mask."""
    sims = similarity_matrix[psalm_a, psalm_b]
    same = genre_match_matrix[psalm_a, psalm_b]
    if population_mask is not None:
        keep = population_mask[psalm_a, psalm_b]
        sims, same = sims[keep], same[keep]
    return sims[same], sims[~same]


def _upper_triangle_same_and_different(
    similarity_matrix: np.ndarray,
    genre_match_matrix: np.ndarray,
    population_mask: np.ndarray | None = None,
) -> Split:
    """Splits the strict upper triangle into same/different sims, restricted to population_mask."""
    rows, cols = np.triu_indices(similarity_matrix.shape[0], k=1)
    return _split_by_pair_indices(
        rows, cols, similarity_matrix, genre_match_matrix, population_mask
    )


def _resample_split(
    psalm_indices: np.ndarray,
    similarity_matrix: np.ndarray,
    genre_match_matrix: np.ndarray,
    population_mask: np.ndarray | None,
) -> Split:
    """One vertex resample's split, dropping pairs of a drawn psalm with its own duplicate copy."""
    rows, cols = np.triu_indices(len(psalm_indices), k=1)
    psalm_a, psalm_b = psalm_indices[rows], psalm_indices[cols]
    # A psalm drawn twice would otherwise pair with itself at similarity 1.0, always same-genre.
    distinct = psalm_a != psalm_b
    return _split_by_pair_indices(
        psalm_a[distinct],
        psalm_b[distinct],
        similarity_matrix,
        genre_match_matrix,
        population_mask,
    )


def _leave_one_out_splits(
    n: int,
    similarity_matrix: np.ndarray,
    genre_match_matrix: np.ndarray,
    population_mask: np.ndarray | None,
) -> Iterator[Split]:
    """Each psalm's leave-one-out split, in psalm order, for the BCa acceleration jackknife."""
    all_idx = np.arange(n)
    rows, cols = np.triu_indices(n - 1, k=1)
    for i in range(n):
        keep = all_idx[all_idx != i]
        yield _split_by_pair_indices(
            keep[rows], keep[cols], similarity_matrix, genre_match_matrix, population_mask
        )


def _vertex_resamples(
    n: int,
    n_resamples: int,
    rng: np.random.Generator,
    similarity_matrix: np.ndarray,
    genre_match_matrix: np.ndarray,
    population_mask: np.ndarray | None,
) -> Iterator[Split]:
    """n_resamples draws of the psalm population itself, with replacement."""
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        yield _resample_split(idx, similarity_matrix, genre_match_matrix, population_mask)


def block_bootstrap_genre_ap_gap_and_auc(
    psalm_ids: list[int],
    similarity_matrix: np.ndarray,
    genre_match_matrix: np.ndarray,
    background: BackgroundStats,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    *,
    rng: np.random.Generator,
    population_mask: np.ndarray | None = None,
) -> ApGapAucCI:
    """BCa 95% CI for AP (primary), gap, and AUC, resampling whole psalms with replacement."""
    n = len(psalm_ids)
    observed = _upper_triangle_same_and_different(
        similarity_matrix, genre_match_matrix, population_mask
    )
    if len(observed[0]) + len(observed[1]) == 0:
        raise InsufficientDataError(f"no genre pairs available among {n} psalms")
    return bootstrap_ap_gap_and_auc(
        observed,
        _vertex_resamples(
            n, n_resamples, rng, similarity_matrix, genre_match_matrix, population_mask
        ),
        _leave_one_out_splits(n, similarity_matrix, genre_match_matrix, population_mask),
        background,
    )
