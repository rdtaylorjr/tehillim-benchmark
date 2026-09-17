"""Vertex-resampling BCa bootstrap CI (Efron 1987) for genre AP, gap, and AUC."""

from dataclasses import dataclass

import numpy as np

from library.ap_gap_auc_bootstrap import (
    MIN_PER_SIDE,
    ApGapAucCI,
    Statistics,
    bootstrap_ci_from_statistics,
    has_enough_per_side,
    point_ap_gap_and_auc,
)
from library.calibration import BackgroundStats
from library.errors import InsufficientDataError
from library.protocol import DEFAULT_N_RESAMPLES
from library.retrieval_metrics import cosine_similarity_matrix
from library.weighted_metrics import WeightedStatistics, sort_pairs, weighted_ap_gap_auc


def psalm_similarity_matrix(
    psalm_ids: list[str], psalm_vectors: dict[str, np.ndarray]
) -> np.ndarray:
    """N x N cosine similarity between item centroids, ordered to match psalm_ids."""
    vectors = np.stack([psalm_vectors[p] for p in psalm_ids])
    return cosine_similarity_matrix(vectors, vectors)


def build_similarity_and_genre_matrices(
    psalm_ids: list[str], psalm_vectors: dict[str, np.ndarray], genre_by_psalm: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """N x N cosine similarity and genre-match matrices, ordered to match psalm_ids."""
    genres = np.array([genre_by_psalm[p] for p in psalm_ids])
    return psalm_similarity_matrix(psalm_ids, psalm_vectors), genres[:, None] == genres[None, :]


@dataclass(frozen=True, slots=True)
class PopulationPairs:
    """The pairs a CI is read over: endpoints, scores, and same-genre flags, in one fixed order."""

    rows: np.ndarray
    cols: np.ndarray
    scores: np.ndarray
    positive: np.ndarray


def population_pairs(
    similarity_matrix: np.ndarray,
    genre_match_matrix: np.ndarray,
    population_mask: np.ndarray | None = None,
) -> PopulationPairs:
    """The strict upper triangle restricted to population_mask, as one pair list."""
    rows, cols = np.triu_indices(similarity_matrix.shape[0], k=1)
    if population_mask is not None:
        keep = population_mask[rows, cols]
        rows, cols = rows[keep], cols[keep]
    return PopulationPairs(
        rows=rows,
        cols=cols,
        scores=similarity_matrix[rows, cols],
        positive=genre_match_matrix[rows, cols],
    )


def own_clusters(n: int) -> np.ndarray:
    """Every item its own cluster, the case of one passage per psalm."""
    return np.arange(n, dtype=np.intp)


def cluster_multiplicities(
    clusters: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """How often each item is drawn per resample, its cluster drawn with replacement each time."""
    codes, item_cluster = np.unique(clusters, return_inverse=True)
    n_clusters = len(codes)
    #: One draw per resample from the one generator, so the draws are those of a sequential loop.
    drawn = (
        np.stack(
            [
                np.bincount(
                    rng.choice(n_clusters, size=n_clusters, replace=True), minlength=n_clusters
                )
                for _ in range(n_resamples)
            ]
        )
        if n_resamples > 0
        else np.empty((0, n_clusters), dtype=np.intp)
    )
    return drawn[:, item_cluster]


def resample_weights(multiplicities: np.ndarray, pairs: PopulationPairs) -> np.ndarray:
    """Each pair's count in each resample: the product of its endpoints' multiplicities."""
    weights: np.ndarray = multiplicities[:, pairs.rows] * multiplicities[:, pairs.cols]
    return weights


def jackknife_weights(clusters: np.ndarray, pairs: PopulationPairs) -> np.ndarray:
    """One row per cluster in code order: the pairs touching neither endpoint of that cluster."""
    codes = np.unique(clusters)
    touched = (clusters[pairs.rows][None, :] == codes[:, None]) | (
        clusters[pairs.cols][None, :] == codes[:, None]
    )
    weights: np.ndarray = (~touched).astype(np.float64)
    return weights


def _valid_statistics(statistics: WeightedStatistics, keep_invalid: bool) -> Statistics:
    """(AP, gap, AUC) where both sides reached MIN_PER_SIDE, dropped or left NaN otherwise."""
    enough = (statistics.positive_weight >= MIN_PER_SIDE) & (
        statistics.negative_weight >= MIN_PER_SIDE
    )
    if keep_invalid:
        blank = np.where(enough, 0.0, np.nan)
        return statistics.ap + blank, statistics.gap + blank, statistics.auc + blank
    return statistics.ap[enough], statistics.gap[enough], statistics.auc[enough]


def block_bootstrap_genre_ap_gap_and_auc(
    psalm_ids: list[str],
    similarity_matrix: np.ndarray,
    genre_match_matrix: np.ndarray,
    background: BackgroundStats,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    *,
    rng: np.random.Generator,
    population_mask: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
) -> ApGapAucCI:
    """BCa 95% CI for AP (primary), gap, and AUC, resampling whole psalms with replacement."""
    n = len(psalm_ids)
    #: Passages of one psalm are not independent, so the psalm is the unit drawn, not the passage.
    cluster_of = own_clusters(n) if clusters is None else clusters
    pairs = population_pairs(similarity_matrix, genre_match_matrix, population_mask)
    if len(pairs.scores) == 0:
        raise InsufficientDataError(f"no genre pairs available among {n} items")
    positive, negative = pairs.scores[pairs.positive], pairs.scores[~pairs.positive]
    if not has_enough_per_side((positive, negative)):
        raise InsufficientDataError(
            f"AP and AUC need at least {MIN_PER_SIDE} values on each side, got "
            f"{len(positive)} positive and {len(negative)} negative"
        )
    point = point_ap_gap_and_auc(positive, negative, background)
    prevalence = len(positive) / len(pairs.scores)

    sorted_pairs = sort_pairs(pairs.scores, pairs.positive)
    multiplicities = cluster_multiplicities(cluster_of, n_resamples, rng)
    resampled = weighted_ap_gap_auc(
        sorted_pairs, resample_weights(multiplicities, pairs), background
    )
    jackknife = weighted_ap_gap_auc(sorted_pairs, jackknife_weights(cluster_of, pairs), background)
    return bootstrap_ci_from_statistics(
        point,
        prevalence,
        _valid_statistics(resampled, keep_invalid=False),
        _valid_statistics(jackknife, keep_invalid=True),
    )
