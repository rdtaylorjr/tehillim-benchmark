"""Psalm-label permutation test for one-vs-rest genre AUC, jointly across genres (maxT)."""

from dataclasses import dataclass

import numpy as np

from library.blocking import row_blocks
from library.errors import InsufficientDataError
from library.fast_metrics import fast_auc
from library.permutation_test import (
    GroupPermutationResult,
    maxt_p_values,
    permuted_label_batches,
)
from library.protocol import DEFAULT_N_GROUP_PERMUTATIONS

_MIN_PSALMS_FOR_PERMUTATION = 2
#: A random gather costs about this many multiply-adds, which sets where the product path wins.
_GATHER_COST = 100


def one_vs_rest_masks(genre_codes: np.ndarray, genre_index: int) -> tuple[np.ndarray, np.ndarray]:
    """same_mask/population_mask (n x n): population is pairs touching genre_index either side."""
    is_target = genre_codes == genre_index
    same_mask = is_target[:, None] & is_target[None, :]
    population_mask = is_target[:, None] | is_target[None, :]
    return same_mask, population_mask


def _one_vs_rest_auc(sims: np.ndarray, same: np.ndarray, population: np.ndarray) -> float:
    """fast_auc restricted to a population mask (Hanley-McNeil 1982); NaN if a side is empty."""
    pop_sims = sims[population]
    pop_same = same[population]
    same_sims = pop_sims[pop_same]
    different_sims = pop_sims[~pop_same]
    if len(same_sims) == 0 or len(different_sims) == 0:
        return float("nan")
    return fast_auc(same_sims, different_sims)


@dataclass(frozen=True, slots=True)
class PairRanks:
    """Per item, twice its mid-rank pair count at every pair score, with a zero column last."""

    doubled_counts: np.ndarray
    degree: np.ndarray
    position: np.ndarray
    n_pairs: int


def pair_ranks(n_items: int, rows: np.ndarray, cols: np.ndarray, sims: np.ndarray) -> PairRanks:
    """Tabulates, once per model, what every draw's rank sum decomposes into."""
    n_pairs = len(sims)
    order = np.argsort(sims, kind="stable")
    sorted_sims = sims[order]
    group_start = np.searchsorted(sorted_sims, sims, side="left")
    group_end = np.searchsorted(sorted_sims, sims, side="right")
    rank_of_pair = np.empty(n_pairs, dtype=np.intp)
    rank_of_pair[order] = np.arange(n_pairs)
    #: float32 holds these small integers exactly and lets a wide draw multiply rather than gather.
    doubled = np.zeros((n_items, n_pairs + 1), dtype=np.float32)
    degree = np.zeros(n_items, dtype=np.int64)
    #: Each item's pairs at their sorted ranks, so a count below any score is one search.
    endpoints = np.concatenate([rows, cols])
    pair_of_endpoint = np.concatenate([np.arange(n_pairs), np.arange(n_pairs)])
    by_item = np.argsort(endpoints, kind="stable")
    bounds = np.searchsorted(endpoints[by_item], np.arange(n_items + 1))
    for item in range(n_items):
        own = np.sort(rank_of_pair[pair_of_endpoint[by_item[bounds[item] : bounds[item + 1]]]])
        degree[item] = len(own)
        below = np.searchsorted(own, group_start, side="left")
        at_or_below = np.searchsorted(own, group_end, side="left")
        doubled[item, :n_pairs] = below + at_or_below
    position = np.full((n_items, n_items), n_pairs, dtype=np.intp)
    position[rows, cols] = np.arange(n_pairs)
    position[cols, rows] = np.arange(n_pairs)
    return PairRanks(doubled_counts=doubled, degree=degree, position=position, n_pairs=n_pairs)


def _separation_for_targets(ranks: PairRanks, targets: np.ndarray) -> np.ndarray:
    """(AUC - 0.5) for draws of equal size, targets as draws x k item indices."""
    n_draws, k = targets.shape
    if k == 0:
        return np.full(n_draws, np.nan)
    upper = np.triu(np.ones((k, k), dtype=bool), k=1)
    pair_index = ranks.position[targets[:, :, None], targets[:, None, :]]
    pair_index = np.where(upper[None], pair_index, ranks.n_pairs)
    n_same = (pair_index < ranks.n_pairs).sum(axis=(1, 2)).astype(np.float64)
    flat = pair_index.reshape(n_draws, k * k)
    n_items = ranks.position.shape[0]
    if k**3 * _GATHER_COST > n_items * ranks.n_pairs:
        #: A wide draw sums every item's column at once, then reads its same pairs off the product.
        membership = np.zeros((n_draws, n_items), dtype=np.float32)
        membership[np.arange(n_draws)[:, None], targets] = 1.0
        column_sums = membership @ ranks.doubled_counts
        doubled_rank_sum = np.take_along_axis(column_sums, flat, axis=1).sum(
            axis=1, dtype=np.float64
        )
    else:
        gathered = ranks.doubled_counts[targets[:, :, None], flat[:, None, :]]
        doubled_rank_sum = gathered.sum(axis=(1, 2), dtype=np.float64)
    n_population = ranks.degree[targets].sum(axis=1) - n_same
    n_different = n_population - n_same
    u_statistic = doubled_rank_sum / 2 - n_same * n_same
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = u_statistic / (n_same * n_different)
    invalid = (n_same == 0) | (n_different == 0)
    return np.where(invalid, np.nan, auc - 0.5)


def _draw_width(ranks: PairRanks, k: int) -> int:
    """The widest array one draw allocates on the path its size takes."""
    n_items = ranks.position.shape[0]
    if k**3 * _GATHER_COST > n_items * ranks.n_pairs:
        return ranks.n_pairs + 1
    return max(1, k**3)


def batched_separation(ranks: PairRanks, is_target_batch: np.ndarray) -> np.ndarray:
    """Batched one-vs-rest (AUC - 0.5) for many draws at once, from the tabulated ranks."""
    is_target_batch = np.asarray(is_target_batch, dtype=bool)
    n_draws, n_items = is_target_batch.shape
    counts = is_target_batch.sum(axis=1)
    separation = np.full(n_draws, np.nan)
    #: Draws of one size share one gather shape; a permutation gives every draw the same size.
    for k in np.unique(counts):
        draws = np.flatnonzero(counts == k)
        targets = np.flatnonzero(is_target_batch[draws]).reshape(len(draws), int(k)) % n_items
        #: Draws are independent, so blocking is exact and bounds the draws x k x k x k gather.
        separation[draws] = np.concatenate(
            [
                _separation_for_targets(ranks, targets[span])
                for span in row_blocks(len(draws), _draw_width(ranks, int(k)), itemsize=4)
            ]
        )
    return separation


def _batched_separation(
    sims: np.ndarray, rows: np.ndarray, cols: np.ndarray, is_target_batch: np.ndarray
) -> np.ndarray:
    """batched_separation from raw pairs, tabulating first."""
    n_items = is_target_batch.shape[1]
    return batched_separation(pair_ranks(n_items, rows, cols, sims), is_target_batch)


def joint_psalm_label_permutation_test(
    similarity_matrix: np.ndarray,
    genre_codes: np.ndarray,
    genres: tuple[str, ...],
    n_permutations: int = DEFAULT_N_GROUP_PERMUTATIONS,
    *,
    rng: np.random.Generator,
    admissible: np.ndarray | None = None,
) -> GroupPermutationResult:
    """One-sided permutation p per genre's one-vs-rest AUC, plus a Westfall-Young (1993) maxT."""
    n = similarity_matrix.shape[0]
    if n < _MIN_PSALMS_FOR_PERMUTATION:
        raise InsufficientDataError(
            f"a one-vs-rest permutation test needs at least "
            f"{_MIN_PSALMS_FOR_PERMUTATION} items, got {n}"
        )
    rows, cols = np.triu_indices(n, k=1)
    #: Passages sharing text are never a pair, under the real labels or any permutation of them.
    if admissible is not None:
        keep = admissible[rows, cols]
        rows, cols = rows[keep], cols[keep]
    sims = similarity_matrix[rows, cols]
    n_genres = len(genres)

    auc_observed = np.full(n_genres, np.nan)
    for g in range(n_genres):
        same_mask, population_mask = one_vs_rest_masks(genre_codes, g)
        auc_observed[g] = _one_vs_rest_auc(sims, same_mask[rows, cols], population_mask[rows, cols])
    # Signed, not |AUC-0.5|, so a genre separated in the opposite direction is not flagged.
    separation_observed = auc_observed - 0.5

    permuted_codes = permuted_label_batches(genre_codes, n_permutations, rng)
    ranks = pair_ranks(n, rows, cols, sims)
    null_separation = np.full((n_permutations, n_genres), np.nan)
    for g in range(n_genres):
        null_separation[:, g] = batched_separation(ranks, permuted_codes == g)
    permutation = maxt_p_values(separation_observed, null_separation)

    return GroupPermutationResult(
        genres=genres,
        observed=tuple(auc_observed.tolist()),
        p_perm=tuple(permutation.p_per_group.tolist()),
        p_maxt=tuple(permutation.p_maxt.tolist()),
        n_permutations=n_permutations,
    )
