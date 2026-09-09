"""Psalm-label permutation test for one-vs-rest genre AUC, jointly across genres (maxT)."""

import numpy as np

from library.blocking import row_blocks, rows_per_block
from library.errors import InsufficientDataError
from library.fast_metrics import fast_auc
from library.permutation_test import (
    GroupPermutationResult,
    maxt_p_values,
    permuted_label_batches,
)
from library.protocol import DEFAULT_N_GROUP_PERMUTATIONS

_MIN_PSALMS_FOR_PERMUTATION = 2


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


def _batched_separation(
    sims: np.ndarray, rows: np.ndarray, cols: np.ndarray, is_target_batch: np.ndarray
) -> np.ndarray:
    """Batched one-vs-rest (AUC - 0.5) for many draws at once, over one fixed sort of sims."""
    #: Draws are independent, so chunking is exact and caps two draws-by-pairs float64 arrays.
    n_pairs_total = len(rows)
    if len(is_target_batch) > rows_per_block(n_pairs_total):
        return np.concatenate(
            [
                _batched_separation(sims, rows, cols, is_target_batch[span])
                for span in row_blocks(len(is_target_batch), n_pairs_total)
            ]
        )
    n_pairs = sims.shape[0]
    same_batch = is_target_batch[:, rows] & is_target_batch[:, cols]
    population_batch = is_target_batch[:, rows] | is_target_batch[:, cols]

    order = np.argsort(sims, kind="stable")
    sims_sorted = sims[order]
    group_id = np.concatenate(([0], np.cumsum(sims_sorted[1:] != sims_sorted[:-1])))
    _, first_idx_of_group = np.unique(group_id, return_index=True)
    last_idx_of_group = np.concatenate([first_idx_of_group[1:] - 1, [n_pairs - 1]])
    group_start_idx = first_idx_of_group[group_id]
    group_end_idx = last_idx_of_group[group_id]

    same_sorted = same_batch[:, order]
    population_sorted = population_batch[:, order].astype(np.float64)

    # One cumsum per draw, then fancy indexing to each position's tie-group boundary.
    cum_pop_sorted = np.cumsum(population_sorted, axis=1)
    count_le = cum_pop_sorted[:, group_end_idx]
    start_before = np.clip(group_start_idx - 1, 0, None)
    count_lt = np.where(group_start_idx > 0, cum_pop_sorted[:, start_before], 0.0)
    count_eq = count_le - count_lt
    avg_rank = count_lt + (count_eq + 1) / 2

    n_same = same_batch.sum(axis=1)
    n_population = population_batch.sum(axis=1)
    n_different = n_population - n_same
    rank_sum_same = np.where(same_sorted, avg_rank, 0.0).sum(axis=1)
    u_statistic = rank_sum_same - n_same * (n_same + 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = u_statistic / (n_same * n_different)
    invalid = (n_same == 0) | (n_different == 0)
    return np.where(invalid, np.nan, auc - 0.5)


def joint_psalm_label_permutation_test(
    similarity_matrix: np.ndarray,
    genre_codes: np.ndarray,
    genres: tuple[str, ...],
    n_permutations: int = DEFAULT_N_GROUP_PERMUTATIONS,
    *,
    rng: np.random.Generator,
) -> GroupPermutationResult:
    """One-sided permutation p per genre's one-vs-rest AUC, plus a Westfall-Young (1993) maxT."""
    n = similarity_matrix.shape[0]
    if n < _MIN_PSALMS_FOR_PERMUTATION:
        raise InsufficientDataError(
            f"a one-vs-rest permutation test needs at least "
            f"{_MIN_PSALMS_FOR_PERMUTATION} psalms, got {n}"
        )
    rows, cols = np.triu_indices(n, k=1)
    sims = similarity_matrix[rows, cols]
    n_genres = len(genres)

    auc_observed = np.full(n_genres, np.nan)
    for g in range(n_genres):
        same_mask, population_mask = one_vs_rest_masks(genre_codes, g)
        auc_observed[g] = _one_vs_rest_auc(sims, same_mask[rows, cols], population_mask[rows, cols])
    # Signed, not |AUC-0.5|, so a genre separated in the opposite direction is not flagged.
    separation_observed = auc_observed - 0.5

    permuted_codes = permuted_label_batches(genre_codes, n_permutations, rng)
    null_separation = np.full((n_permutations, n_genres), np.nan)
    for g in range(n_genres):
        null_separation[:, g] = _batched_separation(sims, rows, cols, permuted_codes == g)
    permutation = maxt_p_values(separation_observed, null_separation)

    return GroupPermutationResult(
        genres=genres,
        observed=tuple(auc_observed.tolist()),
        p_perm=tuple(permutation.p_per_group.tolist()),
        p_maxt=tuple(permutation.p_maxt.tolist()),
        n_permutations=n_permutations,
    )
