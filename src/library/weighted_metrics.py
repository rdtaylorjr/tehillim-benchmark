"""AP, calibrated gap, and AUC for many weightings of one pair set, over a single sort."""

from dataclasses import dataclass

import numpy as np

from library.blocking import row_blocks
from library.calibration import BackgroundStats, calibrated_z_scores


@dataclass(frozen=True, slots=True)
class SortedPairs:
    """A pair set sorted once by score, with its tie groups, scored under many weightings."""

    order: np.ndarray
    scores: np.ndarray
    positive: np.ndarray
    group_start: np.ndarray
    group_end: np.ndarray

    @property
    def n_pairs(self) -> int:
        """How many pairs the set holds."""
        return len(self.scores)


def sort_pairs(scores: np.ndarray, positive: np.ndarray) -> SortedPairs:
    """Sorts ascending by score and marks each position's tie group, once per pair set."""
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    n = len(sorted_scores)
    new_group = np.concatenate(([True], sorted_scores[1:] != sorted_scores[:-1]))
    group_id = np.cumsum(new_group) - 1
    first = np.flatnonzero(new_group)
    last = np.concatenate([first[1:] - 1, [n - 1]])
    return SortedPairs(
        order=order,
        scores=sorted_scores,
        positive=np.asarray(positive, dtype=bool)[order],
        group_start=first[group_id],
        group_end=last[group_id],
    )


@dataclass(frozen=True, slots=True)
class WeightedStatistics:
    """AP, gap, and AUC per weighting, with the weight each side carried."""

    ap: np.ndarray
    gap: np.ndarray
    auc: np.ndarray
    positive_weight: np.ndarray
    negative_weight: np.ndarray


def _weighted_block(
    pairs: SortedPairs, weights: np.ndarray, background: BackgroundStats
) -> tuple[np.ndarray, ...]:
    """The five statistics for one block of weightings, weights given in the original pair order."""
    w = np.asarray(weights, dtype=np.float64)[:, pairs.order]
    w_pos = np.where(pairs.positive, w, 0.0)
    w_neg = w - w_pos
    positive_weight = w_pos.sum(axis=1)
    negative_weight = w_neg.sum(axis=1)

    # AUC as the weighted Mann-Whitney count, ties at half, read off one cumulative negative weight.
    cum_neg = np.cumsum(w_neg, axis=1)
    neg_le = cum_neg[:, pairs.group_end]
    neg_lt = cum_neg[:, pairs.group_start] - w_neg[:, pairs.group_start]
    u_statistic = (w_pos * (neg_lt + 0.5 * (neg_le - neg_lt))).sum(axis=1)

    # AP as sklearn's step area: each positive weight times the precision at its threshold.
    above = np.cumsum(w[:, ::-1], axis=1)[:, ::-1]
    above_pos = np.cumsum(w_pos[:, ::-1], axis=1)[:, ::-1]
    n_at = above[:, pairs.group_start]
    tp_at = above_pos[:, pairs.group_start]
    precision = np.divide(tp_at, n_at, out=np.zeros_like(tp_at), where=n_at > 0)
    ap_numerator = (w_pos * precision).sum(axis=1)

    mean_pos = (w_pos * pairs.scores).sum(axis=1)
    mean_neg = (w_neg * pairs.scores).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = u_statistic / (positive_weight * negative_weight)
        ap = ap_numerator / positive_weight
        mean_pos = mean_pos / positive_weight
        mean_neg = mean_neg / negative_weight
    gap = calibrated_z_scores(mean_pos, background) - calibrated_z_scores(mean_neg, background)
    return ap, gap, auc, positive_weight, negative_weight


def weighted_ap_gap_auc(
    pairs: SortedPairs, weights: np.ndarray, background: BackgroundStats
) -> WeightedStatistics:
    """Scores every row of weights (draws x pairs, original order) in blocks bounded in memory."""
    weights = np.atleast_2d(weights)
    #: Rows are independent draws, so blocking is exact and bounds the widened float64 arrays.
    blocks = [
        _weighted_block(pairs, weights[span], background)
        for span in row_blocks(len(weights), pairs.n_pairs)
    ]
    if not blocks:
        empty = np.empty(0)
        return WeightedStatistics(empty, empty, empty, empty, empty)
    columns = [np.concatenate(parts) for parts in zip(*blocks, strict=True)]
    return WeightedStatistics(*columns)
