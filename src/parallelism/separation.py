"""Measures whether true parallel half-verses are closer in embedding space than unrelated ones."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu

from library.errors import InsufficientDataError

_MIN_PAIRS_FOR_SEPARATION = 2


@dataclass(frozen=True, slots=True)
class SeparationResult:
    """AUC separating two score populations, with the counts it was computed from."""

    auc: float
    p_value: float
    n_positive: int
    n_negative: int


def similarity_separation(
    similarity_matrix: np.ndarray, row_mask: np.ndarray | None = None
) -> SeparationResult:
    """AUC of true-pair vs. other-pair similarity; row_mask restricts positives only."""
    n = similarity_matrix.shape[0]
    if n < _MIN_PAIRS_FOR_SEPARATION:
        raise InsufficientDataError(
            f"separation AUC needs at least {_MIN_PAIRS_FOR_SEPARATION} pairs, got {n}"
        )
    rows = np.arange(n) if row_mask is None else np.flatnonzero(row_mask)
    positive = similarity_matrix[rows, rows]
    # Gathering only the masked rows keeps a per-type call off the full n x n off-diagonal copy.
    negative = similarity_matrix[rows][~np.eye(n, dtype=bool)[rows]]
    statistic, p_value = mannwhitneyu(positive, negative, alternative="greater")
    auc = statistic / (len(positive) * len(negative))
    return SeparationResult(
        auc=float(auc),
        p_value=float(p_value),
        n_positive=len(positive),
        n_negative=len(negative),
    )
