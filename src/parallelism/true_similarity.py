"""Raw cosine similarity of true parallel half-verses, with no comparison to any other candidate."""

from dataclasses import dataclass

import numpy as np

from library.errors import InsufficientDataError


@dataclass(frozen=True, slots=True)
class TruePairSimilarity:
    """Location and spread of the similarities of genuinely parallel pairs."""

    mean: float
    median: float
    std: float
    n: int


def summarize_true_pair_similarity(similarities: np.ndarray) -> TruePairSimilarity:
    """Location and spread of the genuinely parallel pairs' similarities, and their count."""
    if len(similarities) == 0:
        raise InsufficientDataError("summarising true pair similarity needs at least one pair")
    return TruePairSimilarity(
        mean=float(np.mean(similarities)),
        median=float(np.median(similarities)),
        std=float(np.std(similarities)),
        n=len(similarities),
    )
