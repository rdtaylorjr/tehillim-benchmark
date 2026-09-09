import numpy as np
import pytest


def test_summarising_no_similarities_raises_rather_than_warning() -> None:
    """Its siblings raise a typed error for an empty side, and np.mean would warn instead."""
    from library.errors import InsufficientDataError
    from parallelism.true_similarity import summarize_true_pair_similarity

    with pytest.raises(InsufficientDataError):
        summarize_true_pair_similarity(np.array([]))
