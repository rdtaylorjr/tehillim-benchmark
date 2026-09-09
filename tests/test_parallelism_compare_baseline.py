import numpy as np
import pytest


def test_an_empty_side_is_reported_as_insufficient_data_not_a_crash() -> None:
    """Every other scorer raises a typed error here, so the skip policy can catch this one too."""
    from library.calibration import BackgroundStats
    from library.errors import InsufficientDataError
    from parallelism.baseline_comparison import compare_to_baseline_from_similarities

    background = BackgroundStats(mean=0.0, std=1.0, n_vectors=10)

    with pytest.raises(InsufficientDataError):
        compare_to_baseline_from_similarities(np.array([0.5, 0.6]), np.array([]), background)
    with pytest.raises(InsufficientDataError):
        compare_to_baseline_from_similarities(np.array([]), np.array([0.5]), background)
