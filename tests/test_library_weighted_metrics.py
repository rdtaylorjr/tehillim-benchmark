"""The one-sort weighted statistics agree with the per-split fast path on expanded arrays."""

import numpy as np
import pytest

from library.ap_gap_auc_bootstrap import resample_ap_gap_and_auc
from library.calibration import BackgroundStats
from library.weighted_metrics import sort_pairs, weighted_ap_gap_auc

BACKGROUND = BackgroundStats(mean=0.1, std=0.3, n_vectors=50)


def _expanded(scores: np.ndarray, positive: np.ndarray, weights: np.ndarray) -> tuple:
    """The split a weighting stands for: each pair repeated as many times as its weight."""
    counts = weights.astype(int)
    repeated = np.repeat(scores, counts)
    label = np.repeat(positive, counts)
    return repeated[label], repeated[~label]


def _pairs(rng: np.random.Generator, n: int, *, tied: bool) -> tuple[np.ndarray, np.ndarray]:
    scores = rng.normal(size=n)
    if tied:
        scores = np.round(scores, 1)
    positive = rng.random(n) < 0.3
    return scores, positive


@pytest.mark.parametrize("tied", [False, True])
def test_matches_the_fast_path_on_expanded_arrays_for_integer_weights(tied: bool) -> None:
    rng = np.random.default_rng(3)
    scores, positive = _pairs(rng, 300, tied=tied)
    weights = rng.integers(0, 4, size=(20, 300)).astype(float)
    result = weighted_ap_gap_auc(sort_pairs(scores, positive), weights, BACKGROUND)
    for i in range(20):
        pos, neg = _expanded(scores, positive, weights[i])
        ap, gap, auc = resample_ap_gap_and_auc(pos, neg, BACKGROUND)
        assert result.ap[i] == pytest.approx(ap, abs=1e-12)
        assert result.gap[i] == pytest.approx(gap, abs=1e-12)
        assert result.auc[i] == pytest.approx(auc, abs=1e-12)
        assert result.positive_weight[i] == len(pos)
        assert result.negative_weight[i] == len(neg)


def test_unit_weights_reproduce_the_observed_split() -> None:
    rng = np.random.default_rng(5)
    scores, positive = _pairs(rng, 100, tied=True)
    result = weighted_ap_gap_auc(sort_pairs(scores, positive), np.ones((1, 100)), BACKGROUND)
    ap, gap, auc = resample_ap_gap_and_auc(scores[positive], scores[~positive], BACKGROUND)
    assert result.ap[0] == pytest.approx(ap)
    assert result.gap[0] == pytest.approx(gap)
    assert result.auc[0] == pytest.approx(auc)


def test_a_side_with_no_weight_yields_nan_not_an_error() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    positive = np.array([True, True, False, False])
    weights = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0]])
    result = weighted_ap_gap_auc(sort_pairs(scores, positive), weights, BACKGROUND)
    assert np.isnan(result.auc).all()
    assert result.negative_weight[0] == 0
    assert result.positive_weight[1] == 0


def test_blocks_of_draws_concatenate_in_order() -> None:
    rng = np.random.default_rng(9)
    scores, positive = _pairs(rng, 5000, tied=False)
    weights = rng.integers(0, 3, size=(2000, 5000)).astype(float)
    pairs = sort_pairs(scores, positive)
    whole = weighted_ap_gap_auc(pairs, weights, BACKGROUND)
    halves = [
        weighted_ap_gap_auc(pairs, weights[span], BACKGROUND)
        for span in (slice(0, 1000), slice(1000, 2000))
    ]
    assert np.array_equal(whole.auc, np.concatenate([h.auc for h in halves]))
    assert np.array_equal(whole.ap, np.concatenate([h.ap for h in halves]))


def test_no_draws_yields_empty_statistics() -> None:
    pairs = sort_pairs(np.array([0.5, 0.2]), np.array([True, False]))
    result = weighted_ap_gap_auc(pairs, np.empty((0, 2)), BACKGROUND)
    assert result.ap.shape == (0,)
