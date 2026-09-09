import numpy as np
import pytest
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from library.fast_metrics import fast_auc, fast_average_precision


def _random_scores(rng: np.random.Generator, n_true: int, n_baseline: int) -> tuple:
    true_sims = rng.normal(0.6, 0.1, size=n_true)
    baseline_sims = rng.normal(0.5, 0.1, size=n_baseline)
    return true_sims, baseline_sims


def _sklearn_ap(true_sims: np.ndarray, baseline_sims: np.ndarray) -> float:
    labels = np.concatenate([np.ones(len(true_sims)), np.zeros(len(baseline_sims))])
    scores = np.concatenate([true_sims, baseline_sims])
    return float(average_precision_score(labels, scores))


def _scipy_auc(true_sims: np.ndarray, baseline_sims: np.ndarray) -> float:
    statistic, _ = mannwhitneyu(true_sims, baseline_sims, alternative="greater")
    return float(statistic / (len(true_sims) * len(baseline_sims)))


@pytest.mark.parametrize("seed", range(30))
def test_fast_average_precision_matches_sklearn_on_random_continuous_scores(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_true = rng.integers(2, 50)
    n_baseline = rng.integers(2, 200)
    true_sims, baseline_sims = _random_scores(rng, n_true, n_baseline)

    assert fast_average_precision(true_sims, baseline_sims) == pytest.approx(
        _sklearn_ap(true_sims, baseline_sims), abs=1e-9
    )


@pytest.mark.parametrize("seed", range(30))
def test_fast_auc_matches_scipy_mannwhitneyu_on_random_continuous_scores(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_true = rng.integers(2, 50)
    n_baseline = rng.integers(2, 200)
    true_sims, baseline_sims = _random_scores(rng, n_true, n_baseline)

    assert fast_auc(true_sims, baseline_sims) == pytest.approx(
        _scipy_auc(true_sims, baseline_sims), abs=1e-9
    )


def test_fast_average_precision_matches_sklearn_with_heavy_ties() -> None:
    """Real cosine similarities tie exactly, so the fast path must match sklearn under ties."""
    rng = np.random.default_rng(0)
    true_sims = rng.choice([0.3, 0.5, 0.5, 0.7, 0.9], size=40)
    baseline_sims = rng.choice([0.3, 0.5, 0.5, 0.7, 0.9], size=60)

    assert fast_average_precision(true_sims, baseline_sims) == pytest.approx(
        _sklearn_ap(true_sims, baseline_sims), abs=1e-9
    )


def test_fast_auc_matches_scipy_with_heavy_ties() -> None:
    rng = np.random.default_rng(0)
    true_sims = rng.choice([0.3, 0.5, 0.5, 0.7, 0.9], size=40)
    baseline_sims = rng.choice([0.3, 0.5, 0.5, 0.7, 0.9], size=60)

    assert fast_auc(true_sims, baseline_sims) == pytest.approx(
        _scipy_auc(true_sims, baseline_sims), abs=1e-9
    )


def test_fast_average_precision_perfect_separation_is_one() -> None:
    true_sims = np.array([0.9, 0.8])
    baseline_sims = np.array([0.2, 0.1])
    assert fast_average_precision(true_sims, baseline_sims) == pytest.approx(1.0)


def test_fast_average_precision_chance_level_is_prevalence() -> None:
    true_sims = np.array([1.0])
    baseline_sims = np.array([1.0, 1.0, 1.0])
    assert fast_average_precision(true_sims, baseline_sims) == pytest.approx(0.25)


def test_fast_auc_perfect_separation_is_one() -> None:
    true_sims = np.array([0.9, 0.8])
    baseline_sims = np.array([0.2, 0.1])
    assert fast_auc(true_sims, baseline_sims) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(20))
def test_fast_metrics_match_at_full_corpus_scale_with_rounded_tie_prone_scores(seed: int) -> None:
    """Corpus-scale n (hundreds to thousands) with 3-decimal rounding."""
    rng = np.random.default_rng(seed)
    n_true = rng.integers(20, 1200)
    n_baseline = rng.integers(500, 3500)
    true_sims = np.round(rng.normal(0.6, 0.15, size=n_true), 3)
    baseline_sims = np.round(rng.normal(0.5, 0.15, size=n_baseline), 3)

    assert fast_average_precision(true_sims, baseline_sims) == pytest.approx(
        _sklearn_ap(true_sims, baseline_sims), abs=1e-9
    )
    assert fast_auc(true_sims, baseline_sims) == pytest.approx(
        _scipy_auc(true_sims, baseline_sims), abs=1e-9
    )
