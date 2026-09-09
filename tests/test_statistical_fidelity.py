"""Each statistic against its published definition, transcribed independently."""

import numpy as np
from scipy.stats import mannwhitneyu, norm

from library.bootstrap_ci import bca_ci
from library.fast_metrics import fast_auc
from library.multiple_comparisons import benjamini_hochberg, benjamini_yekutieli


def _bh_from_the_1995_paper(p_values: np.ndarray) -> np.ndarray:
    """Benjamini and Hochberg (1995) step-up, transcribed from the definition."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adjusted[i] = running
    result = np.empty(n)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def _by_from_the_2001_paper(p_values: np.ndarray) -> np.ndarray:
    """Benjamini and Yekutieli (2001): the BH threshold scaled by the harmonic number."""
    harmonic = np.sum(1.0 / np.arange(1, len(p_values) + 1))
    return np.clip(_bh_from_the_1995_paper(p_values) * harmonic, 0, 1)


def test_benjamini_hochberg_matches_the_published_step_up() -> None:
    rng = np.random.default_rng(0)

    for _ in range(200):
        p = rng.random(int(rng.integers(2, 40)))
        np.testing.assert_allclose(
            benjamini_hochberg(p), _bh_from_the_1995_paper(p), rtol=0, atol=1e-12
        )


def test_benjamini_yekutieli_matches_the_published_scaling() -> None:
    rng = np.random.default_rng(1)

    for _ in range(200):
        p = rng.random(int(rng.integers(2, 40)))
        np.testing.assert_allclose(
            benjamini_yekutieli(p), _by_from_the_2001_paper(p), rtol=0, atol=1e-12
        )


def test_auc_equals_the_mann_whitney_statistic_over_the_pair_count() -> None:
    """AUC is U/(n1*n2) by definition, which is what makes it a rank statistic."""
    rng = np.random.default_rng(2)

    for _ in range(200):
        a, b = rng.random(int(rng.integers(2, 50))), rng.random(int(rng.integers(2, 50)))
        statistic = mannwhitneyu(a, b, alternative="greater").statistic

        assert fast_auc(a, b) == statistic / (len(a) * len(b))


def _bca_from_efron_1987(point, bootstrap, jackknife, alpha=0.05):
    """Efron (1987) equation 2.2, transcribed from the paper."""
    n = len(bootstrap)
    proportion = np.clip(np.mean(bootstrap < point), 1 / (n + 1), n / (n + 1))
    bias = norm.ppf(proportion)
    centred = jackknife.mean() - jackknife
    denominator = 6.0 * (np.sum(centred**2) ** 1.5)
    acceleration = np.sum(centred**3) / denominator if denominator != 0 else 0.0
    low, high = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    lower = norm.cdf(bias + (bias + low) / (1 - acceleration * (bias + low)))
    upper = norm.cdf(bias + (bias + high) / (1 - acceleration * (bias + high)))
    return tuple(np.percentile(bootstrap, [100 * lower, 100 * upper]))


def test_bca_interval_matches_efron_1987() -> None:
    rng = np.random.default_rng(3)

    for _ in range(100):
        bootstrap = rng.normal(0.5, 0.1, 400)
        jackknife = rng.normal(0.5, 0.05, 30)

        np.testing.assert_allclose(
            bca_ci(0.5, bootstrap, jackknife),
            _bca_from_efron_1987(0.5, bootstrap, jackknife),
            rtol=0,
            atol=1e-12,
        )
