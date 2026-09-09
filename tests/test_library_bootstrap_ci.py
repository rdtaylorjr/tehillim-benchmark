import warnings

import numpy as np
import pytest

from library.bootstrap_ci import bca_ci


def test_bca_ci_reduces_to_percentile_when_unbiased_and_unaccelerated() -> None:
    """With z0=0 and a=0 the interval must reproduce the plain percentile interval."""
    bootstrap_values = np.concatenate(
        [-np.arange(1, 5001, dtype=float), np.arange(1, 5001, dtype=float)]
    )
    jackknife_values = np.full(60, 7.0)

    lo, hi = bca_ci(0.0, bootstrap_values, jackknife_values)
    perc_lo, perc_hi = np.percentile(bootstrap_values, [2.5, 97.5])

    assert lo == pytest.approx(perc_lo)
    assert hi == pytest.approx(perc_hi)


def test_bca_ci_shifts_when_point_is_biased_relative_to_bootstrap_distribution() -> None:
    bootstrap_values = np.concatenate(
        [-np.arange(1, 5001, dtype=float), np.arange(1, 5001, dtype=float)]
    )
    jackknife_values = np.full(60, 7.0)

    lo, hi = bca_ci(4999.0, bootstrap_values, jackknife_values)
    perc_lo, perc_hi = np.percentile(bootstrap_values, [2.5, 97.5])

    assert (lo, hi) != pytest.approx((perc_lo, perc_hi))


def test_bca_ci_falls_back_to_the_point_for_a_zero_variance_bootstrap_distribution() -> None:
    bootstrap_values = np.full(50, 0.42)
    jackknife_values = np.full(10, 0.1)

    lo, hi = bca_ci(0.42, bootstrap_values, jackknife_values)

    assert lo == pytest.approx(0.42)
    assert hi == pytest.approx(0.42)


def _skewed_bootstrap_values() -> np.ndarray:
    return np.random.default_rng(0).gamma(shape=2.0, scale=1.0, size=4000)


def test_bca_ci_applies_a_nonzero_acceleration_without_warning_for_a_skewed_jackknife() -> None:
    bootstrap_values = _skewed_bootstrap_values()
    jackknife_values = np.array([1.0, 1.1, 1.2, 1.3, 9.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        accelerated = bca_ci(2.0, bootstrap_values, jackknife_values)

    unaccelerated = bca_ci(2.0, bootstrap_values, np.full(5, 3.0))
    assert caught == []
    assert accelerated != pytest.approx(unaccelerated)


def test_bca_ci_warns_when_too_few_jackknife_values_are_valid_to_accelerate() -> None:
    bootstrap_values = _skewed_bootstrap_values()
    jackknife_values = np.array([np.nan, np.nan, 1.0])

    with pytest.warns(RuntimeWarning, match="bias-corrected only"):
        degraded = bca_ci(2.0, bootstrap_values, jackknife_values)

    assert degraded == pytest.approx(bca_ci(2.0, bootstrap_values, np.full(5, 3.0)))
