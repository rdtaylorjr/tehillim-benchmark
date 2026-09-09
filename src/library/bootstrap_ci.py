"""Efron (1987) BCa confidence interval from a bootstrap distribution plus a jackknife."""

import warnings

import numpy as np
from scipy.stats import norm

from library.protocol import DEFAULT_ALPHA

# Acceleration is a jackknife skewness estimate, undefined below two values, where BCa decays to BC.
MIN_JACKKNIFE_FOR_ACCELERATION = 2
# Below this spread every resample landed on one value, leaving no distribution to quantile.
_ZERO_VARIANCE_RANGE = 1e-12


def bca_ci(
    point: float,
    bootstrap_values: np.ndarray,
    jackknife_values: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Efron (1987) BCa interval; falls back to the point itself for a zero-variance bootstrap."""
    if np.ptp(bootstrap_values) < _ZERO_VARIANCE_RANGE:
        return float(bootstrap_values[0]), float(bootstrap_values[0])

    n_boot = len(bootstrap_values)
    prop_less = np.mean(bootstrap_values < point)
    prop_less = np.clip(prop_less, 1.0 / (n_boot + 1), n_boot / (n_boot + 1))
    z0 = norm.ppf(prop_less)

    valid_jack = jackknife_values[~np.isnan(jackknife_values)]
    if len(valid_jack) < MIN_JACKKNIFE_FOR_ACCELERATION:
        warnings.warn(
            f"BCa acceleration needs at least {MIN_JACKKNIFE_FOR_ACCELERATION} valid jackknife "
            f"values and got {len(valid_jack)}, so the returned interval is bias-corrected only.",
            RuntimeWarning,
            stacklevel=2,
        )
        a = 0.0
    else:
        jack_mean = valid_jack.mean()
        numerator = np.sum((jack_mean - valid_jack) ** 3)
        denominator = 6.0 * (np.sum((jack_mean - valid_jack) ** 2) ** 1.5)
        a = numerator / denominator if denominator != 0 else 0.0

    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    alpha1 = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    alpha2 = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

    lo, hi = np.percentile(bootstrap_values, [100 * alpha1, 100 * alpha2])
    return float(lo), float(hi)
