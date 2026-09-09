"""BCa bootstrap CI (Efron 1987) for AP, calibrated gap, and AUC, over any resampling scheme."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from library.bootstrap_ci import bca_ci
from library.calibration import BackgroundStats, calibrated_effect_size
from library.errors import InsufficientDataError
from library.fast_metrics import fast_auc, fast_average_precision

# AP, AUC, and the jackknife acceleration are all undefined below two values on a side.
MIN_PER_SIDE = 2

Split = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class ApGapAucCI:
    """BCa intervals for average precision, gap and AUC, with the resamples behind them."""

    point_ap: float
    ap_ci_low: float
    ap_ci_high: float
    ap_ci_low_pct: float
    ap_ci_high_pct: float
    point_gap: float
    gap_ci_low: float
    gap_ci_high: float
    gap_ci_low_pct: float
    gap_ci_high_pct: float
    point_auc: float
    auc_ci_low: float
    auc_ci_high: float
    auc_ci_low_pct: float
    auc_ci_high_pct: float
    prevalence: float
    n_valid_resamples: int
    n_valid_jackknife: int


def calibrated_gap(
    positive: np.ndarray, negative: np.ndarray, background: BackgroundStats
) -> float:
    """Difference in calibrated effect size between the positive and negative similarity groups."""
    return calibrated_effect_size(float(positive.mean()), background) - calibrated_effect_size(
        float(negative.mean()), background
    )


def point_ap_gap_and_auc(
    positive: np.ndarray, negative: np.ndarray, background: BackgroundStats
) -> tuple[float, float, float]:
    """Point estimate via official sklearn/scipy; resampling and jackknife use the fast path."""
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    scores = np.concatenate([positive, negative])
    ap = float(average_precision_score(labels, scores))
    statistic, _ = mannwhitneyu(positive, negative, alternative="greater")
    auc = float(statistic / (len(positive) * len(negative)))
    return ap, calibrated_gap(positive, negative, background), auc


def resample_ap_gap_and_auc(
    positive: np.ndarray, negative: np.ndarray, background: BackgroundStats
) -> tuple[float, float, float]:
    """The same three statistics as point_ap_gap_and_auc, via the verified vectorized fast path."""
    ap = fast_average_precision(positive, negative)
    auc = fast_auc(positive, negative)
    return ap, calibrated_gap(positive, negative, background), auc


def has_enough_per_side(split: Split) -> bool:
    """Whether both sides of a split carry the MIN_PER_SIDE values the statistics need."""
    return len(split[0]) >= MIN_PER_SIDE and len(split[1]) >= MIN_PER_SIDE


def _nan_result(point: tuple[float, float, float], prevalence: float) -> ApGapAucCI:
    """Point estimates with every interval NaN, for a scheme that yielded no usable resample."""
    point_ap, point_gap, point_auc = point
    nan = float("nan")
    return ApGapAucCI(
        point_ap=point_ap,
        ap_ci_low=nan,
        ap_ci_high=nan,
        ap_ci_low_pct=nan,
        ap_ci_high_pct=nan,
        point_gap=point_gap,
        gap_ci_low=nan,
        gap_ci_high=nan,
        gap_ci_low_pct=nan,
        gap_ci_high_pct=nan,
        point_auc=point_auc,
        auc_ci_low=nan,
        auc_ci_high=nan,
        auc_ci_low_pct=nan,
        auc_ci_high_pct=nan,
        prevalence=prevalence,
        n_valid_resamples=0,
        n_valid_jackknife=0,
    )


def jackknife_statistics(
    jackknife: Iterable[Split | None], background: BackgroundStats
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-cluster-out AP/gap/AUC, NaN wherever the reduced split is too small to score."""
    rows = [
        resample_ap_gap_and_auc(split[0], split[1], background)
        if split is not None and has_enough_per_side(split)
        else (np.nan, np.nan, np.nan)
        for split in jackknife
    ]
    if not rows:
        return np.empty(0), np.empty(0), np.empty(0)
    columns = np.array(rows, dtype=float)
    return columns[:, 0], columns[:, 1], columns[:, 2]


def bootstrap_ap_gap_and_auc(
    observed: Split,
    resamples: Iterable[Split],
    jackknife: Iterable[Split | None],
    background: BackgroundStats,
) -> ApGapAucCI:
    """BCa 95% CI for AP (primary), gap, and AUC from a caller-supplied resampling scheme."""
    if not has_enough_per_side(observed):
        raise InsufficientDataError(
            f"AP and AUC need at least {MIN_PER_SIDE} values on each side, got "
            f"{len(observed[0])} positive and {len(observed[1])} negative"
        )
    prevalence = len(observed[0]) / (len(observed[0]) + len(observed[1]))
    point = point_ap_gap_and_auc(observed[0], observed[1], background)

    valid = [
        resample_ap_gap_and_auc(split[0], split[1], background)
        for split in resamples
        if has_enough_per_side(split)
    ]
    if not valid:
        return _nan_result(point, prevalence)

    ap_arr, gap_arr, auc_arr = (np.array(column) for column in zip(*valid, strict=True))
    ap_low_pct, ap_high_pct = np.percentile(ap_arr, [2.5, 97.5])
    gap_low_pct, gap_high_pct = np.percentile(gap_arr, [2.5, 97.5])
    auc_low_pct, auc_high_pct = np.percentile(auc_arr, [2.5, 97.5])

    jack_aps, jack_gaps, jack_aucs = jackknife_statistics(jackknife, background)
    ap_low, ap_high = bca_ci(point[0], ap_arr, jack_aps)
    gap_low, gap_high = bca_ci(point[1], gap_arr, jack_gaps)
    auc_low, auc_high = bca_ci(point[2], auc_arr, jack_aucs)

    return ApGapAucCI(
        point_ap=point[0],
        ap_ci_low=ap_low,
        ap_ci_high=ap_high,
        ap_ci_low_pct=float(ap_low_pct),
        ap_ci_high_pct=float(ap_high_pct),
        point_gap=point[1],
        gap_ci_low=gap_low,
        gap_ci_high=gap_high,
        gap_ci_low_pct=float(gap_low_pct),
        gap_ci_high_pct=float(gap_high_pct),
        point_auc=point[2],
        auc_ci_low=auc_low,
        auc_ci_high=auc_high,
        auc_ci_low_pct=float(auc_low_pct),
        auc_ci_high_pct=float(auc_high_pct),
        prevalence=prevalence,
        n_valid_resamples=len(valid),
        n_valid_jackknife=int(np.sum(~np.isnan(jack_aps))),
    )


def ci_row(
    model: str, result: ApGapAucCI, scope: str | None = None
) -> dict[str, str | int | float]:
    """One CSV row for a CI result; column order is the header, so it is fixed here."""
    row: dict[str, str | int | float] = {"model": model}
    if scope is not None:
        row["scope"] = scope
    row["prevalence"] = result.prevalence
    for name in ApGapAucCI.__dataclass_fields__:
        if name != "prevalence":
            row[name] = getattr(result, name)
    return row
