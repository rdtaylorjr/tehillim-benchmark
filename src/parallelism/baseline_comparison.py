"""True parallel pairs against the unmarked baseline, calibrated on a shared background."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from library.calibration import BackgroundStats, calibrated_effect_size
from library.errors import InsufficientDataError
from parallelism.node_pairs import NodePairs, pair_similarities


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """True parallel pairs against the unmarked baseline, calibrated on one background."""

    n_true: int
    n_baseline: int
    prevalence: float
    mean_true_similarity: float
    mean_baseline_similarity: float
    true_effect_size: float
    baseline_effect_size: float
    average_precision: float
    separation_auc: float
    separation_p: float


def compare_to_baseline(
    true_pairs: NodePairs,
    baseline_pairs: NodePairs,
    node_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> BaselineComparison:
    """True-pair vs baseline similarity: Average Precision is primary, AUC/effect size secondary."""
    return compare_to_baseline_from_similarities(
        pair_similarities(true_pairs, node_vectors),
        pair_similarities(baseline_pairs, node_vectors),
        background,
    )


def compare_to_baseline_from_similarities(
    true_sims: np.ndarray, baseline_sims: np.ndarray, background: BackgroundStats
) -> BaselineComparison:
    """Same comparison as compare_to_baseline, from already-computed pair similarities."""
    if len(true_sims) == 0 or len(baseline_sims) == 0:
        raise InsufficientDataError(
            f"a baseline comparison needs pairs on both sides, got "
            f"{len(true_sims)} true and {len(baseline_sims)} baseline"
        )
    statistic, p_value = mannwhitneyu(true_sims, baseline_sims, alternative="greater")
    auc = statistic / (len(true_sims) * len(baseline_sims))
    mean_true = float(true_sims.mean())
    mean_baseline = float(baseline_sims.mean())

    labels = np.concatenate([np.ones(len(true_sims)), np.zeros(len(baseline_sims))])
    scores = np.concatenate([true_sims, baseline_sims])
    ap = average_precision_score(labels, scores)

    return BaselineComparison(
        n_true=len(true_sims),
        n_baseline=len(baseline_sims),
        prevalence=len(true_sims) / (len(true_sims) + len(baseline_sims)),
        mean_true_similarity=mean_true,
        mean_baseline_similarity=mean_baseline,
        true_effect_size=calibrated_effect_size(mean_true, background),
        baseline_effect_size=calibrated_effect_size(mean_baseline, background),
        average_precision=float(ap),
        separation_auc=float(auc),
        separation_p=float(p_value),
    )


def baseline_metric_fields(result: BaselineComparison) -> dict[str, float]:
    """The metrics both the summary row and the detail row carry, with gap derived once."""
    return {
        "n_true": result.n_true,
        "n_baseline": result.n_baseline,
        "prevalence": result.prevalence,
        "average_precision": result.average_precision,
        "true_effect_size": result.true_effect_size,
        "baseline_effect_size": result.baseline_effect_size,
        "gap": result.true_effect_size - result.baseline_effect_size,
    }
