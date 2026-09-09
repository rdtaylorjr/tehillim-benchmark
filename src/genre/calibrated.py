"""Same-genre against different-genre similarity, calibrated on the whole-population background."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from genre.evaluate import pair_similarities
from genre.pairs import GenrePair
from library.calibration import BackgroundStats, calibrated_effect_size


@dataclass(frozen=True, slots=True)
class GenreCalibratedComparison:
    """Same-genre against different-genre similarity, with each group's calibrated effect size."""

    n_same_genre: int
    n_different_genre: int
    prevalence: float
    mean_same_genre_similarity: float
    mean_different_genre_similarity: float
    same_genre_effect_size: float
    different_genre_effect_size: float
    average_precision: float
    separation_auc: float
    separation_p: float


def compare_genre_calibrated(
    pairs: list[GenrePair], psalm_vectors: dict[int, np.ndarray], background: BackgroundStats
) -> GenreCalibratedComparison:
    """Same-genre vs different-genre similarity: AP/AUC plus each group's calibrated effect size."""
    usable, similarities = pair_similarities(pairs, psalm_vectors)
    labels = np.array([p.same_genre for p in usable], dtype=int)

    same_sims = similarities[labels == 1]
    different_sims = similarities[labels == 0]

    statistic, p_value = mannwhitneyu(same_sims, different_sims, alternative="greater")
    auc = statistic / (len(same_sims) * len(different_sims))
    ap = average_precision_score(labels, similarities)
    mean_same = float(same_sims.mean())
    mean_different = float(different_sims.mean())

    return GenreCalibratedComparison(
        n_same_genre=len(same_sims),
        n_different_genre=len(different_sims),
        prevalence=len(same_sims) / len(usable),
        mean_same_genre_similarity=mean_same,
        mean_different_genre_similarity=mean_different,
        same_genre_effect_size=calibrated_effect_size(mean_same, background),
        different_genre_effect_size=calibrated_effect_size(mean_different, background),
        average_precision=float(ap),
        separation_auc=float(auc),
        separation_p=float(p_value),
    )


def genre_calibrated_row(
    model: str, result: GenreCalibratedComparison
) -> dict[str, str | int | float]:
    """The published calibrated row for one model, with its gap derived in one place."""
    return {
        "model": model,
        "n_same_genre": result.n_same_genre,
        "n_different_genre": result.n_different_genre,
        "prevalence": result.prevalence,
        "average_precision": result.average_precision,
        "same_genre_effect_size": result.same_genre_effect_size,
        "different_genre_effect_size": result.different_genre_effect_size,
        "gap": result.same_genre_effect_size - result.different_genre_effect_size,
        "separation_auc": result.separation_auc,
        "separation_p": result.separation_p,
    }
