"""What passage length alone predicts about a pair, the reference a model's AP is read against."""

from collections.abc import Callable

import numpy as np

from genre.evaluate import GenreEvaluationReport, report_from_similarities
from genre.pairs import GenrePair
from genre.passages import Passage

#: Each predictor ranks pairs as a similarity would: a higher value claims a same-genre pair.
LENGTH_PREDICTORS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "shorter_side": np.minimum,
    "length_agreement": lambda a, b: -np.abs(a - b),
}


def pair_lengths(pairs: list[GenrePair], passages: list[Passage]) -> tuple[np.ndarray, np.ndarray]:
    """The word count on each side of every pair."""
    length_of = {passage.id: len(passage.words) for passage in passages}
    lengths_a = np.fromiter((length_of[p.item_a] for p in pairs), dtype=np.int64, count=len(pairs))
    lengths_b = np.fromiter((length_of[p.item_b] for p in pairs), dtype=np.int64, count=len(pairs))
    return lengths_a, lengths_b


def _row(predictor: str, report: GenreEvaluationReport) -> dict[str, str | int | float]:
    """One baseline row in the shape of a model's summary row, named for the predictor."""
    return {
        "predictor": predictor,
        "n_same_genre": report.n_same_genre,
        "n_different_genre": report.n_different_genre,
        "prevalence": report.prevalence,
        "average_precision": report.average_precision,
        "separation_auc": report.separation_auc,
        "separation_p": report.separation_p,
    }


def length_baseline_rows(
    pairs: list[GenrePair], passages: list[Passage]
) -> list[dict[str, str | int | float]]:
    """AP and AUC of each length predictor over the pairs, under the protocol the models face."""
    lengths_a, lengths_b = pair_lengths(pairs, passages)
    return [
        _row(name, report_from_similarities(pairs, predictor(lengths_a, lengths_b).astype(float)))
        for name, predictor in LENGTH_PREDICTORS.items()
    ]
