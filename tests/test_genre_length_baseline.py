"""The length baseline: what passage length alone predicts about a pair's genre."""

import math

import numpy as np

from genre.length_baseline import LENGTH_PREDICTORS, length_baseline_rows, pair_lengths
from genre.pairs import build_genre_pairs
from genre.passages import Passage

#: Two long same-genre songs, two short different-genre motifs, so length tracks genre.
PASSAGES = [
    Passage("1:1-6:Hymnus", 1, "Hymnus", (10, 11, 12, 13, 14, 15), "1:1-6:Hymnus"),
    Passage("2:1-6:Hymnus", 2, "Hymnus", (20, 21, 22, 23, 24, 25), "2:1-6:Hymnus"),
    Passage("3:2-2:Fluch", 3, "Fluch", (32,), "3:2-2:Fluch"),
    Passage("4:5-5:Segensspruch", 4, "Segensspruch", (45,), "4:5-5:Segensspruch"),
]


def test_pair_lengths_read_each_sides_half_verse_count() -> None:
    pairs = build_genre_pairs(PASSAGES)
    lengths_a, lengths_b = pair_lengths(pairs, PASSAGES)

    assert lengths_a.tolist() == [6, 6, 6, 6, 6, 1]
    assert lengths_b.tolist() == [6, 1, 1, 1, 1, 1]


def test_reports_one_row_per_predictor_on_the_same_pairs_a_model_is_scored_on() -> None:
    rows = length_baseline_rows(build_genre_pairs(PASSAGES), PASSAGES)

    assert [row["predictor"] for row in rows] == list(LENGTH_PREDICTORS)
    assert all(row["n_same_genre"] == 1 and row["n_different_genre"] == 5 for row in rows)
    assert all(row["prevalence"] == 1 / 6 for row in rows)


def test_a_predictor_that_tracks_genre_scores_high_and_one_that_does_not_stays_lower() -> None:
    """Here the only same-genre pair is the longest, so the shorter side ranks it first."""
    by_name = {
        row["predictor"]: row for row in length_baseline_rows(build_genre_pairs(PASSAGES), PASSAGES)
    }

    assert by_name["shorter_side"]["average_precision"] == 1.0
    assert by_name["shorter_side"]["separation_auc"] == 1.0
    #: Length difference is zero for the same-genre pair and for the motif pair alike.
    assert by_name["length_agreement"]["separation_auc"] < 1.0


def test_an_empty_pair_set_yields_nan_rows_rather_than_a_crash() -> None:
    rows = length_baseline_rows([], PASSAGES)

    assert len(rows) == len(LENGTH_PREDICTORS)
    assert all(math.isnan(row["average_precision"]) for row in rows)
    assert all(isinstance(row["n_same_genre"], int) for row in rows)


def test_rows_are_json_and_csv_safe_floats() -> None:
    for row in length_baseline_rows(build_genre_pairs(PASSAGES), PASSAGES):
        assert all(not isinstance(v, np.generic) for v in row.values())
