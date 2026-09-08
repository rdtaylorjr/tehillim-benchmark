"""A separation AUC dominated by one tie block is decided by rounding, so report the tie mass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from library.tie_diagnostics import (
    TIE_FRACTION_METRIC,
    max_tie_fraction,
    tie_fractions_by_model,
)


class TestMaxTieFraction:
    def test_all_distinct_values_score_the_minimum(self) -> None:
        assert max_tie_fraction(np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(0.25)

    def test_one_value_shared_by_every_pair_scores_one(self) -> None:
        """The degenerate case: the statistic has nothing left to rank."""
        assert max_tie_fraction(np.array([0.5, 0.5, 0.5, 0.5])) == 1.0

    def test_it_reports_the_largest_block_not_the_count_of_blocks(self) -> None:
        assert max_tie_fraction(np.array([0.5, 0.5, 0.5, 0.1, 0.2])) == pytest.approx(0.6)

    def test_exact_zeros_from_disjoint_vocabulary_still_count(self) -> None:
        """Sparse lexical pairs genuinely tie at zero, and that mass still drives the AUC."""
        assert max_tie_fraction(np.array([0.0, 0.0, 0.0, 0.4])) == pytest.approx(0.75)

    def test_an_empty_input_has_no_tie_fraction(self) -> None:
        assert np.isnan(max_tie_fraction(np.array([])))

    def test_nan_similarities_are_not_counted_as_a_tie_block(self) -> None:
        """A NaN is missing data, not a shared value."""
        assert max_tie_fraction(np.array([np.nan, np.nan, 0.1, 0.2])) == pytest.approx(0.5)


class TestTieFractionsByModel:
    def _pair_detail(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": ["a", "a", "a", "a", "b", "b", "b", "b"],
                "raw_similarity": [0.5, 0.5, 0.5, 0.1, 0.1, 0.2, 0.3, 0.4],
            }
        )

    def test_one_row_per_model(self) -> None:
        out = tie_fractions_by_model(self._pair_detail())

        assert sorted(out["model"]) == ["a", "b"]

    def test_it_scores_each_model_independently(self) -> None:
        out = tie_fractions_by_model(self._pair_detail()).set_index("model")

        assert out.loc["a", "value"] == pytest.approx(0.75)
        assert out.loc["b", "value"] == pytest.approx(0.25)

    def test_it_emits_the_long_schema_the_master_report_expects(self) -> None:
        out = tie_fractions_by_model(self._pair_detail())

        assert {"model", "scope", "scope_kind", "source", "metric", "value"} <= set(out.columns)
        assert set(out["metric"]) == {TIE_FRACTION_METRIC}
        assert set(out["scope"]) == {"overall"}

    def test_an_empty_frame_yields_no_rows(self) -> None:
        empty = pd.DataFrame({"model": [], "raw_similarity": []})

        assert len(tie_fractions_by_model(empty)) == 0


class TestBothMasterReportsCarryIt:
    """Genre scores the same similarities as parallelism, so it reports the same caveat."""

    def _frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        from genre.scripts.build_master_report import _BOOTSTRAP_METRICS, _SUMMARY_METRICS

        summary = pd.DataFrame({"model": ["a"], **{m: [0.5] for m in _SUMMARY_METRICS}})
        bootstrap = pd.DataFrame({"model": ["a"], **{m: [0.5] for m in _BOOTSTRAP_METRICS}})
        return summary, bootstrap

    def test_genre_build_long_metrics_accepts_pair_detail(self) -> None:
        from genre.scripts.build_master_report import build_long_metrics

        summary, bootstrap = self._frames()
        pair_detail = pd.DataFrame(
            {"model": ["a", "a", "a", "a"], "raw_similarity": [0.3, 0.3, 0.3, 0.9]}
        )

        out = build_long_metrics(summary, bootstrap, pair_detail)

        tie = out[out["metric"] == TIE_FRACTION_METRIC]
        assert len(tie) == 1
        assert tie["value"].iloc[0] == pytest.approx(0.75)

    def test_genre_without_pair_detail_omits_the_metric(self) -> None:
        from genre.scripts.build_master_report import build_long_metrics

        summary, bootstrap = self._frames()

        out = build_long_metrics(summary, bootstrap)

        assert TIE_FRACTION_METRIC not in set(out["metric"])
