import pandas as pd

from library.master_report import (
    LONG_COLUMNS,
    finalise_long_metrics,
    melt_to_long,
    pivot_metrics_wide,
)


def test_melt_to_long_tags_every_row_with_its_scope_and_source() -> None:
    frame = pd.DataFrame({"model": ["a", "b"], "average_precision": [0.5, 0.6], "ignored": [1, 2]})

    long = melt_to_long(frame, ["average_precision"], "genre_discrimination")

    assert list(long["metric"]) == ["average_precision"] * 2
    assert set(long["scope"]) == {"overall"}
    assert set(long["source"]) == {"genre_discrimination"}
    assert "ignored" not in long["metric"].to_numpy()


def test_melt_to_long_carries_a_named_scope_for_a_per_type_breakdown() -> None:
    frame = pd.DataFrame({"model": ["a"], "n_pairs": [10]})

    long = melt_to_long(frame, ["n_pairs"], "retrieval", scope="Synonymous", scope_kind="type")

    assert list(long["scope"]) == ["Synonymous"]
    assert list(long["scope_kind"]) == ["type"]


def test_finalise_splits_the_model_name_and_emits_the_agreed_columns() -> None:
    """Both master reports emitted the same ten columns in the same order, twice over."""
    part = melt_to_long(
        pd.DataFrame({"model": ["alephbert_consonantal"], "separation_p": [0.01]}),
        ["separation_p"],
        "retrieval",
    )

    result = finalise_long_metrics([part])

    assert list(result.columns) == list(LONG_COLUMNS)
    assert result["model_base"].iloc[0] == "alephbert"
    assert result["text_variant"].iloc[0] == "consonantal"


def test_finalise_attaches_q_values_for_a_p_value_metric() -> None:
    part = melt_to_long(
        pd.DataFrame({"model": ["a", "b"], "separation_p": [0.01, 0.9]}),
        ["separation_p"],
        "retrieval",
    )

    result = finalise_long_metrics([part])

    assert result["q_value"].notna().all()


def test_pivot_puts_each_metric_in_its_own_column_with_its_q_columns() -> None:
    part = melt_to_long(
        pd.DataFrame({"model": ["a"], "separation_p": [0.01]}), ["separation_p"], "retrieval"
    )
    long = finalise_long_metrics([part])

    wide = pivot_metrics_wide(long, ["model", "model_base", "text_variant"])

    assert "separation_p" in wide.columns
    assert "separation_p_q" in wide.columns
    assert "separation_p_q_by" in wide.columns


def test_pivot_omits_q_columns_when_no_metric_carries_one() -> None:
    part = melt_to_long(
        pd.DataFrame({"model": ["a"], "average_precision": [0.4]}), ["average_precision"], "genre"
    )
    long = finalise_long_metrics([part])

    wide = pivot_metrics_wide(long, ["model", "model_base", "text_variant"])

    assert "average_precision" in wide.columns
    assert not any(str(c).endswith("_q") for c in wide.columns)
