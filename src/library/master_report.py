"""Shapes a benchmark's scored rows into the long and wide master tables both reports emit."""

import pandas as pd

from library.embeddings import split_model_name
from library.multiple_comparisons import add_fdr_q_values

#: The column set and order both master reports publish, agreed here rather than in each.
LONG_COLUMNS = (
    "model",
    "model_base",
    "text_variant",
    "scope",
    "scope_kind",
    "source",
    "metric",
    "value",
    "q_value",
    "q_value_by",
)


def melt_to_long(
    frame: pd.DataFrame,
    metrics: list[str],
    source: str,
    *,
    scope: str = "overall",
    scope_kind: str = "overall",
) -> pd.DataFrame:
    """One row per (model, metric), tagged with the scope and the run that produced it."""
    long = frame.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="value")
    long["scope"] = scope
    long["scope_kind"] = scope_kind
    long["source"] = source
    return long


def finalise_long_metrics(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Joins the melted parts, splits each model name, corrects p-values, and orders the columns."""
    long_frame = pd.concat(parts, ignore_index=True)
    base_variant = long_frame["model"].apply(lambda name: pd.Series(split_model_name(name)))
    long_frame["model_base"] = base_variant[0]
    long_frame["text_variant"] = base_variant[1]
    return add_fdr_q_values(long_frame)[list(LONG_COLUMNS)]


def pivot_metrics_wide(long_frame: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    """Pivots wide on metric, adding a `<metric>_q` and `_q_by` column wherever one exists."""
    values_wide = long_frame.pivot_table(index=index_cols, columns="metric", values="value")
    for q_column, suffix in (("q_value", "_q"), ("q_value_by", "_q_by")):
        q_subset = long_frame.dropna(subset=[q_column])
        if q_subset.empty:
            continue
        q_wide = q_subset.pivot_table(index=index_cols, columns="metric", values=q_column)
        q_wide.columns = [f"{column}{suffix}" for column in q_wide.columns]
        values_wide = values_wide.join(q_wide)
    return values_wide.reset_index()
