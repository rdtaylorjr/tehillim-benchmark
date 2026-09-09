"""FDR correction (BH and BY), applied within each (source, metric, scope_kind) family."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

_P_VALUE_METRICS = frozenset({"separation_p", "discrimination_p", "type_gap_p", "p_vs_baseline"})


def _adjust(p_values: np.ndarray, weight_at_rank: Callable[[int], np.ndarray]) -> np.ndarray:
    """Step-up adjustment shared by BH and BY, differing only in the weight at each rank."""
    p = np.asarray(p_values, dtype=float)
    result = np.full(len(p), np.nan)
    #: NaN leaves before the monotone pass, which accumulates from the end where argsort puts it.
    tested = np.flatnonzero(~np.isnan(p))
    if tested.size == 0:
        return result
    ranked_within_tested = np.argsort(p[tested])
    ranked = p[tested][ranked_within_tested]
    raw_q = ranked * weight_at_rank(tested.size)
    monotone_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    result[tested[ranked_within_tested]] = np.clip(monotone_q, 0, 1)
    return result


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """FDR-adjusted q-values, valid under independence or positive regression dependence."""
    return _adjust(p_values, lambda n: n / np.arange(1, n + 1))


def benjamini_yekutieli(p_values: np.ndarray) -> np.ndarray:
    """FDR-adjusted q-values, valid under arbitrary dependence; more conservative than BH."""

    def weight(n: int) -> np.ndarray:
        c_n = np.sum(1.0 / np.arange(1, n + 1))
        return n * c_n / np.arange(1, n + 1)

    return _adjust(p_values, weight)


def add_fdr_q_values(long_df: pd.DataFrame) -> pd.DataFrame:
    """Adds BH/BY q-values, corrected within each (source, metric, scope_kind) family."""
    result = long_df.reset_index(drop=True)
    result["q_value"] = np.nan
    result["q_value_by"] = np.nan
    is_p_value = result["metric"].isin(_P_VALUE_METRICS)
    group_cols = ["source", "metric", "scope_kind"]
    # dropna=False so a family with a missing key still gets q-values instead of vanishing.
    for _key, group in result[is_p_value].groupby(group_cols, dropna=False):
        values = group["value"].to_numpy()
        result.loc[group.index, "q_value"] = benjamini_hochberg(values)
        result.loc[group.index, "q_value_by"] = benjamini_yekutieli(values)
    return result


def add_source_q_columns(
    rows: list[dict[str, Any]],
    *,
    sources: tuple[str, ...],
    scope_column: str,
    p_value_template: str,
) -> pd.DataFrame:
    """Adds each source's BH/BY q-values, corrected within the family its scope column defines."""
    frame = pd.DataFrame(rows)
    long_parts = [
        pd.DataFrame(
            {
                "model": frame["model"],
                "scope_kind": frame[scope_column],
                "source": source,
                "metric": "separation_p",
                "value": frame[p_value_template.format(source=source)],
            }
        )
        for source in sources
    ]
    long_frame = add_fdr_q_values(pd.concat(long_parts, ignore_index=True))

    result = frame.copy()
    for source in sources:
        q_columns = long_frame[long_frame["source"] == source][
            ["model", "scope_kind", "q_value", "q_value_by"]
        ].rename(
            columns={
                "scope_kind": scope_column,
                "q_value": f"{source}_q",
                "q_value_by": f"{source}_q_by",
            }
        )
        result = result.merge(q_columns, on=["model", scope_column], how="left")
    return result
