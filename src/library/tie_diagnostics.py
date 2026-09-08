"""Reports how much of a model's similarity mass sits on one value, where ranking loses meaning."""

import numpy as np
import pandas as pd

#: Named like the other long-format metrics, so it pivots into the wide tables unchanged.
TIE_FRACTION_METRIC = "max_tie_fraction"


def max_tie_fraction(similarities: np.ndarray) -> float:
    """Share of pairs holding the single most common similarity value, NaN when there are none."""
    finite = np.asarray(similarities, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    _, counts = np.unique(finite, return_counts=True)
    return float(counts.max() / finite.size)


def tie_fractions_by_model(pair_detail: pd.DataFrame) -> pd.DataFrame:
    """One tie fraction per model, from the per-pair similarities the detail stage already wrote."""
    if pair_detail.empty:
        return pd.DataFrame(columns=["model", "scope", "scope_kind", "source", "metric", "value"])
    grouped = (
        pair_detail.groupby("model")["raw_similarity"]
        .apply(lambda s: max_tie_fraction(s.to_numpy()))
        .reset_index(name="value")
    )
    grouped["scope"] = "overall"
    grouped["scope_kind"] = "overall"
    grouped["source"] = "vs_baseline"
    grouped["metric"] = TIE_FRACTION_METRIC
    return grouped[["model", "scope", "scope_kind", "source", "metric", "value"]]
