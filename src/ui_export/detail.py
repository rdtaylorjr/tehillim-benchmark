"""Builds one model's row-click detail payload: raincloud/ROC/PR/heatmap views, real pair data."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

# Canonical scholarly ordering, matching tehillim-ui's genre-tab section-note, not alphabetical.
_PARALLELISM_TYPE_ORDER = ["Synonymous", "Antithetic", "Synthetic", "Emblematic", "Staircase"]


def raincloud_group(values: pd.Series) -> dict[str, Any]:
    """Full raw value array plus n and mean, for a client-side KDE/box/point raincloud render."""
    arr = values.to_numpy()
    return {
        "values": [round(float(v), 4) for v in arr],
        "n": len(arr),
        "mean": round(float(arr.mean()), 4),
    }


def roc_pr_series(name: str, labels: np.ndarray, scores: np.ndarray, n: int) -> dict[str, Any]:
    """One named ROC+PR series (the combined series, or a one-vs-rest breakdown), plus its n."""
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "name": name,
        "n": int(n),
        "roc": [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr, strict=True)],
        "pr": [
            {"recall": float(r), "precision": float(p)}
            for r, p in zip(recall, precision, strict=True)
        ],
    }


def genre_mean_matrix(
    pair_df: pd.DataFrame, value_col: str, genres: list[str]
) -> list[dict[str, Any]]:
    """Mean value_col for every (genre_a, genre_b) cell, averaged over both orderings."""
    means = pair_df.groupby(["genre_a", "genre_b"])[value_col].mean()
    cells = []
    for ga in genres:
        for gb in genres:
            vals = [means.get((ga, gb)), means.get((gb, ga))]
            vals = [v for v in vals if v is not None and not pd.isna(v)]
            if vals:
                cells.append(
                    {"genre_a": ga, "genre_b": gb, "value": round(float(np.mean(vals)), 4)}
                )
    return cells


def heatmap_cells(pair_df: pd.DataFrame, value_col: str) -> list[dict[str, Any]]:
    """One cell per row: the pair's psalm ids and its rounded value."""
    return [
        {
            "psalm_a": int(row.psalm_a),
            "psalm_b": int(row.psalm_b),
            "value": round(float(getattr(row, value_col)), 4),
        }
        for row in pair_df.itertuples()
    ]


def order_psalms_by_own_stat(
    same_genre_df: pd.DataFrame, value_col: str, genre_by_psalm: dict[int, str]
) -> list[dict[str, Any]]:
    """Groups psalms by genre, ordered within a genre by that psalm's own mean value, descending."""
    per_psalm_mean = (
        pd.concat(
            [
                same_genre_df.groupby("psalm_a")[value_col].mean(),
                same_genre_df.groupby("psalm_b")[value_col].mean(),
            ]
        )
        .groupby(level=0)
        .mean()
    )
    psalms_sorted = sorted(
        genre_by_psalm.keys(),
        key=lambda p: (genre_by_psalm[p], -per_psalm_mean.get(p, 0.0)),
    )
    return [{"psalm": p, "genre": genre_by_psalm[p]} for p in psalms_sorted]


def auc_ap_ci_for(ci_df: pd.DataFrame, model: str, scope: str | None) -> dict[str, Any] | None:
    """The already-bootstrapped AUC/AP estimate and BCa CI for one model, or None if absent."""
    row = (
        ci_df[ci_df.model == model]
        if scope is None
        else ci_df[(ci_df.model == model) & (ci_df.scope == scope)]
    )
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "auc": float(row["point_auc"]),
        "auc_ci_low": float(row["auc_ci_low"]),
        "auc_ci_high": float(row["auc_ci_high"]),
        "ap": float(row["point_ap"]),
        "ap_ci_low": float(row["ap_ci_low"]),
        "ap_ci_high": float(row["ap_ci_high"]),
    }


def validated_gap_stats_for(
    validation_df: pd.DataFrame, model: str, metric: str
) -> dict[str, dict[str, float]] | None:
    """The already-tested gap, p and effect size for one model and metric, or None."""
    row = validation_df[(validation_df.model == model) & (validation_df.metric == metric)]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        source: {
            "gap": float(row[f"{source}_gap"]),
            "p": float(row[f"{source}_p"]),
            "effect_size": float(row[f"{source}_effect_size"]),
        }
        for source in ("length_controlled", "length_and_content_controlled")
    }


def build_parallelism_detail(
    pair_detail_df: pd.DataFrame,
    baseline_detail_df: pd.DataFrame,
    auc_ap_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """Marked-parallel vs. baseline: raincloud groups, combined + per-type ROC/PR curves."""
    baseline_scores = baseline_detail_df.calibrated_z.to_numpy()
    observed_types = set(pair_detail_df.parallelism_type.unique())
    types = [t for t in _PARALLELISM_TYPE_ORDER if t in observed_types]

    def series_for(positive_scores: np.ndarray, name: str) -> dict[str, Any]:
        labels = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(baseline_scores))])
        scores = np.concatenate([positive_scores, baseline_scores])
        return roc_pr_series(name, labels, scores, len(positive_scores))

    return {
        "raincloud_groups": [
            {
                "key": "baseline",
                "label": "Baseline",
                **raincloud_group(baseline_detail_df.calibrated_z),
            },
            {
                "key": "combined",
                "label": "Marked-parallel (combined)",
                **raincloud_group(pair_detail_df.calibrated_z),
            },
        ]
        + [
            {
                "key": t,
                "label": t,
                **raincloud_group(
                    pair_detail_df[pair_detail_df.parallelism_type == t].calibrated_z
                ),
            }
            for t in types
        ],
        "series": [series_for(pair_detail_df.calibrated_z.to_numpy(), "Combined")]
        + [
            series_for(
                pair_detail_df[pair_detail_df.parallelism_type == t].calibrated_z.to_numpy(), t
            )
            for t in types
        ],
        "auc_ap_stats": auc_ap_stats,
    }


def _same_genre_scores(genre_pair_df: "pd.DataFrame", genre: str) -> "pd.Series":
    """Calibrated scores of pairs where both psalms carry the given genre."""
    return genre_pair_df[(genre_pair_df.genre_a == genre) & genre_pair_df.same_genre].calibrated_z


def _genre_by_psalm_from_pairs(genre_pair_df: "pd.DataFrame") -> dict[int, str]:
    """Rebuilds each psalm's genre from the pair table, where it appears on either side."""
    sides = [
        genre_pair_df[["psalm_a", "genre_a"]].rename(
            columns={"psalm_a": "psalm", "genre_a": "genre"}
        ),
        genre_pair_df[["psalm_b", "genre_b"]].rename(
            columns={"psalm_b": "psalm", "genre_b": "genre"}
        ),
    ]
    return dict(pd.concat(sides).drop_duplicates("psalm").set_index("psalm")["genre"])


def _raincloud_groups(
    genre_pair_df: "pd.DataFrame", observed_genres: list[str]
) -> list[dict[str, Any]]:
    """The different-genre baseline, the combined same-genre group, then one group per genre."""
    groups = [
        {
            "key": "different",
            "label": "Different genre",
            **raincloud_group(genre_pair_df[~genre_pair_df.same_genre].calibrated_z),
        },
        {
            "key": "combined",
            "label": "Same genre (combined)",
            **raincloud_group(genre_pair_df[genre_pair_df.same_genre].calibrated_z),
        },
    ]
    return groups + [
        {"key": g, "label": g, **raincloud_group(_same_genre_scores(genre_pair_df, g))}
        for g in observed_genres
    ]


def build_genre_detail(
    genre_pair_df: pd.DataFrame,
    genres: list[str],
    auc_ap_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """Same- vs. different-genre separation, plus the full genre-grouped pairwise matrix."""
    different_scores = genre_pair_df[~genre_pair_df.same_genre].calibrated_z.to_numpy()
    observed_genres = [
        g for g in genres if ((genre_pair_df.genre_a == g) & genre_pair_df.same_genre).any()
    ]

    def series_for(positive_scores: np.ndarray, name: str) -> dict[str, Any]:
        labels = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(different_scores))])
        scores = np.concatenate([positive_scores, different_scores])
        return roc_pr_series(name, labels, scores, len(positive_scores))

    genre_by_psalm = _genre_by_psalm_from_pairs(genre_pair_df)

    return {
        "genre_order": order_psalms_by_own_stat(
            genre_pair_df[genre_pair_df.same_genre], "calibrated_z", genre_by_psalm
        ),
        "raincloud_groups": _raincloud_groups(genre_pair_df, observed_genres),
        "series": [
            series_for(genre_pair_df[genre_pair_df.same_genre].calibrated_z.to_numpy(), "Combined")
        ]
        + [series_for(_same_genre_scores(genre_pair_df, g).to_numpy(), g) for g in observed_genres],
        "heatmap": heatmap_cells(genre_pair_df.assign(value=genre_pair_df.calibrated_z), "value"),
        "heatmap_genre_mean": genre_mean_matrix(genre_pair_df, "calibrated_z", genres),
        "auc_ap_stats": auc_ap_stats,
    }


def build_trajectory_detail(
    traj_df: pd.DataFrame,
    metric: str,
    genres: list[str],
    gap_stats: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    """Within-genre vs. across-genre pairwise distance, for both length-controlled sources."""
    genre_by_psalm = dict(
        pd.concat(
            [
                traj_df[["psalm_a", "genre_a"]].rename(
                    columns={"psalm_a": "psalm", "genre_a": "genre"}
                ),
                traj_df[["psalm_b", "genre_b"]].rename(
                    columns={"psalm_b": "psalm", "genre_b": "genre"}
                ),
            ]
        )
        .drop_duplicates("psalm")
        .set_index("psalm")["genre"]
    )
    same = traj_df[traj_df.same_genre]
    different = traj_df[~traj_df.same_genre]
    return {
        "metric": metric,
        "order": order_psalms_by_own_stat(same, "length_controlled", genre_by_psalm),
        "sources": {
            source: {
                "raincloud": {
                    "same": raincloud_group(same[source]),
                    "different": raincloud_group(different[source]),
                },
                "heatmap": heatmap_cells(traj_df.assign(value=traj_df[source]), "value"),
                "heatmap_genre_mean": genre_mean_matrix(traj_df, source, genres),
                "gap_stats": (gap_stats or {}).get(source),
            }
            for source in ("length_controlled", "length_and_content_controlled")
        },
    }
