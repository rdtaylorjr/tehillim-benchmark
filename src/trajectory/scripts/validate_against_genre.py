"""Permutation test of within vs between-genre psalm distance, apart from the AP/AUC benchmark."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from genre.genre_labels import load_genre_by_psalm
from library.bhsa import list_psalms_half_verses_by_psalm, load_bhsa_api
from library.cli import add_genre_csv_argument, add_scoring_arguments
from library.multiple_comparisons import (
    add_source_q_columns,
    benjamini_hochberg,
    benjamini_yekutieli,
)
from library.rows_output import write_rows_csv
from library.scoring import skipping_unscorable
from library.worker_pool import map_in_order
from trajectory.genre_breakdown import joint_genre_breakdown_permutation_test
from trajectory.residualize import residualize_by_length, residualize_on_covariates
from trajectory.validation import permutation_test, same_genre_matrix

_METRICS = (
    "content_distance",
    "structural_distance",
    "adjacent_similarity_distance",
    "step_magnitude_distance",
    "turning_angle_distance",
)
_SOURCES = ("raw", "length_controlled", "length_and_content_controlled")
# Both sides need two pairs before a gap between their means means anything.
_MIN_PAIRS_PER_SIDE = 2


def _subset_and_length_diff(
    metric: str,
    base_subset: pd.DataFrame,
    index_of: dict[int, int],
    n_half_verses: dict[int, int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Filters to non-NaN pairs for this metric, returning subset, idx_a/idx_b, |length diff|."""
    has_content_covariate = metric != "content_distance"
    required = [metric, "content_distance"] if has_content_covariate else [metric]
    subset = base_subset[base_subset[required].notna().all(axis=1)]
    idx_a = subset["psalm_a"].map(index_of).to_numpy()
    idx_b = subset["psalm_b"].map(index_of).to_numpy()
    length_diff = (
        (subset["psalm_a"].map(n_half_verses) - subset["psalm_b"].map(n_half_verses))
        .abs()
        .to_numpy()
    )
    return subset, idx_a, idx_b, length_diff


def _values_by_source(
    metric: str, subset: pd.DataFrame, length_diff: np.ndarray
) -> dict[str, np.ndarray | None]:
    """raw/length_controlled/length_and_content_controlled value arrays for this metric's subset."""
    has_content_covariate = metric != "content_distance"
    raw = subset[metric].to_numpy()
    values_by_source: dict[str, np.ndarray | None] = {
        "raw": raw,
        "length_controlled": residualize_by_length(raw, length_diff),
        "length_and_content_controlled": None,
    }
    if has_content_covariate:
        content = subset["content_distance"].to_numpy()
        covariates = np.column_stack([length_diff, content])
        values_by_source["length_and_content_controlled"] = residualize_on_covariates(
            raw, covariates
        )
    return values_by_source


def build_validation_row(
    model: str,
    metric: str,
    base_subset: pd.DataFrame,
    index_of: dict[int, int],
    genre_labels: np.ndarray,
    n_half_verses: dict[int, int],
    n_permutations: int,
    seed: int,
) -> dict[str, str | int | float]:
    """One (model, metric) row of gap+p per source in _SOURCES, excluding NaN-valued pairs."""
    subset, idx_a, idx_b, length_diff = _subset_and_length_diff(
        metric, base_subset, index_of, n_half_verses
    )
    same_genre = genre_labels[idx_a] == genre_labels[idx_b]

    row: dict[str, str | int | float] = {
        "model": model,
        "metric": metric,
        "n_pairs_total": len(base_subset),
        "n_pairs_valid": len(subset),
    }
    if (
        int(same_genre.sum()) < _MIN_PAIRS_PER_SIDE
        or int((~same_genre).sum()) < _MIN_PAIRS_PER_SIDE
    ):
        for source in _SOURCES:
            row[f"{source}_gap"] = float("nan")
            row[f"{source}_p"] = float("nan")
            row[f"{source}_effect_size"] = float("nan")
        return row

    values_by_source = _values_by_source(metric, subset, length_diff)
    # The shuffle depends only on the pairs and the seed, so all three sources share one matrix.
    shared_matrix = same_genre_matrix(
        idx_a, idx_b, genre_labels, n_permutations, np.random.default_rng(seed)
    )

    for source in _SOURCES:
        values = values_by_source[source]
        if values is None:
            row[f"{source}_gap"] = float("nan")
            row[f"{source}_p"] = float("nan")
            row[f"{source}_effect_size"] = float("nan")
            continue
        gap, p_value, effect_size = permutation_test(
            idx_a,
            idx_b,
            values,
            genre_labels,
            n_permutations=n_permutations,
            rng=np.random.default_rng(seed),
            same_matrix=shared_matrix,
        )
        row[f"{source}_gap"] = gap
        row[f"{source}_p"] = p_value
        row[f"{source}_effect_size"] = effect_size
    return row


def build_genre_breakdown_rows(
    model: str,
    metric: str,
    base_subset: pd.DataFrame,
    index_of: dict[int, int],
    genre_labels: np.ndarray,
    n_half_verses: dict[int, int],
    n_permutations: int,
    seed: int,
) -> list[dict[str, str | int | float]]:
    """One row per (genre, available source): one-vs-rest distance gap, perm p, and maxT p."""
    subset, idx_a, idx_b, length_diff = _subset_and_length_diff(
        metric, base_subset, index_of, n_half_verses
    )
    if len(subset) == 0:
        return []

    genres_array, genre_codes = np.unique(genre_labels, return_inverse=True)
    genres = tuple(genres_array.tolist())
    values_by_source = _values_by_source(metric, subset, length_diff)

    rows: list[dict[str, str | int | float]] = []
    for source in _SOURCES:
        values = values_by_source[source]
        if values is None:
            continue
        result = joint_genre_breakdown_permutation_test(
            values,
            idx_a,
            idx_b,
            genre_codes,
            genres,
            n_permutations=n_permutations,
            rng=np.random.default_rng(seed),
        )
        for genre, gap, p_perm, p_maxt in zip(
            result.genres, result.observed, result.p_perm, result.p_maxt, strict=True
        ):
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "source": source,
                    "genre": genre,
                    "gap": gap,
                    "p_perm": p_perm,
                    "p_maxT": p_maxt,
                }
            )
    return rows


def add_fdr_columns(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Adds BH/BY q-values to every source, corrected within each scope's family."""
    return add_source_q_columns(
        rows,
        sources=_SOURCES,
        scope_column="metric",
        p_value_template="{source}_p",
    )


def add_genre_breakdown_fdr_columns(rows: list[dict[str, str | int | float]]) -> pd.DataFrame:
    """Adds BH/BY q-values to p_perm/p_maxT, corrected within each (metric, source, genre)."""
    df = pd.DataFrame(rows)
    df["perm_q"] = np.nan
    df["perm_q_by"] = np.nan
    df["maxT_q"] = np.nan
    df["maxT_q_by"] = np.nan
    group_cols = ["metric", "source", "genre"]
    for _key, group in df.groupby(group_cols):
        df.loc[group.index, "perm_q"] = benjamini_hochberg(group["p_perm"].to_numpy())
        df.loc[group.index, "perm_q_by"] = benjamini_yekutieli(group["p_perm"].to_numpy())
        df.loc[group.index, "maxT_q"] = benjamini_hochberg(group["p_maxT"].to_numpy())
        df.loc[group.index, "maxT_q_by"] = benjamini_yekutieli(group["p_maxT"].to_numpy())
    return df


def breakdown_path_for(output: Path) -> Path:
    """The per-genre breakdown sits beside its validation CSV, since the two are always one run."""
    return output.with_name(f"{output.stem}_by_genre{output.suffix}")


def validate_one_model(
    model: str,
    group: pd.DataFrame,
    genre_by_psalm: dict[int, str],
    n_half_verses: dict[int, int],
    n_permutations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str | int | float]]]:
    """Every validation and breakdown row for one model, independent of every other model."""
    psalms = sorted(set(group["psalm_a"]) | set(group["psalm_b"]))
    usable_psalms = [p for p in psalms if p in genre_by_psalm]
    index_of = {psalm: i for i, psalm in enumerate(usable_psalms)}
    mask = group["psalm_a"].isin(usable_psalms) & group["psalm_b"].isin(usable_psalms)
    base_subset = group[mask]
    genre_labels = np.array([genre_by_psalm[psalm] for psalm in usable_psalms])

    rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, str | int | float]] = []
    for metric in _METRICS:
        rows.append(
            build_validation_row(
                model,
                metric,
                base_subset,
                index_of,
                genre_labels,
                n_half_verses,
                n_permutations,
                seed,
            )
        )
        breakdown_rows.extend(
            build_genre_breakdown_rows(
                model,
                metric,
                base_subset,
                index_of,
                genre_labels,
                n_half_verses,
                n_permutations,
                seed,
            )
        )
    return rows, breakdown_rows


class _ValidationTask(NamedTuple):
    """Everything one worker needs to validate one model's trajectory distances."""

    model: str
    group: pd.DataFrame
    genre_by_psalm: dict[int, str]
    n_half_verses: dict[int, int]
    n_permutations: int
    seed: int


def _validate_task(
    task: _ValidationTask,
) -> tuple[list[dict[str, Any]], list[dict[str, str | int | float]]]:
    """Validates one model, returning its rows and its per-genre breakdown rows."""
    return validate_one_model(
        task.model,
        task.group,
        task.genre_by_psalm,
        task.n_half_verses,
        task.n_permutations,
        task.seed,
    )


def _task_model(task: _ValidationTask) -> str:
    """Names a task by its model, so a skipped one is reported the way every batch reports one."""
    return task.model


def validate_models(
    distances_df: pd.DataFrame,
    genre_by_psalm: dict[int, str],
    n_half_verses: dict[int, int],
    n_permutations: int,
    seed: int,
    max_workers: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str | int | float]]]:
    """Models are independent, so they run across workers and are reassembled in submit order."""
    tasks = [
        _ValidationTask(model, group, genre_by_psalm, n_half_verses, n_permutations, seed)
        for model, group in distances_df.groupby("model")
    ]
    rows: list[dict[str, Any]] = []
    breakdown: list[dict[str, str | int | float]] = []
    scored = map_in_order(
        skipping_unscorable(_validate_task, label=_task_model), tasks, max_workers
    )
    for result in scored:
        if result is None:
            continue
        model_rows, model_breakdown = result
        rows.extend(model_rows)
        breakdown.extend(model_breakdown)
    return rows, breakdown


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_genre_csv_argument(parser)
    parser.add_argument("distances_parquet", type=Path, help="trajectory_distances.parquet")
    # The breakdown is computed either way, so it is written beside --output by default.
    parser.add_argument("--breakdown-output", type=Path, default=None)
    add_scoring_arguments(parser, with_seed=True, with_permutations=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    n_half_verses = {p: len(nodes) for p, nodes in list_psalms_half_verses_by_psalm(api).items()}
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    distances_df = pd.read_parquet(args.distances_parquet)

    rows, all_breakdown_rows = validate_models(
        distances_df, genre_by_psalm, n_half_verses, args.n_permutations, args.seed, args.workers
    )

    result_df = add_fdr_columns(rows)

    for _, row in result_df.iterrows():
        header = f"{row['model']:55s} {row['metric']:28s}"
        counts = f"n_valid={row['n_pairs_valid']}/{row['n_pairs_total']}"
        parts = [f"{header} {counts}"]
        parts.extend(
            f"{source}_p={row[f'{source}_p']:.4f} q={row[f'{source}_q']:.4f} "
            f"z={row[f'{source}_effect_size']:.3f}"
            for source in _SOURCES
        )
        print(" ".join(parts))

    if args.output:
        write_rows_csv(args.output, result_df.to_dict("records"))

    if all_breakdown_rows:
        breakdown_output = args.breakdown_output or (
            breakdown_path_for(args.output) if args.output else None
        )
        if breakdown_output:
            breakdown_df = add_genre_breakdown_fdr_columns(all_breakdown_rows)
            write_rows_csv(breakdown_output, breakdown_df.to_dict("records"))


if __name__ == "__main__":
    main()
