"""Computes each psalm's content centroid and half-verses sequence, all models."""

import argparse
from collections.abc import Callable
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from library.bhsa import list_psalms_half_verses_by_psalm, load_bhsa_api
from library.centroid import psalm_centroids, sparse_psalm_centroids
from library.cli import add_embeddings_dir_argument, add_scoring_arguments, report_reuse
from library.embeddings import (
    dataset_identifier,
    is_sparse_embeddings,
    load_embeddings,
    load_sparse_embeddings,
)
from library.incremental_cache import load_cached_parquet_set
from library.model_files import uncached_model_paths
from library.rows_output import write_dataframe_parquet
from library.scoring import skipping_unscorable
from library.worker_pool import map_in_order
from trajectory.distance import content_distance, dtw_curve_distance, structural_distance_dtw
from trajectory.geometry import adjacent_similarity, step_magnitude, turning_angle
from trajectory.self_similarity import self_similarity_matrix
from trajectory.sequence import (
    normalize_sequence,
    psalm_half_verse_sequences,
    psalm_half_verse_sequences_sparse,
)

_MIN_HALF_VERSES = 4


def compute_psalm_profiles(
    sequences_by_psalm: dict[int, np.ndarray],
    centroids_by_psalm: dict[int, np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    """One profile per psalm with a centroid and at least _MIN_HALF_VERSES half-verses."""
    profiles = {}
    for psalm, sequence in sequences_by_psalm.items():
        if psalm not in centroids_by_psalm or len(sequence) < _MIN_HALF_VERSES:
            continue
        profiles[psalm] = {
            "centroid": centroids_by_psalm[psalm],
            "sequence": normalize_sequence(sequence),
        }
    return profiles


def profile_rows(model: str, profiles: dict[int, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    """Flattens each psalm's profile into one row: the real, un-resampled half-verses sequence."""
    rows = []
    for psalm, profile in profiles.items():
        sequence = profile["sequence"]
        rows.append(
            {
                "model": model,
                "psalm": psalm,
                "centroid": profile["centroid"].astype(np.float32),
                "sequence": sequence.flatten().astype(np.float32),
                "n_half_verses": sequence.shape[0],
                "dim": sequence.shape[1],
            }
        )
    return rows


def distance_rows(model: str, profiles: dict[int, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    """One row per unordered psalm pair: content, structural, and geometry-curve DTW distances."""
    self_similarity = {p: self_similarity_matrix(v["sequence"]) for p, v in profiles.items()}
    adjacent = {p: adjacent_similarity(v["sequence"]) for p, v in profiles.items()}
    step = {p: step_magnitude(v["sequence"]) for p, v in profiles.items()}
    turning = {p: turning_angle(v["sequence"]) for p, v in profiles.items()}

    rows = []
    for a, b in combinations(sorted(profiles), 2):
        seq_a, seq_b = profiles[a]["sequence"], profiles[b]["sequence"]
        rows.append(
            {
                "model": model,
                "psalm_a": a,
                "psalm_b": b,
                "content_distance": content_distance(
                    profiles[a]["centroid"], profiles[b]["centroid"]
                ),
                "structural_distance": structural_distance_dtw(
                    seq_a, seq_b, self_similarity[a], self_similarity[b]
                ),
                "adjacent_similarity_distance": dtw_curve_distance(adjacent[a], adjacent[b]),
                "step_magnitude_distance": dtw_curve_distance(step[a], step[b]),
                "turning_angle_distance": dtw_curve_distance(turning[a], turning[b]),
            }
        )
    return rows


def profile_shard_path(output_dir: Path, model: str) -> Path:
    """One parquet file per model, so no single shard risks GitHub's 100MB per-file limit."""
    return output_dir / f"{model}.parquet"


def score_model(
    path: Path, half_verses_by_psalm: dict[int, list[int]], output_dir: Path
) -> tuple[int, list[dict[str, Any]]]:
    """Writes one model file's profile shard and returns its row count plus its distance rows."""
    model = dataset_identifier(path)
    if is_sparse_embeddings(path):
        node_ids, matrix = load_sparse_embeddings(path)
        sequences_by_psalm = psalm_half_verse_sequences_sparse(
            half_verses_by_psalm, node_ids, matrix
        )
        psalms, centroids = sparse_psalm_centroids(half_verses_by_psalm, node_ids, matrix)
        dense_centroids = centroids.toarray().astype("<f4", copy=False)
        centroids_by_psalm = {p: dense_centroids[i] for i, p in enumerate(psalms)}
    else:
        node_vectors = load_embeddings(path)
        sequences_by_psalm = psalm_half_verse_sequences(half_verses_by_psalm, node_vectors)
        centroids_by_psalm = psalm_centroids(half_verses_by_psalm, node_vectors)
    profiles = compute_psalm_profiles(sequences_by_psalm, centroids_by_psalm)
    rows = profile_rows(model, profiles)
    write_dataframe_parquet(
        profile_shard_path(output_dir, model),
        pd.DataFrame(rows),
        compression="zstd",
        compression_level=19,
    )
    return len(rows), distance_rows(model, profiles)


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_embeddings_dir_argument(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_scoring_arguments(parser)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = api_factory(args.checkout)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)

    (cached_distance_rows,), cached_models = load_cached_parquet_set(
        args.output_dir, ("trajectory_distances.parquet",)
    )
    report_reuse(cached_models, args.output_dir)

    # A cached model still needs rescoring when its own profile shard is missing.
    complete = {
        model for model in cached_models if profile_shard_path(args.output_dir, model).exists()
    }
    model_paths = uncached_model_paths(args.embeddings_dir, complete)
    all_distance_rows: list[dict[str, Any]] = list(cached_distance_rows)
    n_profile_rows = 0
    score = partial(
        score_model, half_verses_by_psalm=half_verses_by_psalm, output_dir=args.output_dir
    )
    for scored in map_in_order(skipping_unscorable(score), model_paths, args.workers):
        if scored is None:
            continue
        model_n_rows, model_distance_rows = scored
        n_profile_rows += model_n_rows
        all_distance_rows.extend(model_distance_rows)

    write_dataframe_parquet(
        args.output_dir / "trajectory_distances.parquet",
        pd.DataFrame(all_distance_rows),
        compression="zstd",
        compression_level=19,
    )
    print(f"wrote {n_profile_rows} profile rows, {len(all_distance_rows)} distance rows")


if __name__ == "__main__":
    main()
