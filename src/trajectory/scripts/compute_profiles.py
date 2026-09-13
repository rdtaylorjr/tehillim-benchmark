"""Computes psalm trajectory profiles in memory and writes the pairwise distances, all models."""

import argparse
from collections.abc import Callable
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from library.bhsa import list_psalms_half_verses_by_psalm, load_bhsa_api
from library.centroid import psalm_centroids, sparse_psalm_centroids
from library.cli import add_embeddings_dir_argument, add_scoring_arguments
from library.embeddings import (
    dataset_identifier,
    is_sparse_embeddings,
    load_embeddings,
    load_sparse_embeddings,
)
from library.frame_accumulator import FrameAccumulator
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


def score_model(
    path: Path, half_verses_by_psalm: dict[int, list[int]]
) -> tuple[int, list[dict[str, Any]]]:
    """Builds one model's psalm profiles in memory and returns their count and distance rows."""
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
    return len(profiles), distance_rows(model, profiles)


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_embeddings_dir_argument(parser)
    add_scoring_arguments(parser)
    args = parser.parse_args(argv)
    if args.output is None:
        parser.error("--output is required")

    api = api_factory(args.checkout)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)
    model_paths = uncached_model_paths(args.embeddings_dir, set())
    distances = FrameAccumulator()
    n_profiles = 0
    score = partial(score_model, half_verses_by_psalm=half_verses_by_psalm)
    for scored in map_in_order(skipping_unscorable(score), model_paths, args.workers):
        if scored is None:
            continue
        model_n_profiles, model_distance_rows = scored
        n_profiles += model_n_profiles
        distances.extend(model_distance_rows)

    write_dataframe_parquet(
        args.output, distances.frame(), compression="zstd", compression_level=19
    )
    print(f"profiled {n_profiles} psalm sequences, wrote {len(distances)} distance rows")


if __name__ == "__main__":
    main()
