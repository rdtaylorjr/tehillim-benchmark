"""Ranks embedding files by true-pair similarity calibrated against each model's own background."""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from library.bhsa import list_psalms_half_verse_nodes
from library.calibration import (
    background_similarity_stats,
    background_similarity_stats_sparse,
    calibrated_effect_size,
)
from library.cli import add_embeddings_dir_argument, add_scoring_arguments, resume_from_cache
from library.embeddings import (
    dataset_identifier,
    is_sparse_embeddings,
    load_embeddings,
    load_sparse_embeddings,
)
from library.retrieval_metrics import paired_cosine_similarity, sparse_paired_cosine_similarity
from library.rows_output import write_rows_csv
from library.scoring import skipping_unscorable
from library.worker_pool import map_in_order
from parallelism.evaluate import build_side_vectors, build_side_vectors_sparse
from parallelism.pairs import RetrievalPair, build_retrieval_pairs, filter_pairs_with_vectors
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups
from parallelism.true_similarity import summarize_true_pair_similarity


def score_model(
    path: Path,
    pairs: list[RetrievalPair],
    background_node_ids: list[int],
) -> dict[str, str | int | float]:
    """One model file's row: raw and calibrated true-pair similarity, overall and per type."""
    model = dataset_identifier(path)
    if is_sparse_embeddings(path):
        node_ids, matrix = load_sparse_embeddings(path)
        node_index = {n: i for i, n in enumerate(node_ids)}
        model_pairs = filter_pairs_with_vectors(pairs, node_index)
        types = np.array([p.parallelism_type for p in model_pairs])
        similarities = sparse_paired_cosine_similarity(
            build_side_vectors_sparse(model_pairs, "source", node_ids, matrix),
            build_side_vectors_sparse(model_pairs, "target", node_ids, matrix),
        )
        summary = summarize_true_pair_similarity(similarities)
        background_rows = [node_index[n] for n in background_node_ids if n in node_index]
        background = background_similarity_stats_sparse(matrix[background_rows])
    else:
        node_vectors = load_embeddings(path)
        model_pairs = filter_pairs_with_vectors(pairs, node_vectors)
        types = np.array([p.parallelism_type for p in model_pairs])
        source_vecs = build_side_vectors(model_pairs, "source", node_vectors)
        target_vecs = build_side_vectors(model_pairs, "target", node_vectors)
        similarities = paired_cosine_similarity(source_vecs, target_vecs)
        summary = summarize_true_pair_similarity(similarities)
        background_vecs = np.stack(
            [node_vectors[n] for n in background_node_ids if n in node_vectors]
        )
        background = background_similarity_stats(background_vecs)
    effect_size = calibrated_effect_size(summary.mean, background)

    row: dict[str, str | int | float] = {
        "model": model,
        "n_pairs": summary.n,
        "mean_true_similarity": summary.mean,
        "median_true_similarity": summary.median,
        "std_true_similarity": summary.std,
        "background_mean": background.mean,
        "background_std": background.std,
        "background_n_vectors": background.n_vectors,
        "calibrated_effect_size": effect_size,
    }
    for ptype in sorted(set(types.tolist())):
        type_summary = summarize_true_pair_similarity(similarities[types == ptype])
        row[f"n_pairs_{ptype}"] = type_summary.n
        row[f"mean_true_similarity_{ptype}"] = type_summary.mean
        row[f"calibrated_effect_size_{ptype}"] = calibrated_effect_size(
            type_summary.mean, background
        )
    return row


def compare_true_similarity(
    pairs: list[RetrievalPair],
    model_paths: list[Path],
    background_node_ids: list[int],
    max_workers: int | None = None,
) -> list[dict[str, str | int | float]]:
    """Scores every model file across workers, rows sorted by calibrated effect size descending."""
    score = partial(score_model, pairs=pairs, background_node_ids=background_node_ids)
    scored = map_in_order(skipping_unscorable(score), model_paths, max_workers)
    rows = [row for row in scored if row is not None]
    rows.sort(key=lambda r: r["calibrated_effect_size"], reverse=True)
    return rows


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_embeddings_dir_argument(parser)
    add_scoring_arguments(parser)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    pairs = build_retrieval_pairs(groups)
    marked_nodes = {n for p in pairs for n in p.source_nodes} | {
        n for p in pairs for n in p.target_nodes
    }
    background_node_ids = [n for n in list_psalms_half_verse_nodes(api) if n not in marked_nodes]

    cached_rows, model_paths = resume_from_cache(args.embeddings_dir, args.output)
    new_rows = compare_true_similarity(
        pairs, model_paths, background_node_ids, max_workers=args.workers
    )

    rows = sorted(cached_rows + new_rows, key=lambda r: r["calibrated_effect_size"], reverse=True)

    for row in rows:
        print(
            f"{row['model']:55s} effect_size={row['calibrated_effect_size']:.3f} "
            f"mean_sim={row['mean_true_similarity']:.4f} "
            f"background_mean={row['background_mean']:.4f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
