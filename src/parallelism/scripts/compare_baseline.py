"""Compares true-pair similarity against adjacent half-verse pairs never marked as parallel."""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from library.bhsa import (
    list_psalms_half_verse_nodes,
    list_psalms_half_verses_by_psalm,
)
from library.calibration import (
    background_similarity_stats,
    background_similarity_stats_sparse,
)
from library.cli import add_embeddings_dir_argument, add_scoring_arguments, resume_from_cache
from library.embeddings import (
    dataset_identifier,
    is_sparse_embeddings,
    load_embeddings,
    load_sparse_embeddings,
)
from library.rows_output import write_rows_csv
from library.scoring import skipping_unscorable
from library.worker_pool import map_in_order
from parallelism.baseline import build_unmarked_half_verse_pairs
from parallelism.baseline_comparison import (
    baseline_metric_fields,
    compare_to_baseline,
    compare_to_baseline_from_similarities,
)
from parallelism.node_pairs import (
    NodePairs,
    as_node_pairs,
    filter_node_pairs_with_vectors,
    pair_similarities_sparse,
    retrieval_pairs_as_node_pairs,
)
from parallelism.pairs import build_retrieval_pairs
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups


def score_model(
    path: Path,
    true_pairs: NodePairs,
    baseline_pairs: NodePairs,
    background_node_ids: list[int],
) -> dict[str, str | int | float]:
    """One model file's true-vs-baseline row; each model is scored independently of every other."""
    model = dataset_identifier(path)
    if is_sparse_embeddings(path):
        node_ids, matrix = load_sparse_embeddings(path)
        node_index = {n: i for i, n in enumerate(node_ids)}
        background_rows = [node_index[n] for n in background_node_ids if n in node_index]
        background = background_similarity_stats_sparse(matrix[background_rows])
        result = compare_to_baseline_from_similarities(
            pair_similarities_sparse(
                filter_node_pairs_with_vectors(true_pairs, node_index), node_ids, matrix
            ),
            pair_similarities_sparse(
                filter_node_pairs_with_vectors(baseline_pairs, node_index), node_ids, matrix
            ),
            background,
        )
    else:
        node_vectors = load_embeddings(path)
        background_vecs = np.stack(
            [node_vectors[n] for n in background_node_ids if n in node_vectors]
        )
        background = background_similarity_stats(background_vecs)
        result = compare_to_baseline(
            filter_node_pairs_with_vectors(true_pairs, node_vectors),
            filter_node_pairs_with_vectors(baseline_pairs, node_vectors),
            node_vectors,
            background,
        )
    return {
        "model": model,
        **baseline_metric_fields(result),
        "separation_auc": result.separation_auc,
        "separation_p": result.separation_p,
    }


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
    retrieval_pairs = build_retrieval_pairs(groups)
    true_pairs = retrieval_pairs_as_node_pairs(retrieval_pairs)

    marked_nodes = {n for p in retrieval_pairs for n in p.source_nodes} | {
        n for p in retrieval_pairs for n in p.target_nodes
    }
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)
    baseline_pairs = as_node_pairs(
        build_unmarked_half_verse_pairs(half_verses_by_psalm, marked_nodes)
    )
    background_node_ids = [n for n in list_psalms_half_verse_nodes(api) if n not in marked_nodes]

    rows, model_paths = resume_from_cache(args.embeddings_dir, args.output)
    score = partial(
        score_model,
        true_pairs=true_pairs,
        baseline_pairs=baseline_pairs,
        background_node_ids=background_node_ids,
    )
    scored = map_in_order(skipping_unscorable(score), model_paths, args.workers)
    rows.extend(row for row in scored if row is not None)
    rows.sort(key=lambda r: r["average_precision"], reverse=True)

    for row in rows:
        print(
            f"{row['model']:55s} AP={row['average_precision']:.3f} "
            f"(chance={row['prevalence']:.3f}) auc={row['separation_auc']:.3f} gap={row['gap']:.3f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
