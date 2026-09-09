"""Psalm-clustered BCa bootstrap 95% CIs for AP (primary), gap, and AUC, every model x scope."""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from library.ap_gap_auc_bootstrap import ci_row
from library.bhsa import (
    list_psalms_half_verse_nodes,
    list_psalms_half_verses_by_psalm,
    node_to_psalm_map,
)
from library.calibration import background_similarity_stats, background_similarity_stats_sparse
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
from parallelism.bootstrap import (
    block_bootstrap_ap_gap_and_auc,
    block_bootstrap_ap_gap_and_auc_from_similarities,
)
from parallelism.node_pairs import (
    NodePairs,
    as_node_pairs,
    filter_node_pairs_with_vectors,
    pair_similarities_sparse,
    retrieval_pairs_as_node_pairs,
)
from parallelism.pairs import build_retrieval_pairs, filter_pairs_by_type
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups

_TYPES = ("Antithetic", "Emblematic", "Staircase", "Synonymous", "Synthetic")


def score_model(
    path: Path,
    scopes: dict[str, NodePairs],
    baseline_pairs: NodePairs,
    background_node_ids: list[int],
    node_to_psalm: dict[int, int],
    n_resamples: int,
    seed: int,
) -> list[dict[str, str | int | float]]:
    """One row per scope for one model file, scored independently of every other model."""
    model = dataset_identifier(path)
    rng = np.random.default_rng(seed)
    rows = []
    if is_sparse_embeddings(path):
        node_ids, matrix = load_sparse_embeddings(path)
        node_index = {n: i for i, n in enumerate(node_ids)}
        background_rows = [node_index[n] for n in background_node_ids if n in node_index]
        background = background_similarity_stats_sparse(matrix[background_rows])
        model_baseline_pairs = filter_node_pairs_with_vectors(baseline_pairs, node_index)
        baseline_sims = pair_similarities_sparse(model_baseline_pairs, node_ids, matrix)
        for scope_name, true_pairs in scopes.items():
            model_true_pairs = filter_node_pairs_with_vectors(true_pairs, node_index)
            #: A type this corpus never annotates is an absent scope, not a model that failed.
            if not model_true_pairs:
                continue
            result = block_bootstrap_ap_gap_and_auc_from_similarities(
                model_true_pairs,
                model_baseline_pairs,
                pair_similarities_sparse(model_true_pairs, node_ids, matrix),
                baseline_sims,
                background,
                node_to_psalm,
                n_resamples=n_resamples,
                rng=rng,
            )
            rows.append(ci_row(model, result, scope=scope_name))
        return rows

    node_vectors = load_embeddings(path)
    background_vecs = np.stack([node_vectors[n] for n in background_node_ids if n in node_vectors])
    background = background_similarity_stats(background_vecs)
    model_baseline_pairs = filter_node_pairs_with_vectors(baseline_pairs, node_vectors)
    for scope_name, true_pairs in scopes.items():
        model_true_pairs = filter_node_pairs_with_vectors(true_pairs, node_vectors)
        #: A type this corpus never annotates is an absent scope, not a model that failed.
        if not model_true_pairs:
            continue
        result = block_bootstrap_ap_gap_and_auc(
            model_true_pairs,
            model_baseline_pairs,
            node_vectors,
            background,
            node_to_psalm,
            n_resamples=n_resamples,
            rng=rng,
        )
        rows.append(ci_row(model, result, scope=scope_name))
    return rows


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_embeddings_dir_argument(parser)
    add_scoring_arguments(parser, with_seed=True, with_resamples=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    all_pairs = build_retrieval_pairs(groups)

    marked_nodes = {n for p in all_pairs for n in p.source_nodes} | {
        n for p in all_pairs for n in p.target_nodes
    }
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)
    baseline_pairs = as_node_pairs(
        build_unmarked_half_verse_pairs(half_verses_by_psalm, marked_nodes)
    )
    background_node_ids = [n for n in list_psalms_half_verse_nodes(api) if n not in marked_nodes]
    node_to_psalm = node_to_psalm_map(half_verses_by_psalm)

    scopes: dict[str, NodePairs] = {"overall": retrieval_pairs_as_node_pairs(all_pairs)}
    for ptype in _TYPES:
        scopes[ptype] = retrieval_pairs_as_node_pairs(
            filter_pairs_by_type(all_pairs, frozenset({ptype}))
        )

    rows, model_paths = resume_from_cache(args.embeddings_dir, args.output)
    score = partial(
        score_model,
        scopes=scopes,
        baseline_pairs=baseline_pairs,
        background_node_ids=background_node_ids,
        node_to_psalm=node_to_psalm,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    for model_rows in map_in_order(skipping_unscorable(score), model_paths, args.workers):
        if model_rows is None:
            continue
        rows.extend(model_rows)

    for row in rows:
        print(
            f"{row['model']:55s} {row['scope']:12s} "
            f"AP={row['point_ap']:.3f} [{row['ap_ci_low']:.3f}, {row['ap_ci_high']:.3f}] "
            f"(chance={row['prevalence']:.3f}) "
            f"auc={row['point_auc']:.3f} gap={row['point_gap']:.3f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
