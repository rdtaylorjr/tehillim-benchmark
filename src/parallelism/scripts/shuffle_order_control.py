"""Order-shuffle null: does half-verse order carry parallelism signal beyond a shuffled null."""

import argparse
import csv
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from library.cli import add_scoring_arguments
from library.embeddings import (
    is_sparse_embeddings,
    load_embeddings,
    load_sparse_embeddings,
)
from library.order_shuffle import order_shuffle_result
from library.retrieval_metrics import cosine_similarity_matrix, sparse_cosine_similarity_matrix
from library.shuffle_draws import select_shuffle_draws
from library.worker_pool import map_in_order
from parallelism.evaluate import build_side_vectors, build_side_vectors_sparse
from parallelism.pairs import RetrievalPair, build_retrieval_pairs, filter_pairs_with_vectors
from parallelism.separation import similarity_separation
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups


def score_separation_auc(path: Path, all_pairs: list[RetrievalPair]) -> float:
    """Separation AUC (no permutation testing) for one embeddings file against all_pairs."""
    if is_sparse_embeddings(path):
        node_ids, matrix = load_sparse_embeddings(path)
        pairs = filter_pairs_with_vectors(all_pairs, set(node_ids))
        similarities = sparse_cosine_similarity_matrix(
            build_side_vectors_sparse(pairs, "source", node_ids, matrix),
            build_side_vectors_sparse(pairs, "target", node_ids, matrix),
        )
        return similarity_separation(similarities).auc
    node_vectors = load_embeddings(path)
    pairs = filter_pairs_with_vectors(all_pairs, node_vectors)
    source_vecs = build_side_vectors(pairs, "source", node_vectors)
    target_vecs = build_side_vectors(pairs, "target", node_vectors)
    similarities = cosine_similarity_matrix(source_vecs, target_vecs)
    return similarity_separation(similarities).auc


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_embeddings", type=Path)
    parser.add_argument("shuffled_embeddings_dir", type=Path)
    add_scoring_arguments(parser, with_shuffles=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    all_pairs = build_retrieval_pairs(groups)

    auc_real = score_separation_auc(args.real_embeddings, all_pairs)
    shuffled_paths = select_shuffle_draws(args.shuffled_embeddings_dir, args.n_shuffles)
    score = partial(score_separation_auc, all_pairs=all_pairs)
    auc_shuffled = map_in_order(score, shuffled_paths, args.workers)

    result = order_shuffle_result(real_score=auc_real, shuffled_scores=np.array(auc_shuffled))
    print(
        f"auc_real={auc_real:.4f} auc_shuffled_mean={np.mean(auc_shuffled):.4f} "
        f"delta_order={result.delta_order:+.4f} p={result.p_value:.4f}"
    )

    if args.output:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "auc_real",
                    "auc_shuffled_mean",
                    "n_shuffles",
                    "delta_order",
                    "p_value",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "auc_real": auc_real,
                    "auc_shuffled_mean": float(np.mean(auc_shuffled)),
                    "n_shuffles": len(auc_shuffled),
                    "delta_order": result.delta_order,
                    "p_value": result.p_value,
                }
            )


if __name__ == "__main__":
    main()
