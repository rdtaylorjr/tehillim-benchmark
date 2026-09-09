"""Order-shuffle null: does half-verse order carry parallelism signal beyond a shuffled null."""

import argparse
import csv
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from families.shuffle import Draws, draw, load_draws

from library.cli import add_scoring_arguments, add_shuffle_family_arguments
from library.embeddings import (
    drop_zero_norm_vectors,
    is_sparse_embeddings,
    load_embeddings,
    load_sparse_embeddings,
    sparse_vectors_to_csr,
)
from library.order_shuffle import order_shuffle_result
from library.retrieval_metrics import cosine_similarity_matrix, sparse_cosine_similarity_matrix
from library.worker_pool import map_in_order
from parallelism.evaluate import build_side_vectors, build_side_vectors_sparse
from parallelism.pairs import RetrievalPair, build_retrieval_pairs, filter_pairs_with_vectors
from parallelism.separation import similarity_separation
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups


def score_separation_auc(path: Path, all_pairs: list[RetrievalPair]) -> float:
    """Separation AUC (no permutation testing) for one embeddings file against all_pairs."""
    if is_sparse_embeddings(path):
        node_ids, matrix = load_sparse_embeddings(path)
        return sparse_separation_auc(node_ids, matrix, all_pairs)
    return dense_separation_auc(load_embeddings(path), all_pairs)


def sparse_separation_auc(
    node_ids: list[int], matrix: sp.csr_matrix, all_pairs: list[RetrievalPair]
) -> float:
    """Separation AUC over sparse node vectors already in memory, file or freshly built."""
    pairs = filter_pairs_with_vectors(all_pairs, set(node_ids))
    similarities = sparse_cosine_similarity_matrix(
        build_side_vectors_sparse(pairs, "source", node_ids, matrix),
        build_side_vectors_sparse(pairs, "target", node_ids, matrix),
    )
    return similarity_separation(similarities).auc


def dense_separation_auc(
    node_vectors: dict[int, np.ndarray], all_pairs: list[RetrievalPair]
) -> float:
    """Separation AUC over dense node vectors already in memory, file or freshly built."""
    pairs = filter_pairs_with_vectors(all_pairs, node_vectors)
    similarities = cosine_similarity_matrix(
        build_side_vectors(pairs, "source", node_vectors),
        build_side_vectors(pairs, "target", node_vectors),
    )
    return similarity_separation(similarities).auc


def score_built_seed(draws: Draws, all_pairs: list[RetrievalPair], seed: int) -> float:
    """Builds one seed's draw, scores it, and releases the vectors before the next seed."""
    vectors = draw(draws, seed)
    if draws.sparse_width is None:
        return dense_separation_auc(drop_zero_norm_vectors(vectors), all_pairs)
    return sparse_separation_auc(*sparse_vectors_to_csr(vectors, draws.sparse_width), all_pairs)


def built_null_scores(
    draws: Draws, n_shuffles: int, all_pairs: list[RetrievalPair], workers: int | None
) -> list[float]:
    """One AUC per seed, each draw built in memory and never written to a file."""
    seeds = list(range(1, n_shuffles + 1))
    return map_in_order(partial(score_built_seed, draws, all_pairs), seeds, workers)


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_api,
    draws_factory: Callable[[str, Path], Draws] = load_draws,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_embeddings", type=Path)
    add_shuffle_family_arguments(parser)
    add_scoring_arguments(parser, with_shuffles=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    all_pairs = build_retrieval_pairs(groups)

    auc_real = score_separation_auc(args.real_embeddings, all_pairs)
    draws = draws_factory(args.family, args.config_root)
    auc_shuffled = built_null_scores(draws, args.n_shuffles, all_pairs, args.workers)

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
