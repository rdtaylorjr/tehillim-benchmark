"""Scores a tehillim-embeddings Parquet vector file against the parallelism benchmark."""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from library.cli import add_scoring_arguments
from library.embeddings import is_sparse_embeddings, load_embeddings, load_sparse_embeddings
from library.errors import BenchmarkDataError, InsufficientDataError
from library.protocol import DEFAULT_N_PERMUTATIONS
from library.retrieval_metrics import (
    DiscriminationResult,
    PermutationResult,
    cosine_similarity_matrix,
    mean_reciprocal_rank,
    paired_discrimination_test,
    ranks_from_similarity_matrix,
    recall_at_k,
    sparse_cosine_similarity_matrix,
    stratified_mean_gap_test,
)
from parallelism.pairs import (
    RetrievalPair,
    build_retrieval_pairs,
    filter_pairs_with_vectors,
)
from parallelism.separation import SeparationResult, similarity_separation
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups

_MISSING_NODES_SHOWN = 10


def _span_lengths(node_lists: list[tuple[int, ...]]) -> np.ndarray:
    """Node count per span; a segmented reduction would misread an empty run as the next span."""
    counts = np.array([len(nodes) for nodes in node_lists])
    if counts.size and counts.min() == 0:
        raise InsufficientDataError("every retrieval pair side needs at least one node")
    return counts


def side_node_lists(
    pairs: list[RetrievalPair], side: Literal["source", "target"]
) -> list[tuple[int, ...]]:
    """Each pair's node span for the requested side, kept explicit so the type survives."""
    return [pair.source_nodes if side == "source" else pair.target_nodes for pair in pairs]


def build_side_vectors(
    pairs: list[RetrievalPair],
    side: Literal["source", "target"],
    node_vectors: dict[int, np.ndarray],
) -> np.ndarray:
    """Mean-pools each pair's source/target node tuple into one vector."""
    node_lists = side_node_lists(pairs, side)
    missing = sorted({n for nodes in node_lists for n in nodes} - set(node_vectors))
    if missing:
        shown = missing[:_MISSING_NODES_SHOWN]
        suffix = "..." if len(missing) > _MISSING_NODES_SHOWN else ""
        raise BenchmarkDataError(
            f"embedding file is missing {len(missing)} node id(s): {shown}{suffix}"
        )
    counts = _span_lengths(node_lists)
    flat_vectors = np.stack([node_vectors[n] for nodes in node_lists for n in nodes])
    # A segmented reduction over the same elements in the same order as scatter-add, 5x faster.
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    sums = np.add.reduceat(flat_vectors.astype(np.float64), starts, axis=0)
    return np.asarray(sums / counts[:, None])


def build_side_vectors_sparse(
    pairs: list[RetrievalPair],
    side: Literal["source", "target"],
    node_ids: list[int],
    node_vectors: sp.csr_matrix,
) -> sp.csr_matrix:
    """Sparse mean-pool analogue of build_side_vectors: pools via matmul, never densifies."""
    node_index = {n: i for i, n in enumerate(node_ids)}
    node_lists = side_node_lists(pairs, side)
    missing = sorted({n for nodes in node_lists for n in nodes} - set(node_index))
    if missing:
        shown = missing[:_MISSING_NODES_SHOWN]
        suffix = "..." if len(missing) > _MISSING_NODES_SHOWN else ""
        raise BenchmarkDataError(
            f"embedding file is missing {len(missing)} node id(s): {shown}{suffix}"
        )
    row_lengths = _span_lengths(node_lists)
    group_ids = np.repeat(np.arange(len(node_lists)), row_lengths)
    flat_cols = np.array([node_index[n] for nodes in node_lists for n in nodes])
    weights = 1.0 / np.repeat(row_lengths, row_lengths)
    pooling = sp.csr_matrix(
        (weights, (group_ids, flat_cols)), shape=(len(node_lists), len(node_ids))
    )
    return sp.csr_matrix(pooling @ node_vectors)


@dataclass
class TypeReport:
    """One parallelism type's retrieval and separation scores for a single model."""

    parallelism_type: str
    n_pairs: int
    separation: SeparationResult
    discrimination: DiscriminationResult
    mrr_forward: float
    mrr_backward: float
    recall_at_1_forward: float
    recall_at_5_forward: float
    recall_at_1_backward: float
    recall_at_5_backward: float


@dataclass
class EvaluationReport:
    """One model's parallelism scores overall and broken down by parallelism type."""

    n_pairs: int
    separation: SeparationResult
    discrimination: DiscriminationResult
    type_gap: PermutationResult
    by_type: list[TypeReport]
    mrr_forward: float
    mrr_backward: float
    recall_at_1_forward: float
    recall_at_5_forward: float
    recall_at_10_forward: float
    recall_at_1_backward: float
    recall_at_5_backward: float
    recall_at_10_backward: float


def _report_from_pair_similarities(
    pairs: list[RetrievalPair],
    similarities: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> EvaluationReport:
    """Shared retrieval/discrimination/type-stratified report step for the dense and sparse paths"""
    pair_ids = [p.pair_id for p in pairs]
    ranks_forward = ranks_from_similarity_matrix(similarities, pair_ids, true_target_ids=pair_ids)
    ranks_backward = ranks_from_similarity_matrix(
        similarities.T, pair_ids, true_target_ids=pair_ids
    )

    n = len(pair_ids)
    if n < 2:
        raise InsufficientDataError("retrieval scoring needs at least two retrieval pairs")
    true_similarities = np.diag(similarities)
    off_diagonal_by_row = similarities[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    null_similarities = off_diagonal_by_row.mean(axis=1)
    discrimination = paired_discrimination_test(true_similarities, null_similarities)
    separation = similarity_separation(similarities)

    types = np.array([p.parallelism_type for p in pairs])
    type_gap = stratified_mean_gap_test(similarities, types, n_permutations=n_permutations, rng=rng)

    ranks_forward_arr = np.asarray(ranks_forward)
    ranks_backward_arr = np.asarray(ranks_backward)
    by_type = []
    for ptype in sorted(set(types.tolist())):
        idx = np.flatnonzero(types == ptype)
        by_type.append(
            TypeReport(
                parallelism_type=ptype,
                n_pairs=len(idx),
                separation=similarity_separation(similarities, row_mask=types == ptype),
                discrimination=paired_discrimination_test(
                    true_similarities[idx], null_similarities[idx]
                ),
                mrr_forward=mean_reciprocal_rank(ranks_forward_arr[idx]),
                mrr_backward=mean_reciprocal_rank(ranks_backward_arr[idx]),
                recall_at_1_forward=recall_at_k(ranks_forward_arr[idx], 1),
                recall_at_5_forward=recall_at_k(ranks_forward_arr[idx], 5),
                recall_at_1_backward=recall_at_k(ranks_backward_arr[idx], 1),
                recall_at_5_backward=recall_at_k(ranks_backward_arr[idx], 5),
            )
        )

    return EvaluationReport(
        n_pairs=n,
        separation=separation,
        discrimination=discrimination,
        type_gap=type_gap,
        by_type=by_type,
        mrr_forward=mean_reciprocal_rank(ranks_forward),
        mrr_backward=mean_reciprocal_rank(ranks_backward),
        recall_at_1_forward=recall_at_k(ranks_forward, 1),
        recall_at_5_forward=recall_at_k(ranks_forward, 5),
        recall_at_10_forward=recall_at_k(ranks_forward, 10),
        recall_at_1_backward=recall_at_k(ranks_backward, 1),
        recall_at_5_backward=recall_at_k(ranks_backward, 5),
        recall_at_10_backward=recall_at_k(ranks_backward, 10),
    )


def run_evaluation(
    pairs: list[RetrievalPair],
    node_vectors: dict[int, np.ndarray],
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    *,
    rng: np.random.Generator,
) -> EvaluationReport:
    """Runs bidirectional retrieval, discrimination, and type-stratified tests for one model."""
    source_vecs = build_side_vectors(pairs, "source", node_vectors)
    target_vecs = build_side_vectors(pairs, "target", node_vectors)
    similarities = cosine_similarity_matrix(source_vecs, target_vecs)
    return _report_from_pair_similarities(pairs, similarities, n_permutations, rng)


def run_evaluation_sparse(
    pairs: list[RetrievalPair],
    node_ids: list[int],
    node_vectors: sp.csr_matrix,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    *,
    rng: np.random.Generator,
) -> EvaluationReport:
    """Same report as run_evaluation, pooling and comparing sparse vectors without densifying."""
    source_vecs = build_side_vectors_sparse(pairs, "source", node_ids, node_vectors)
    target_vecs = build_side_vectors_sparse(pairs, "target", node_ids, node_vectors)
    similarities = sparse_cosine_similarity_matrix(source_vecs, target_vecs)
    return _report_from_pair_similarities(pairs, similarities, n_permutations, rng)


def score_embedding_file(
    path: Path,
    pairs: list[RetrievalPair],
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[list[RetrievalPair], EvaluationReport]:
    """Scores one embeddings file, pooling sparsely when the file is stored in the sparse layout."""
    if is_sparse_embeddings(path):
        node_ids, matrix = load_sparse_embeddings(path)
        usable = filter_pairs_with_vectors(pairs, set(node_ids))
        return usable, run_evaluation_sparse(
            usable, node_ids, matrix, n_permutations=n_permutations, rng=rng
        )
    node_vectors = load_embeddings(path)
    usable = filter_pairs_with_vectors(pairs, node_vectors)
    return usable, run_evaluation(usable, node_vectors, n_permutations=n_permutations, rng=rng)


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding_file", type=Path, help="a tehillim-embeddings Parquet file")
    add_scoring_arguments(parser, with_permutations=True, with_seed=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    pairs = build_retrieval_pairs(groups)
    rng = np.random.default_rng(args.seed)
    _, report = score_embedding_file(
        args.embedding_file, pairs, n_permutations=args.n_permutations, rng=rng
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
