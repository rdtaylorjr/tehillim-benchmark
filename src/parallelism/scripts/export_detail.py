"""Exports row-per-observation detail: pair/baseline similarity and per-type vs-baseline stats."""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from library.bhsa import (
    list_psalms_half_verse_nodes,
    list_psalms_half_verses_by_psalm,
)
from library.calibration import (
    BackgroundStats,
    background_similarity_stats,
    background_similarity_stats_sparse,
    calibrated_z_score,
)
from library.cli import add_embeddings_dir_argument, add_scoring_arguments, report_reuse
from library.embeddings import (
    dataset_identifier,
    is_sparse_embeddings,
    load_embeddings,
    load_sparse_embeddings,
)
from library.incremental_cache import load_cached_parquet_set
from library.model_files import uncached_model_paths
from library.retrieval_metrics import (
    cosine_similarity_matrix,
    paired_cosine_similarity,
    ranks_from_similarity_matrix,
    sparse_cosine_similarity_matrix,
    sparse_paired_cosine_similarity,
)
from library.rows_output import write_dataframe_parquet
from library.scoring import skipping_unscorable
from library.worker_pool import map_in_order
from parallelism.baseline import build_unmarked_half_verse_pairs
from parallelism.baseline_comparison import (
    BaselineComparison,
    baseline_metric_fields,
    compare_to_baseline,
    compare_to_baseline_from_similarities,
)
from parallelism.evaluate import build_side_vectors, build_side_vectors_sparse
from parallelism.node_pairs import (
    NodePairs,
    as_node_pairs,
    filter_node_pairs_with_vectors,
    pair_similarities_sparse,
    retrieval_pairs_as_node_pairs,
)
from parallelism.pairs import (
    RetrievalPair,
    build_retrieval_pairs,
    filter_pairs_by_type,
    filter_pairs_with_vectors,
)
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups

_TYPES = frozenset({"Synonymous", "Staircase", "Emblematic", "Synthetic", "Antithetic"})
_OUTPUT_FILES = ("pair_detail.parquet", "baseline_detail.parquet", "type_vs_baseline.parquet")


def build_pair_detail_rows(
    model: str,
    pairs: list[RetrievalPair],
    node_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """One row per pair: raw similarity, per-pair calibrated z, and bidirectional rank."""
    return _pair_detail_rows_from_matrix(
        model,
        pairs,
        cosine_similarity_matrix(
            build_side_vectors(pairs, "source", node_vectors),
            build_side_vectors(pairs, "target", node_vectors),
        ),
        background,
    )


def build_pair_detail_rows_sparse(
    model: str,
    pairs: list[RetrievalPair],
    node_ids: list[int],
    node_vectors: sp.csr_matrix,
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """Sparse analogue of build_pair_detail_rows, pooling and comparing without densifying."""
    return _pair_detail_rows_from_matrix(
        model,
        pairs,
        sparse_cosine_similarity_matrix(
            build_side_vectors_sparse(pairs, "source", node_ids, node_vectors),
            build_side_vectors_sparse(pairs, "target", node_ids, node_vectors),
        ),
        background,
    )


def _pair_detail_rows_from_matrix(
    model: str,
    pairs: list[RetrievalPair],
    similarities: np.ndarray,
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """Same rows as build_pair_detail_rows, from an already-computed similarity matrix."""
    pair_ids = [p.pair_id for p in pairs]
    true_similarities = np.diag(similarities)
    ranks_forward = ranks_from_similarity_matrix(similarities, pair_ids, true_target_ids=pair_ids)
    ranks_backward = ranks_from_similarity_matrix(
        similarities.T, pair_ids, true_target_ids=pair_ids
    )

    rows = []
    for pair, sim, rank_f, rank_b in zip(
        pairs, true_similarities, ranks_forward, ranks_backward, strict=True
    ):
        rows.append(
            {
                "model": model,
                "pair_id": pair.pair_id,
                "group_range": pair.group_range,
                "parallelism_type": pair.parallelism_type,
                "signature": pair.signature,
                "source_nodes": ",".join(map(str, pair.source_nodes)),
                "target_nodes": ",".join(map(str, pair.target_nodes)),
                "source_indicator": pair.source_indicator,
                "target_indicator": pair.target_indicator,
                "n_source_nodes": len(pair.source_nodes),
                "n_target_nodes": len(pair.target_nodes),
                "raw_similarity": float(sim),
                "calibrated_z": calibrated_z_score(float(sim), background),
                "rank_forward": rank_f,
                "reciprocal_rank_forward": 1.0 / rank_f,
                "rank_backward": rank_b,
                "reciprocal_rank_backward": 1.0 / rank_b,
            }
        )
    return rows


def build_baseline_detail_rows(
    model: str,
    baseline_pairs: list[tuple[int, int]],
    node_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """One row per unmarked adjacent half-verse pair: similarity and calibrated z, no rank."""
    return _baseline_detail_rows_from_similarities(
        model,
        baseline_pairs,
        paired_cosine_similarity(
            np.stack([node_vectors[a] for a, _ in baseline_pairs]),
            np.stack([node_vectors[b] for _, b in baseline_pairs]),
        ),
        background,
    )


def build_baseline_detail_rows_sparse(
    model: str,
    baseline_pairs: list[tuple[int, int]],
    node_ids: list[int],
    node_vectors: sp.csr_matrix,
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """Sparse analogue of build_baseline_detail_rows, comparing sparse rows directly."""
    node_index = {n: i for i, n in enumerate(node_ids)}
    sims = sparse_paired_cosine_similarity(
        node_vectors[[node_index[a] for a, _ in baseline_pairs]],
        node_vectors[[node_index[b] for _, b in baseline_pairs]],
    )
    return _baseline_detail_rows_from_similarities(model, baseline_pairs, sims, background)


def _baseline_detail_rows_from_similarities(
    model: str,
    baseline_pairs: list[tuple[int, int]],
    sims: np.ndarray,
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """Same rows as build_baseline_detail_rows, from already-computed pair similarities."""
    rows = []
    for (a, b), sim in zip(baseline_pairs, sims, strict=True):
        rows.append(
            {
                "model": model,
                "node_a": a,
                "node_b": b,
                "raw_similarity": float(sim),
                "calibrated_z": calibrated_z_score(float(sim), background),
            }
        )
    return rows


def _scope_row(
    model: str, scope: str, scope_kind: str, result: BaselineComparison
) -> dict[str, str | int | float]:
    """One detail row for a model within one scope, overall or a single parallelism type."""
    return {
        "model": model,
        "scope": scope,
        "scope_kind": scope_kind,
        **baseline_metric_fields(result),
        "auc_vs_baseline": result.separation_auc,
        "p_vs_baseline": result.separation_p,
    }


def build_type_vs_baseline_rows(
    model: str,
    all_pairs: list[RetrievalPair],
    baseline_pairs: NodePairs,
    node_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """One row per (model, scope): each of the 5 Lowth parallelism types alone, plus overall."""
    rows = []
    for ptype in sorted(_TYPES):
        type_pairs = retrieval_pairs_as_node_pairs(
            filter_pairs_by_type(all_pairs, frozenset({ptype}))
        )
        if not type_pairs:
            continue
        result = compare_to_baseline(type_pairs, baseline_pairs, node_vectors, background)
        rows.append(_scope_row(model, ptype, "type", result))

    result_all = compare_to_baseline(
        retrieval_pairs_as_node_pairs(all_pairs), baseline_pairs, node_vectors, background
    )
    rows.append(_scope_row(model, "overall", "overall", result_all))
    return rows


def build_type_vs_baseline_rows_sparse(
    model: str,
    all_pairs: list[RetrievalPair],
    baseline_pairs: NodePairs,
    node_ids: list[int],
    node_vectors: sp.csr_matrix,
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """Sparse analogue of build_type_vs_baseline_rows, pooling each scope without densifying."""
    baseline_sims = pair_similarities_sparse(baseline_pairs, node_ids, node_vectors)
    rows = []
    for ptype in sorted(_TYPES):
        type_pairs = retrieval_pairs_as_node_pairs(
            filter_pairs_by_type(all_pairs, frozenset({ptype}))
        )
        if not type_pairs:
            continue
        result = compare_to_baseline_from_similarities(
            pair_similarities_sparse(type_pairs, node_ids, node_vectors),
            baseline_sims,
            background,
        )
        rows.append(_scope_row(model, ptype, "type", result))

    result_all = compare_to_baseline_from_similarities(
        pair_similarities_sparse(retrieval_pairs_as_node_pairs(all_pairs), node_ids, node_vectors),
        baseline_sims,
        background,
    )
    rows.append(_scope_row(model, "overall", "overall", result_all))
    return rows


def score_model(
    path: Path,
    all_pairs: list[RetrievalPair],
    baseline_pairs: NodePairs,
    baseline_pairs_raw: list[tuple[int, int]],
    background_node_ids: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """One model file's (pair, baseline, type-scope) rows, independent of every other model."""
    model = dataset_identifier(path)
    if is_sparse_embeddings(path):
        node_ids, matrix = load_sparse_embeddings(path)
        node_index = {n: i for i, n in enumerate(node_ids)}
        background_rows = [node_index[n] for n in background_node_ids if n in node_index]
        background = background_similarity_stats_sparse(matrix[background_rows])
        model_pairs = filter_pairs_with_vectors(all_pairs, node_index)
        model_baseline_pairs_raw = [
            (a, b) for a, b in baseline_pairs_raw if a in node_index and b in node_index
        ]
        model_baseline_pairs = filter_node_pairs_with_vectors(baseline_pairs, node_index)
        return (
            build_pair_detail_rows_sparse(model, model_pairs, node_ids, matrix, background),
            build_baseline_detail_rows_sparse(
                model, model_baseline_pairs_raw, node_ids, matrix, background
            ),
            build_type_vs_baseline_rows_sparse(
                model, model_pairs, model_baseline_pairs, node_ids, matrix, background
            ),
        )

    node_vectors = load_embeddings(path)
    background_vecs = np.stack([node_vectors[n] for n in background_node_ids if n in node_vectors])
    background = background_similarity_stats(background_vecs)
    model_pairs = filter_pairs_with_vectors(all_pairs, node_vectors)
    model_baseline_pairs_raw = [
        (a, b) for a, b in baseline_pairs_raw if a in node_vectors and b in node_vectors
    ]
    model_baseline_pairs = filter_node_pairs_with_vectors(baseline_pairs, node_vectors)
    return (
        build_pair_detail_rows(model, model_pairs, node_vectors, background),
        build_baseline_detail_rows(model, model_baseline_pairs_raw, node_vectors, background),
        build_type_vs_baseline_rows(
            model, model_pairs, model_baseline_pairs, node_vectors, background
        ),
    )


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_embeddings_dir_argument(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_scoring_arguments(parser)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = api_factory(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    all_pairs = build_retrieval_pairs(groups)

    marked_nodes = {n for p in all_pairs for n in p.source_nodes} | {
        n for p in all_pairs for n in p.target_nodes
    }
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)
    baseline_pairs_raw = build_unmarked_half_verse_pairs(half_verses_by_psalm, marked_nodes)
    baseline_pairs = as_node_pairs(baseline_pairs_raw)
    background_node_ids = [n for n in list_psalms_half_verse_nodes(api) if n not in marked_nodes]

    (cached_pair_rows, cached_baseline_rows, cached_scope_rows), cached_models = (
        load_cached_parquet_set(args.output_dir, _OUTPUT_FILES)
    )
    report_reuse(cached_models, args.output_dir)

    model_paths = uncached_model_paths(args.embeddings_dir, cached_models)
    pair_rows: list[dict[str, Any]] = list(cached_pair_rows)
    baseline_rows: list[dict[str, Any]] = list(cached_baseline_rows)
    scope_rows: list[dict[str, Any]] = list(cached_scope_rows)
    score = partial(
        score_model,
        all_pairs=all_pairs,
        baseline_pairs=baseline_pairs,
        baseline_pairs_raw=baseline_pairs_raw,
        background_node_ids=background_node_ids,
    )
    for scored in map_in_order(skipping_unscorable(score), model_paths, args.workers):
        if scored is None:
            continue
        model_pairs, model_baselines, model_scopes = scored
        pair_rows.extend(model_pairs)
        baseline_rows.extend(model_baselines)
        scope_rows.extend(model_scopes)

    write_dataframe_parquet(args.output_dir / "pair_detail.parquet", pd.DataFrame(pair_rows))
    write_dataframe_parquet(
        args.output_dir / "baseline_detail.parquet",
        pd.DataFrame(baseline_rows),
    )
    write_dataframe_parquet(args.output_dir / "type_vs_baseline.parquet", pd.DataFrame(scope_rows))
    print(f"wrote {len(pair_rows)} pair rows, {len(baseline_rows)} baseline rows")
    print(f"wrote {len(scope_rows)} type scope rows")


if __name__ == "__main__":
    main()
