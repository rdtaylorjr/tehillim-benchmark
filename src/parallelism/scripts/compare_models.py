"""Runs run_evaluation across every embedding file in a directory, ranked by separation AUC."""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np

from library.cli import add_embeddings_dir_argument, add_scoring_arguments, resume_from_cache
from library.embeddings import dataset_identifier
from library.protocol import DEFAULT_N_GROUP_PERMUTATIONS
from library.rows_output import write_rows_csv
from library.scoring import skipping_unscorable
from library.worker_pool import map_in_order
from parallelism.evaluate import score_embedding_file
from parallelism.pairs import RetrievalPair, build_retrieval_pairs
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups


def score_model(
    path: Path,
    pairs: list[RetrievalPair],
    n_permutations: int,
    seed: int,
) -> dict[str, str | int | float]:
    """One model file's metric row; each model is scored independently of every other."""
    model = dataset_identifier(path)
    rng = np.random.default_rng(seed)
    _, report = score_embedding_file(path, pairs, n_permutations=n_permutations, rng=rng)
    row: dict[str, str | int | float] = {
        "model": model,
        "n_pairs": report.n_pairs,
        "separation_auc": report.separation.auc,
        "separation_p": report.separation.p_value,
        "discrimination_p": report.discrimination.p_value,
        "discrimination_rank_biserial": report.discrimination.rank_biserial,
        "type_gap_z": report.type_gap.z_score,
        "type_gap_p": report.type_gap.p_value,
        "mrr_forward": report.mrr_forward,
        "mrr_backward": report.mrr_backward,
        "recall_at_1_forward": report.recall_at_1_forward,
        "recall_at_5_forward": report.recall_at_5_forward,
        "recall_at_10_forward": report.recall_at_10_forward,
        "recall_at_1_backward": report.recall_at_1_backward,
        "recall_at_5_backward": report.recall_at_5_backward,
        "recall_at_10_backward": report.recall_at_10_backward,
    }
    for type_report in report.by_type:
        suffix = type_report.parallelism_type
        row[f"n_pairs_{suffix}"] = type_report.n_pairs
        row[f"separation_auc_{suffix}"] = type_report.separation.auc
        row[f"separation_p_{suffix}"] = type_report.separation.p_value
        row[f"discrimination_p_{suffix}"] = type_report.discrimination.p_value
        row[f"discrimination_rank_biserial_{suffix}"] = type_report.discrimination.rank_biserial
        row[f"mrr_forward_{suffix}"] = type_report.mrr_forward
        row[f"mrr_backward_{suffix}"] = type_report.mrr_backward
        row[f"recall_at_1_forward_{suffix}"] = type_report.recall_at_1_forward
        row[f"recall_at_5_forward_{suffix}"] = type_report.recall_at_5_forward
        row[f"recall_at_1_backward_{suffix}"] = type_report.recall_at_1_backward
        row[f"recall_at_5_backward_{suffix}"] = type_report.recall_at_5_backward
    return row


def compare_models(
    pairs: list[RetrievalPair],
    model_paths: list[Path],
    n_permutations: int = DEFAULT_N_GROUP_PERMUTATIONS,
    seed: int = 0,
    max_workers: int | None = None,
) -> list[dict[str, str | int | float]]:
    """Scores every model file across workers, rows sorted by separation AUC descending."""
    score = partial(score_model, pairs=pairs, n_permutations=n_permutations, seed=seed)
    scored = map_in_order(skipping_unscorable(score), model_paths, max_workers)
    rows = [row for row in scored if row is not None]
    rows.sort(key=lambda r: cast("float", r["separation_auc"]), reverse=True)
    return rows


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_embeddings_dir_argument(parser)
    add_scoring_arguments(parser, with_seed=True, with_group_permutations=True)
    args = parser.parse_args(argv)

    api = api_factory(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    pairs = build_retrieval_pairs(groups)

    cached_rows, model_paths = resume_from_cache(args.embeddings_dir, args.output)

    new_rows = compare_models(
        pairs,
        model_paths,
        n_permutations=args.n_permutations,
        seed=args.seed,
        max_workers=args.workers,
    )
    rows = sorted(cached_rows + new_rows, key=lambda r: r["separation_auc"], reverse=True)

    for row in rows:
        print(
            f"{row['model']:55s} auc={row['separation_auc']:.4f} "
            f"discrimination_r={row['discrimination_rank_biserial']:.4f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
