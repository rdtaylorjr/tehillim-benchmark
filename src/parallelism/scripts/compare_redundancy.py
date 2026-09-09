"""Conditional redundancy: what a representation adds beyond a reference it may repeat."""

import argparse
from pathlib import Path
from typing import Any

from library.cli import add_scoring_arguments
from library.rows_output import write_rows_csv
from parallelism.pairs import build_retrieval_pairs
from parallelism.redundancy import conditional_redundancy
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups


def compare_redundancy(
    subject_paths: list[Path], reference_paths: list[Path], pairs: list[Any]
) -> list[dict[str, str | int | float]]:
    """Every subject scored against every reference, on the colons the two share."""
    rows: list[dict[str, str | int | float]] = []
    for reference_path in reference_paths:
        for subject_path in subject_paths:
            result = conditional_redundancy(subject_path, reference_path, pairs)
            rows.append(
                {
                    "subject": result.subject,
                    "reference": result.reference,
                    "n_pairs": result.n_pairs,
                    "auc_subject": result.auc_subject,
                    "auc_reference": result.auc_reference,
                    "auc_joined": result.auc_joined,
                    "delta_over_reference": result.delta_over_reference,
                    "delta_over_subject": result.delta_over_subject,
                }
            )
    rows.sort(key=lambda row: float(row["delta_over_reference"]), reverse=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    add_scoring_arguments(parser)
    args = parser.parse_args(argv)

    api = load_api(args.checkout)
    pairs = build_retrieval_pairs(reconstruct_groups(read_node_feature_values(api)))
    rows = compare_redundancy(args.subject, args.reference, pairs)

    for row in rows:
        print(
            f"{row['subject']:28s} over {row['reference']:22s} "
            f"alone={row['auc_subject']:.4f} ref={row['auc_reference']:.4f} "
            f"joined={row['auc_joined']:.4f} adds={row['delta_over_reference']:+.4f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
