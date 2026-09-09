"""Diffs two pipeline output trees, separating the expected-change ledger from real regressions."""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Columns RERUN.md's documented causes move, so anything outside this ledger is a regression.
_CI_COLUMNS = frozenset(
    {
        "ap_ci_low",
        "ap_ci_high",
        "ap_ci_low_pct",
        "ap_ci_high_pct",
        "gap_ci_low",
        "gap_ci_high",
        "gap_ci_low_pct",
        "gap_ci_high_pct",
        "auc_ci_low",
        "auc_ci_high",
        "auc_ci_low_pct",
        "auc_ci_high_pct",
        "n_valid_resamples",
        "n_valid_jackknife",
        "raw_effect_size",
        "length_controlled_effect_size",
        "length_and_content_controlled_effect_size",
    }
)
#: Causes 4 and 12: the hypothesis set changed, so every corrected p-value moves with it.
_Q_VALUE_COLUMNS = frozenset(
    {
        "q_value",
        "q_value_by",
        "naive_q",
        "naive_q_by",
        "perm_q",
        "perm_q_by",
        "maxT_q",
        "maxT_q_by",
    }
)
#: Causes 5 and 13: regenerated embeddings and one similarity route move rank-derived estimates.
_POINT_ESTIMATE_PREFIXES = (
    "average_precision",
    "separation_auc",
    "separation_p",
    "discrimination_p",
    "point_ap",
    "point_auc",
    "point_gap",
    "gap",
    "prevalence",
    "mrr_",
    "recall_at_",
    "calibrated_effect_size",
    "auc_vs_baseline",
    "type_gap_z",
    "mean_true_similarity",
    "median_true_similarity",
    "std_true_similarity",
    "background_mean",
    "background_std",
    "true_effect_size",
    "baseline_effect_size",
    "same_genre_effect_size",
    "different_genre_effect_size",
)
EXPECTED_DIFF_COLUMNS = _CI_COLUMNS | _Q_VALUE_COLUMNS

#: Cause 4: draws of the order-shuffle null were scored as models and no longer are.
_SHUFFLE_DRAW = re.compile(r"shuffle\d+$")
#: Cause 9: the hive path refactor moved the syntax models under level=phrase.
_RENAMED_PREFIXES = (("syntax_", "phrase_"),)


def _explained_row_change(only_old: set[object], only_new: set[object]) -> bool:
    """Whether every row that appeared or vanished is one a documented cause accounts for."""

    def name(key: object) -> str:
        return str(key[0] if isinstance(key, tuple) else key)

    renamed_away = {
        name(k).replace(old, new, 1)
        for k in only_old
        for old, new in _RENAMED_PREFIXES
        if name(k).startswith(old)
    }
    unexplained_old = {
        k
        for k in only_old
        if not _SHUFFLE_DRAW.search(name(k))
        and not name(k).startswith(tuple(o for o, _ in _RENAMED_PREFIXES))
    }
    unexplained_new = {k for k in only_new if name(k) not in renamed_away}
    return not unexplained_old and not unexplained_new


_KEY_CANDIDATES = ("model", "scope", "genre", "metric", "source", "psalm_a", "psalm_b", "pair_id")
_TOLERANCE = 0.0


@dataclass(frozen=True)
class ColumnDiff:
    """One column's disagreement between an old and a new run of the same output file."""

    column: str
    kind: str
    n_changed: int
    max_abs_delta: float


@dataclass
class TreeReport:
    """Every disagreement found between two output trees."""

    files: dict[str, list[ColumnDiff]] = field(default_factory=dict)
    missing_in_new: list[str] = field(default_factory=list)
    missing_in_old: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Clean means every difference was one the ledger predicted, and no file went missing."""
        if self.missing_in_new or self.missing_in_old:
            return False
        return all(d.kind == "expected" for diffs in self.files.values() for d in diffs)


def classify_column(column: str) -> str:
    """Whether a differing column is one the ledger predicted or a genuine regression."""
    if column in EXPECTED_DIFF_COLUMNS:
        return "expected"
    if column.startswith(_POINT_ESTIMATE_PREFIXES):
        return "expected"
    return "unexpected"


def key_columns_for(frame: pd.DataFrame) -> list[str]:
    """The identifying columns to align two runs' rows on, so row order never reads as a diff."""
    return [column for column in _KEY_CANDIDATES if column in frame.columns]


def _both_nan(old: pd.Series, new: pd.Series) -> pd.Series:
    return old.isna() & new.isna()


def compare_frames(
    old: pd.DataFrame, new: pd.DataFrame, key_columns: list[str]
) -> list[ColumnDiff]:
    """Per-column differences between two runs of one output, aligned on key_columns."""
    if not key_columns:
        return []
    old_indexed = old.set_index(key_columns).sort_index()
    new_indexed = new.set_index(key_columns).sort_index()

    shared = old_indexed.index.intersection(new_indexed.index)
    only_old = set(old_indexed.index.difference(shared))
    only_new = set(new_indexed.index.difference(shared))
    diffs = []
    if only_old or only_new:
        kind = "expected" if _explained_row_change(only_old, only_new) else "unexpected"
        diffs.append(ColumnDiff("<rows>", kind, len(only_old) + len(only_new), float("nan")))

    old_rows, new_rows = old_indexed.loc[shared], new_indexed.loc[shared]
    for column in old_rows.columns.intersection(new_rows.columns):
        left, right = old_rows[column], new_rows[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            delta = (left - right).abs()
            changed = ~(_both_nan(left, right) | (delta <= _TOLERANCE))
            max_delta = float(delta[changed].max()) if changed.any() else 0.0
        else:
            changed = ~(_both_nan(left, right) | (left == right))
            max_delta = float("nan")
        if changed.any():
            diffs.append(
                ColumnDiff(str(column), classify_column(str(column)), int(changed.sum()), max_delta)
            )
    return diffs


def _read(path: Path) -> pd.DataFrame:
    """Reads one output file, whichever of the pipeline's three formats it uses."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _relative_outputs(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root)): path
        for suffix in ("*.csv", "*.parquet", "*.json")
        for path in root.rglob(suffix)
    }


def compare_trees(old_root: Path, new_root: Path) -> TreeReport:
    """Compares every output file common to both trees, reporting any that exists in only one."""
    old_files, new_files = _relative_outputs(old_root), _relative_outputs(new_root)
    report = TreeReport(
        missing_in_new=sorted(set(old_files) - set(new_files)),
        missing_in_old=sorted(set(new_files) - set(old_files)),
    )
    for name in sorted(set(old_files) & set(new_files)):
        old_frame, new_frame = _read(old_files[name]), _read(new_files[name])
        diffs = compare_frames(old_frame, new_frame, key_columns_for(old_frame))
        if diffs:
            report.files[name] = diffs
    return report


def main() -> None:
    """Compares two output trees and reports every column that moved."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_dir", type=Path, help="outputs from before the change")
    parser.add_argument("new_dir", type=Path, help="outputs from the rerun")
    args = parser.parse_args()

    report = compare_trees(args.old_dir, args.new_dir)
    for name in report.missing_in_new:
        print(f"MISSING in new: {name}")
    for name in report.missing_in_old:
        print(f"MISSING in old: {name}")
    for name, diffs in report.files.items():
        for diff in diffs:
            marker = "ok  " if diff.kind == "expected" else "FAIL"
            print(
                f"{marker} {name}::{diff.column} "
                f"changed={diff.n_changed} max={diff.max_abs_delta:.3g}"
            )
    verdict = "CLEAN: every difference was predicted" if report.is_clean else "UNEXPECTED diffs"
    print(f"\n{verdict}")
    sys.exit(0 if report.is_clean else 1)


if __name__ == "__main__":
    main()
