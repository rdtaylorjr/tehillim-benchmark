"""Westfall and Young (1993) maxT permutation test over permuted group labels."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class MaxTResult(NamedTuple):
    """Per-group permutation p and the family-wise maxT p, one entry per group."""

    p_per_group: np.ndarray
    p_maxt: np.ndarray


def permuted_label_batches(
    codes: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """One relabelling per row, drawn once so every group is tested against the same draws."""
    return np.asarray(rng.permuted(np.tile(codes, (n_permutations, 1)), axis=1))


def _rank_p(observed: float, null: np.ndarray) -> float:
    """(count + 1) / (n + 1), the rank convention whose floor bounds the smallest reachable p."""
    valid = null[~np.isnan(null)]
    if valid.size == 0:
        return float("nan")
    return float((np.sum(valid >= observed) + 1) / (valid.size + 1))


def _per_draw_max(null: np.ndarray) -> np.ndarray:
    """Largest statistic in each draw, NaN where no group could be scored and without warning."""
    per_draw = np.full(len(null), np.nan)
    scored = ~np.all(np.isnan(null), axis=1) if null.shape[1] else np.zeros(len(null), dtype=bool)
    if np.any(scored):
        per_draw[scored] = np.nanmax(null[scored], axis=1)
    return per_draw


def maxt_p_values(observed: np.ndarray, null: np.ndarray) -> MaxTResult:
    """Per-group p from that group's null, and maxT p from the null's per-draw maximum."""
    max_null = _per_draw_max(null)
    per_group = np.array([_rank_p(observed[g], null[:, g]) for g in range(len(observed))])
    family_wise = np.array([_rank_p(value, max_null) for value in observed])
    return MaxTResult(p_per_group=per_group, p_maxt=family_wise)


@dataclass(frozen=True)
class GroupPermutationResult:
    """One-vs-rest permutation outcome per group, whatever statistic the caller measured."""

    genres: tuple[str, ...]
    observed: tuple[float, ...]
    p_perm: tuple[float, ...]
    p_maxt: tuple[float, ...]
    n_permutations: int
