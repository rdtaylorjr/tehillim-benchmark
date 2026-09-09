"""Psalm-clustered BCa bootstrap CI (Efron 1987) for AP, AUC, and the effect-size gap."""

from collections.abc import Iterator

import numpy as np

from library.ap_gap_auc_bootstrap import ApGapAucCI, Split, bootstrap_ap_gap_and_auc
from library.calibration import BackgroundStats
from library.protocol import DEFAULT_N_RESAMPLES
from parallelism.node_pairs import NodePairs, pair_similarities

PsalmIndices = dict[int, np.ndarray]


def _psalm_indices(pair_psalms: np.ndarray, psalms: np.ndarray) -> PsalmIndices:
    """Which pair rows belong to each psalm, so a resample can draw whole psalms at once."""
    return {int(psalm): np.flatnonzero(pair_psalms == psalm) for psalm in psalms}


def _cluster_resamples(
    n_resamples: int,
    rng: np.random.Generator,
    psalms: np.ndarray,
    true_sims: np.ndarray,
    true_idx_by_psalm: PsalmIndices,
    baseline_sims: np.ndarray,
    baseline_idx_by_psalm: PsalmIndices,
) -> Iterator[Split]:
    """n_resamples draws of whole psalms with replacement, carrying all of each psalm's pairs."""
    for _ in range(n_resamples):
        chosen = rng.choice(psalms, size=len(psalms), replace=True)
        true_idx = np.concatenate([true_idx_by_psalm[p] for p in chosen])
        baseline_idx = np.concatenate([baseline_idx_by_psalm[p] for p in chosen])
        yield true_sims[true_idx], baseline_sims[baseline_idx]


def _leave_one_psalm_out_splits(
    psalms: np.ndarray,
    true_sims: np.ndarray,
    true_idx_by_psalm: PsalmIndices,
    baseline_sims: np.ndarray,
    baseline_idx_by_psalm: PsalmIndices,
) -> Iterator[Split]:
    """Each psalm's leave-one-out split, in psalm order, for the BCa acceleration jackknife."""
    all_true_idx = np.arange(len(true_sims))
    all_baseline_idx = np.arange(len(baseline_sims))
    for psalm in psalms:
        true_idx = np.setdiff1d(all_true_idx, true_idx_by_psalm[int(psalm)], assume_unique=True)
        baseline_idx = np.setdiff1d(
            all_baseline_idx, baseline_idx_by_psalm[int(psalm)], assume_unique=True
        )
        yield true_sims[true_idx], baseline_sims[baseline_idx]


def block_bootstrap_ap_gap_and_auc(
    true_pairs: NodePairs,
    baseline_pairs: NodePairs,
    node_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
    node_to_psalm: dict[int, int],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    *,
    rng: np.random.Generator,
) -> ApGapAucCI:
    """BCa 95% CI for AP (primary), gap, and AUC, resampling whole psalms."""
    return block_bootstrap_ap_gap_and_auc_from_similarities(
        true_pairs,
        baseline_pairs,
        pair_similarities(true_pairs, node_vectors),
        pair_similarities(baseline_pairs, node_vectors),
        background,
        node_to_psalm,
        n_resamples=n_resamples,
        rng=rng,
    )


def block_bootstrap_ap_gap_and_auc_from_similarities(
    true_pairs: NodePairs,
    baseline_pairs: NodePairs,
    true_sims: np.ndarray,
    baseline_sims: np.ndarray,
    background: BackgroundStats,
    node_to_psalm: dict[int, int],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    *,
    rng: np.random.Generator,
) -> ApGapAucCI:
    """Same CI as block_bootstrap_ap_gap_and_auc, from already-computed pair similarities."""
    true_psalms = np.array([node_to_psalm[pair[0][0]] for pair in true_pairs])
    baseline_psalms = np.array([node_to_psalm[pair[0][0]] for pair in baseline_pairs])

    psalms = np.array(sorted(set(true_psalms.tolist()) | set(baseline_psalms.tolist())))
    true_idx_by_psalm = _psalm_indices(true_psalms, psalms)
    baseline_idx_by_psalm = _psalm_indices(baseline_psalms, psalms)

    return bootstrap_ap_gap_and_auc(
        (true_sims, baseline_sims),
        _cluster_resamples(
            n_resamples,
            rng,
            psalms,
            true_sims,
            true_idx_by_psalm,
            baseline_sims,
            baseline_idx_by_psalm,
        ),
        _leave_one_psalm_out_splits(
            psalms, true_sims, true_idx_by_psalm, baseline_sims, baseline_idx_by_psalm
        ),
        background,
    )
