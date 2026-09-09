from pathlib import Path

import numpy as np
import pytest

from library.calibration import BackgroundStats, background_similarity_stats
from parallelism.node_pairs import as_node_pairs, pair_similarities
from parallelism.scripts.compare_baseline import compare_to_baseline, score_model


def test_compare_to_baseline_reports_higher_effect_size_when_true_pairs_are_closer() -> None:
    true_pairs = as_node_pairs([(1, 2), (3, 4)])
    baseline_pairs = as_node_pairs([(5, 6), (7, 8), (9, 10)])
    node_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),  # identical: similarity 1.0
        3: np.array([1.0, 0.0]),
        4: np.array([0.99, 0.14107]),  # nearly identical: high similarity
        5: np.array([1.0, 0.0]),
        6: np.array([0.0, 1.0]),  # orthogonal: similarity 0.0
        7: np.array([1.0, 0.0]),
        8: np.array([0.0, 1.0]),
        9: np.array([1.0, 0.0]),
        10: np.array([0.0, 1.0]),
    }
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=10)

    result = compare_to_baseline(true_pairs, baseline_pairs, node_vectors, background)

    assert result.n_true == 2
    assert result.n_baseline == 3
    assert result.mean_true_similarity > result.mean_baseline_similarity
    assert result.true_effect_size > result.baseline_effect_size
    assert result.separation_auc == 1.0
    assert result.separation_p < 0.1  # n=2 vs n=3 caps Mann-Whitney's minimum p just above 0.05
    # Perfect separation gives AP = 1.0, the MTEB Pair Classification convention.
    assert result.average_precision == 1.0
    assert result.prevalence == 2 / 5


def test_compare_to_baseline_mean_pools_multi_node_spans() -> None:
    """A member spanning two half-verse nodes must be pooled, not silently dropped."""
    true_pairs = as_node_pairs([(1, 2)])
    true_pairs_multi = [((1, 10), (2,))]  # source spans nodes 1 and 10
    baseline_pairs = as_node_pairs([(5, 6)])
    node_vectors = {
        1: np.array([1.0, 0.0]),
        10: np.array([1.0, 0.0]),  # same direction as node 1, so pooling changes nothing here
        2: np.array([1.0, 0.0]),
        5: np.array([1.0, 0.0]),
        6: np.array([0.0, 1.0]),
    }
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=10)

    single = compare_to_baseline(true_pairs, baseline_pairs, node_vectors, background)
    multi = compare_to_baseline(true_pairs_multi, baseline_pairs, node_vectors, background)

    assert multi.n_true == 1
    assert multi.mean_true_similarity == single.mean_true_similarity


def test_compare_to_baseline_average_precision_chance_level_is_prevalence_not_half() -> None:
    """With zero discrimination (every pair, true or baseline, has identical similarity)."""
    true_pairs = as_node_pairs([(1, 2)])
    baseline_pairs = as_node_pairs([(3, 4), (5, 6), (7, 8)])
    node_vectors = {n: np.array([1.0, 0.0]) for n in range(1, 9)}
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=10)

    result = compare_to_baseline(true_pairs, baseline_pairs, node_vectors, background)

    assert result.prevalence == pytest.approx(1 / 4)
    assert result.average_precision == pytest.approx(1 / 4)


def test_score_model_loads_a_file_and_names_the_row_after_its_dataset_identifier(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=mine" / "v.parquet",
        {1: [1.0, 0.0], 2: [1.0, 0.0], 5: [1.0, 0.0], 6: [0.0, 1.0], 7: [0.7, 0.7]},
    )

    row = score_model(path, as_node_pairs([(1, 2)]), as_node_pairs([(5, 6)]), [5, 6, 7])

    assert row["model"] == "mine"
    assert row["n_true"] == 1
    assert row["n_baseline"] == 1


def test_score_model_reports_the_gap_between_true_and_baseline_effect_sizes(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """True pairs are identical vectors and baseline pairs orthogonal, so the gap is positive."""
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=m" / "v.parquet",
        {
            1: [1.0, 0.0],
            2: [1.0, 0.0],
            3: [1.0, 0.0],
            4: [1.0, 0.0],
            5: [1.0, 0.0],
            6: [0.0, 1.0],
            7: [0.7071, 0.7071],
            8: [-1.0, 0.0],
        },
    )

    row = score_model(path, as_node_pairs([(1, 2), (3, 4)]), as_node_pairs([(5, 6)]), [5, 6, 7, 8])

    assert row["gap"] == row["true_effect_size"] - row["baseline_effect_size"]
    assert row["gap"] > 0


def test_compare_to_baseline_from_similarities_matches_the_vector_entry_point() -> None:
    """Splitting similarity computation from the statistics must not change the statistics."""
    from parallelism.scripts.compare_baseline import compare_to_baseline_from_similarities

    rng = np.random.default_rng(9)
    node_vectors = {n: rng.standard_normal(16) for n in range(1, 21)}
    true_pairs = [((n,), (n + 1,)) for n in range(1, 10, 2)]
    baseline_pairs = [((n,), (n + 2,)) for n in range(11, 19, 2)]
    background = background_similarity_stats(np.stack(list(node_vectors.values())))

    expected = compare_to_baseline(true_pairs, baseline_pairs, node_vectors, background)
    actual = compare_to_baseline_from_similarities(
        pair_similarities(true_pairs, node_vectors),
        pair_similarities(baseline_pairs, node_vectors),
        background,
    )

    assert actual == expected
