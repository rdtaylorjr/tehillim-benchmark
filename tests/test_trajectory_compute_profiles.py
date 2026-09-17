from pathlib import Path

import numpy as np
import pytest
from conftest import semantic_file

from library.errors import InsufficientDataError
from trajectory.scripts.compute_profiles import (
    compute_psalm_profiles,
    distance_rows,
    score_model,
)


def _sequences_and_centroids() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    long_psalm = np.array([[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0.0, 1.0]])
    short_psalm = np.array([[1.0, 0.0], [0.0, 1.0]])
    sequences = {1: long_psalm, 2: short_psalm}
    centroids = {1: long_psalm.mean(axis=0), 2: short_psalm.mean(axis=0)}
    return sequences, centroids


def test_compute_psalm_profiles_skips_psalms_with_too_few_half_verses() -> None:
    sequences, centroids = _sequences_and_centroids()

    profiles = compute_psalm_profiles(sequences, centroids)

    assert set(profiles) == {1}


def test_compute_psalm_profiles_stores_the_real_length_normalized_sequence() -> None:
    sequences, centroids = _sequences_and_centroids()

    profiles = compute_psalm_profiles(sequences, centroids)

    assert profiles[1]["sequence"].shape == (5, 2)
    assert np.allclose(np.linalg.norm(profiles[1]["sequence"], axis=1), 1.0)


def test_compute_psalm_profiles_skips_a_psalm_missing_its_centroid() -> None:
    sequences, centroids = _sequences_and_centroids()
    del centroids[1]

    profiles = compute_psalm_profiles(sequences, centroids)

    assert profiles == {}


def test_distance_rows_has_one_row_per_unordered_psalm_pair() -> None:
    long_a = np.array([[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    long_b = np.array([[0.0, 1.0], [0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    long_c = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    sequences = {1: long_a, 2: long_b, 3: long_c}
    centroids = {p: s.mean(axis=0) for p, s in sequences.items()}
    profiles = compute_psalm_profiles(sequences, centroids)

    rows = distance_rows("model_a", profiles)

    assert len(rows) == 3
    pairs = {(r["psalm_a"], r["psalm_b"]) for r in rows}
    assert pairs == {(1, 2), (1, 3), (2, 3)}
    expected_keys = {
        "content_distance",
        "structural_distance",
        "adjacent_similarity_distance",
        "step_magnitude_distance",
        "turning_angle_distance",
    }
    assert all(expected_keys <= r.keys() for r in rows)


def test_distance_rows_handles_psalms_of_different_lengths_without_resampling() -> None:
    """The whole point of DTW: no shared fixed grid size is needed across psalms."""
    short = np.array([[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    long = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9], [0.0, 1.0]]
    )
    sequences = {1: short, 2: long}
    centroids = {p: s.mean(axis=0) for p, s in sequences.items()}
    profiles = compute_psalm_profiles(sequences, centroids)

    rows = distance_rows("model_a", profiles)

    assert len(rows) == 1
    assert np.isfinite(rows[0]["structural_distance"])


def test_score_model_profiles_in_memory_and_returns_its_distance_rows(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Nothing is written per model: the profile lives only long enough to yield distances."""
    path = write_embeddings_parquet(
        semantic_file(tmp_path, "mine", "v.parquet"),
        {
            1: [1.0, 0.0],
            2: [0.9, 0.1],
            3: [0.7, 0.3],
            4: [0.5, 0.5],
            5: [0.0, 1.0],
            6: [0.1, 0.9],
            7: [0.3, 0.7],
            8: [0.5, 0.5],
        },
    )
    n_profiles, rows = score_model(path, {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]})

    assert list(tmp_path.rglob("*.parquet")) == [path]
    assert n_profiles == 2
    assert [row["model"] for row in rows] == ["mine_consonantal"]
    assert {row["psalm_a"] for row in rows} == {1}
    assert {row["psalm_b"] for row in rows} == {2}


def test_score_model_raises_when_no_psalm_has_a_complete_sequence(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """A zero-vector half-verse drops from the file, so its psalm has no sequence to profile."""
    path = write_embeddings_parquet(
        semantic_file(tmp_path, "mine", "v.parquet"),
        {1: [1.0, 0.0], 2: [0.0, 0.0], 3: [0.7, 0.3], 4: [0.5, 0.5]},
    )
    with pytest.raises(InsufficientDataError, match="mine"):
        score_model(path, {1: [1, 2, 3, 4]})
