from pathlib import Path

import numpy as np

from trajectory.scripts.compute_profiles import (
    compute_psalm_profiles,
    distance_rows,
    profile_rows,
    profile_shard_path,
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


def test_profile_rows_flattens_one_row_per_psalm_with_its_real_half_verse_count() -> None:
    sequences, centroids = _sequences_and_centroids()
    profiles = compute_psalm_profiles(sequences, centroids)

    rows = profile_rows("model_a", profiles)

    assert len(rows) == 1
    assert rows[0]["model"] == "model_a"
    assert rows[0]["psalm"] == 1
    assert rows[0]["n_half_verses"] == 5
    assert rows[0]["dim"] == 2
    assert len(rows[0]["sequence"]) == 10


def test_profile_rows_stores_centroid_and_sequence_as_float32_to_halve_parquet_size() -> None:
    """The source embeddings are float32 already; float64 storage would add no precision."""
    sequences, centroids = _sequences_and_centroids()
    profiles = compute_psalm_profiles(sequences, centroids)

    rows = profile_rows("model_a", profiles)

    assert rows[0]["centroid"].dtype == np.float32
    assert rows[0]["sequence"].dtype == np.float32
    assert np.allclose(rows[0]["sequence"], profiles[1]["sequence"].flatten(), atol=1e-6)


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


def test_profile_shard_path_writes_one_file_per_model_directly_under_output_dir() -> None:
    """One file per model keeps every shard under GitHub's 100MB per-file limit."""
    path = profile_shard_path(Path("results/trajectory"), "bge_m3_vocalized")

    assert path == Path("results/trajectory/bge_m3_vocalized.parquet")


def test_score_model_writes_a_profile_shard_and_returns_its_distance_rows(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=mine" / "v.parquet",
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
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    n_profile_rows, rows = score_model(path, {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}, output_dir)

    assert profile_shard_path(output_dir, "mine").exists()
    assert n_profile_rows == 2
    assert [row["model"] for row in rows] == ["mine"]
    assert {row["psalm_a"] for row in rows} == {1}
    assert {row["psalm_b"] for row in rows} == {2}
