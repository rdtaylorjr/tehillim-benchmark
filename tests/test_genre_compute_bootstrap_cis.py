from functools import partial
from pathlib import Path

import pytest

from genre.scripts.compute_bootstrap_cis import score_model
from library.errors import BenchmarkDataError
from library.scoring import skipping_unscorable

_GENRE_BY_PSALM = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"}
_HALF_VERSES_BY_PSALM = {psalm: [psalm] for psalm in _GENRE_BY_PSALM}
_SEPARABLE = {
    1: [1.0, 0.0],
    2: [0.95, 0.05],
    3: [0.9, 0.1],
    4: [0.0, 1.0],
    5: [0.05, 0.95],
    6: [0.1, 0.9],
}


def test_score_model_names_the_row_after_the_files_dataset_identifier(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    path = write_embeddings_parquet(tmp_path / "domain=d" / "model=mine" / "v.parquet", _SEPARABLE)

    row = score_model(path, _HALF_VERSES_BY_PSALM, _GENRE_BY_PSALM, n_resamples=50, seed=0)

    assert row is not None
    assert row["model"] == "mine"
    assert row["ap_ci_low"] <= row["point_ap"] <= row["ap_ci_high"]


def test_score_model_raises_when_the_population_cannot_support_a_ci(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Two psalms of different genres leave zero same-genre pairs, so AP and AUC are undefined."""
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=tiny" / "v.parquet", {1: [1.0, 0.0], 2: [0.0, 1.0]}
    )

    with pytest.raises(BenchmarkDataError):
        score_model(path, {1: [1], 2: [2]}, {1: "A", 2: "B"}, n_resamples=20, seed=0)


def test_score_model_raises_when_only_one_psalm_vector_survives(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """A background needs two vectors, and one syntactic model leaves only one."""
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=lonely" / "v.parquet", {1: [1.0, 0.0]}
    )

    with pytest.raises(BenchmarkDataError):
        score_model(path, {1: [1]}, {1: "A"}, n_resamples=20, seed=0)


def test_the_shared_policy_turns_that_raise_into_a_skip(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """The scorer computes or raises; deciding to skip belongs to one shared policy."""
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=lonely" / "v.parquet", {1: [1.0, 0.0]}
    )
    score = partial(
        score_model, half_verses_by_psalm={1: [1]}, genre_by_psalm={1: "A"}, n_resamples=20, seed=0
    )

    assert skipping_unscorable(score)(path) is None
