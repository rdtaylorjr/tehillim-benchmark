from functools import partial
from pathlib import Path

import pytest
from conftest import fake_words, semantic_file, whole_psalm_passages

from genre.passages import Passage, half_verse_weights
from genre.scripts.compute_bootstrap_cis import score_model
from library.errors import BenchmarkDataError
from library.scoring import skipping_unscorable

_PASSAGES = whole_psalm_passages({1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"})
_SEPARABLE = {
    1: [1.0, 0.0],
    2: [0.95, 0.05],
    3: [0.9, 0.1],
    4: [0.0, 1.0],
    5: [0.05, 0.95],
    6: [0.1, 0.9],
}


def _weights(passages: list[Passage]) -> dict[str, dict[int, float]]:
    """Whole-half-verse weights for test passages whose words are numbered by fake_words."""
    return half_verse_weights(
        passages, {w // 10: fake_words(w // 10) for p in passages for w in p.words}
    )


def test_score_model_names_the_row_after_the_files_dataset_identifier(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    path = write_embeddings_parquet(semantic_file(tmp_path, "mine", "v.parquet"), _SEPARABLE)

    row = score_model(path, _PASSAGES, _weights(_PASSAGES), n_resamples=50, seed=0)

    assert row is not None
    assert row["model"] == "mine_consonantal"
    assert row["ap_ci_low"] <= row["point_ap"] <= row["ap_ci_high"]


def test_score_model_raises_when_the_population_cannot_support_a_ci(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Two psalms of different genres leave zero same-genre pairs, so AP and AUC are undefined."""
    path = write_embeddings_parquet(
        semantic_file(tmp_path, "tiny", "v.parquet"), {1: [1.0, 0.0], 2: [0.0, 1.0]}
    )

    with pytest.raises(BenchmarkDataError):
        score_model(
            path,
            whole_psalm_passages({1: "A", 2: "B"}),
            _weights(whole_psalm_passages({1: "A", 2: "B"})),
            n_resamples=20,
            seed=0,
        )


def test_score_model_raises_when_only_one_psalm_vector_survives(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """A background needs two vectors, and one syntactic model leaves only one."""
    path = write_embeddings_parquet(semantic_file(tmp_path, "lonely", "v.parquet"), {1: [1.0, 0.0]})

    with pytest.raises(BenchmarkDataError):
        score_model(
            path,
            whole_psalm_passages({1: "A"}),
            _weights(whole_psalm_passages({1: "A"})),
            n_resamples=20,
            seed=0,
        )


def test_the_shared_policy_turns_that_raise_into_a_skip(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """The scorer computes or raises; deciding to skip belongs to one shared policy."""
    path = write_embeddings_parquet(semantic_file(tmp_path, "lonely", "v.parquet"), {1: [1.0, 0.0]})
    passages = whole_psalm_passages({1: "A"})
    score = partial(
        score_model, passages=passages, weights=_weights(passages), n_resamples=20, seed=0
    )

    assert skipping_unscorable(score)(path) is None


def test_a_passage_shares_no_pair_with_text_it_overlaps(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """A motif inside its host pairs with other psalms only, and the CI still resolves."""
    path = write_embeddings_parquet(semantic_file(tmp_path, "nested", "v.parquet"), _SEPARABLE)
    passages = [
        Passage("1:1-2:A", 1, "A", tuple(fake_words(1) + fake_words(2)), "Ps 1:1-2"),
        Passage("1:2-2:B", 1, "B", tuple(fake_words(2)), "Ps 1:2"),
        Passage("3", 3, "A", tuple(fake_words(3)), "Ps 3"),
        Passage("4", 4, "B", tuple(fake_words(4)), "Ps 4"),
        Passage("5", 5, "B", tuple(fake_words(5)), "Ps 5"),
        Passage("6", 6, "A", tuple(fake_words(6)), "Ps 6"),
    ]

    row = score_model(path, passages, _weights(passages), n_resamples=50, seed=0)

    assert row["ap_ci_low"] <= row["point_ap"] <= row["ap_ci_high"]
