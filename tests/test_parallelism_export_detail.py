from pathlib import Path

import numpy as np
import pytest

from library.calibration import BackgroundStats
from parallelism.node_pairs import as_node_pairs
from parallelism.pairs import RetrievalPair
from parallelism.scripts.export_detail import (
    build_baseline_detail_rows,
    build_pair_detail_rows,
    build_type_vs_baseline_rows,
    score_model,
)


def _pair(
    pair_id: str,
    parallelism_type: str,
    source_nodes: tuple[int, ...],
    target_nodes: tuple[int, ...],
    group_range: str = "g",
    signature: str = "AB",
    source_indicator: str = "A",
    target_indicator: str = "B",
) -> RetrievalPair:
    return RetrievalPair(
        pair_id=pair_id,
        group_range=group_range,
        parallelism_type=parallelism_type,
        signature=signature,
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        source_indicator=source_indicator,
        target_indicator=target_indicator,
    )


def _pairs_and_vectors() -> tuple[list[RetrievalPair], dict[int, np.ndarray]]:
    pairs = [
        _pair("p1", "Synonymous", (1,), (2,)),
        _pair("p2", "Antithetic", (3,), (4,)),
    ]
    node_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.9, 0.1]),
        3: np.array([0.0, 1.0]),
        4: np.array([0.1, 0.9]),
    }
    return pairs, node_vectors


class TestBuildPairDetailRows:
    def test_has_one_row_per_pair(self) -> None:
        pairs, node_vectors = _pairs_and_vectors()
        background = BackgroundStats(mean=0.5, std=0.2, n_vectors=4)

        rows = build_pair_detail_rows("model_a", pairs, node_vectors, background)

        assert len(rows) == 2

    def test_carries_the_pair_s_identity_fields_and_raw_similarity(self) -> None:
        pairs, node_vectors = _pairs_and_vectors()
        background = BackgroundStats(mean=0.5, std=0.2, n_vectors=4)

        rows = build_pair_detail_rows("model_a", pairs, node_vectors, background)

        row = next(r for r in rows if r["pair_id"] == "p1")
        assert row["model"] == "model_a"
        assert row["parallelism_type"] == "Synonymous"
        assert row["source_nodes"] == "1"
        assert row["target_nodes"] == "2"
        assert row["raw_similarity"] == pytest.approx(0.99388373, abs=1e-6)

    def test_gives_the_only_pair_a_perfect_forward_and_backward_rank(self) -> None:
        pairs, node_vectors = _pairs_and_vectors()
        background = BackgroundStats(mean=0.5, std=0.2, n_vectors=4)

        # With only one candidate target per source (and vice versa), rank is always 1.
        rows = build_pair_detail_rows("model_a", pairs[:1], node_vectors, background)

        assert rows[0]["rank_forward"] == 1
        assert rows[0]["rank_backward"] == 1
        assert rows[0]["reciprocal_rank_forward"] == 1.0
        assert rows[0]["reciprocal_rank_backward"] == 1.0

    def test_ranks_a_worse_matching_pair_below_a_better_one(self) -> None:
        # p1's source is closer to p2's target than to its own, so p1 ranks below the true match.
        pairs = [
            _pair("p1", "Synonymous", (1,), (2,)),
            _pair("p2", "Synonymous", (5,), (4,)),
        ]
        node_vectors = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
            4: np.array([0.99, 0.01]),
            5: np.array([0.98, 0.02]),
        }
        background = BackgroundStats(mean=0.5, std=0.2, n_vectors=4)

        rows = build_pair_detail_rows("model_a", pairs, node_vectors, background)

        row_p1 = next(r for r in rows if r["pair_id"] == "p1")
        assert row_p1["rank_forward"] > 1


class TestBuildBaselineDetailRows:
    def test_has_one_row_per_baseline_pair_with_no_rank_fields(self) -> None:
        node_vectors = {
            10: np.array([1.0, 0.0]),
            11: np.array([0.9, 0.1]),
            12: np.array([0.0, 1.0]),
            13: np.array([0.1, 0.9]),
        }
        background = BackgroundStats(mean=0.5, std=0.2, n_vectors=4)

        rows = build_baseline_detail_rows("model_a", [(10, 11), (12, 13)], node_vectors, background)

        assert len(rows) == 2
        assert "rank_forward" not in rows[0]
        first = next(r for r in rows if r["node_a"] == 10)
        assert first["node_b"] == 11
        assert first["raw_similarity"] == pytest.approx(0.99388373, abs=1e-6)


class TestBuildTypeVsBaselineRows:
    def test_writes_one_row_per_present_type_plus_one_overall_row(self) -> None:
        pairs, node_vectors = _pairs_and_vectors()
        baseline_pairs = [((10,), (11,)), ((12,), (13,))]
        node_vectors[10] = np.array([1.0, 0.0])
        node_vectors[11] = np.array([0.0, 1.0])
        node_vectors[12] = np.array([1.0, 0.0])
        node_vectors[13] = np.array([0.0, 1.0])
        background = BackgroundStats(mean=0.5, std=0.2, n_vectors=6)

        rows = build_type_vs_baseline_rows(
            "model_a", pairs, baseline_pairs, node_vectors, background
        )

        # Two types present (Synonymous, Antithetic) plus the "overall" row.
        assert len(rows) == 3
        assert {r["scope"] for r in rows} == {"Synonymous", "Antithetic", "overall"}
        overall = next(r for r in rows if r["scope"] == "overall")
        assert overall["scope_kind"] == "overall"

    def test_omits_a_type_with_zero_pairs(self) -> None:
        pairs = [_pair("p1", "Synonymous", (1,), (2,))]
        node_vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.9, 0.1])}
        baseline_pairs = [((10,), (11,))]
        node_vectors[10] = np.array([1.0, 0.0])
        node_vectors[11] = np.array([0.0, 1.0])
        background = BackgroundStats(mean=0.5, std=0.2, n_vectors=4)

        rows = build_type_vs_baseline_rows(
            "model_a", pairs, baseline_pairs, node_vectors, background
        )

        type_scopes = {r["scope"] for r in rows if r["scope_kind"] == "type"}
        assert type_scopes == {"Synonymous"}

    def test_the_gap_is_true_effect_size_minus_baseline_effect_size(self) -> None:
        pairs, node_vectors = _pairs_and_vectors()
        baseline_pairs = [((10,), (11,)), ((12,), (13,))]
        node_vectors[10] = np.array([1.0, 0.0])
        node_vectors[11] = np.array([0.0, 1.0])
        node_vectors[12] = np.array([1.0, 0.0])
        node_vectors[13] = np.array([0.0, 1.0])
        background = BackgroundStats(mean=0.5, std=0.2, n_vectors=6)

        rows = build_type_vs_baseline_rows(
            "model_a", pairs, baseline_pairs, node_vectors, background
        )

        overall = next(r for r in rows if r["scope"] == "overall")
        assert overall["gap"] == pytest.approx(
            overall["true_effect_size"] - overall["baseline_effect_size"]
        )


def test_score_model_returns_pair_baseline_and_type_scope_rows(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=mine" / "v.parquet",
        {
            1: [1.0, 0.0],
            2: [1.0, 0.0],
            3: [1.0, 0.0],
            4: [0.98, 0.2],
            5: [1.0, 0.0],
            6: [0.0, 1.0],
            7: [1.0, 0.0],
            8: [0.1, 0.99],
            9: [0.7, 0.71],
            10: [-1.0, 0.0],
            11: [0.0, 1.0],
            12: [0.3, 0.95],
        },
    )
    all_pairs = [
        RetrievalPair("p1", "g", "Synonymous", "AB", (1,), (2,), "A", "B"),
        RetrievalPair("p2", "g", "Synonymous", "AB", (3,), (4,), "A", "B"),
    ]
    baseline_raw = [(5, 6), (7, 8)]

    pair_rows, baseline_rows, scope_rows = score_model(
        path, all_pairs, as_node_pairs(baseline_raw), baseline_raw, [9, 10, 11, 12]
    )

    assert [row["pair_id"] for row in pair_rows] == ["p1", "p2"]
    assert len(baseline_rows) == 2
    assert {row["model"] for row in scope_rows} == {"mine"}
