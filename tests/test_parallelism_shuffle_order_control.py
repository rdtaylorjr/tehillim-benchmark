from pathlib import Path

from conftest import _write_embeddings_parquet as _write_parquet

from parallelism.pairs import build_retrieval_pairs
from parallelism.scripts.shuffle_order_control import score_separation_auc
from parallelism.tf_features import ReconstructedGroup


def _group(
    signature: str,
    member_ids: tuple[int, ...],
    member_indicators: tuple[str, ...],
    member_nodes: tuple[tuple[int, ...], ...],
    member_ambiguous: tuple[bool, ...] | None = None,
    group_range: str = "g",
    parallelism_type: str = "Synonymous",
) -> ReconstructedGroup:
    return ReconstructedGroup(
        group_range=group_range,
        parallelism_type=parallelism_type,
        signature=signature,
        member_ids=member_ids,
        member_indicators=member_indicators,
        member_nodes=member_nodes,
        member_ambiguous=member_ambiguous or tuple(False for _ in member_ids),
    )


class TestScoreSeparationAuc:
    def test_perfect_alignment_scores_auc_one(self, tmp_path: Path) -> None:
        groups = [
            _group("AB", (0, 1), ("A", "B"), ((1,), (2,)), group_range="g1"),
            _group("AB", (0, 1), ("A", "B"), ((3,), (4,)), group_range="g2"),
        ]
        pairs = build_retrieval_pairs(groups)
        path = tmp_path / "embeddings.parquet"
        _write_parquet(
            path,
            {
                1: [1.0, 0.0],
                2: [1.0, 0.0],
                3: [0.0, 1.0],
                4: [0.0, 1.0],
            },
        )

        auc = score_separation_auc(path, pairs)

        assert auc == 1.0

    def test_orthogonal_pairs_score_a_low_auc(self, tmp_path: Path) -> None:
        groups = [
            _group("AB", (0, 1), ("A", "B"), ((1,), (2,)), group_range="g1"),
            _group("AB", (0, 1), ("A", "B"), ((3,), (4,)), group_range="g2"),
        ]
        pairs = build_retrieval_pairs(groups)
        path = tmp_path / "embeddings.parquet"
        # Source and target are orthogonal in a pair but identical across pairs, so AUC is chance.
        _write_parquet(
            path,
            {
                1: [1.0, 0.0],
                2: [0.0, 1.0],
                3: [1.0, 0.0],
                4: [0.0, 1.0],
            },
        )

        auc = score_separation_auc(path, pairs)

        assert auc <= 0.5

    def test_drops_pairs_with_a_zero_norm_vector(self, tmp_path: Path) -> None:
        groups = [
            _group("AB", (0, 1), ("A", "B"), ((1,), (2,)), group_range="g1"),
            _group("AB", (0, 1), ("A", "B"), ((3,), (4,)), group_range="g2"),
            _group("AB", (0, 1), ("A", "B"), ((5,), (6,)), group_range="g3"),
        ]
        pairs = build_retrieval_pairs(groups)
        path = tmp_path / "embeddings.parquet"
        # Node 2's zero vector is excluded, dropping g1 and leaving g2 and g3 for a defined AUC.
        _write_parquet(
            path,
            {
                1: [1.0, 0.0],
                2: [0.0, 0.0],
                3: [1.0, 0.0],
                4: [1.0, 0.0],
                5: [0.0, 1.0],
                6: [0.0, 1.0],
            },
        )

        auc = score_separation_auc(path, pairs)

        assert auc == 1.0
