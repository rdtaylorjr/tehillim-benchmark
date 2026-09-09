import pandas as pd


def test_pivot_scope_keeps_the_overall_and_per_type_tables_apart() -> None:
    """The two published tables come from one long frame, split on scope_kind."""
    from parallelism.scripts.build_master_report import _pivot_scope

    long_frame = pd.DataFrame(
        {
            "model": ["a", "a"],
            "model_base": ["a", "a"],
            "text_variant": ["", ""],
            "scope": ["overall", "Synonymous"],
            "scope_kind": ["overall", "type"],
            "source": ["retrieval", "retrieval"],
            "metric": ["n_pairs", "n_pairs"],
            "value": [10.0, 4.0],
            "q_value": [float("nan"), float("nan")],
            "q_value_by": [float("nan"), float("nan")],
        }
    )

    overall = _pivot_scope(long_frame, "overall", ["model", "model_base", "text_variant"])
    by_type = _pivot_scope(long_frame, "type", ["model", "model_base", "text_variant"])

    assert overall["n_pairs"].tolist() == [10.0]
    assert by_type["n_pairs"].tolist() == [4.0]
