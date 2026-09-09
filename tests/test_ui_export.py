import json

import pandas as pd

from library.rows_output import json_safe
from ui_export import export
from ui_export.export import build_domain_data


def _parallelism_overall_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "bge_m3_vocalized",
                "model_base": "bge_m3",
                "text_variant": "vocalized",
                "separation_auc": 0.66,
                "separation_p_q": 0.03,
                "auc_vs_baseline": 0.58,
                "p_vs_baseline_q": 0.04,
                "average_precision": 0.33,
                "calibrated_effect_size": 0.44,
                "mrr_forward": 0.06,
                "n_true": 1110.0,
                "unused_column": "should be dropped",
            },
            {
                "model": "form_icf_posmean_psalm",
                "model_base": "form_icf_posmean_psalm",
                "text_variant": "unknown",
                "separation_auc": 0.99,
                "separation_p_q": 0.001,
                "auc_vs_baseline": 0.95,
                "p_vs_baseline_q": 0.001,
                "average_precision": 0.98,
                "calibrated_effect_size": 5.0,
                "mrr_forward": 0.9,
                "n_true": 1110.0,
                "unused_column": "should be dropped",
            },
        ]
    )


def _parallelism_by_type_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "bge_m3_vocalized",
                "model_base": "bge_m3",
                "text_variant": "vocalized",
                "scope": "Synonymous",
                "separation_auc": 0.7,
                "separation_p_q": 0.02,
                "auc_vs_baseline": 0.6,
                "p_vs_baseline_q": 0.03,
                "average_precision": 0.35,
                "calibrated_effect_size": 0.5,
                "mrr_forward": 0.05,
                "n_true": 48.0,
            },
            {
                "model": "form_icf_posmean_psalm",
                "model_base": "form_icf_posmean_psalm",
                "text_variant": "unknown",
                "scope": "Synonymous",
                "separation_auc": 0.99,
                "separation_p_q": 0.001,
                "auc_vs_baseline": 0.96,
                "p_vs_baseline_q": 0.001,
                "average_precision": 0.98,
                "calibrated_effect_size": 5.0,
                "mrr_forward": 0.85,
                "n_true": 48.0,
            },
        ]
    )


def _genre_overall_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "bge_m3_vocalized",
                "model_base": "bge_m3",
                "text_variant": "vocalized",
                "separation_auc": 0.6,
                "auc_ci_low": 0.55,
                "auc_ci_high": 0.65,
                "separation_p_q": 0.01,
                "average_precision": 0.3,
                "ap_ci_low": 0.25,
                "ap_ci_high": 0.35,
                "gap": 0.39,
                "gap_ci_low": 0.23,
                "gap_ci_high": 0.5,
                "same_genre_effect_size": 0.4,
                "prevalence": 0.28,
                "n_same_genre": 500,
                "n_different_genre": 4700,
            }
        ]
    )


def _genre_by_genre_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "bge_m3_vocalized",
                "genre": "Wisdom",
                "separation_auc": 0.65,
                "auc_ci_low": 0.6,
                "auc_ci_high": 0.7,
                "average_precision": 0.32,
                "ap_ci_low": 0.27,
                "ap_ci_high": 0.37,
                "separation_p_perm": 0.01,
                "separation_p_maxT": 0.05,
                "perm_q": 0.02,
                "maxT_q": 0.08,
                "prevalence": 0.29,
                "n_same_genre": 200,
                "n_different_genre": 1900,
            }
        ]
    )


def _trajectory_rows() -> list[dict]:
    return [{"model": "bge_m3_vocalized", "metric": "content_distance", "raw_p": 0.001}]


def test_build_domain_data_selects_only_the_uis_parallelism_overall_columns() -> None:
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    row = data["parallelism_overall"][0]
    assert set(row) == {
        "model",
        "model_base",
        "text_variant",
        "separation_auc",
        "separation_p_q",
        "auc_vs_baseline",
        "p_vs_baseline_q",
        "average_precision",
        "calibrated_effect_size",
        "mrr_forward",
        "n_true",
    }
    assert "unused_column" not in row


def test_build_domain_data_keeps_scope_in_parallelism_by_type() -> None:
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    row = data["parallelism_by_type"][0]
    assert row["scope"] == "Synonymous"


def test_build_domain_data_selects_parallelism_by_type_columns() -> None:
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    row = data["parallelism_by_type"][0]
    assert set(row) == {
        "model",
        "model_base",
        "text_variant",
        "scope",
        "separation_auc",
        "separation_p_q",
        "auc_vs_baseline",
        "p_vs_baseline_q",
        "average_precision",
        "calibrated_effect_size",
        "mrr_forward",
        "n_true",
    }


def test_build_domain_data_drops_psalm_level_models_from_parallelism_overall() -> None:
    """Psalm-broadcast representations are architecturally degenerate for a half-verse-pair task."""
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    models = {row["model"] for row in data["parallelism_overall"]}
    assert models == {"bge_m3_vocalized"}


def test_build_domain_data_drops_psalm_level_shuffle_control_models_from_parallelism_overall() -> (
    None
):
    """A _psalm_shuffleNN model is just as degenerate as its unshuffled _psalm base."""
    parallelism_overall = _parallelism_overall_df()
    parallelism_overall.loc[len(parallelism_overall)] = {
        "model": "morph_signature_1_2gram_psalm_shuffle03",
        "model_base": "morph_signature_1_2gram_psalm_shuffle03",
        "text_variant": "unknown",
        "separation_auc": 0.995,
        "separation_p_q": 0.001,
        "average_precision": 0.99,
        "calibrated_effect_size": 5.0,
        "mrr_forward": 0.9,
        "n_true": 1110.0,
        "unused_column": "should be dropped",
    }
    data = build_domain_data(
        parallelism_overall,
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    models = {row["model"] for row in data["parallelism_overall"]}
    assert "morph_signature_1_2gram_psalm_shuffle03" not in models


def test_build_domain_data_drops_psalm_level_models_from_parallelism_by_type() -> None:
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    models = {row["model"] for row in data["parallelism_by_type"]}
    assert models == {"bge_m3_vocalized"}


def test_build_domain_data_keeps_psalm_level_models_in_genre_tables() -> None:
    """Psalm-broadcast representations are exactly the right granularity for a psalm-level task."""
    genre_overall = _genre_overall_df()
    genre_overall.loc[len(genre_overall)] = {
        "model": "form_icf_posmean_psalm",
        "model_base": "form_icf_posmean_psalm",
        "text_variant": "unknown",
        "separation_auc": 0.7,
        "auc_ci_low": 0.6,
        "auc_ci_high": 0.8,
        "average_precision": 0.4,
        "same_genre_effect_size": 0.5,
        "n_same_genre": 500,
    }
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        genre_overall,
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    models = {row["model"] for row in data["genre_overall"]}
    assert "form_icf_posmean_psalm" in models


def test_build_domain_data_selects_genre_overall_columns() -> None:
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    row = data["genre_overall"][0]
    assert set(row) == {
        "model",
        "model_base",
        "text_variant",
        "separation_auc",
        "auc_ci_low",
        "auc_ci_high",
        "average_precision",
        "ap_ci_low",
        "ap_ci_high",
        "prevalence",
        "n_same_genre",
        "n_different_genre",
    }


def test_build_domain_data_selects_genre_by_genre_columns() -> None:
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    row = data["genre_by_genre"][0]
    assert set(row) == {
        "model",
        "model_base",
        "text_variant",
        "genre",
        "separation_auc",
        "auc_ci_low",
        "auc_ci_high",
        "average_precision",
        "ap_ci_low",
        "ap_ci_high",
        "prevalence",
        "n_same_genre",
        "n_different_genre",
    }


def test_build_domain_data_derives_model_base_and_text_variant_for_genre_by_genre() -> None:
    """genre_by_genre.csv has no model_base/text_variant columns."""
    genre_by_genre = _genre_by_genre_df()
    genre_by_genre.loc[0, "model"] = "word_consonantal_binary"
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        genre_by_genre,
        _trajectory_rows(),
    )

    row = data["genre_by_genre"][0]
    assert row["model_base"] == "word_binary"
    assert row["text_variant"] == "consonantal"


def test_build_domain_data_passes_trajectory_rows_through_unchanged() -> None:
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    assert data["trajectory"] == _trajectory_rows()


def test_build_domain_data_has_exactly_the_six_ui_keys() -> None:
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    assert set(data) == {
        "parallelism_overall",
        "parallelism_by_type",
        "genre_overall",
        "genre_by_genre",
        "trajectory",
        "trajectory_by_genre",
    }


def test_build_domain_data_defaults_trajectory_by_genre_to_an_empty_list() -> None:
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    assert data["trajectory_by_genre"] == []


def test_build_domain_data_passes_trajectory_by_genre_rows_through_unchanged() -> None:
    by_genre_rows = [{"model": "bge_m3_vocalized", "metric": "content_distance", "genre": "Wisdom"}]
    data = build_domain_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
        by_genre_rows,
    )

    assert data["trajectory_by_genre"] == by_genre_rows


def test_build_domain_data_drops_shuffle_control_models_from_every_table() -> None:
    """A _shuffleNN model is a null-order control checked against its real base, not rankable."""
    shuffle_row = {
        "model": "phrase_signature_1_2gram_shuffle03",
        "model_base": "phrase_signature_1_2gram_shuffle03",
        "text_variant": "unknown",
        "separation_auc": 0.9,
        "separation_p_q": 0.001,
        "auc_vs_baseline": 0.85,
        "p_vs_baseline_q": 0.001,
        "average_precision": 0.9,
        "calibrated_effect_size": 5.0,
        "mrr_forward": 0.9,
        "n_true": 1110.0,
        "auc_ci_low": 0.85,
        "auc_ci_high": 0.95,
        "ap_ci_low": 0.85,
        "ap_ci_high": 0.95,
        "gap": 0.4,
        "gap_ci_low": 0.3,
        "gap_ci_high": 0.5,
        "same_genre_effect_size": 0.9,
        "prevalence": 0.28,
        "n_same_genre": 500,
        "n_different_genre": 4700,
        "scope": "Synonymous",
    }
    parallelism_overall = _parallelism_overall_df()
    parallelism_overall.loc[len(parallelism_overall)] = shuffle_row
    parallelism_by_type = _parallelism_by_type_df()
    parallelism_by_type.loc[len(parallelism_by_type)] = shuffle_row
    genre_overall = _genre_overall_df()
    genre_overall.loc[len(genre_overall)] = shuffle_row
    genre_by_genre = _genre_by_genre_df()
    genre_by_genre.loc[len(genre_by_genre)] = {
        "model": "phrase_signature_1_2gram_shuffle03",
        "genre": "Wisdom",
        "separation_auc": 0.9,
        "auc_ci_low": 0.85,
        "auc_ci_high": 0.95,
        "average_precision": 0.9,
        "ap_ci_low": 0.85,
        "ap_ci_high": 0.95,
        "separation_p_perm": 0.01,
        "separation_p_maxT": 0.05,
        "perm_q": 0.02,
        "maxT_q": 0.08,
        "prevalence": 0.29,
        "n_same_genre": 200,
        "n_different_genre": 1900,
    }
    trajectory_rows = [
        *_trajectory_rows(),
        {
            "model": "phrase_signature_1_2gram_shuffle03",
            "metric": "content_distance",
            "raw_p": 0.9,
        },
    ]
    trajectory_by_genre_rows = [
        {
            "model": "phrase_signature_1_2gram_shuffle03",
            "metric": "content_distance",
            "genre": "Wisdom",
        }
    ]

    data = build_domain_data(
        parallelism_overall,
        parallelism_by_type,
        genre_overall,
        genre_by_genre,
        trajectory_rows,
        trajectory_by_genre_rows,
    )

    for table in (
        "parallelism_overall",
        "parallelism_by_type",
        "genre_overall",
        "genre_by_genre",
        "trajectory",
        "trajectory_by_genre",
    ):
        models = {row["model"] for row in data[table]}
        assert "phrase_signature_1_2gram_shuffle03" not in models, table


def test_json_safe_replaces_non_finite_floats_with_none() -> None:
    """NaN and infinities are legal Python but not legal JSON, so they become null."""
    payload = {"gap": float("nan"), "hi": float("inf"), "lo": float("-inf"), "ok": 0.5}
    assert json_safe(payload) == {"gap": None, "hi": None, "lo": None, "ok": 0.5}


def test_json_safe_walks_nested_lists_and_dicts() -> None:
    payload = {"rows": [{"p": float("nan")}, {"p": 0.25}]}
    assert json_safe(payload) == {"rows": [{"p": None}, {"p": 0.25}]}


def test_json_safe_leaves_other_values_untouched() -> None:
    payload = {"model": "alephbert", "n": 1110, "flag": True, "missing": None}
    assert json_safe(payload) == payload


def test_exported_payload_is_strict_json() -> None:
    """allow_nan=False is the guard: a stray non-finite value fails the export loudly."""
    payload = json_safe({"a": float("nan")})
    assert json.dumps(payload, allow_nan=False) == '{"a": null}'


def test_split_payloads_separates_the_per_genre_trajectory_rows() -> None:
    """The by-genre section is 79% of the payload and is read by one view, so it ships apart."""
    data = {
        "genre_overall": [{"model": "a"}],
        "trajectory_by_genre": [
            {"model": "a", "metric": "structural_distance", "genre": "Hymn"},
            {"model": "a", "metric": "turning_angle_distance", "genre": "Hymn"},
        ],
    }
    core, slices = export.split_payloads("semantic", data)

    assert "trajectory_by_genre" not in core["semantic"]
    assert core["semantic"]["genre_overall"] == [{"model": "a"}]
    assert set(slices) == {"structural_distance", "turning_angle_distance"}
    assert slices["structural_distance"]["semantic"]["trajectory_by_genre"] == [
        {"model": "a", "metric": "structural_distance", "genre": "Hymn"}
    ]


def test_split_payloads_yields_no_slices_when_there_are_no_by_genre_rows() -> None:
    core, slices = export.split_payloads(
        "syntactic", {"genre_overall": [], "trajectory_by_genre": []}
    )
    assert slices == {}
    assert core["syntactic"]["genre_overall"] == []


def test_split_payloads_leaves_a_payload_without_the_section_alone() -> None:
    core, slices = export.split_payloads("syntactic", {"genre_overall": []})
    assert core == {"syntactic": {"genre_overall": []}}
    assert slices == {}
