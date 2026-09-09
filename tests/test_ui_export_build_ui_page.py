import json

from ui_export.scripts.build_ui_page import (
    ALL_DOMAIN_IDS,
    build_ui_page_html,
    merge_domain_json_files,
)

_TEMPLATE = (
    "<html><body>\n"
    'const DATA = /*UI_DATA_JSON*/{"families": {}}/*END_UI_DATA_JSON*/;\n'
    "/*UI_BUNDLE_JS*/\n"
    "</body></html>"
)


def test_build_ui_page_html_injects_the_bundle_js_verbatim() -> None:
    html = build_ui_page_html(_TEMPLATE, "var x = 1;", {})

    assert "var x = 1;" in html
    assert "/*UI_BUNDLE_JS*/" not in html


def test_build_ui_page_html_injects_families_as_json() -> None:
    families = {"semantic": {"parallelism_overall": [{"model": "bge_m3"}]}}

    html = build_ui_page_html(_TEMPLATE, "", families)

    assert '"model": "bge_m3"' in html
    assert "/*UI_DATA_JSON*/" not in html


def test_build_ui_page_html_fills_in_every_domain_missing_from_the_input() -> None:
    html = build_ui_page_html(_TEMPLATE, "", {"semantic": {"parallelism_overall": []}})

    start = html.index("const DATA = ") + len("const DATA = ")
    end = html.index(";\n", start)
    data = json.loads(html[start:end])

    assert set(data["domains"]) == set(ALL_DOMAIN_IDS)
    assert data["domains"]["phonological"] == {
        "parallelism_overall": [],
        "parallelism_by_type": [],
        "genre_overall": [],
        "genre_by_genre": [],
        "trajectory": [],
        "trajectory_by_genre": [],
    }


def test_merge_domain_json_files_keys_by_the_single_domain_each_file_carries(tmp_path) -> None:
    semantic_path = tmp_path / "ui_semantic.json"
    semantic_path.write_text(json.dumps({"semantic": {"parallelism_overall": [1]}}))
    lexical_path = tmp_path / "ui_lexical.json"
    lexical_path.write_text(json.dumps({"lexical": {"parallelism_overall": [2]}}))

    merged = merge_domain_json_files([semantic_path, lexical_path])

    assert merged == {
        "semantic": {"parallelism_overall": [1]},
        "lexical": {"parallelism_overall": [2]},
    }
