"""Assembles the results UI's HTML page from the built bundle, domain JSON, and a template."""

import argparse
import json
import re
from pathlib import Path

from library.rows_output import write_text

DomainData = dict[str, list[dict[str, object]]]
Domains = dict[str, DomainData]

ALL_DOMAIN_IDS = ("semantic", "lexical", "phonological", "morphological", "syntactic", "discourse")

_EMPTY_DOMAIN_DATA: DomainData = {
    "parallelism_overall": [],
    "parallelism_by_type": [],
    "genre_overall": [],
    "genre_by_genre": [],
    "trajectory": [],
    "trajectory_by_genre": [],
}

_DATA_PATTERN = re.compile(r"/\*UI_DATA_JSON\*/.*?/\*END_UI_DATA_JSON\*/", re.DOTALL)
_BUNDLE_MARKER = "/*UI_BUNDLE_JS*/"


def merge_domain_json_files(paths: list[Path]) -> Domains:
    """Merges each ui_<domain>.json's single top-level domain key into one dict."""
    merged: Domains = {}
    for path in paths:
        merged.update(json.loads(path.read_text()))
    return merged


def build_ui_page_html(template: str, bundle_js: str, domains: Domains) -> str:
    """Fills any missing domain with empty tables, then injects DATA and the bundle JS."""
    complete = {
        domain_id: domains.get(domain_id, _EMPTY_DOMAIN_DATA) for domain_id in ALL_DOMAIN_IDS
    }
    html = _DATA_PATTERN.sub(json.dumps({"domains": complete}), template)
    return html.replace(_BUNDLE_MARKER, bundle_js)


def main(argv: list[str] | None = None) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_json", type=Path, nargs="+")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    domains = merge_domain_json_files(args.domain_json)
    html = build_ui_page_html(args.template.read_text(), args.bundle.read_text(), domains)
    write_text(args.output, html)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
