import csv
import json
from pathlib import Path

import pytest

from library.rows_output import write_json, write_rows_csv, write_text


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_writes_a_header_and_one_line_per_row(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"

    write_rows_csv(path, [{"model": "a", "auc": 0.5}, {"model": "b", "auc": 0.6}])

    header, rows = _read(path)
    assert header == ["model", "auc"]
    assert [row["model"] for row in rows] == ["a", "b"]


def test_columns_follow_first_appearance_across_all_rows(tmp_path: Path) -> None:
    """Per-type metrics appear only on the models that have that type, so the header is a union."""
    path = tmp_path / "out.csv"

    write_rows_csv(path, [{"model": "a", "x": 1}, {"model": "b", "y": 2}])

    header, _ = _read(path)
    assert header == ["model", "x", "y"]


def test_a_row_missing_a_column_writes_an_empty_cell(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"

    write_rows_csv(path, [{"model": "a", "x": 1}, {"model": "b"}])

    _, rows = _read(path)
    assert rows[1]["x"] == ""


def test_writes_only_a_header_for_an_empty_row_list(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"

    write_rows_csv(path, [])

    assert path.read_text() == ""


def test_creates_the_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.csv"

    write_rows_csv(path, [{"model": "a"}])

    assert path.exists()


def test_write_json_replaces_the_target_only_once_the_write_succeeds(tmp_path: Path) -> None:
    """Nine call sites wrote JSON directly, leaving a truncated payload when a run was killed."""
    path = tmp_path / "out" / "payload.json"

    write_json(path, {"rows": [1, 2, 3]})

    assert json.loads(path.read_text()) == {"rows": [1, 2, 3]}


def test_write_json_keeps_the_previous_file_when_serialising_fails(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    write_json(path, {"kept": True})

    with pytest.raises(TypeError, match="not JSON serializable"):
        write_json(path, {"bad": {1, 2}})

    assert json.loads(path.read_text()) == {"kept": True}


def test_write_json_writes_a_non_finite_float_as_null_which_json_can_express(
    tmp_path: Path,
) -> None:
    """One rule for every JSON writer: an uncomputable statistic reaches the reader as null."""
    path = tmp_path / "p.json"

    write_json(path, {"q": float("nan"), "hi": float("inf"), "ok": 0.5})

    assert json.loads(path.read_text()) == {"q": None, "hi": None, "ok": 0.5}


def test_write_json_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    write_json(tmp_path / "p.json", {"a": 1})

    assert [p.name for p in tmp_path.iterdir()] == ["p.json"]


def test_write_text_replaces_atomically(tmp_path: Path) -> None:
    path = tmp_path / "page" / "index.html"

    write_text(path, "<html></html>")

    assert path.read_text() == "<html></html>"


def test_a_failed_write_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    """The rename consumes the temp on success, so only a failure exercises the cleanup."""
    path = tmp_path / "payload.json"

    with pytest.raises(TypeError, match="not JSON serializable"):
        write_json(path, {"bad": {1, 2}})

    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_leaves_an_existing_file_untouched_and_no_temp(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    write_json(path, {"kept": True})

    with pytest.raises(TypeError, match="not JSON serializable"):
        write_json(path, {"bad": {1, 2}})

    assert [p.name for p in tmp_path.iterdir()] == ["payload.json"]
    assert json.loads(path.read_text()) == {"kept": True}


def test_a_writer_that_fails_after_creating_the_temp_leaves_nothing_behind(tmp_path: Path) -> None:
    """Serialisation fails before the temp exists, so only a mid-write failure cleans up."""
    from library.rows_output import _replace_atomically

    path = tmp_path / "out.bin"

    def write_then_fail(target: Path) -> None:
        target.write_text("half written")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        _replace_atomically(path, write_then_fail)

    assert list(tmp_path.iterdir()) == []


def test_a_mid_write_failure_keeps_the_previous_file(tmp_path: Path) -> None:
    from library.rows_output import _replace_atomically

    path = tmp_path / "out.bin"
    path.write_text("original")

    def write_then_fail(target: Path) -> None:
        target.write_text("half written")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        _replace_atomically(path, write_then_fail)

    assert path.read_text() == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]
