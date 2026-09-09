from pathlib import Path

from library.model_files import uncached_model_paths


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_returns_every_embeddings_file_when_nothing_is_cached(tmp_path: Path) -> None:
    a = _touch(tmp_path / "domain=d" / "model=a" / "part-0.parquet")
    b = _touch(tmp_path / "domain=d" / "model=b" / "part-0.parquet")

    assert uncached_model_paths(tmp_path, set()) == [a, b]


def test_skips_a_model_already_present_in_the_cache(tmp_path: Path) -> None:
    _touch(tmp_path / "domain=d" / "model=a" / "part-0.parquet")
    b = _touch(tmp_path / "domain=d" / "model=b" / "part-0.parquet")

    assert uncached_model_paths(tmp_path, {"a"}) == [b]


def test_orders_paths_canonically_so_reruns_submit_work_the_same_way(tmp_path: Path) -> None:
    """Row order follows submission order, so the file order must not depend on the filesystem."""
    _touch(tmp_path / "domain=d" / "model=c" / "part-0.parquet")
    _touch(tmp_path / "domain=d" / "model=a" / "part-0.parquet")
    _touch(tmp_path / "domain=d" / "model=b" / "part-0.parquet")

    found = uncached_model_paths(tmp_path, set())

    assert found == sorted(found)


def test_finds_files_nested_at_any_partition_depth(tmp_path: Path) -> None:
    deep = _touch(tmp_path / "domain=d" / "level=x" / "feature=y" / "c=z" / "part-0.parquet")

    assert uncached_model_paths(tmp_path, set()) == [deep]


def test_ignores_a_directory_that_merely_ends_in_parquet(tmp_path: Path) -> None:
    (tmp_path / "domain=d" / "stray.parquet").mkdir(parents=True)
    real = _touch(tmp_path / "domain=d" / "model=a" / "part-0.parquet")

    assert uncached_model_paths(tmp_path, set()) == [real]


def test_returns_nothing_for_an_empty_directory(tmp_path: Path) -> None:
    assert uncached_model_paths(tmp_path, set()) == []


def test_excludes_order_shuffle_draws_which_are_the_null_not_models(tmp_path: Path) -> None:
    """A shuffle draw is one sample of the order-shuffle null, so scoring it as a model is wrong."""
    real = _touch(tmp_path / "domain=d" / "construction=icf_position4" / "part-0.parquet")
    _touch(tmp_path / "domain=d" / "construction=icf_position4_shuffle0001" / "part-0.parquet")
    _touch(tmp_path / "domain=d" / "construction=icf_position4_shuffle1000" / "part-0.parquet")

    assert uncached_model_paths(tmp_path, set()) == [real]


def test_excludes_the_earlier_two_digit_shuffle_generation_as_well(tmp_path: Path) -> None:
    real = _touch(tmp_path / "domain=d" / "feature=sp" / "construction=1_2gram" / "part-0.parquet")
    _touch(
        tmp_path / "domain=d" / "feature=sp" / "construction=1_2gram_shuffle15" / "part-0.parquet"
    )

    assert uncached_model_paths(tmp_path, set()) == [real]


def test_keeps_a_model_whose_name_contains_shuffle_without_a_draw_number(tmp_path: Path) -> None:
    """Only a trailing draw number marks a null sample, so a name ending in the bare word stays."""
    kept = _touch(tmp_path / "domain=d" / "construction=order_shuffle" / "part-0.parquet")

    assert uncached_model_paths(tmp_path, set()) == [kept]


def test_uncached_model_paths_passes_over_a_parquet_outside_any_hive_partition(
    tmp_path: Path,
) -> None:
    (tmp_path / "domain=semantic/unit=psalm").mkdir(parents=True)
    (tmp_path / "domain=semantic/unit=psalm/vectors.parquet").touch()
    (tmp_path / "stray.parquet").touch()

    paths = uncached_model_paths(tmp_path, set())

    assert [p.name for p in paths] == ["vectors.parquet"]
