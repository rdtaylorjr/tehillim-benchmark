import argparse
from pathlib import Path

from library.bhsa import DEFAULT_CHECKOUT
from library.cli import add_scoring_arguments, load_cache
from library.rows_output import write_rows_csv
from library.worker_pool import default_max_workers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_scoring_arguments(parser)
    return parser


def test_every_batch_script_gets_the_same_three_arguments() -> None:
    """These were defined fifteen times, so a changed default meant editing fifteen files."""
    args = _parser().parse_args([])

    assert args.checkout == DEFAULT_CHECKOUT
    assert args.workers == default_max_workers()
    assert args.output is None


def test_the_arguments_still_parse_when_given() -> None:
    args = _parser().parse_args(["--checkout", "v1.9", "--workers", "7", "--output", "out.csv"])

    assert (args.checkout, args.workers, args.output) == ("v1.9", 7, Path("out.csv"))


def test_seed_and_resample_options_are_opt_in() -> None:
    """Only the scripts that resample take these, so the shared helper must not force them."""
    parser = argparse.ArgumentParser()
    add_scoring_arguments(parser, with_seed=True, with_resamples=True)

    args = parser.parse_args([])

    assert args.seed == 0
    assert args.n_resamples > 0
    assert not hasattr(_parser().parse_args([]), "seed")


def test_load_cache_returns_nothing_for_a_path_that_is_not_there(tmp_path: Path) -> None:
    assert load_cache(tmp_path / "absent.csv") == ([], set())


def test_load_cache_returns_nothing_when_no_output_was_asked_for() -> None:
    assert load_cache(None) == ([], set())


def test_load_cache_reads_back_rows_and_their_model_names(tmp_path: Path) -> None:
    path = tmp_path / "prior.csv"
    write_rows_csv(path, [{"model": "a", "score": "1"}, {"model": "b", "score": "2"}])

    rows, models = load_cache(path)

    assert models == {"a", "b"}
    assert [row["model"] for row in rows] == ["a", "b"]


def test_load_cache_reports_what_it_reused_on_stderr(tmp_path: Path, capsys) -> None:
    """Five scripts printed this line themselves, each with its own wording."""
    path = tmp_path / "prior.csv"
    write_rows_csv(path, [{"model": "a"}])

    load_cache(path)

    assert "1" in capsys.readouterr().err


def test_resampling_counts_come_from_the_shared_protocol_not_a_second_copy() -> None:
    """The CLI held its own hard-coded defaults, two of which had already drifted."""
    from library.order_shuffle import DEFAULT_N_SHUFFLES
    from library.protocol import (
        DEFAULT_N_GROUP_PERMUTATIONS,
        DEFAULT_N_PERMUTATIONS,
        DEFAULT_N_RESAMPLES,
    )

    single = argparse.ArgumentParser()
    add_scoring_arguments(single, with_resamples=True, with_permutations=True, with_shuffles=True)
    grouped = argparse.ArgumentParser()
    add_scoring_arguments(grouped, with_group_permutations=True)

    parsed = single.parse_args([])
    assert parsed.n_resamples == DEFAULT_N_RESAMPLES
    assert parsed.n_permutations == DEFAULT_N_PERMUTATIONS
    assert parsed.n_shuffles == DEFAULT_N_SHUFFLES
    assert grouped.parse_args([]).n_permutations == DEFAULT_N_GROUP_PERMUTATIONS


def test_resume_reads_the_cache_and_lists_only_what_is_left(tmp_path: Path) -> None:
    """Reading the cache and listing the remainder always happen together."""
    from library.cli import resume_from_cache

    embeddings = tmp_path / "data"
    for model in ("a", "b"):
        target = embeddings / f"domain=d/model={model}/part-0.parquet"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"")
    cache = tmp_path / "prior.csv"
    write_rows_csv(cache, [{"model": "a", "score": "1"}])

    cached_rows, pending = resume_from_cache(embeddings, cache)

    assert [row["model"] for row in cached_rows] == ["a"]
    assert [p.parent.name for p in pending] == ["model=b"]


def test_resume_scores_everything_when_there_is_no_cache(tmp_path: Path) -> None:
    from library.cli import resume_from_cache

    embeddings = tmp_path / "data"
    target = embeddings / "domain=d/model=a/part-0.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")

    cached_rows, pending = resume_from_cache(embeddings, None)

    assert cached_rows == []
    assert len(pending) == 1


def test_the_genre_csv_positional_is_defined_once_for_every_genre_script() -> None:
    """The five-line block was repeated in every genre script's parser."""
    from library.cli import add_genre_csv_argument

    parser = argparse.ArgumentParser()
    add_genre_csv_argument(parser)
    add_scoring_arguments(parser)

    args = parser.parse_args(["labels.csv"])

    assert args.genre_csv == Path("labels.csv")


def test_the_embeddings_directory_positional_is_defined_once() -> None:
    """Declared eleven times before, so a change to it meant editing eleven parsers."""
    from library.cli import add_embeddings_dir_argument

    parser = argparse.ArgumentParser()
    add_embeddings_dir_argument(parser)
    add_scoring_arguments(parser)

    assert parser.parse_args(["data/domain=lexical"]).embeddings_dir == Path("data/domain=lexical")
