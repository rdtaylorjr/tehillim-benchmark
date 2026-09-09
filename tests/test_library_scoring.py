import pickle
from functools import partial
from pathlib import Path

import pytest

from library.errors import BenchmarkDataError, InsufficientDataError
from library.scoring import skipping_unscorable


def _score(item, fail_on: str) -> str:
    path = Path(item[0]) if isinstance(item, tuple) else item
    if path.name == fail_on:
        raise InsufficientDataError(f"{path.name} has too few vectors")
    return path.name


def test_returns_the_scorers_result_when_the_model_is_scorable() -> None:
    scored = skipping_unscorable(partial(_score, fail_on="none"))

    assert scored(Path("domain=d/model=a/part-0.parquet")) == "part-0.parquet"


def test_returns_none_when_the_model_cannot_support_the_statistic() -> None:
    """One degenerate model must not end the batch, which is what six scripts used to do."""
    scored = skipping_unscorable(partial(_score, fail_on="part-0.parquet"))

    assert scored(Path("domain=d/model=a/part-0.parquet")) is None


def test_reports_the_skipped_model_on_stderr(capsys) -> None:
    scored = skipping_unscorable(partial(_score, fail_on="part-0.parquet"))

    scored(Path("domain=d/model=lonely/part-0.parquet"))

    assert "lonely" in capsys.readouterr().err


def test_lets_an_unrelated_error_through_rather_than_hiding_a_bug() -> None:
    def explode(path: Path) -> str:
        raise ValueError("not a data problem")

    try:
        skipping_unscorable(explode)(Path("x/part-0.parquet"))
    except ValueError:
        return
    raise AssertionError("a non-data error must not be swallowed")


def test_is_picklable_so_a_process_pool_can_carry_it() -> None:
    """Workers receive the callable by pickle, which a closure would not survive."""
    scored = skipping_unscorable(partial(_score, fail_on="none"))

    assert pickle.loads(pickle.dumps(scored))(Path("d/part-0.parquet")) == "part-0.parquet"


def test_catches_every_benchmark_data_error_not_one_subclass() -> None:
    def raise_base(path: Path) -> str:
        raise BenchmarkDataError("degenerate")

    assert skipping_unscorable(raise_base)(Path("d/part-0.parquet")) is None


def _first(item: tuple[str, int]) -> str:
    return item[0]


def test_a_non_path_item_can_name_itself_for_the_report(capsys) -> None:
    """Not every batch maps over files, so the policy takes how an item names itself."""

    def score(item: tuple[str, int]) -> int:
        raise InsufficientDataError("nothing to score")

    scored = skipping_unscorable(score, label=_first)

    assert scored(("phrase_det_1gram", 3)) is None
    assert "phrase_det_1gram" in capsys.readouterr().err


def test_a_non_path_policy_is_still_picklable() -> None:
    scored = skipping_unscorable(partial(_score, fail_on="none"), label=_first)

    assert pickle.loads(pickle.dumps(scored))(("model", 1)) == "model"


def _raise(error: BaseException, _item: object = None) -> None:
    raise error


@pytest.mark.parametrize(
    "error",
    [ZeroDivisionError("bug"), KeyError("bug"), TypeError("bug"), AttributeError("bug")],
)
def test_only_a_data_error_is_skipped_and_every_other_bug_propagates(error) -> None:
    """Widening this policy would turn a programming bug into a silently skipped model."""
    with pytest.raises(type(error)):
        skipping_unscorable(partial(_raise, error))(Path("d/part-0.parquet"))
