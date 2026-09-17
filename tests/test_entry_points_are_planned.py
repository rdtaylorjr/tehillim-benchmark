"""Check 2 of the driver design: every entry point is a cell the driver plans, or the driver."""

import re
from pathlib import Path

from library.stages import Roots, plan_cells

SRC = Path(__file__).resolve().parents[1] / "src"
#: Entry points that are the driver itself rather than cells it plans, with the reason.
DRIVER_ENTRY_POINTS = {"library.driver": "the cell runner every rendered rule invokes"}


def _modules_with_main() -> set[str]:
    """Dotted names of every shipped module defining a main()."""
    found = set()
    for path in SRC.rglob("*.py"):
        if "egg-info" in path.parts:
            continue
        if re.search(r"^def main\(", path.read_text(), re.MULTILINE):
            found.add(".".join(path.relative_to(SRC).with_suffix("").parts))
    return found


def _planned_modules(tmp_path: Path) -> set[str]:
    """Modules the stage declarations invoke for a tree holding one domain."""
    embeddings = tmp_path / "emb"
    (embeddings / "corpus=bhsa/unit=half_verse/domain=lexical/type=lexeme/construction=icf").mkdir(
        parents=True
    )
    roots = Roots(
        data_root=tmp_path / "d",
        embeddings_root=embeddings,
        config_root=tmp_path / "c",
        genre_csv=tmp_path / "g.csv",
        ui_root=tmp_path / "u",
        workers=1,
    )
    return {cell.module for cell in plan_cells(roots)}


def test_every_main_is_planned_by_a_stage(tmp_path: Path) -> None:
    """A module with main() and no stage would be runnable by hand and invisible to the driver."""
    assert _modules_with_main() - _planned_modules(tmp_path) - set(DRIVER_ENTRY_POINTS) == set()


def test_every_planned_module_has_a_main(tmp_path: Path) -> None:
    assert _planned_modules(tmp_path) - _modules_with_main() == set()
