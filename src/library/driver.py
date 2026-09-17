"""Plans the benchmark cells for a Snakefile and runs one cell with provenance recorded."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import core
from core.cli import add_workers_argument
from core.driver import Cell, Plan, provenance_of, run_cell, write_run_manifest
from core.provenance import code_hash

from library.parity import check_parity
from library.stages import BENCHMARK_ROOT, SHUFFLE_MODULE_SUFFIX, Roots, plan_cells

BENCHMARK_SRC = Path(__file__).resolve().parents[1]
EMBEDDINGS_SRC = Path(core.__file__).resolve().parents[1]
#: The shared readers and pool live in core, so a scoring cell's identity follows them there.
BENCHMARK_PACKAGES: tuple[str, ...] = (
    "genre",
    "parallelism",
    "trajectory",
    "ui_export",
    "library",
    "core",
)
#: A shuffle control draws through families.shuffle and its domain's vectorizers, so those count.
SHUFFLE_PACKAGES: tuple[str, ...] = ("families",)
ROOT_KEYS = ("data_root", "embeddings_root", "config_root", "genre_csv", "gunkel_csv", "ui_root")


def benchmark_code_hash(
    module: str, domain: str | None = None, hasher: Callable[..., str] = code_hash
) -> str:
    """Code identity of a benchmark module, reaching into embeddings for a shuffle control."""
    packages = BENCHMARK_PACKAGES
    if module.endswith(SHUFFLE_MODULE_SUFFIX) and domain is not None:
        packages = (*BENCHMARK_PACKAGES, *SHUFFLE_PACKAGES, domain)
    return hasher(module, (BENCHMARK_SRC, EMBEDDINGS_SRC), packages)


def cell_domain(cell: Cell) -> str:
    """The domain segment of a benchmark cell name, `benchmark.domain.stage`."""
    return cell.name.split(".")[1]


def cell_provenance(cell: Cell) -> str:
    """The params string a benchmark rule carries."""
    domain = cell_domain(cell)
    return provenance_of(cell, code_hash_of=lambda module: benchmark_code_hash(module, domain))


def roots_from_config(config: Mapping[str, Any], *, default_workers: int) -> Roots:
    """Builds the roots from a Snakemake config mapping, naming the first key that is absent."""
    values = {key: Path(config[key]) for key in ROOT_KEYS}
    return Roots(**values, workers=int(config.get("workers", default_workers)))


def _roots_from_args(args: argparse.Namespace) -> Roots:
    """The roots as the command line named them."""
    return Roots(
        data_root=args.data_root,
        embeddings_root=args.embeddings_root,
        config_root=args.config_root,
        genre_csv=args.genre_csv,
        gunkel_csv=args.gunkel_csv,
        ui_root=args.ui_root,
        workers=args.workers,
    )


def _module_main(module: str) -> Callable[[list[str]], None]:
    """The script's main, imported when the cell runs."""
    main_fn: Callable[[list[str]], None] = importlib.import_module(module).main
    return main_fn


def main(
    argv: list[str] | None = None,
    *,
    cells_factory: Callable[[Roots], Sequence[Cell]] = plan_cells,
    roots_factory: Callable[[argparse.Namespace], Roots] = _roots_from_args,
    module_main: Callable[[str], Callable[[list[str]], None]] = _module_main,
    parity: Callable[[Roots, Path], dict[str, dict[str, object]]] = check_parity,
) -> None:
    """`run --cell NAME` executes one benchmark cell, `manifest` writes the run-level record."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "manifest"):
        command = sub.add_parser(name)
        for key in ROOT_KEYS:
            command.add_argument(f"--{key.replace('_', '-')}", type=Path, default=Path(key))
        add_workers_argument(command)
        if name == "run":
            command.add_argument("--cell", required=True)
        else:
            command.add_argument("--log-root", type=Path, default=Path("logs"))
    args = parser.parse_args(argv)
    roots = roots_factory(args)
    cells = list(cells_factory(roots))
    if args.command == "manifest":
        #: Under the analysis root, since other drivers write their manifests beside this tree.
        manifest_root = args.data_root / BENCHMARK_ROOT
        path = write_run_manifest(Plan(runnable=tuple(cells), blocked=()), manifest_root)
        report = parity(roots, args.log_root)
        record = json.loads(path.read_text())
        record["parity"] = report
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return
    by_name = {cell.name: cell for cell in cells}
    if args.cell not in by_name:
        parser.error(f"unknown cell {args.cell}")
    cell = by_name[args.cell]
    domain = cell_domain(cell)
    run_cell(
        cell,
        main=module_main(cell.module),
        code_hash_of=lambda module: benchmark_code_hash(module, domain),
    )


if __name__ == "__main__":
    main()
