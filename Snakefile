# One command scores every benchmark for every dataset: `snakemake -j 4 --rerun-triggers params`.
import sys
from pathlib import Path

HERE = Path(str(workflow.current_basedir))
PYTHON = str(HERE / ".venv" / "bin" / "python")
sys.path.insert(0, str(HERE / "src"))

from core.driver import render_rules
from library.driver import ROOT_KEYS, cell_provenance, roots_from_config
from library.stages import plan_cells
from library.worker_pool import default_max_workers

ROOTS = roots_from_config(config, default_workers=int(config.get("workers", 4)))
CELLS = plan_cells(ROOTS)
PROVENANCE = {cell.name: cell_provenance(cell) for cell in CELLS}
ROOT_FLAGS = " ".join(
    f"--{key.replace('_', '-')} {getattr(ROOTS, key)}" for key in ROOT_KEYS
) + f" --workers {ROOTS.workers}"

RULES = HERE / ".snakemake" / "cells.smk"
RULES.parent.mkdir(exist_ok=True)
RULES.write_text(
    render_rules(
        CELLS,
        python=PYTHON,
        provenance=PROVENANCE,
        roots=ROOT_FLAGS,
        runner="library.driver",
        threads=ROOTS.workers,
    )
)

include: str(RULES)


# No declared output, so the manifest step runs on every invocation and demands every cell's outputs.
rule all:
    input:
        [str(p) for cell in CELLS for p in cell.outputs],
    shell:
        f"{PYTHON} -m library.driver manifest {ROOT_FLAGS} --log-root {Path('logs').resolve()}"
