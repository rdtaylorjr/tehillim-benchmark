"""Declares every benchmark cell: the scripts, what each reads, and what each writes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.datasets import discover_domains, domain_root
from core.driver import Cell
from core.partition import BHSA_HALF_VERSE, PART_FILE, Scope
from families.shuffle import FAMILIES, dataset_source

from genre.taxonomies import TAXONOMIES
from trajectory.scripts.validate_against_genre import METRICS

BENCHMARK_ROOT = "analysis=benchmark"
#: Every order-shuffle control script ends in this, whichever benchmark it belongs to.
SHUFFLE_MODULE_SUFFIX = ".shuffle_order_control"
#: Dropped from a control's filename, because the committed tree omits it.
IMPLICIT_LEVEL = "phrase"

#: The benchmarks score the Masoretic Psalms at the accentual half-verse, since their labels
#: (parallel pairs, genres, trajectories) are keyed to BHSA half-verse nodes; no other scope of
#: the embeddings tree is read.
SCOPE: Scope = BHSA_HALF_VERSE


@dataclass(frozen=True, slots=True)
class Roots:
    """Every location a cell resolves against, supplied once per run."""

    data_root: Path
    embeddings_root: Path
    config_root: Path
    genre_csv: Path
    gunkel_csv: Path
    ui_root: Path
    workers: int

    def labels_of(self, taxonomy: str) -> Path:
        """The table a taxonomy's passages are read from."""
        return {"logos": self.genre_csv, "gunkel": self.gunkel_csv}[taxonomy]


def genre_registers() -> list[tuple[str, str | None]]:
    """Every (taxonomy, unit) the genre benchmark scores, unit None where the taxonomy has none."""
    return [
        (taxonomy.name, unit)
        for taxonomy in TAXONOMIES.values()
        for unit in (taxonomy.units or (None,))
    ]


def register_name(taxonomy: str, unit: str | None) -> str:
    """The dotted segment a register carries in cell names: the taxonomy, then its unit if any."""
    return taxonomy if unit is None else f"{taxonomy}.{unit}"


def register_key(taxonomy: str, unit: str | None) -> str:
    """The segment a register carries in file names, the taxonomy then its unit if any."""
    return taxonomy if unit is None else f"{taxonomy}_{unit}"


def domain_dir(roots: Roots, domain: str) -> Path:
    """The directory of one domain's datasets under the scope the benchmarks read."""
    return domain_root(roots.embeddings_root, SCOPE, domain)


def domain_datasets(roots: Roots, domain: str) -> tuple[Path, ...]:
    """Every dataset file of one domain, which every domain-sweeping stage reads."""
    return tuple(sorted(domain_dir(roots, domain).rglob(PART_FILE)))


def stage_dir(roots: Roots, benchmark: str, domain: str, stage: str) -> Path:
    """The partition one stage of one benchmark writes for one domain."""
    return roots.data_root / BENCHMARK_ROOT / f"benchmark={benchmark}/domain={domain}/stage={stage}"


def genre_partition(data_root: Path, taxonomy: str, unit: str | None, domain: str) -> Path:
    """One taxonomy's genre partition in one domain, under its unit register where it has one."""
    levels = [f"taxonomy={taxonomy}"] + ([] if unit is None else [f"unit={unit}"])
    return data_root / BENCHMARK_ROOT / "benchmark=genre" / "/".join(levels) / f"domain={domain}"


def genre_dir(roots: Roots, taxonomy: str, unit: str | None, domain: str) -> Path:
    """The genre partition under the run's data root."""
    return genre_partition(roots.data_root, taxonomy, unit, domain)


def genre_stage_dir(roots: Roots, taxonomy: str, unit: str | None, domain: str, stage: str) -> Path:
    """The partition one stage of the genre benchmark writes under one register."""
    return genre_dir(roots, taxonomy, unit, domain) / f"stage={stage}"


def control_filename(key: str) -> str:
    """The stem a family's control writes under, dropping the domain and the implicit level."""
    return "_".join(part for part in key.split("/")[1:] if part != IMPLICIT_LEVEL)


def shuffle_families_for(domain: str) -> tuple[str, ...]:
    """The registered order-shuffle families whose datasets sit in one domain."""
    return tuple(sorted(key for key in FAMILIES if key.split("/", 1)[0] == domain))


#: Scheduling priority by stage: the long stages start first so the short ones fill the tail.
STAGE_PRIORITY: dict[str, int] = {
    "by_genre": 40,
    "shuffle": 30,
    "detail": 20,
    "bootstrap": 20,
    "retrieval": 20,
    "calibrated": 10,
    "summary": 10,
}


def stage_priority(stage: str) -> int:
    """The priority a stage is scheduled at, read off the stage name inside a dotted cell stage."""
    return max(
        (STAGE_PRIORITY[part] for part in stage.split(".") if part in STAGE_PRIORITY), default=0
    )


def _cell(
    benchmark: str,
    domain: str,
    stage: str,
    module: str,
    inputs: tuple[Path, ...],
    outputs: tuple[Path, ...],
    args: list[str],
) -> Cell:
    """A benchmark cell, named benchmark.domain.stage, scheduled by the weight of its stage."""
    return Cell(
        name=f"{benchmark}.{domain}.{stage}",
        module=module,
        inputs=inputs,
        outputs=outputs,
        command_args=args,
        resource=None,
        priority=stage_priority(stage),
    )


def _workers(roots: Roots) -> list[str]:
    """The worker-count flag every scoring script takes."""
    return ["--workers", str(roots.workers)]


def _parallelism_cells(roots: Roots, domain: str) -> list[Cell]:
    """The parallelism chain for one domain: raw scores, detail, shuffle controls, master."""
    datasets = domain_datasets(roots, domain)
    embeddings_dir = domain_dir(roots, domain)
    raw = stage_dir(roots, "parallelism", domain, "raw")
    detail = stage_dir(roots, "parallelism", domain, "detail")
    master = stage_dir(roots, "parallelism", domain, "master")
    shuffle = stage_dir(roots, "parallelism", domain, "shuffle_control")
    scoring = (
        ("retrieval", "parallelism.scripts.compare_models", "retrieval.csv"),
        ("baseline", "parallelism.scripts.compare_baseline", "baseline.csv"),
        ("bootstrap", "parallelism.scripts.compute_bootstrap_cis", "bootstrap_cis.csv"),
        ("calibration", "parallelism.scripts.compare_true_similarity", "calibration.csv"),
    )
    cells = [
        _cell(
            "parallelism",
            domain,
            stage,
            module,
            datasets,
            (raw / filename,),
            [str(embeddings_dir), "--output", str(raw / filename), *_workers(roots)],
        )
        for stage, module, filename in scoring
    ]
    detail_files = ("pair_detail.parquet", "baseline_detail.parquet", "type_vs_baseline.parquet")
    cells.append(
        _cell(
            "parallelism",
            domain,
            "detail",
            "parallelism.scripts.export_detail",
            datasets,
            tuple(detail / name for name in detail_files),
            [str(embeddings_dir), "--output-dir", str(detail), *_workers(roots)],
        )
    )
    support = tuple(sorted(roots.config_root.glob("*.csv")))
    cells.extend(
        _cell(
            "parallelism",
            domain,
            f"shuffle.{control_filename(key)}",
            "parallelism.scripts.shuffle_order_control",
            (dataset_source(key, roots.embeddings_root), *support),
            (shuffle / f"{control_filename(key)}.csv",),
            [
                str(dataset_source(key, roots.embeddings_root)),
                "--family",
                key,
                "--config-root",
                str(roots.config_root),
                "--output",
                str(shuffle / f"{control_filename(key)}.csv"),
                *_workers(roots),
            ],
        )
        for key in shuffle_families_for(domain)
    )
    master_files = (
        "model_metrics_long.parquet",
        "model_metrics_overall.parquet",
        "model_metrics_by_type.parquet",
    )
    cells.append(
        _cell(
            "parallelism",
            domain,
            "master",
            "parallelism.scripts.build_master_report",
            (raw / "retrieval.csv", raw / "calibration.csv", *(detail / n for n in detail_files)),
            tuple(master / name for name in master_files),
            [
                "--retrieval-csv",
                str(raw / "retrieval.csv"),
                "--calibration-csv",
                str(raw / "calibration.csv"),
                "--detail-dir",
                str(detail),
                "--output-dir",
                str(master),
            ],
        )
    )
    return cells


def _register_cells(roots: Roots, domain: str, taxonomy: str, unit: str | None) -> list[Cell]:
    """One register's genre chain in one domain: baseline, scores, detail, master."""
    datasets = domain_datasets(roots, domain)
    embeddings_dir = domain_dir(roots, domain)
    labels = roots.labels_of(taxonomy)
    register = [str(labels), "--taxonomy", taxonomy]
    if unit is not None:
        register.extend(["--unit", unit])
    prefix = register_name(taxonomy, unit)

    def at(stage: str) -> Path:
        return genre_stage_dir(roots, taxonomy, unit, domain, stage)

    def cell(
        stage: str,
        module: str,
        inputs: tuple[Path, ...],
        outputs: tuple[Path, ...],
        args: list[str],
    ) -> Cell:
        """A cell of this register, named genre.domain.taxonomy[.unit].stage."""
        return _cell("genre", domain, f"{prefix}.{stage}", module, inputs, outputs, args)

    raw, detail, master = at("raw"), at("detail"), at("master")
    cells = [
        cell(
            "baseline",
            "genre.scripts.compare_baseline",
            (labels,),
            (raw / "baseline.csv",),
            [*register, "--output", str(raw / "baseline.csv")],
        )
    ]
    scoring = (
        ("summary", "genre.scripts.compare_models", "summary.csv"),
        ("by_genre", "genre.scripts.compare_by_genre", "by_genre.csv"),
        ("calibrated", "genre.scripts.compare_calibrated", "calibrated.csv"),
        ("bootstrap", "genre.scripts.compute_bootstrap_cis", "bootstrap_cis.csv"),
    )
    cells.extend(
        cell(
            stage,
            module,
            (labels, *datasets),
            (raw / filename,),
            [*register, str(embeddings_dir), "--output", str(raw / filename), *_workers(roots)],
        )
        for stage, module, filename in scoring
    )
    detail_files = ("genre_pair_detail.parquet", "genre_summary.parquet")
    cells.append(
        cell(
            "detail",
            "genre.scripts.export_detail",
            (labels, *datasets),
            tuple(detail / name for name in detail_files),
            [*register, str(embeddings_dir), "--output-dir", str(detail), *_workers(roots)],
        )
    )
    master_files = ("genre_metrics_long.parquet", "genre_metrics_wide.parquet")
    cells.append(
        cell(
            "master",
            "genre.scripts.build_master_report",
            (
                raw / "calibrated.csv",
                raw / "bootstrap_cis.csv",
                *(detail / n for n in detail_files),
            ),
            tuple(master / name for name in master_files),
            [
                "--summary-csv",
                str(raw / "calibrated.csv"),
                "--bootstrap-csv",
                str(raw / "bootstrap_cis.csv"),
                "--detail-dir",
                str(detail),
                "--output-dir",
                str(master),
            ],
        )
    )
    return cells


def _shuffle_cells(roots: Roots, domain: str, taxonomy: str) -> list[Cell]:
    """One order-shuffle control per family, its draws scored once for every register at once."""
    labels = roots.labels_of(taxonomy)
    units = [unit for name, unit in genre_registers() if name == taxonomy]
    register = [str(labels), "--taxonomy", taxonomy]
    for unit in units:
        if unit is not None:
            register.extend(["--unit", unit])
    outputs = {
        key: tuple(
            genre_stage_dir(roots, taxonomy, unit, domain, "shuffle_control")
            / f"{control_filename(key)}.csv"
            for unit in units
        )
        for key in shuffle_families_for(domain)
    }
    support = tuple(sorted(roots.config_root.glob("*.csv")))
    return [
        _cell(
            "genre",
            domain,
            f"{taxonomy}.shuffle.{control_filename(key)}",
            "genre.scripts.shuffle_order_control",
            (labels, dataset_source(key, roots.embeddings_root), *support),
            outputs[key],
            [
                *register,
                str(dataset_source(key, roots.embeddings_root)),
                "--family",
                key,
                "--config-root",
                str(roots.config_root),
                *(flag for path in outputs[key] for flag in ("--output", str(path))),
                *_workers(roots),
            ],
        )
        for key in shuffle_families_for(domain)
    ]


def _genre_cells(roots: Roots, domain: str) -> list[Cell]:
    """The genre chains for one domain: every register's stages, then each taxonomy's shuffles."""
    registers = genre_registers()
    cells = [
        cell
        for taxonomy, unit in registers
        for cell in _register_cells(roots, domain, taxonomy, unit)
    ]
    for taxonomy in dict.fromkeys(taxonomy for taxonomy, _ in registers):
        cells.extend(_shuffle_cells(roots, domain, taxonomy))
    return cells


def _trajectory_cells(roots: Roots, domain: str) -> list[Cell]:
    """The trajectory chain for one domain: distances, genre validation, interface rows."""
    datasets = domain_datasets(roots, domain)
    embeddings_dir = domain_dir(roots, domain)
    raw = stage_dir(roots, "trajectory", domain, "raw")
    ui = stage_dir(roots, "trajectory", domain, "ui")
    distances = raw / "trajectory_distances.parquet"
    validation = raw / "validate_against_genre.csv"
    breakdown = raw / "validate_against_genre_by_genre.csv"
    return [
        _cell(
            "trajectory",
            domain,
            "distances",
            "trajectory.scripts.compute_profiles",
            datasets,
            (distances,),
            [str(embeddings_dir), "--output", str(distances), *_workers(roots)],
        ),
        _cell(
            "trajectory",
            domain,
            "validate",
            "trajectory.scripts.validate_against_genre",
            (roots.genre_csv, distances),
            (validation, breakdown),
            [
                str(roots.genre_csv),
                str(distances),
                "--output",
                str(validation),
                "--breakdown-output",
                str(breakdown),
                *_workers(roots),
            ],
        ),
        _cell(
            "trajectory",
            domain,
            "ui_rows",
            "trajectory.scripts.export_ui_rows",
            (validation, breakdown),
            (ui / "ui_rows.json", ui / "ui_rows_by_genre.json"),
            [
                str(validation),
                "--breakdown-csv",
                str(breakdown),
                "--output",
                str(ui / "ui_rows.json"),
                "--breakdown-output",
                str(ui / "ui_rows_by_genre.json"),
            ],
        ),
    ]


def _ui_cells(roots: Roots, domain: str) -> list[Cell]:
    """The front-end payloads for one domain: the summary JSON with its slices, and the details."""
    parallelism_dir = roots.data_root / BENCHMARK_ROOT / f"benchmark=parallelism/domain={domain}"
    registers = [(genre_dir(roots, t, u, domain), register_key(t, u)) for t, u in genre_registers()]
    trajectory_ui = stage_dir(roots, "trajectory", domain, "ui")
    public = roots.ui_root / "public" / "data"
    payload = public / f"ui_{domain}.json"
    slices = (
        *(public / f"ui_{domain}_trajectory_{metric}.json" for metric in METRICS),
        *(public / f"ui_{domain}_genre_{key}.json" for _, key in registers),
    )
    payload_inputs = (
        parallelism_dir / "stage=master" / "model_metrics_overall.parquet",
        parallelism_dir / "stage=master" / "model_metrics_by_type.parquet",
        *(
            genre / stage / name
            for genre, _ in registers
            for stage, name in (
                ("stage=master", "genre_metrics_wide.parquet"),
                ("stage=raw", "by_genre.csv"),
                ("stage=raw", "baseline.csv"),
            )
        ),
        trajectory_ui / "ui_rows.json",
        trajectory_ui / "ui_rows_by_genre.json",
    )
    detail_inputs = (
        roots.genre_csv,
        roots.gunkel_csv,
        payload,
        parallelism_dir / "stage=detail" / "pair_detail.parquet",
        parallelism_dir / "stage=detail" / "baseline_detail.parquet",
        parallelism_dir / "stage=raw" / "bootstrap_cis.csv",
        *(
            genre / stage / name
            for genre, _ in registers
            for stage, name in (
                ("stage=detail", "genre_pair_detail.parquet"),
                ("stage=raw", "bootstrap_cis.csv"),
            )
        ),
        stage_dir(roots, "trajectory", domain, "raw") / "trajectory_distances.parquet",
        stage_dir(roots, "trajectory", domain, "raw") / "validate_against_genre.csv",
    )
    detail_dir = roots.ui_root / "detail-data"
    return [
        _cell(
            "ui",
            domain,
            "payload",
            "ui_export.export",
            payload_inputs,
            (payload, *slices),
            [
                domain,
                "--parallelism-dir",
                str(parallelism_dir),
                "--data-root",
                str(roots.data_root),
                "--trajectory-ui-rows",
                str(trajectory_ui / "ui_rows.json"),
                "--trajectory-by-genre-rows",
                str(trajectory_ui / "ui_rows_by_genre.json"),
                "--output",
                str(payload),
            ],
        ),
        _cell(
            "ui",
            domain,
            "detail",
            "ui_export.scripts.build_detail_json",
            detail_inputs,
            (detail_dir / f"detail_{domain}_index.json",),
            [
                str(roots.genre_csv),
                "--gunkel-csv",
                str(roots.gunkel_csv),
                "--data-dir",
                str(roots.data_root),
                "--ui-dir",
                str(public),
                "--domains",
                domain,
                "--output-dir",
                str(detail_dir),
                *_workers(roots),
            ],
        ),
    ]


BUILDERS: tuple[Callable[[Roots, str], list[Cell]], ...] = (
    _parallelism_cells,
    _genre_cells,
    _trajectory_cells,
    _ui_cells,
)


def plan_cells(roots: Roots) -> list[Cell]:
    """Every cell for every domain the embeddings tree holds."""
    return [
        cell
        for domain in discover_domains(roots.embeddings_root, SCOPE)
        for build in BUILDERS
        for cell in build(roots, domain)
    ]
