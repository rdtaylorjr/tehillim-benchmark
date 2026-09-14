"""Declares every benchmark cell: the scripts, what each reads, and what each writes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.driver import Cell
from families.shuffle import FAMILIES, dataset_source

from library.shuffle_control_sweep import control_filename
from trajectory.scripts.validate_against_genre import METRICS

BENCHMARK_ROOT = "analysis=benchmark"
PART_FILE = "part-0.parquet"


@dataclass(frozen=True, slots=True)
class Roots:
    """Every location a cell resolves against, supplied once per run."""

    data_root: Path
    embeddings_root: Path
    config_root: Path
    genre_csv: Path
    ui_root: Path
    workers: int


def discover_domains(embeddings_root: Path) -> tuple[str, ...]:
    """The representation domains present in the embeddings tree."""
    return tuple(
        sorted(p.name.split("=", 1)[1] for p in embeddings_root.glob("domain=*") if p.is_dir())
    )


def domain_datasets(roots: Roots, domain: str) -> tuple[Path, ...]:
    """Every dataset file of one domain, which every domain-sweeping stage reads."""
    return tuple(sorted((roots.embeddings_root / f"domain={domain}").rglob(PART_FILE)))


def stage_dir(roots: Roots, benchmark: str, domain: str, stage: str) -> Path:
    """The partition one stage of one benchmark writes for one domain."""
    return roots.data_root / BENCHMARK_ROOT / f"benchmark={benchmark}/domain={domain}/stage={stage}"


def shuffle_families_for(domain: str) -> tuple[str, ...]:
    """The registered order-shuffle families whose datasets sit in one domain."""
    return tuple(sorted(key for key in FAMILIES if key.split("/", 1)[0] == domain))


def _cell(
    benchmark: str,
    domain: str,
    stage: str,
    module: str,
    inputs: tuple[Path, ...],
    outputs: tuple[Path, ...],
    args: list[str],
) -> Cell:
    """A benchmark cell, named benchmark.domain.stage."""
    return Cell(
        name=f"{benchmark}.{domain}.{stage}",
        module=module,
        inputs=inputs,
        outputs=outputs,
        command_args=args,
        resource=None,
    )


def _workers(roots: Roots) -> list[str]:
    """The worker-count flag every scoring script takes."""
    return ["--workers", str(roots.workers)]


def _parallelism_cells(roots: Roots, domain: str) -> list[Cell]:
    """The parallelism chain for one domain: raw scores, detail, shuffle controls, master."""
    datasets = domain_datasets(roots, domain)
    embeddings_dir = roots.embeddings_root / f"domain={domain}"
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


def _genre_cells(roots: Roots, domain: str) -> list[Cell]:
    """The genre chain for one domain: raw scores, detail, shuffle controls, master."""
    datasets = domain_datasets(roots, domain)
    embeddings_dir = roots.embeddings_root / f"domain={domain}"
    raw = stage_dir(roots, "genre", domain, "raw")
    detail = stage_dir(roots, "genre", domain, "detail")
    master = stage_dir(roots, "genre", domain, "master")
    shuffle = stage_dir(roots, "genre", domain, "shuffle_control")
    scoring = (
        ("summary", "genre.scripts.compare_models", "summary.csv"),
        ("by_genre", "genre.scripts.compare_by_genre", "by_genre.csv"),
        ("calibrated", "genre.scripts.compare_calibrated", "calibrated.csv"),
        ("bootstrap", "genre.scripts.compute_bootstrap_cis", "bootstrap_cis.csv"),
    )
    cells = [
        _cell(
            "genre",
            domain,
            stage,
            module,
            (roots.genre_csv, *datasets),
            (raw / filename,),
            [
                str(roots.genre_csv),
                str(embeddings_dir),
                "--output",
                str(raw / filename),
                *_workers(roots),
            ],
        )
        for stage, module, filename in scoring
    ]
    detail_files = ("genre_pair_detail.parquet", "genre_summary.parquet")
    cells.append(
        _cell(
            "genre",
            domain,
            "detail",
            "genre.scripts.export_detail",
            (roots.genre_csv, *datasets),
            tuple(detail / name for name in detail_files),
            [
                str(roots.genre_csv),
                str(embeddings_dir),
                "--output-dir",
                str(detail),
                *_workers(roots),
            ],
        )
    )
    support = tuple(sorted(roots.config_root.glob("*.csv")))
    cells.extend(
        _cell(
            "genre",
            domain,
            f"shuffle.{control_filename(key)}",
            "genre.scripts.shuffle_order_control",
            (roots.genre_csv, dataset_source(key, roots.embeddings_root), *support),
            (shuffle / f"{control_filename(key)}.csv",),
            [
                str(roots.genre_csv),
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
    master_files = ("genre_metrics_long.parquet", "genre_metrics_wide.parquet")
    cells.append(
        _cell(
            "genre",
            domain,
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


def _trajectory_cells(roots: Roots, domain: str) -> list[Cell]:
    """The trajectory chain for one domain: distances, genre validation, interface rows."""
    datasets = domain_datasets(roots, domain)
    embeddings_dir = roots.embeddings_root / f"domain={domain}"
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
    genre_dir = roots.data_root / BENCHMARK_ROOT / f"benchmark=genre/domain={domain}"
    trajectory_ui = stage_dir(roots, "trajectory", domain, "ui")
    public = roots.ui_root / "public" / "data"
    payload = public / f"ui_{domain}.json"
    slices = tuple(public / f"ui_{domain}_trajectory_{metric}.json" for metric in METRICS)
    payload_inputs = (
        parallelism_dir / "stage=master" / "model_metrics_overall.parquet",
        parallelism_dir / "stage=master" / "model_metrics_by_type.parquet",
        genre_dir / "stage=master" / "genre_metrics_wide.parquet",
        genre_dir / "stage=raw" / "by_genre.csv",
        trajectory_ui / "ui_rows.json",
        trajectory_ui / "ui_rows_by_genre.json",
    )
    detail_inputs = (
        roots.genre_csv,
        payload,
        parallelism_dir / "stage=detail" / "pair_detail.parquet",
        parallelism_dir / "stage=detail" / "baseline_detail.parquet",
        parallelism_dir / "stage=raw" / "bootstrap_cis.csv",
        genre_dir / "stage=detail" / "genre_pair_detail.parquet",
        genre_dir / "stage=raw" / "bootstrap_cis.csv",
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
                "--genre-dir",
                str(genre_dir),
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
        for domain in discover_domains(roots.embeddings_root)
        for build in BUILDERS
        for cell in build(roots, domain)
    ]
