"""Batch-generates one row-click detail JSON per model, for every model any ui table links to."""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from core.parallel import map_in_order

from genre.genre_labels import load_genre_by_psalm
from genre.passages import Passage, load_passages
from library.bhsa import list_psalms_half_verses_by_psalm, load_bhsa_api
from library.cli import add_scoring_arguments
from library.rows_output import write_json
from library.scoring import skipping_unscorable
from library.stages import genre_partition, genre_registers, register_key
from trajectory.residualize import residualize_by_length, residualize_on_covariates
from ui_export.detail import (
    auc_ap_ci_for,
    build_genre_detail,
    build_parallelism_detail,
    build_trajectory_detail,
    validated_gap_stats_for,
)

_TRAJECTORY_SOURCES = ("length_controlled", "length_and_content_controlled")


@dataclass(frozen=True, slots=True)
class GenreSource:
    """One taxonomy read at one register: the section it names and the passages it compares."""

    taxonomy: str
    unit: str | None
    passages: list[Passage]

    @property
    def section(self) -> str:
        """The detail section, and file suffix, this source's genre view is served under."""
        return f"genre_{register_key(self.taxonomy, self.unit)}"

    @property
    def genres(self) -> list[str]:
        """Every label the register assigns, in sorted order."""
        return sorted({passage.gattung for passage in self.passages})

    @property
    def genre_by_item(self) -> dict[str, str]:
        """Each passage's label by its id, the join the pair table needs."""
        return {passage.id: passage.gattung for passage in self.passages}

    @property
    def items(self) -> dict[str, tuple[int, str]]:
        """Each passage's psalm and on-screen name by its id."""
        return {passage.id: (passage.psalm, passage.label) for passage in self.passages}


def genre_sources(
    tables: dict[str, Path],
    api: Any,  # noqa: ANN401
    *,
    loader: Callable[[str, str | None, Path, Any], list[Passage]] = load_passages,
) -> list[GenreSource]:
    """Every register the genre benchmark scores, with its passages read against the corpus."""
    return [
        GenreSource(taxonomy, unit, loader(taxonomy, unit, tables[taxonomy], api))
        for taxonomy, unit in genre_registers()
    ]


def table_model_sets(
    domain_json: dict[str, Any], sources: list[GenreSource]
) -> dict[str, set[str]]:
    """Each table's own model set, kept separate so a section is built only where used."""
    genre_rows = domain_json["genre_overall"]
    return {
        "parallelism": {r["model"] for r in domain_json["parallelism_overall"]},
        **{
            source.section: {
                r["model"]
                for r in genre_rows
                if r["taxonomy"] == source.taxonomy and r["unit"] == source.unit
            }
            for source in sources
        },
        "trajectory": {r["model"] for r in domain_json["trajectory"]},
    }


def choose_primary_metric(validate_df: pd.DataFrame, model: str) -> str | None:
    """Picks the metric with the strongest (smallest) length-controlled p-value for one model."""
    # content_distance excluded: its content-controlled source residualizes against itself, NaN.
    rows = validate_df[
        (validate_df.model == model) & (validate_df.metric != "content_distance")
    ].dropna(subset=["length_controlled_p"])
    if rows.empty:
        return None
    return str(rows.loc[rows.length_controlled_p.idxmin(), "metric"])


def attach_genre_columns[K](
    pair_df: pd.DataFrame, genre_by_item: dict[K, str], key: str = "psalm"
) -> pd.DataFrame:
    """Adds genre_a/genre_b/same_genre by `{key}_a` and `{key}_b`, joined in memory only."""
    pair_df = pair_df.copy()
    pair_df["genre_a"] = pair_df[f"{key}_a"].map(genre_by_item)
    pair_df["genre_b"] = pair_df[f"{key}_b"].map(genre_by_item)
    pair_df["same_genre"] = pair_df.genre_a == pair_df.genre_b
    return pair_df


def residualize_trajectory_metric(
    traj_df: pd.DataFrame, metric: str, n_half_verses: dict[int, int]
) -> pd.DataFrame:
    """Adds length_controlled/length_and_content_controlled columns for one metric, per model."""
    df = traj_df[traj_df[[metric, "content_distance"]].notna().all(axis=1)].copy()
    length_diff = (df.psalm_a.map(n_half_verses) - df.psalm_b.map(n_half_verses)).abs().to_numpy()
    raw = df[metric].to_numpy()
    content = df["content_distance"].to_numpy()
    df["length_controlled"] = residualize_by_length(raw, length_diff)
    df["length_and_content_controlled"] = residualize_on_covariates(
        raw, np.column_stack([length_diff, content])
    )
    return df


# A payload holding only "model" and "domain" gained no section worth writing.
_EMPTY_PAYLOAD_KEYS = 2


def split_sections(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    """Writes one file per section, since the detail view renders exactly one of them at a time."""
    written: list[Path] = []
    for section, content in payload.items():
        if section in ("model", "domain"):
            continue
        body = {"model": payload["model"], "domain": payload["domain"], section: content}
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"detail_{payload['domain']}_{payload['model']}_{section}.json"
        write_json(path, body)
        written.append(path)
    return written


@dataclass(frozen=True, slots=True)
class ModelDetailInputs:
    """One model's per-section frames and precomputed statistics, absent where it has no data."""

    pair_detail: pd.DataFrame | None = None
    baseline_detail: pd.DataFrame | None = None
    #: One pair frame and CI per genre section the model appears in, keyed by section.
    genre_pair: dict[str, pd.DataFrame] = field(default_factory=dict)
    genre_auc_ap: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    trajectory: pd.DataFrame | None = None
    trajectory_metric: str | None = None
    parallelism_auc_ap: dict[str, Any] | None = None
    gap_stats: dict[str, dict[str, float]] | None = None


@dataclass(frozen=True, slots=True)
class GenreMeta:
    """What a genre section needs beyond its pairs: the labels it ranks and the items it names."""

    genres: list[str]
    items: dict[str, tuple[int, str]]


def build_one_model(
    model: str,
    domain: str,
    output_dir: Path,
    genres: list[str],
    genre_meta: dict[str, GenreMeta],
    inputs: ModelDetailInputs,
) -> list[Path]:
    """Writes whichever sections have real data for one model, returning the files it wrote."""
    payload: dict[str, Any] = {"model": model, "domain": domain}
    if (
        inputs.pair_detail is not None
        and not inputs.pair_detail.empty
        and inputs.baseline_detail is not None
    ):
        payload["parallelism"] = build_parallelism_detail(
            inputs.pair_detail, inputs.baseline_detail, inputs.parallelism_auc_ap
        )
    for section, pairs in inputs.genre_pair.items():
        if pairs.empty:
            continue
        meta = genre_meta[section]
        payload[section] = build_genre_detail(
            pairs, meta.genres, inputs.genre_auc_ap.get(section), meta.items
        )
    if (
        inputs.trajectory is not None
        and not inputs.trajectory.empty
        and inputs.trajectory_metric is not None
    ):
        payload["trajectory"] = build_trajectory_detail(
            inputs.trajectory, inputs.trajectory_metric, genres, inputs.gap_stats
        )
    if len(payload) <= _EMPTY_PAYLOAD_KEYS:
        return []
    return split_sections(payload, output_dir)


class _DomainSources(NamedTuple):
    """Every frame one domain's detail files are built from, loaded once and shared by model."""

    genres: list[str]
    genre_meta: dict[str, GenreMeta]
    table_models: dict[str, set[str]]
    pair_detail: dict[str, pd.DataFrame]
    baseline_detail: dict[str, pd.DataFrame]
    #: Per genre section, one pair frame per model.
    genre_pair: dict[str, dict[str, pd.DataFrame]]
    trajectory: dict[str, pd.DataFrame]
    parallelism_ci: pd.DataFrame | None
    genre_ci: dict[str, pd.DataFrame | None]
    validate: pd.DataFrame


def _dataset(data_dir: Path, benchmark: str, domain: str, stage: str, name: str) -> Path:
    """The hive path of one dataset, so the layout is written once rather than at every call."""
    partition = f"analysis=benchmark/benchmark={benchmark}/domain={domain}/stage={stage}"
    return data_dir / partition / name


def _genre_dataset(data_dir: Path, source: GenreSource, domain: str, stage: str, name: str) -> Path:
    """The hive path of one genre dataset under the source's taxonomy and unit."""
    return genre_partition(data_dir, source.taxonomy, source.unit, domain) / f"stage={stage}" / name


def _grouped_by_model(
    path: Path, genre_by_item: dict[Any, str] | None = None, key: str = "psalm"
) -> dict[str, "pd.DataFrame"]:
    """One frame per model, empty when the dataset was never produced for this domain."""
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if genre_by_item is not None:
        frame = attach_genre_columns(frame, genre_by_item, key)
    return dict(tuple(frame.groupby("model")))


def _genre_pairs_by_model(path: Path, source: GenreSource) -> dict[str, "pd.DataFrame"]:
    """One frame per model with the passages as items, the labels joined from the source."""
    if not path.exists():
        return {}
    frame = pd.read_parquet(path).rename(columns={"passage_a": "item_a", "passage_b": "item_b"})
    frame = attach_genre_columns(frame, source.genre_by_item, key="item")
    return dict(tuple(frame.groupby("model")))


def _load_domain_sources(
    domain: str,
    data_dir: Path,
    domain_json: dict[str, Any],
    genre_by_psalm: dict[int, str],
    sources: list[GenreSource],
) -> _DomainSources:
    """Reads every input for one domain up front, so each model is assembled from memory."""
    parallelism_ci = _dataset(data_dir, "parallelism", domain, "raw", "bootstrap_cis.csv")
    genre_cis = {
        source.section: _genre_dataset(data_dir, source, domain, "raw", "bootstrap_cis.csv")
        for source in sources
    }
    return _DomainSources(
        genres=sorted(set(genre_by_psalm.values())),
        genre_meta={source.section: GenreMeta(source.genres, source.items) for source in sources},
        table_models=table_model_sets(domain_json, sources),
        pair_detail=_grouped_by_model(
            _dataset(data_dir, "parallelism", domain, "detail", "pair_detail.parquet")
        ),
        baseline_detail=_grouped_by_model(
            _dataset(data_dir, "parallelism", domain, "detail", "baseline_detail.parquet")
        ),
        genre_pair={
            source.section: _genre_pairs_by_model(
                _genre_dataset(data_dir, source, domain, "detail", "genre_pair_detail.parquet"),
                source,
            )
            for source in sources
        },
        trajectory=_grouped_by_model(
            _dataset(data_dir, "trajectory", domain, "raw", "trajectory_distances.parquet"),
            genre_by_psalm,
        ),
        parallelism_ci=pd.read_csv(parallelism_ci) if parallelism_ci.exists() else None,
        genre_ci={
            section: pd.read_csv(path) if path.exists() else None
            for section, path in genre_cis.items()
        },
        validate=pd.read_csv(
            _dataset(data_dir, "trajectory", domain, "raw", "validate_against_genre.csv")
        ),
    )


def _model_inputs(
    model: str, sources: _DomainSources, n_half_verses: dict[int, int]
) -> ModelDetailInputs:
    """One model's slice of every source, present only for the tables that list it."""
    trajectory = (
        sources.trajectory.get(model) if model in sources.table_models["trajectory"] else None
    )
    metric = choose_primary_metric(sources.validate, model) if trajectory is not None else None
    if trajectory is not None and metric is not None:
        trajectory = residualize_trajectory_metric(trajectory, metric, n_half_verses)
    in_parallelism = model in sources.table_models["parallelism"]
    return ModelDetailInputs(
        pair_detail=sources.pair_detail.get(model) if in_parallelism else None,
        baseline_detail=sources.baseline_detail.get(model) if in_parallelism else None,
        genre_pair={
            section: pairs[model]
            for section, pairs in sources.genre_pair.items()
            if model in sources.table_models[section] and model in pairs
        },
        genre_auc_ap={
            section: auc_ap_ci_for(ci, model, None) if ci is not None else None
            for section, ci in sources.genre_ci.items()
        },
        trajectory=trajectory,
        trajectory_metric=metric,
        parallelism_auc_ap=auc_ap_ci_for(sources.parallelism_ci, model, "overall")
        if sources.parallelism_ci is not None
        else None,
        gap_stats=validated_gap_stats_for(sources.validate, model, metric)
        if metric is not None
        else None,
    )


class _ModelTask(NamedTuple):
    """Everything one worker needs to write one model's detail files."""

    model: str
    domain: str
    output_dir: Path
    genres: list[str]
    genre_meta: dict[str, GenreMeta]
    inputs: ModelDetailInputs


def _write_task(task: _ModelTask) -> list[Path]:
    """Writes one model's detail files, returning their paths."""
    return build_one_model(
        task.model, task.domain, task.output_dir, task.genres, task.genre_meta, task.inputs
    )


def index_path(output_dir: Path, domain: str) -> Path:
    """The per-domain listing of detail files, which is the one output a driver declares."""
    return output_dir / f"detail_{domain}_index.json"


def clear_domain(output_dir: Path, domain: str) -> None:
    """Removes a domain's earlier detail files so a rebuild never leaves a retired model behind."""
    for path in output_dir.glob(f"detail_{domain}_*.json"):
        path.unlink()


def _task_model(task: _ModelTask) -> str:
    """Names a task by its model, so a skipped one is reported the same way every batch is."""
    return task.model


def build_domain(
    domain: str,
    data_dir: Path,
    domain_json: dict[str, Any],
    genre_by_psalm: dict[int, str],
    genre_sources_: list[GenreSource],
    n_half_verses: dict[int, int],
    output_dir: Path,
    max_workers: int,
) -> int:
    """Rebuilds one domain's detail JSON from a clean slate and writes its index of files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_domain(output_dir, domain)
    sources = _load_domain_sources(domain, data_dir, domain_json, genre_by_psalm, genre_sources_)
    #: Each task is assembled here, so a worker carries one model's slice and not every frame.
    tasks = [
        _ModelTask(
            model=model,
            domain=domain,
            output_dir=output_dir,
            genres=sources.genres,
            genre_meta=sources.genre_meta,
            inputs=_model_inputs(model, sources, n_half_verses),
        )
        for model in sorted(set().union(*sources.table_models.values()))
    ]
    written = map_in_order(
        skipping_unscorable(_write_task, label=_task_model), tasks, max_workers, label="models"
    )
    files = sorted(path.name for paths in written if paths is not None for path in paths)
    write_json(index_path(output_dir, domain), files)
    return len(files)


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "genre_csv", type=Path, help="third-party genre CSV, e.g. psalms-browser.csv"
    )
    parser.add_argument(
        "--gunkel-csv", type=Path, required=True, help="tehillim-gunkel's tables/gunkel.csv"
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="tehillim-data checkout root")
    parser.add_argument(
        "--ui-dir",
        type=Path,
        required=True,
        help="directory holding ui_<domain>.json, which the site carries, not the data repo",
    )
    parser.add_argument(
        "--domains", nargs="+", default=["lexical", "morphological", "semantic", "syntactic"]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    add_scoring_arguments(parser)
    args = parser.parse_args(argv)

    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    api = api_factory(args.checkout)
    sources = genre_sources({"logos": args.genre_csv, "gunkel": args.gunkel_csv}, api)
    n_half_verses = {
        psalm: len(nodes) for psalm, nodes in list_psalms_half_verses_by_psalm(api).items()
    }

    for domain in args.domains:
        domain_json_path = args.ui_dir / f"ui_{domain}.json"
        domain_json = json.loads(domain_json_path.read_text())[domain]
        written = build_domain(
            domain,
            args.data_dir,
            domain_json,
            genre_by_psalm,
            sources,
            n_half_verses,
            args.output_dir,
            args.workers,
        )
        print(f"domain={domain}: wrote {written} detail files")


if __name__ == "__main__":
    main()
