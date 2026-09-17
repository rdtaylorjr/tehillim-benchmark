"""Order-shuffle null: does half-verse order carry more genre signal than a shuffled null."""

import argparse
import csv
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from core.parallel import map_seeds
from families.shuffle import Draws, draw, load_draws

from genre.bootstrap import psalm_similarity_matrix
from genre.evaluate import GenrePairIndex, average_precision_by_genre, index_genre_pairs
from genre.pairs import build_genre_pairs
from genre.passages import load_half_verse_weights, load_passages
from library.bhsa import load_bhsa_api
from library.centroid import Weights
from library.cli import add_scoring_arguments, add_shuffle_family_arguments, add_taxonomy_arguments
from library.errors import BenchmarkDataError
from library.order_shuffle import order_shuffle_result
from library.psalm_vectors import draw_item_vectors, load_item_vectors
from library.retrieval_metrics import sparse_cosine_similarity_matrix


def score_genre_ap(
    path: Path,
    half_verses_by_psalm: Mapping[str, Weights],
    pairs: GenrePairIndex,
    genres: list[str],
) -> dict[str, float]:
    """Per-genre Average Precision (no permutation testing) for one embeddings file."""
    return genre_ap(load_item_vectors(path, half_verses_by_psalm), pairs, genres)


def genre_ap(
    psalm_vectors: dict[str, np.ndarray] | tuple[list[str], sp.csr_matrix],
    pairs: GenrePairIndex,
    genres: list[str],
    *,
    similarity_matrix: Callable[..., np.ndarray] = psalm_similarity_matrix,
) -> dict[str, float]:
    """Per-genre Average Precision over item centroids held as dense rows or one sparse matrix."""
    if isinstance(psalm_vectors, tuple):
        psalm_ids, rows = psalm_vectors
        matrix = sparse_cosine_similarity_matrix(rows, rows)
    else:
        psalm_ids = sorted(psalm_vectors)
        matrix = similarity_matrix(psalm_ids, psalm_vectors)
    return average_precision_by_genre(pairs, matrix, psalm_ids, genres)


def require_scoreable(real_ap: dict[str, float], family: str) -> dict[str, float]:
    """Refuses a family whose real embeddings score on no genre, which a null cannot control."""
    if any(math.isfinite(ap) for ap in real_ap.values()):
        return real_ap
    raise BenchmarkDataError(
        f"no genre could be scored for {family}: every psalm centroid was dropped, "
        "so the shuffle null would compare NaN against NaN"
    )


@dataclass(frozen=True, slots=True)
class Register:
    """One unit register: its passages as weights, pairs and labels, and the file it writes."""

    unit: str | None
    half_verses_by_psalm: Mapping[str, Weights]
    pairs: GenrePairIndex
    genres: list[str]
    output: Path | None


@dataclass(frozen=True, slots=True)
class NullContext:
    """What every seed's draw is built and scored against, shipped to each worker once."""

    draws: Draws
    registers: tuple[Register, ...]


def score_built_seed(context: NullContext, seed: int) -> list[dict[str, float]]:
    """Builds one seed's draw once and scores it for every register, in register order."""
    draws = context.draws
    drawn = draw(draws, seed)
    return [
        genre_ap(
            draw_item_vectors(drawn, draws.sparse_width, register.half_verses_by_psalm),
            register.pairs,
            register.genres,
        )
        for register in context.registers
    ]


def built_null_scores(
    draws: Draws,
    n_shuffles: int,
    registers: tuple[Register, ...],
    workers: int | None,
) -> list[list[dict[str, float]]]:
    """One per-genre score set per seed and register, every draw built once and never written."""
    seeds = list(range(1, n_shuffles + 1))
    context = NullContext(draws, registers)
    return map_seeds(score_built_seed, context, seeds, max_workers=workers, label="shuffles")


def shuffled_scores_by_genre(
    scores_by_seed: list[dict[str, float]], genres: list[str]
) -> dict[str, np.ndarray]:
    """Transposes per-seed genre scores into one null array per genre, keeping seed order."""
    return {
        genre: np.array([scores[genre] for scores in scores_by_seed], dtype=float)
        for genre in genres
    }


def main(
    argv: list[str] | None = None,
    *,
    api_factory: Callable[[str], Any] = load_bhsa_api,
    draws_factory: Callable[[str, Path], Draws] = load_draws,
) -> None:
    """Parses the arguments this module documents, runs the batch, and writes its output."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_taxonomy_arguments(parser, many_units=True)
    parser.add_argument("real_embeddings", type=Path)
    add_shuffle_family_arguments(parser)
    add_scoring_arguments(parser, with_shuffles=True, many_outputs=True)
    args = parser.parse_args(argv)
    units: list[str | None] = args.unit or [None]
    outputs: list[Path | None] = args.output or [None] * len(units)
    if len(outputs) != len(units):
        parser.error("one --output per --unit, in the same order")

    api = api_factory(args.checkout)
    registers = tuple(
        _register(args.taxonomy, unit, args.labels_csv, api, output)
        for unit, output in zip(units, outputs, strict=True)
    )
    real_ap = [
        require_scoreable(
            score_genre_ap(
                args.real_embeddings, register.half_verses_by_psalm, register.pairs, register.genres
            ),
            args.family,
        )
        for register in registers
    ]
    per_seed = built_null_scores(
        draws_factory(args.family, args.config_root), args.n_shuffles, registers, args.workers
    )
    for index, register in enumerate(registers):
        shuffled_ap = shuffled_scores_by_genre(
            [scores[index] for scores in per_seed], register.genres
        )
        rows = control_rows(real_ap[index], shuffled_ap, register.genres)
        for row in rows:
            print(
                f"{row['genre']:15s} ap_real={row['ap_real']:.4f} "
                f"delta_order={row['delta_order']:+.4f} p={row['p_value']:.4f}"
            )
        if register.output:
            with register.output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def _register(
    taxonomy: str, unit: str | None, labels_csv: Path, api: object, output: Path | None
) -> Register:
    """One register's passages read against the corpus, with everything a seed scores it by."""
    passages = load_passages(taxonomy, unit, labels_csv, api)
    return Register(
        unit=unit,
        half_verses_by_psalm=load_half_verse_weights(passages, api),
        pairs=index_genre_pairs(build_genre_pairs(passages)),
        genres=sorted({passage.gattung for passage in passages}),
        output=output,
    )


def control_rows(
    real_ap: dict[str, float], shuffled_ap: dict[str, np.ndarray], genres: list[str]
) -> list[dict[str, str | int | float]]:
    """One row per genre: the real AP against its shuffle null, corrected across the genres."""
    rows: list[dict[str, str | int | float]] = []
    for genre in genres:
        result = order_shuffle_result(
            real_score=real_ap[genre],
            shuffled_scores=shuffled_ap[genre],
            n_hypotheses=len(genres),
        )
        rows.append(
            {
                "genre": genre,
                "ap_real": real_ap[genre],
                "ap_shuffled_mean": float(np.mean(shuffled_ap[genre])),
                "n_shuffles": len(shuffled_ap[genre]),
                "delta_order": result.delta_order,
                "p_value": result.p_value,
            }
        )
    return rows


if __name__ == "__main__":
    main()
