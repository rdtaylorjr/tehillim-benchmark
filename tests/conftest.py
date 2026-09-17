from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from genre.passages import Passage


def _write_embeddings_parquet(path: Path, vectors: dict[int, list[float]]) -> Path:
    """Writes a dense tehillim-embeddings Parquet file, the shape load_embeddings expects."""
    node_ids = sorted(vectors)
    dim = len(vectors[node_ids[0]])
    matrix = np.array([vectors[n] for n in node_ids], dtype="<f4")
    table = pa.table(
        {
            "node_id": pa.array(node_ids, type=pa.int32()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.flatten(), type=pa.float32()), dim
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def _write_sparse_embeddings_parquet(
    path: Path, vectors: dict[int, list[float]], dim: int | None = None
) -> Path:
    """Writes the sparse embeddings layout that load_sparse_embeddings dispatches on."""
    node_ids = sorted(vectors)
    rows = [np.asarray(vectors[node], dtype="<f4") for node in node_ids]
    width = dim if dim is not None else len(rows[0])
    table = pa.table(
        {
            "node_id": pa.array(node_ids, type=pa.int32()),
            "indices": pa.array(
                [np.flatnonzero(row).astype("<i4").tolist() for row in rows],
                type=pa.list_(pa.int32()),
            ),
            "values": pa.array(
                [row[np.flatnonzero(row)].tolist() for row in rows], type=pa.list_(pa.float32())
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata({"dim": str(width), "sparse": "true"}), path)
    return path


#: The scope every benchmark reads: the Masoretic Psalms at the accentual half-verse.
SCOPE_DIR = "corpus=bhsa/unit=half_verse"


#: Every fake half-verse holds this many words, numbered from ten times its node.
WORDS_PER_HALF_VERSE = 3


def fake_words(half_verse: int) -> list[int]:
    """The word nodes of a fake half-verse, in text order."""
    return [half_verse * 10 + k for k in range(1, WORDS_PER_HALF_VERSE + 1)]


def whole_psalm_passages(genre_by_psalm: dict[int, str]) -> list[Passage]:
    """One whole-psalm passage per labelled psalm, its one half-verse node numbered as the psalm."""
    return [
        Passage(str(p), p, genre, tuple(fake_words(p)), f"Ps {p}")
        for p, genre in sorted(genre_by_psalm.items())
    ]


def semantic_file(root: Path, model: str, name: str = "part-0.parquet") -> Path:
    """A fixture's place in the tree: one semantic model at the consonantal tier."""
    return root / SCOPE_DIR / "domain=semantic" / f"model={model}" / "text=consonantal" / name


@pytest.fixture
def write_sparse_embeddings_parquet():
    """Factory writing a sparse embeddings Parquet file at a given path."""
    return _write_sparse_embeddings_parquet


@pytest.fixture
def write_embeddings_parquet():
    """Factory writing a dense embeddings Parquet file at a given path."""
    return _write_embeddings_parquet


class _FakeOtype:
    """Text-Fabric's otype feature, which lists the nodes of a given type."""

    def __init__(self, book_types: dict[int, str]) -> None:
        self._book_types = book_types

    def s(self, otype: str) -> list[int]:
        return list(self._book_types) if otype == "book" else []


class _FakeFeature:
    """A Text-Fabric node feature, which maps a node to its value."""

    def __init__(self, values: dict[int, str]) -> None:
        self._values = values

    def v(self, node: int) -> str | None:
        return self._values.get(node)


class _FakeF:
    """Text-Fabric's node-feature namespace."""

    def __init__(self, book_names: dict[int, str]) -> None:
        self.otype = _FakeOtype(book_names)
        self.book = _FakeFeature(book_names)


class _FakeL:
    """Text-Fabric's locality namespace, which walks from a node to its children."""

    def __init__(self, children: dict[tuple[int, str], list[int]]) -> None:
        self._children = children

    def d(self, node: int, otype: str) -> list[int]:
        return self._children.get((node, otype), [])


class _FakeT:
    """Text-Fabric's text namespace, which resolves a chapter or verse node to its section."""

    def __init__(
        self, chapter_to_psalm: dict[int, int], verse_to_section: dict[int, tuple[int, int]]
    ) -> None:
        self._chapter_to_psalm = chapter_to_psalm
        self._verse_to_section = verse_to_section

    def sectionFromNode(self, node: int) -> tuple[str, int] | tuple[str, int, int]:  # noqa: N802
        if node in self._verse_to_section:
            return ("Psalmi", *self._verse_to_section[node])
        return ("Psalmi", self._chapter_to_psalm[node])


class _FakeApi:
    """The Text-Fabric surface every benchmark reads the corpus through."""

    def __init__(self, F: _FakeF, L: _FakeL, T: _FakeT) -> None:  # noqa: N803
        self.F = F
        self.L = L
        self.T = T


def _bhsa_api_over(half_verses_by_psalm: dict[int, list[int]]) -> _FakeApi:
    """Builds a fake api whose Psalms book yields the given half-verse nodes, one per verse."""
    chapter_of = {psalm: 1000 + psalm for psalm in half_verses_by_psalm}
    children: dict[tuple[int, str], list[int]] = {
        (1, "chapter"): [chapter_of[psalm] for psalm in sorted(half_verses_by_psalm)]
    }
    verse_to_section: dict[int, tuple[int, int]] = {}
    for psalm, nodes in half_verses_by_psalm.items():
        children[chapter_of[psalm], "half_verse"] = list(nodes)
        verses = [100_000 + 100 * psalm + number for number in range(1, len(nodes) + 1)]
        children[chapter_of[psalm], "verse"] = verses
        for number, (verse, node) in enumerate(zip(verses, nodes, strict=True), start=1):
            children[verse, "half_verse"] = [node]
            children[verse, "word"] = fake_words(node)
            children[node, "word"] = fake_words(node)
            verse_to_section[verse] = (psalm, number)
    return _FakeApi(
        _FakeF({1: "Psalmi"}),
        _FakeL(children),
        _FakeT({node: psalm for psalm, node in chapter_of.items()}, verse_to_section),
    )


@pytest.fixture
def bhsa_api_over():
    """Factory building a fake BHSA api from a psalm-to-half-verse-nodes mapping."""
    return _bhsa_api_over


PARALLEL_FEATURES = (
    "parallel_group_id",
    "parallel_member_id",
    "parallel_type",
    "parallel_group",
    "parallel_member",
    "parallel_signature",
    "parallel_ambiguous",
)


def _parallel_bhsa_api_over(
    half_verses_by_psalm: dict[int, list[int]], annotations: dict[int, dict[str, str]]
) -> _FakeApi:
    """A fake api that also carries tehillim-logos's parallel_* features on its annotated nodes."""
    api = _bhsa_api_over(half_verses_by_psalm)
    for feature in PARALLEL_FEATURES:
        values = {node: row[feature] for node, row in annotations.items() if feature in row}
        setattr(api.F, feature, _FakeFeature(values))
    return api


@pytest.fixture
def parallel_bhsa_api_over():
    """Factory building a fake BHSA api carrying parallelism annotations on the given nodes."""
    return _parallel_bhsa_api_over
