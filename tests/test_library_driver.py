import json
from pathlib import Path

import pytest
from core.driver import Cell

from library.driver import benchmark_code_hash, main, roots_from_config


class TestRootsFromConfig:
    def test_reads_every_root_and_defaults_the_worker_count(self, tmp_path: Path) -> None:
        roots = roots_from_config(
            {
                "data_root": str(tmp_path / "d"),
                "embeddings_root": str(tmp_path / "e"),
                "config_root": str(tmp_path / "c"),
                "genre_csv": str(tmp_path / "g.csv"),
                "ui_root": str(tmp_path / "u"),
            },
            default_workers=3,
        )
        assert roots.workers == 3
        assert roots.genre_csv == tmp_path / "g.csv"

    def test_a_missing_root_is_a_usage_error(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="embeddings_root"):
            roots_from_config({"data_root": str(tmp_path)}, default_workers=1)


class TestBenchmarkCodeHash:
    def test_a_shuffle_control_reaches_into_its_domain_vectorizers_only(self) -> None:
        """Its draws come from families.shuffle and the domain's builders, nothing else upstream."""
        from library.driver import BENCHMARK_PACKAGES, SHUFFLE_PACKAGES

        seen: list[tuple[str, ...]] = []

        def fake_hasher(module: str, roots: tuple[Path, ...], packages: tuple[str, ...]) -> str:
            seen.append(packages)
            return "h" * 64

        benchmark_code_hash("parallelism.scripts.shuffle_order_control", "lexical", fake_hasher)
        benchmark_code_hash("parallelism.scripts.compare_models", "lexical", fake_hasher)
        assert seen == [(*BENCHMARK_PACKAGES, *SHUFFLE_PACKAGES, "lexical"), BENCHMARK_PACKAGES]

    def test_a_scoring_cell_depends_on_generators_only_through_its_dataset_bytes(self) -> None:
        """Dataset files are hashed inputs, so generator code stays out of a scoring cell."""
        from library.driver import BENCHMARK_PACKAGES

        assert "core" in BENCHMARK_PACKAGES
        assert "families" not in BENCHMARK_PACKAGES
        assert "syntactic" not in BENCHMARK_PACKAGES

    def test_differs_between_scripts(self) -> None:
        assert benchmark_code_hash("genre.scripts.compare_models") != benchmark_code_hash(
            "parallelism.scripts.compare_models"
        )


class TestMain:
    def test_manifest_subcommand_seals_a_run(self, tmp_path: Path) -> None:
        out = tmp_path / "d/x.csv"
        out.parent.mkdir(parents=True)
        out.write_text("a\n")
        cell = Cell("c", "m", (), (out,), [], None)
        main(
            ["manifest", "--data-root", str(tmp_path / "d")],
            cells_factory=lambda roots: [cell],
            roots_factory=lambda args: None,
            parity=lambda roots, log_root: {"genre.lexical": {"unexplained": []}},
        )
        record = json.loads((tmp_path / "d/analysis=benchmark/_manifest.json").read_text())
        assert record["missing"] == []
        assert record["expected_cells"] == 1
        assert record["parity"] == {"genre.lexical": {"unexplained": []}}

    def test_manifest_fails_when_parity_does(self, tmp_path: Path) -> None:
        """The run manifest is written, then the coverage gap ends the run non-zero."""
        from library.parity import ParityError

        out = tmp_path / "d/x.csv"
        out.parent.mkdir(parents=True)
        out.write_text("a\n")
        cell = Cell("c", "m", (), (out,), [], None)

        def failing(roots, log_root):
            raise ParityError("genre.lexical: unexplained ['z']")

        with pytest.raises(ParityError, match=r"genre\.lexical"):
            main(
                ["manifest", "--data-root", str(tmp_path / "d")],
                cells_factory=lambda roots: [cell],
                roots_factory=lambda args: None,
                parity=failing,
            )
        assert (tmp_path / "d/analysis=benchmark/_manifest.json").exists()

    def test_run_executes_the_named_cell_only(self, tmp_path: Path) -> None:
        out = tmp_path / "d/y.csv"
        calls: list[list[str]] = []

        def fake_main(argv: list[str]) -> None:
            calls.append(argv)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("b\n")

        cell = Cell("b.lexical.wanted", "m", (), (out,), ["--flag"], None)
        other = Cell("b.lexical.other", "m", (), (tmp_path / "d/z.csv",), ["--no"], None)
        main(
            ["run", "--cell", "b.lexical.wanted", "--data-root", str(tmp_path / "d")],
            cells_factory=lambda roots: [cell, other],
            roots_factory=lambda args: None,
            module_main=lambda module: fake_main,
            parity=lambda roots, log_root: {},
        )
        assert calls == [["--flag"]]
        assert (out.parent / "_manifest.json").exists()


class TestCellProvenance:
    def test_carries_the_cross_repository_code_hash(self, tmp_path: Path) -> None:
        from library.driver import cell_provenance

        cell = Cell(
            "parallelism.lexical.retrieval", "parallelism.scripts.compare_models", (), (), [], None
        )
        expected = benchmark_code_hash(cell.module, "lexical")
        assert json.loads(cell_provenance(cell))["code"] == expected
