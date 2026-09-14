from pathlib import Path

import pytest

from library.stages import (
    BENCHMARK_ROOT,
    Roots,
    discover_domains,
    plan_cells,
    shuffle_families_for,
    stage_dir,
)


@pytest.fixture
def roots(tmp_path: Path) -> Roots:
    """Every root the stages resolve against, with two discoverable domains in the tree."""
    embeddings = tmp_path / "emb"
    for domain, unit in (
        ("lexical", "unit=lexeme/construction=icf"),
        ("syntactic", "level=phrase/feature=typ/construction=1gram"),
    ):
        part = embeddings / f"domain={domain}" / unit / "part-0.parquet"
        part.parent.mkdir(parents=True)
        part.write_bytes(b"x")
    genre = tmp_path / "genre.csv"
    genre.write_text("Psalm,Genre\n")
    return Roots(
        data_root=tmp_path / "data",
        embeddings_root=embeddings,
        config_root=tmp_path / "config",
        genre_csv=genre,
        ui_root=tmp_path / "ui",
        workers=2,
    )


class TestDiscoverDomains:
    def test_lists_domain_partitions_present_in_the_embeddings_tree(self, roots: Roots) -> None:
        assert discover_domains(roots.embeddings_root) == ("lexical", "syntactic")

    def test_ignores_files_and_non_domain_directories(self, roots: Roots) -> None:
        (roots.embeddings_root / "_manifest.json").write_text("{}")
        (roots.embeddings_root / "notes").mkdir()
        assert discover_domains(roots.embeddings_root) == ("lexical", "syntactic")


class TestStageDir:
    def test_places_a_stage_under_the_benchmark_and_domain_partition(self, roots: Roots) -> None:
        assert stage_dir(roots, "genre", "lexical", "raw") == (
            roots.data_root / BENCHMARK_ROOT / "benchmark=genre/domain=lexical/stage=raw"
        )


class TestShuffleFamilies:
    def test_returns_only_the_registered_families_of_one_domain(self) -> None:
        """Families are keyed `domain/...`, so a domain's controls are the keys under its prefix."""
        keys = shuffle_families_for("morphological")
        assert keys
        assert all(k.startswith("morphological/") for k in keys)
        assert keys == tuple(sorted(keys))


class TestPlanCells:
    def test_every_cell_has_outputs_under_a_declared_root(self, roots: Roots) -> None:
        """Outputs land in tehillim-data or the front-end tree, nowhere else."""
        allowed = (roots.data_root, roots.ui_root)
        for cell in plan_cells(roots):
            assert cell.outputs
            for output in cell.outputs:
                assert any(output.is_relative_to(root) for root in allowed), (cell.name, output)

    def test_no_two_cells_write_one_file(self, roots: Roots) -> None:
        owners: dict[Path, str] = {}
        for cell in plan_cells(roots):
            for output in cell.outputs:
                assert output not in owners, (output, owners.get(output), cell.name)
                owners[output] = cell.name

    def test_stage_inputs_are_outputs_of_earlier_cells_or_external_files(
        self, roots: Roots
    ) -> None:
        """A cell may read only what another cell writes or what the roots supply."""
        cells = plan_cells(roots)
        produced = {output for cell in cells for output in cell.outputs}
        external = {roots.genre_csv}
        for cell in cells:
            for path in cell.inputs:
                inside_tree = path.is_relative_to(roots.embeddings_root) or path.is_relative_to(
                    roots.config_root
                )
                assert path in produced or path in external or inside_tree, (cell.name, path)

    def test_each_domain_gets_the_full_chain_for_every_benchmark(self, roots: Roots) -> None:
        names = {cell.name for cell in plan_cells(roots)}
        for domain in ("lexical", "syntactic"):
            assert f"parallelism.{domain}.master" in names
            assert f"genre.{domain}.master" in names
            assert f"trajectory.{domain}.ui_rows" in names
            assert f"ui.{domain}.payload" in names
            assert f"ui.{domain}.detail" in names

    def test_the_parallelism_master_reads_the_three_raw_and_detail_outputs(
        self, roots: Roots
    ) -> None:
        cells = {cell.name: cell for cell in plan_cells(roots)}
        master = cells["parallelism.lexical.master"]
        raw = stage_dir(roots, "parallelism", "lexical", "raw")
        detail = stage_dir(roots, "parallelism", "lexical", "detail")
        assert raw / "retrieval.csv" in master.inputs
        assert raw / "calibration.csv" in master.inputs
        assert detail / "pair_detail.parquet" in master.inputs
        assert master.command_args[:2] == ["--retrieval-csv", str(raw / "retrieval.csv")]

    def test_the_genre_master_reads_the_calibrated_scores(self, roots: Roots) -> None:
        """The master melts effect-size columns that only compare_calibrated writes."""
        cells = {cell.name: cell for cell in plan_cells(roots)}
        master = cells["genre.lexical.master"]
        raw = stage_dir(roots, "genre", "lexical", "raw")
        assert raw / "calibrated.csv" in master.inputs
        assert raw / "summary.csv" not in master.inputs
        assert master.command_args[:2] == ["--summary-csv", str(raw / "calibrated.csv")]

    def test_scoring_cells_read_every_dataset_of_their_domain(self, roots: Roots) -> None:
        """A changed dataset anywhere in the domain makes the domain's scoring stale."""
        cells = {cell.name: cell for cell in plan_cells(roots)}
        retrieval = cells["parallelism.lexical.retrieval"]
        assert (
            roots.embeddings_root / "domain=lexical/unit=lexeme/construction=icf/part-0.parquet"
            in retrieval.inputs
        )
        assert retrieval.command_args[0] == str(roots.embeddings_root / "domain=lexical")

    def test_shuffle_cells_exist_per_registered_family_of_the_domain(self, roots: Roots) -> None:
        cells = plan_cells(roots)
        keys = shuffle_families_for("lexical")
        shuffle = [c for c in cells if c.name.startswith("parallelism.lexical.shuffle.")]
        assert len(shuffle) == len(keys)
        assert all(c.module == "parallelism.scripts.shuffle_order_control" for c in shuffle)

    def test_ui_payload_declares_the_metric_slices(self, roots: Roots) -> None:
        from trajectory.scripts.validate_against_genre import METRICS

        cells = {cell.name: cell for cell in plan_cells(roots)}
        outputs = {p.name for p in cells["ui.lexical.payload"].outputs}
        assert "ui_lexical.json" in outputs
        assert {f"ui_lexical_trajectory_{m}.json" for m in METRICS} <= outputs

    def test_ui_detail_output_is_the_domain_index(self, roots: Roots) -> None:
        cells = {cell.name: cell for cell in plan_cells(roots)}
        assert cells["ui.lexical.detail"].outputs == (
            roots.ui_root / "detail-data" / "detail_lexical_index.json",
        )


@pytest.mark.integration
def test_the_declared_outputs_account_for_every_file_in_the_published_tree() -> None:
    """Check 1 against reality: nothing in tehillim-data's benchmark tree lacks a producing cell."""
    import os

    data_root = Path(os.environ.get("TEHILLIM_DATA_DIR", "../tehillim-data")).resolve()
    embeddings_root = Path(
        os.environ.get("TEHILLIM_EMBEDDINGS_DATA", "../tehillim-embeddings/data")
    ).resolve()
    if not (data_root / BENCHMARK_ROOT).exists() or not embeddings_root.exists():
        pytest.skip("published trees not available")
    roots = Roots(
        data_root=data_root,
        embeddings_root=embeddings_root,
        config_root=embeddings_root.parent / "config",
        genre_csv=Path("genre.csv"),
        ui_root=Path("ui"),
        workers=1,
    )
    declared = {
        o for cell in plan_cells(roots) for o in cell.outputs if o.is_relative_to(data_root)
    }
    on_disk = {
        p
        for p in (data_root / BENCHMARK_ROOT).rglob("*")
        if p.is_file() and p.name not in (".DS_Store", "_manifest.json")
    }
    assert on_disk <= declared, sorted(str(p) for p in on_disk - declared)[:10]
