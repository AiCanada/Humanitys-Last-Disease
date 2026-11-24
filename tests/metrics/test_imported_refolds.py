"""Importing a structure folded somewhere else.

Not everyone can run a refolding oracle: AF3 weights are licence-gated, a
container runtime may be missing, and some services hand a file to a browser
rather than to a program. In all of those the fold exists -- it is just on disk.

The contract is two files at two canonical paths. Every oracle calls a design
refolded once its confidence JSON is there, and reads the pair back through
`_load_compute_result`, so a placed fold flows into the refolding metrics and
the declared pass criteria with nothing downstream aware it came from outside.

These tests use a stub oracle rather than AF3, because the point is the
*contract*, and a test that needed licensed weights could never run.
"""

import json
from pathlib import Path

import pytest

from rfproteina.metrics.oracles.imported_refolds import (
    IMPORT_MARK,
    NO_CONFIDENCE,
    PathOracle,
    import_refolds,
    place_refold,
)
from rfproteina.utils.common import design_output_dir
from tests.helpers.disease import REFOLDED_CIF as _CIF


class _StubOracle:
    """The paths and the cache predicate, exactly as OracleMetric defines them.

    Subclassing the real OracleMetric would drag in torch and an abstract fold
    implementation for no gain: this module only ever touches these four
    members, so they are what a stub has to be faithful to.
    """

    def __init__(self, oracle_name: str = "of3", suffix: str = ".cif") -> None:
        self.oracle_name = oracle_name
        self.output_cif_suffix = suffix

    def output_stem(self, example_id: str, save_id: str | None = None) -> str:
        return (
            f"{example_id}_{save_id}_{self.oracle_name}" if save_id is not None else f"{example_id}_{self.oracle_name}"
        )

    def confidence_path(self, out_dir: Path, example_id: str, save_id: str | None = None) -> Path:
        return design_output_dir(out_dir, example_id) / f"{self.output_stem(example_id, save_id)}_confidences.json"

    def refolded_cif_path(self, out_dir: Path, example_id: str, save_id: str | None = None) -> Path:
        return (
            design_output_dir(out_dir, example_id)
            / f"{self.output_stem(example_id, save_id)}_refolded{self.output_cif_suffix}"
        )

    def already_refolded(self, out_dir: Path, example_id: str, save_id: str | None = None) -> bool:
        return self.confidence_path(out_dir, example_id, save_id).exists()


@pytest.fixture
def oracle() -> _StubOracle:
    return _StubOracle()


def _fold(tmp_path: Path, name: str = "binder_0.cif", text: str = _CIF) -> Path:
    source = tmp_path / "elsewhere"
    source.mkdir(exist_ok=True)
    path = source / name
    path.write_text(text, encoding="utf-8")
    return path


class TestPlaceOne:
    def test_a_placed_fold_makes_the_oracle_call_the_design_refolded(self, tmp_path: Path, oracle) -> None:
        """This is the whole contract. Everything else follows from it."""
        out = tmp_path / "run"
        assert not oracle.already_refolded(out, "binder_0")
        found = place_refold(oracle, out, "binder_0", _fold(tmp_path))
        assert found.placed, found.blocker
        assert oracle.already_refolded(out, "binder_0")

    def test_the_structure_lands_where_the_oracle_looks(self, tmp_path: Path, oracle) -> None:
        out = tmp_path / "run"
        found = place_refold(oracle, out, "binder_0", _fold(tmp_path))
        assert found.structure == out / "binder_0" / "binder_0_of3_refolded.cif"
        assert found.structure.read_text(encoding="utf-8") == _CIF

    def test_the_caller_s_file_is_copied_not_moved(self, tmp_path: Path, oracle) -> None:
        """Their file stays theirs, and editing it later cannot change the run."""
        source = _fold(tmp_path)
        place_refold(oracle, tmp_path / "run", "binder_0", source)
        assert source.is_file()

    def test_an_extension_the_oracle_does_not_write_is_still_found(self, tmp_path: Path) -> None:
        """Lookup is by stem, so re-encoding the file would be work for nothing."""
        oracle = _StubOracle(suffix=".cif.zst")
        found = place_refold(oracle, tmp_path / "run", "binder_0", _fold(tmp_path))
        assert found.placed
        assert found.structure.name == "binder_0_of3_refolded.cif"

    def test_the_oracle_name_is_honoured(self, tmp_path: Path) -> None:
        found = place_refold(_StubOracle(oracle_name="af2"), tmp_path / "run", "binder_0", _fold(tmp_path))
        assert found.structure.name == "binder_0_af2_refolded.cif"

    def test_a_save_id_fans_out_under_its_own_name(self, tmp_path: Path, oracle) -> None:
        found = place_refold(oracle, tmp_path / "run", "binder_0", _fold(tmp_path), save_id="apo")
        assert found.structure.name == "binder_0_apo_of3_refolded.cif"
        assert oracle.already_refolded(tmp_path / "run", "binder_0", "apo")


class TestConfidence:
    def test_no_supplied_confidence_is_recorded_as_such(self, tmp_path: Path, oracle) -> None:
        """The confidence metrics are then absent, which a pass criterion reports
        as a bound it could not check rather than one that failed."""
        found = place_refold(oracle, tmp_path / "run", "binder_0", _fold(tmp_path))
        record = json.loads(found.confidence.read_text(encoding="utf-8"))
        assert record[IMPORT_MARK] is True
        assert record["confidence"] == NO_CONFIDENCE
        assert record["oracle_name"] == "of3"

    def test_supplied_confidence_values_are_kept(self, tmp_path: Path, oracle) -> None:
        source = _fold(tmp_path)
        (source.parent / "binder_0_confidences.json").write_text(
            json.dumps({"ptm_coi": 86.2, "min_ipae_coi": 1.1}), encoding="utf-8"
        )
        report = import_refolds(oracle, tmp_path / "run", source.parent, example_ids=["binder_0"])
        assert len(report.placed) == 1
        record = json.loads(report.placed[0].confidence.read_text(encoding="utf-8"))
        assert record["ptm_coi"] == 86.2
        assert record["min_ipae_coi"] == 1.1

    def test_the_import_mark_survives_supplied_values(self, tmp_path: Path, oracle) -> None:
        """A reader can tell the numbers came from outside even when they are real."""
        confidence = tmp_path / "c.json"
        confidence.write_text(json.dumps({"ptm_coi": 90.0, IMPORT_MARK: False}), encoding="utf-8")
        found = place_refold(oracle, tmp_path / "run", "binder_0", _fold(tmp_path), confidence=confidence)
        record = json.loads(found.confidence.read_text(encoding="utf-8"))
        assert record[IMPORT_MARK] is True, "the mark is not something a supplied file can clear"
        assert record["ptm_coi"] == 90.0

    @pytest.mark.parametrize(
        ("label", "text", "blocker"),
        [
            ("not json", "{not json", "confidence_unreadable"),
            ("a list", "[1, 2]", "confidence_not_an_object"),
            ("a bare number", "7", "confidence_not_an_object"),
        ],
    )
    def test_a_confidence_file_that_cannot_be_trusted_is_refused(
        self, tmp_path: Path, oracle, label: str, text: str, blocker: str
    ) -> None:
        confidence = tmp_path / "c.json"
        confidence.write_text(text, encoding="utf-8")
        found = place_refold(oracle, tmp_path / "run", "binder_0", _fold(tmp_path), confidence=confidence)
        assert found.blocker == blocker


class TestRefusalsWriteNothing:
    """A half-placed refold is worse than none: the confidence JSON is what
    marks a design done, so one beside an unreadable structure would make the
    design look refolded and measure nothing."""

    @pytest.mark.parametrize(
        ("label", "maker", "blocker"),
        [
            ("a missing file", lambda d: d / "absent.cif", "structure_not_found"),
            ("something not a structure", lambda d: _write(d / "notes.txt", "hello"), "not_a_structure_file"),
            ("an empty structure", lambda d: _write(d / "empty.cif", ""), "structure_empty"),
        ],
    )
    def test_a_bad_structure_is_refused(self, tmp_path: Path, oracle, label, maker, blocker) -> None:
        found = place_refold(oracle, tmp_path / "run", "binder_0", maker(tmp_path))
        assert found.blocker == blocker
        assert not found.placed

    def test_nothing_is_written_when_it_is_refused(self, tmp_path: Path, oracle) -> None:
        out = tmp_path / "run"
        place_refold(oracle, out, "binder_0", tmp_path / "absent.cif")
        assert not oracle.already_refolded(out, "binder_0")
        assert not (out / "binder_0").exists()

    def test_an_id_that_would_escape_the_run_is_refused(self, tmp_path: Path, oracle) -> None:
        found = place_refold(oracle, tmp_path / "run", "../elsewhere", _fold(tmp_path))
        assert found.blocker == "example_id_not_a_directory_name"


class TestImportDirectory:
    def test_each_design_is_matched_by_its_stem(self, tmp_path: Path, oracle) -> None:
        source = tmp_path / "elsewhere"
        source.mkdir()
        for name in ("binder_0", "binder_1"):
            (source / f"{name}.cif").write_text(_CIF, encoding="utf-8")
        report = import_refolds(oracle, tmp_path / "run", source, example_ids=["binder_0", "binder_1"])
        assert [item.example_id for item in report.placed] == ["binder_0", "binder_1"]
        assert report.refused == []
        assert report.unmatched == []

    def test_a_decorated_filename_still_finds_its_design(self, tmp_path: Path, oracle) -> None:
        """External tools add their own suffixes: _model_1, _rank0."""
        source = tmp_path / "elsewhere"
        source.mkdir()
        (source / "binder_0_model_1.cif").write_text(_CIF, encoding="utf-8")
        report = import_refolds(oracle, tmp_path / "run", source, example_ids=["binder_0"])
        assert [item.example_id for item in report.placed] == ["binder_0"]

    def test_binder_1_is_never_taken_for_binder_10(self, tmp_path: Path, oracle) -> None:
        """The longest matching id wins, so a prefix cannot steal a sibling's fold."""
        source = tmp_path / "elsewhere"
        source.mkdir()
        (source / "binder_10.cif").write_text(_CIF, encoding="utf-8")
        report = import_refolds(oracle, tmp_path / "run", source, example_ids=["binder_1", "binder_10"])
        assert [item.example_id for item in report.placed] == ["binder_10"]

    def test_a_file_matching_nothing_is_named(self, tmp_path: Path, oracle) -> None:
        """An import that silently did nothing is the likeliest failure here."""
        source = tmp_path / "elsewhere"
        source.mkdir()
        (source / "someone_elses_protein.cif").write_text(_CIF, encoding="utf-8")
        report = import_refolds(oracle, tmp_path / "run", source, example_ids=["binder_0"])
        assert report.placed == []
        assert report.unmatched == ["someone_elses_protein.cif"]

    def test_designs_already_under_the_run_are_the_default_candidates(self, tmp_path: Path, oracle) -> None:
        out = tmp_path / "run"
        (out / "binder_7").mkdir(parents=True)
        source = tmp_path / "elsewhere"
        source.mkdir()
        (source / "binder_7.cif").write_text(_CIF, encoding="utf-8")
        report = import_refolds(oracle, out, source)
        assert [item.example_id for item in report.placed] == ["binder_7"]

    def test_a_missing_source_directory_is_named_not_ignored(self, tmp_path: Path, oracle) -> None:
        report = import_refolds(oracle, tmp_path / "run", tmp_path / "nowhere")
        assert report.placed == []
        assert report.unmatched == [str(tmp_path / "nowhere")]

    def test_the_report_records_every_reason_by_name(self, tmp_path: Path, oracle) -> None:
        source = tmp_path / "elsewhere"
        source.mkdir()
        (source / "binder_0.cif").write_text(_CIF, encoding="utf-8")
        (source / "binder_1.cif").write_text("", encoding="utf-8")
        (source / "stranger.cif").write_text(_CIF, encoding="utf-8")
        report = import_refolds(oracle, tmp_path / "run", source, example_ids=["binder_0", "binder_1"])
        assert report.as_dict() == {
            "imported": 1,
            "refused": [{"example_id": "binder_1", "blocker": "structure_empty"}],
            "unmatched_files": ["stranger.cif"],
        }


class TestAlphaFoldServerDownload:
    """The shape the AlphaFold Server actually returns.

    Its download is an archive that unpacks to a directory per job, holding
    five ranked models named ``fold_<job>_model_<rank>.cif`` beside
    ``fold_<job>_summary_confidences_<rank>.json``. Every one of those three
    facts -- the leading word, the nesting, the five models -- broke an import
    written against a flat directory of files named after the design.
    """

    def _download(self, tmp_path: Path, job: str = "fold_binder_0", models: int = 5) -> Path:
        folder = tmp_path / "downloads" / job
        folder.mkdir(parents=True)
        for rank in range(models):
            (folder / f"{job}_model_{rank}.cif").write_text(
                _CIF.replace("data_refold", f"data_model_{rank}"), encoding="utf-8"
            )
            (folder / f"{job}_summary_confidences_{rank}.json").write_text(
                json.dumps({"ptm_coi": 90.0 - rank * 10}), encoding="utf-8"
            )
        (folder / "terms_of_use.md").write_text("terms", encoding="utf-8")
        return tmp_path / "downloads"

    def test_the_server_s_own_filename_finds_the_design(self, tmp_path: Path, oracle) -> None:
        """`fold_` in front defeats any prefix rule; the id is in the middle."""
        report = import_refolds(oracle, tmp_path / "run", self._download(tmp_path), example_ids=["binder_0"])
        assert [item.example_id for item in report.placed] == ["binder_0"]
        assert report.unmatched == [], "nothing in the download should look unrecognised"

    def test_the_archive_s_subdirectory_is_walked(self, tmp_path: Path, oracle) -> None:
        source = self._download(tmp_path)
        assert not any(path.is_file() for path in source.iterdir()), "the fixture nests, as the real one does"
        report = import_refolds(oracle, tmp_path / "run", source, example_ids=["binder_0"])
        assert len(report.placed) == 1

    def test_the_top_ranked_model_is_the_one_kept(self, tmp_path: Path, oracle) -> None:
        """Rank 0 is the server's best. Importing all five in turn kept rank 4."""
        report = import_refolds(oracle, tmp_path / "run", self._download(tmp_path), example_ids=["binder_0"])
        assert report.placed[0].structure.read_text(encoding="utf-8") == _CIF.replace("data_refold", "data_model_0")

    def test_the_other_four_are_reported_not_dropped(self, tmp_path: Path, oracle) -> None:
        report = import_refolds(oracle, tmp_path / "run", self._download(tmp_path), example_ids=["binder_0"])
        assert sorted(Path(name).name for name in report.superseded) == [
            f"fold_binder_0_model_{rank}.cif" for rank in (1, 2, 3, 4)
        ]
        assert report.as_dict()["superseded_files"], "a record of the import says four were set aside"

    def test_the_server_s_confidence_file_is_found(self, tmp_path: Path, oracle) -> None:
        """Named for the job, not for the structure, so it needs deriving."""
        report = import_refolds(oracle, tmp_path / "run", self._download(tmp_path), example_ids=["binder_0"])
        record = json.loads(report.placed[0].confidence.read_text(encoding="utf-8"))
        assert record["ptm_coi"] == 90.0, "rank 0's numbers, matching rank 0's structure"
        assert record[IMPORT_MARK] is True

    def test_af3_s_own_unnumbered_pair_is_also_found(self, tmp_path: Path, oracle) -> None:
        """A local AF3 run writes `<job>_model.cif` and no rank anywhere."""
        source = tmp_path / "local"
        source.mkdir()
        (source / "binder_0_model.cif").write_text(_CIF, encoding="utf-8")
        (source / "binder_0_summary_confidences.json").write_text(json.dumps({"ptm_coi": 77.5}), encoding="utf-8")
        report = import_refolds(oracle, tmp_path / "run", source, example_ids=["binder_0"])
        record = json.loads(report.placed[0].confidence.read_text(encoding="utf-8"))
        assert record["ptm_coi"] == 77.5

    def test_a_flat_import_can_still_be_asked_for(self, tmp_path: Path, oracle) -> None:
        report = import_refolds(
            oracle, tmp_path / "run", self._download(tmp_path), example_ids=["binder_0"], recursive=False
        )
        assert report.placed == [], "the files are one level down and were not looked for there"


class TestContainmentStaysBounded:
    """Matching anywhere in the name is the permissive part. These are the
    limits that keep it from claiming the wrong design's fold."""

    def test_binder_1_is_still_never_taken_for_binder_10(self, tmp_path: Path, oracle) -> None:
        """`binder_1` is a substring of `binder_10`. The boundary is what rejects it."""
        source = tmp_path / "elsewhere"
        source.mkdir()
        (source / "fold_binder_10_model_0.cif").write_text(_CIF, encoding="utf-8")
        report = import_refolds(oracle, tmp_path / "run", source, example_ids=["binder_1", "binder_10"])
        assert [item.example_id for item in report.placed] == ["binder_10"]

    def test_a_longer_id_wins_over_the_one_inside_it(self, tmp_path: Path, oracle) -> None:
        source = tmp_path / "elsewhere"
        source.mkdir()
        (source / "run_a_binder_0_model_0.cif").write_text(_CIF, encoding="utf-8")
        report = import_refolds(oracle, tmp_path / "run", source, example_ids=["binder_0", "a_binder_0"])
        assert [item.example_id for item in report.placed] == ["a_binder_0"]

    def test_an_id_merely_spelled_inside_a_word_does_not_match(self, tmp_path: Path, oracle) -> None:
        source = tmp_path / "elsewhere"
        source.mkdir()
        (source / "rebinder_0x.cif").write_text(_CIF, encoding="utf-8")
        report = import_refolds(oracle, tmp_path / "run", source, example_ids=["binder_0"])
        assert report.placed == []
        assert report.unmatched == ["rebinder_0x.cif"]

    def test_folds_already_in_the_run_are_not_imported_again(self, tmp_path: Path, oracle) -> None:
        """`source` pointed at the run itself would re-import placed refolds,
        which are named for their design and so match it."""
        out = tmp_path / "run"
        place_refold(oracle, out, "binder_0", _fold(tmp_path))
        report = import_refolds(oracle, out, out, example_ids=["binder_0"])
        assert report.placed == []
        assert report.superseded == []


class TestPathOracleMirrorsTheRealOne:
    """`PathOracle` restates OracleMetric's canonical paths so the CLI can ask
    where a refold goes without importing torch. Restating is the risk, so each
    member is checked against the real class's own unbound method, called with
    a PathOracle as `self`. Those methods read nothing but `oracle_name` and
    `output_cif_suffix`, which is exactly why this works -- and if that stops
    being true, or the layout changes, this fails instead of the shim drifting.
    """

    @pytest.fixture
    def real(self):
        return pytest.importorskip(
            "rfproteina.metrics.oracles.oracle_metric",
            reason="the real oracle needs torch; the shim exists so the CLI does not",
        ).OracleMetric

    @pytest.mark.parametrize("save_id", [None, "apo"])
    def test_the_stem_agrees(self, real, save_id) -> None:
        shim = PathOracle()
        assert shim.output_stem("binder_0", save_id) == real.output_stem(shim, "binder_0", save_id)

    @pytest.mark.parametrize("save_id", [None, "apo"])
    def test_the_confidence_path_agrees(self, tmp_path: Path, real, save_id) -> None:
        shim = PathOracle()
        assert shim.confidence_path(tmp_path, "binder_0", save_id) == real.confidence_path(
            shim, tmp_path, "binder_0", save_id
        )

    @pytest.mark.parametrize("suffix", [".cif", ".cif.zst"])
    def test_the_refolded_path_agrees(self, tmp_path: Path, real, suffix) -> None:
        shim = PathOracle(output_cif_suffix=suffix)
        assert shim.refolded_cif_path(tmp_path, "binder_0") == real.refolded_cif_path(shim, tmp_path, "binder_0")

    def test_the_cache_predicate_agrees(self, tmp_path: Path, real) -> None:
        shim = PathOracle()
        assert shim.already_refolded(tmp_path, "binder_0") == real.already_refolded(shim, tmp_path, "binder_0")
        place_refold(shim, tmp_path, "binder_0", _fold(tmp_path))
        assert shim.already_refolded(tmp_path, "binder_0") == real.already_refolded(shim, tmp_path, "binder_0")
        assert shim.already_refolded(tmp_path, "binder_0"), "and it is now true, not merely equal"

    def test_the_oracle_name_reaches_the_filename(self, tmp_path: Path) -> None:
        """It is configurable because it must match the configured oracle."""
        found = place_refold(PathOracle(oracle_name="of3"), tmp_path, "binder_0", _fold(tmp_path))
        assert found.structure.name == "binder_0_of3_refolded.cif"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
