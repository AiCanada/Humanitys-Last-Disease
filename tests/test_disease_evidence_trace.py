"""What the evidence chain is allowed to claim about its own run.

Four claims a research trace made that its run had not earned, and one input
it could not identify.

The chain reported the AlphaFold Database as ``consulted`` because the
structure was *named* ``AF-...`` or carried a saved ``alphafold`` label -- on a
run whose structure list was read from a local file with no service called at
all, and on a structure the user had uploaded themselves.

``generative_model.status`` read ``recorded`` for every run ever written: the
test behind it compared ``code``, a dict present on every record, against a
string, so a command naming no experiment, no hub revision, no checkpoint and
no autoencoder still announced that the model was identified.

``experimental_validation.status`` -- the only place a trace can say a design
reached a bench -- could be replaced by any other key in the same file that
normalised to ``status``, and ten fields of a longer bench record were dropped
with nothing saying the record had been cut.

A trace that could not be read was skipped in silence while the caller printed
how many had been refreshed, and the condition spec that ``contig`` and
``hotspot`` are read out of carried no digest, so nothing said whether the file
beside the run is still the file the design was asked for.
"""

import hashlib
import json
from pathlib import Path

import pytest

from rfproteina.disease.structures import StructureOption
from rfproteina.disease.trace import (
    NOT_PERFORMED,
    NOT_RECORDED,
    restamp_screening,
    search_trace,
    write_research_traces,
)
from tests.helpers.disease import structure_option

_CIF = "data_binder\n#\n"


def _run(root: Path, *files: tuple[str, str]) -> Path:
    """A finished run holding one generated binder, plus any supplied files."""
    output = root / "protein_1ABC_1"
    folder = output / "inference_samples" / "binder_0"
    folder.mkdir(parents=True)
    (folder / "binder_0_generated.cif").write_text(_CIF, encoding="utf-8")
    for name, text in files:
        (output / name).write_text(text, encoding="utf-8")
    return output


def _trace(
    output: Path,
    option: StructureOption,
    *,
    search: object = None,
    argv: list[str] | None = None,
) -> dict:
    written = write_research_traces(
        output,
        disease_name="Asthma",
        disease_id="EFO_0000270",
        option=option,
        binder_kind="protein",
        argv=argv if argv is not None else ["rfd4-generate"],
        screen=None,
        search=search,  # type: ignore[arg-type]
    )
    assert [path.name for path in written] == ["binder_0.json"]
    return json.loads(written[0].read_text(encoding="utf-8"))


def _alphafold_row(record: dict) -> dict:
    rows = [row for row in record["source_databases"]["structure_search"] if row["name"] == "AlphaFold Database"]
    assert len(rows) == 1, rows
    return rows[0]


class TestAlphaFoldConsult:
    def test_an_alphafold_name_on_an_upload_is_not_a_consult(self, tmp_path: Path) -> None:
        """The user supplied the file. Nothing asked the database anything."""
        output = _run(tmp_path)
        record = _trace(
            output,
            structure_option(entry_id="AF-P35225-F1", chain="A", predicted=True, source="upload"),
            search=search_trace(live_structures=False, live_disease=False, focus=None, collected={}),
        )
        assert _alphafold_row(record)["status"] == NOT_PERFORMED

    def test_the_row_follows_what_the_run_called_and_not_the_structure_label(self, tmp_path: Path) -> None:
        """A live search that fetched the prediction record still says so.

        Positive evidence needs no extra coverage, so the live case keeps its
        ``consulted``. A structure list read from a file performed no call, and
        a saved list carries the same ``alphafold`` label into a run that
        touched no service.
        """
        output = _run(tmp_path)
        live = _trace(
            output,
            structure_option(entry_id="AF-P35225-F1", chain="A", predicted=True, source="alphafold"),
            search=search_trace(live_structures=True, live_disease=True, focus=None, collected={}),
        )
        assert _alphafold_row(live)["status"] == "consulted"
        local = _trace(
            output,
            structure_option(entry_id="AF-P35225-F1", chain="A", predicted=True, source="alphafold"),
            search=search_trace(live_structures=False, live_disease=False, focus=None, collected={}),
        )
        assert _alphafold_row(local)["status"] == NOT_PERFORMED

    def test_a_run_that_recorded_no_search_cannot_say_either_way(self, tmp_path: Path) -> None:
        output = _run(tmp_path)
        record = _trace(
            output,
            structure_option(entry_id="AF-P35225-F1", chain="A", predicted=True, source="alphafold"),
            search=None,
        )
        assert _alphafold_row(record)["status"] == NOT_RECORDED


class TestModelIdentity:
    def test_a_command_that_names_no_model_is_not_recorded(self, tmp_path: Path) -> None:
        """``code`` is this checkout's commit. It is not the model that ran."""
        output = _run(tmp_path)
        record = _trace(output, structure_option(), argv=["rfd4-generate", f"out_dir={output}"])
        model = record["generative_model"]
        assert model["status"] == NOT_RECORDED
        assert {key: value for key, value in model["command"].items() if key != "code"} == dict.fromkeys(
            (
                "experiment",
                "hub_repository",
                "hub_revision",
                "checkpoint_name",
                "checkpoint_path",
                "autoencoder_checkpoint_name",
            ),
            NOT_RECORDED,
        )

    def test_a_command_that_names_a_checkpoint_is_recorded(self, tmp_path: Path) -> None:
        """The fix must not start refusing a model the command did name."""
        output = _run(tmp_path)
        record = _trace(
            output,
            structure_option(),
            argv=["rfd4-generate", "ckpt_name=disease.ckpt", f"out_dir={output}"],
        )
        assert record["generative_model"]["status"] == "recorded"
        assert record["generative_model"]["command"]["checkpoint_name"] == "disease.ckpt"


class TestBenchRecord:
    def test_another_key_cannot_replace_the_bench_status(self, tmp_path: Path) -> None:
        """`` status`` normalises to ``status``, and used to overwrite it."""
        bench = json.dumps({"status": "not_performed", " status": "assayed and confirmed", "assay": "SPR"})
        output = _run(tmp_path, ("experimental_validation.json", bench))
        validation = _trace(output, structure_option())["experimental_validation"]
        assert validation["status"] == "not_performed"
        assert validation["assay"] == "SPR"
        assert validation["fields_not_stored"] == 1

    def test_a_bench_record_cut_at_the_limit_says_it_was_cut(self, tmp_path: Path) -> None:
        fields = {f"field_{index:02d}": f"value {index}" for index in range(20)}
        bench = json.dumps({"status": "validated", **fields})
        output = _run(tmp_path, ("experimental_validation.json", bench))
        validation = _trace(output, structure_option())["experimental_validation"]
        assert validation["status"] == "validated"
        stored = [key for key in validation if key.startswith("field_")]
        assert len(stored) == 10
        assert validation["fields_not_stored"] == 10

    def test_a_record_that_fits_is_not_marked_as_cut(self, tmp_path: Path) -> None:
        bench = json.dumps({"status": "validated", "assay": "SPR", "kd_nm": 3.2})
        output = _run(tmp_path, ("experimental_validation.json", bench))
        validation = _trace(output, structure_option())["experimental_validation"]
        assert validation["assay"] == "SPR"
        assert "fields_not_stored" not in validation


class TestRestamp:
    def test_a_trace_that_cannot_be_read_is_named(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """The caller prints how many were refreshed. The rest has to be said."""
        output = _run(tmp_path)
        _trace(output, structure_option())
        broken = output / "research_traces" / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        refreshed = restamp_screening(output, {"call": "untested", "samples": []})
        assert [path.name for path in refreshed] == ["binder_0.json"]
        printed = capsys.readouterr().out
        assert "broken.json" in printed
        assert "not refreshed" in printed


class TestConditionSpec:
    def test_the_spec_the_contig_came_from_is_digested(self, tmp_path: Path) -> None:
        output = _run(tmp_path)
        spec = output.parent / f"{output.name}.yaml"
        spec.write_bytes(b'binder:\r\n  contig: "60-110, /0, A1-9"\r\n')
        design = _trace(output, structure_option())["design_parameters"]
        assert design["contig"] == "60-110, /0, A1-9"
        assert design["spec_digest"] == hashlib.sha256(spec.read_bytes()).hexdigest()[:16]

    def test_a_run_without_a_spec_records_no_digest(self, tmp_path: Path) -> None:
        output = _run(tmp_path)
        design = _trace(output, structure_option())["design_parameters"]
        assert design["contig"] == NOT_RECORDED
        assert design["spec_digest"] == NOT_RECORDED
