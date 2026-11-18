"""Class I windows, and MHCflurry's columns renamed onto the ones the screen reads.

``liability.py`` will only take a concentration from a column that says
nanomolar. The conversion here is the join between a predictor's names and
that contract, so the tests check the join itself rather than any prediction:
no model is installed or called.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfproteina.disease.liability import measure_sequence_liability
from rfproteina.disease.mhc_tables import (
    OUTPUT_COLUMNS,
    PredictorMissingError,
    convert_mhcflurry_csv,
    run_mhcflurry,
    windows,
    write_mhc_i_table,
)

_CSV = (
    "allele,peptide,mhcflurry_affinity,mhcflurry_affinity_percentile\n"
    "HLA-A*02:01,SIINFEKLA,42.5,0.08\n"
    "HLA-A*02:01,AAAAAAAAA,28000.0,62.1\n"
)


def test_windows_are_cut_at_every_width_and_numbered_from_one() -> None:
    found = windows("MKTIIALSYIF", [9])
    assert found[0] == (1, "MKTIIALSY")
    assert found[-1] == (3, "TIIALSYIF")
    assert len(windows("MKTIIALSYIF")) == len(windows("MKTIIALSYIF", [8, 9, 10, 11]))


def test_a_window_holding_a_non_standard_residue_is_dropped() -> None:
    assert all("X" not in peptide for _, peptide in windows("MKTXIALSYIF", [9]))


def test_the_converted_table_carries_a_nanomolar_unit_beside_the_affinity() -> None:
    table = convert_mhcflurry_csv(_CSV)
    header = table.splitlines()[0].split("\t")
    assert header == list(OUTPUT_COLUMNS)
    first = dict(zip(header, table.splitlines()[1].split("\t"), strict=True))
    assert first["ic50_nm"] == "42.5"
    assert first["unit"] == "nM"
    assert first["percentile_rank"] == "0.08"
    assert first["method"] == "mhcflurry"


def test_a_csv_without_a_peptide_or_allele_is_refused() -> None:
    with pytest.raises(ValueError, match="missing_peptide_or_allele"):
        convert_mhcflurry_csv("score\n0.5\n")
    with pytest.raises(ValueError, match="no_rows"):
        convert_mhcflurry_csv("allele,peptide\n")


def test_the_written_table_is_read_by_the_class_one_axis(tmp_path: Path) -> None:
    """The point of the rename: the axis reads the file instead of refusing it."""
    run = tmp_path / "protein_1ABC_1"
    run.mkdir()
    samples = [("binder_0", "SIINFEKLAAAAAAAAAA")]
    without, _ = measure_sequence_liability(samples, directory=run, binder_kind="peptide")
    path = write_mhc_i_table(_CSV, run, starts={"SIINFEKLA": 1})
    assert path.name == "mhc_i.tsv"
    with_table, _ = measure_sequence_liability(samples, directory=run, binder_kind="peptide")
    assert with_table["mhc_i"] != without["mhc_i"]
    text = repr(with_table["mhc_i"])
    assert "unparsed" not in text
    assert "unreadable" not in text


def test_the_runner_says_the_predictor_is_missing_rather_than_inventing_scores(monkeypatch) -> None:
    monkeypatch.setattr("rfproteina.disease.mhc_tables.predictor_command", lambda: None)
    with pytest.raises(PredictorMissingError):
        run_mhcflurry([(1, "SIINFEKLA")], ["HLA-A*02:01"])


def test_allele_comments_and_repeats_are_dropped() -> None:
    from rfproteina.disease.mhc_tables import parse_alleles

    parsed = parse_alleles("# panel\nHLA-A*02:01, HLA-A*02:01\nHLA-B*07:02\n")
    assert parsed == ["HLA-A*02:01", "HLA-B*07:02"]


def test_an_allele_token_that_is_not_a_name_is_refused() -> None:
    from rfproteina.disease.mhc_tables import parse_alleles

    with pytest.raises(ValueError, match="allele_unreadable"):
        parse_alleles("HLA-A*02:01 not an allele\n")


def test_the_program_panel_is_used_when_no_allele_file_is_set(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease.mhc_tables import CLASS_I_ALLELES, configured_alleles

    monkeypatch.delenv("MHC_I_ALLELES", raising=False)
    monkeypatch.setattr("rfproteina.disease.mhc_tables.install_root", lambda: tmp_path / "absent")
    alleles, source = configured_alleles(tmp_path)
    assert source == "program_panel"
    assert alleles == list(CLASS_I_ALLELES)
    assert "HLA-B*08:01" not in alleles


def test_a_missing_predictor_writes_nothing_even_with_the_program_panel(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease.mhc_tables import prepare_class_i_table

    monkeypatch.delenv("MHC_I_ALLELES", raising=False)
    monkeypatch.setattr("rfproteina.disease.mhc_tables.install_root", lambda: tmp_path / "absent")
    monkeypatch.setattr("rfproteina.disease.mhc_tables.predictor_command", lambda: None)
    called = {"run": False}

    def explode(*_args, **_kwargs):
        called["run"] = True
        raise AssertionError("predictor ran")

    monkeypatch.setattr("rfproteina.disease.mhc_tables.run_mhcflurry", explode)
    result = prepare_class_i_table(tmp_path, [("binder_0", "SIINFEKLAA")], binder_kind="peptide")
    assert result["state"] == "predictor_missing"
    assert result["allele_source"] == "program_panel"
    assert called["run"] is False
    assert not (tmp_path / "mhc_i.tsv").exists()


def test_an_existing_table_is_not_replaced(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease.mhc_tables import prepare_class_i_table

    (tmp_path / "mhc_i.tsv").write_text("kept\n", encoding="utf-8")
    monkeypatch.setenv("MHC_I_ALLELES", "HLA-A*02:01")
    monkeypatch.setattr("rfproteina.disease.mhc_tables.run_mhcflurry", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("ran")))
    result = prepare_class_i_table(tmp_path, [("binder_0", "SIINFEKLAA")], binder_kind="peptide")
    assert result["state"] == "table_present"
    assert (tmp_path / "mhc_i.tsv").read_text(encoding="utf-8") == "kept\n"


def test_a_configured_scan_writes_the_table_the_axis_reads(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease.mhc_tables import prepare_class_i_table

    monkeypatch.setenv("MHC_I_ALLELES", "HLA-A*02:01")
    monkeypatch.setattr("rfproteina.disease.mhc_tables.predictor_command", lambda: "mhcflurry-predict")
    monkeypatch.setattr("rfproteina.disease.mhc_tables.weights_present", lambda *_a, **_k: True)
    monkeypatch.setattr("rfproteina.disease.mhc_tables.reference_sequences", lambda: ("absent", {}))

    def fake_run(peptides, alleles, *, timeout_s=1800):
        assert alleles == ["HLA-A*02:01"]
        assert {peptide for _start, peptide in peptides} >= {"SIINFEKLA"}
        return _CSV, {peptide: start for start, peptide in peptides}

    monkeypatch.setattr("rfproteina.disease.mhc_tables.run_mhcflurry", fake_run)
    result = prepare_class_i_table(tmp_path, [("binder_0", "SIINFEKLAAA")], binder_kind="peptide")
    assert result["state"] == "written"
    text = (tmp_path / "mhc_i.tsv").read_text(encoding="utf-8")
    assert "ic50_nm" in text.splitlines()[0]
    assert "nM" in text


def test_the_predict_subprocess_is_pointed_at_the_models_directory(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease import mhc_tables

    monkeypatch.setattr(mhc_tables, "predictor_command", lambda: "mhcflurry-predict")
    monkeypatch.setattr(mhc_tables, "models_directory", lambda: tmp_path / "models")
    captured: dict[str, object] = {}

    class Finished:
        returncode = 0
        stdout = _CSV
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return Finished()

    monkeypatch.setattr(mhc_tables.subprocess, "run", fake_run)
    stdout, _starts = mhc_tables.run_mhcflurry([(1, "SIINFEKLA")], ["HLA-A*02:01"])
    assert stdout == _CSV
    assert captured["env"][mhc_tables.MHCFLURRY_DOWNLOADS_ENV] == str(tmp_path / "models")


def test_weights_are_present_only_when_the_presentation_bundle_is_a_directory(tmp_path: Path) -> None:
    from rfproteina.disease.mhc_tables import MODELS_BUNDLE, weights_present

    assert weights_present(tmp_path) is False
    bundle = tmp_path / "release" / MODELS_BUNDLE
    bundle.mkdir(parents=True)
    assert weights_present(tmp_path) is True


def test_fetch_asks_for_the_pinned_presentation_release(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease import mhc_tables

    downloads = tmp_path / "mhcflurry-downloads"
    downloads.write_text("", encoding="utf-8")
    monkeypatch.setattr(mhc_tables, "predictor_command", lambda: str(tmp_path / "mhcflurry-predict"))
    monkeypatch.setattr(mhc_tables, "_sibling_downloads", lambda _predict: str(downloads))
    monkeypatch.setattr(mhc_tables, "models_directory", lambda: tmp_path / "models")
    captured: dict[str, object] = {}

    class Finished:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        bundle = (tmp_path / "models" / "2.3.0" / mhc_tables.MODELS_BUNDLE)
        bundle.mkdir(parents=True)
        return Finished()

    monkeypatch.setattr(mhc_tables.subprocess, "run", fake_run)
    result = mhc_tables.fetch_weights()
    assert captured["command"] == [
        str(downloads),
        "fetch",
        mhc_tables.MODELS_BUNDLE,
        "--release",
        mhc_tables.MODELS_RELEASE,
    ]
    assert captured["env"][mhc_tables.MHCFLURRY_DOWNLOADS_ENV] == str(tmp_path / "models")
    assert result["weights_present"] is True


def test_a_partial_reference_set_is_not_a_bar(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease import mhc_tables

    monkeypatch.setattr(mhc_tables, "install_root", lambda: tmp_path)
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "P02768.fasta").write_text(">albumin\nMKTI\n", encoding="utf-8")
    status, found = mhc_tables.reference_sequences()
    assert status == "incomplete"
    assert list(found) == ["P02768"]
