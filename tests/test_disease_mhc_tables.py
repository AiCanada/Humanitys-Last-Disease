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
    monkeypatch.setattr("rfproteina.disease.mhc_tables.predictor_available", lambda: False)
    with pytest.raises(PredictorMissingError):
        run_mhcflurry([(1, "SIINFEKLA")], ["HLA-A*02:01"])
