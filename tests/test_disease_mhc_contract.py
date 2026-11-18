"""The MHCflurry contract: the column rename, the units, and the scale.

The existing tests/test_disease_mhc_tables.py pins the conversion's shape.
This file pins the thing that would fail silently instead of loudly: a
rescaling. The numbers below are from a real MHCflurry run on this machine
(mhcflurry 2.3.29, models_class1_presentation release 20260928, 2026-11-16),
recorded in mhc_tables.OBSERVED_EXAMPLE, so a reviewer can re-run them.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from rfproteina.disease import mhc_tables
from rfproteina.disease.liability import _MHC_I_IC50, _MHC_I_RANK
from rfproteina.disease.mhc_tables import (
    COLUMN_MAP,
    OBSERVED_EXAMPLE,
    REFERENCE_ACCESSIONS,
    PredictorMissingError,
    contract,
    convert_mhcflurry_csv,
    convert_reference_csv,
    predict_reference_table,
    predictor_command,
    run_mhcflurry,
)

_HEADER = (
    "allele,peptide,mhcflurry_affinity,mhcflurry_affinity_percentile,"
    "mhcflurry_processing_score,mhcflurry_presentation_score,mhcflurry_presentation_percentile"
)


def _observed_csv() -> str:
    lines = [_HEADER]
    for allele, peptide, affinity, percentile in OBSERVED_EXAMPLE["rows"]:
        lines.append(f"{allele},{peptide},{affinity},{percentile},0.4,0.03,6.9")
    return "\n".join(lines) + "\n"


def _rows(table: str) -> list[dict[str, str]]:
    lines = table.splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"), strict=True)) for line in lines[1:]]


def test_the_conversion_rescales_nothing() -> None:
    """Every number comes out bit-for-bit as MHCflurry wrote it.

    A rescaling is the failure this module can have that looks like success:
    a percentile divided by 100, or an affinity read as micromolar, produces a
    table that parses, reads as numbers, and is wrong by a factor that decides
    the verdict.
    """
    rows = _rows(convert_mhcflurry_csv(_observed_csv()))
    for row, (_allele, peptide, affinity, percentile) in zip(rows, OBSERVED_EXAMPLE["rows"], strict=True):
        assert row["peptide"] == peptide
        assert row["ic50_nm"] == repr(affinity)
        assert row["percentile_rank"] == repr(percentile)
        assert float(row["ic50_nm"]) == affinity
        assert float(row["percentile_rank"]) == percentile
        assert row["unit"] == "nM"


def test_the_percentile_is_a_percentage_and_a_rescale_would_flip_a_verdict() -> None:
    """SIINFEKL is the control: 6249 nM is a non-binder at a rank of 7.96.

    Were the percentile column on 0-1 -- or were something here to divide by
    100 -- that same row would read as 0.0796, which is under liability.py's
    class I rank cut of 1.0, and a 6 uM non-binder would be published as a
    presented ligand. The two assertions below are that pair.
    """
    rows = {row["peptide"]: row for row in _rows(convert_mhcflurry_csv(_observed_csv()))}
    weak = rows["SIINFEKL"]
    assert float(weak["ic50_nm"]) > _MHC_I_IC50
    assert float(weak["percentile_rank"]) > _MHC_I_RANK
    assert float(weak["percentile_rank"]) / 100 < _MHC_I_RANK  # the mistake this guards against
    strong = rows["NLVPMVATV"]
    assert float(strong["ic50_nm"]) < _MHC_I_IC50
    assert float(strong["percentile_rank"]) < _MHC_I_RANK
    assert mhc_tables.PERCENTILE_SCALE == "percent_0_100"


def test_the_column_map_covers_the_header_mhcflurry_actually_writes() -> None:
    written = set(_HEADER.split(","))
    assert written <= set(COLUMN_MAP), written - set(COLUMN_MAP)
    assert COLUMN_MAP["mhcflurry_affinity"] == "ic50_nm"
    assert COLUMN_MAP["mhcflurry_affinity_percentile"] == "percentile_rank"


def test_the_contract_record_names_both_sides() -> None:
    record = contract()
    assert record["affinity_unit"] == "nM"
    assert record["percentile_scale"] == "percent_0_100"
    assert "500" in str(record["read_by"]["ic50"])
    assert record["observed_example"]["mhcflurry_version"]


def test_the_runner_no_longer_depends_on_dev_stdin() -> None:
    """The POSIX-only path flagged in the handoff. A regression here is silent on Linux."""
    source = inspect.getsource(mhc_tables.run_mhcflurry)
    code = [line for line in source.splitlines() if not line.strip().startswith("#")]
    assert all("/dev/stdin" not in line for line in code)
    assert "tempfile.TemporaryDirectory" in source


def test_the_predictor_is_looked_up_rather_than_assumed(monkeypatch, tmp_path) -> None:
    # Clearing PATH and the environment variable is not enough to make the
    # predictor absent: predictor_command() also looks under install_root(),
    # which is DATA_PATH/mhcflurry, and conftest loads .env.local with
    # override=True so that points at a real installation. Without an empty
    # DATA_PATH this asserts that MHCflurry is not installed on the machine
    # running the suite, which is not what it is for.
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "absent"))
    monkeypatch.setattr(mhc_tables.shutil, "which", lambda _name: None)
    monkeypatch.delenv(mhc_tables.MHCFLURRY_COMMAND_ENV, raising=False)
    assert predictor_command() is None
    with pytest.raises(PredictorMissingError, match="not on PATH"):
        run_mhcflurry([(1, "SIINFEKL")], ["HLA-A*02:01"])


def test_a_stale_environment_variable_reads_as_absent_not_as_a_crash(monkeypatch, tmp_path) -> None:
    # Same isolation as above: the stale variable must be the only route left,
    # so the installed copy under DATA_PATH cannot answer in its place.
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "absent"))
    monkeypatch.setattr(mhc_tables.shutil, "which", lambda _name: None)
    monkeypatch.setenv(mhc_tables.MHCFLURRY_COMMAND_ENV, "/no/such/mhcflurry-predict")
    assert predictor_command() is None


def test_an_executable_named_in_the_environment_wins(monkeypatch, tmp_path) -> None:
    fake = tmp_path / "mhcflurry-predict"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(mhc_tables.shutil, "which", lambda _name: "/usr/bin/mhcflurry-predict")
    monkeypatch.setenv(mhc_tables.MHCFLURRY_COMMAND_ENV, str(fake))
    assert predictor_command() == str(fake)


def test_the_reference_table_attributes_every_row_and_states_the_length() -> None:
    rows = convert_reference_csv(_observed_csv(), protein="P02768", length=609, starts={"SIINFEKL": 11})
    assert {row["protein"] for row in rows} == {"P02768"}
    assert {row["length"] for row in rows} == {"609"}
    assert rows[0]["start"] == "11"
    assert rows[0]["unit"] == "nM"


def test_a_protein_liability_py_will_not_accept_is_refused_here(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not_a_tolerated_reference_protein"):
        predict_reference_table({"P12345": "ACDEFGHIKL"}, ["HLA-A*02:01"], directory=tmp_path)
    assert REFERENCE_ACCESSIONS == ("P02768", "P01857", "P0CG47")
