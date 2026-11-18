"""TEPITOPE scores a 9-mer core and does not claim an IEDB consensus rank."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rfproteina.disease.liability import _declares_consensus, measure_sequence_liability
from rfproteina.disease.tepitope import allele_of, load_matrix, prepare_class_ii_table

_MATRIX = """\
Amino acid/Position,P1,P2,P3,P4,P5,P6,P7,P8,P9
A:,1,0,0,0,-,0,0,-,0
L:,0,0,0,0,-,0,0,-,5
Percent Threshold,1%,10%
Numerical Score,6,1
"""


def test_a_filename_becomes_the_panel_spelling() -> None:
    assert allele_of("HLA-DRB1_0301") == "DRB1*03:01"


def test_a_dash_is_an_unscored_pocket_and_a_nine_mer_is_one_core(tmp_path: Path) -> None:
    path = tmp_path / "HLA-DRB1_0301.csv"
    path.write_text(_MATRIX, encoding="utf-8")
    matrix = load_matrix(path)
    # P1 A is 1, P9 L is 5, and the dash positions add zero. One core, not zero.
    assert matrix.score("AAAAAAAAL") == 6
    assert matrix.score("AAAAAAAAA") == 1
    assert matrix.percentile_bin(6) == 1
    assert matrix.percentile_bin(1) == 10
    assert matrix.percentile_bin(0) == 100


def test_a_tepitope_table_scores_class_ii_and_leaves_ada_untested(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "matrices"
    folder.mkdir()
    (folder / "HLA-DRB1_0301.csv").write_text(_MATRIX, encoding="utf-8")
    monkeypatch.setattr("rfproteina.disease.tepitope.matrices_dir", lambda: folder)
    monkeypatch.setattr("rfproteina.disease.tepitope.DR_PANEL", ("DRB1*03:01",))
    peptide = "AAAAAAAAALAAAAA"
    result = prepare_class_ii_table(tmp_path, [("binder_0", peptide)], binder_kind="peptide")
    assert result["state"] == "written"
    assert result["ada"] == "untested_not_consensus_rank"
    table = (tmp_path / "mhc_ii.tsv").read_text(encoding="utf-8")
    assert "tepitope" in table
    assert "consensus" not in table.casefold()
    axes, _ = measure_sequence_liability([("binder_0", peptide)], directory=tmp_path, binder_kind="peptide")
    assert axes["mhc_ii"]["state"] == "hit"
    assert axes["ada"]["state"] == "untested"
    assert axes["ada"]["blocker"] == "rank_scale_unknown"


def test_an_existing_design_table_does_not_suppress_the_reference_table(
    tmp_path: Path, monkeypatch
) -> None:
    """A run screened before the reference sequences arrived must recover.

    ``prepare_class_ii_table`` returns early when ``mhc_ii.tsv`` is already
    there, which is right for the design table -- it may be one the operator
    supplied. It used to return before writing ``mhc_ii_reference.tsv`` as
    well, so the first screen on a machine without the reference sequences
    pinned the class II axis at ``reference_not_configured`` for good: every
    later run saw the design table, returned, and never scanned the tolerated
    proteins even once the sequences were fetched.
    """
    folder = tmp_path / "matrices"
    folder.mkdir()
    (folder / "HLA-DRB1_0301.csv").write_text(_MATRIX, encoding="utf-8")
    monkeypatch.setattr("rfproteina.disease.tepitope.matrices_dir", lambda: folder)
    monkeypatch.setattr("rfproteina.disease.tepitope.DR_PANEL", ("DRB1*03:01",))
    monkeypatch.setattr(
        "rfproteina.disease.tepitope.reference_sequences",
        lambda: ("ready", {"P02768": "AAAAAAAAALAAAAA", "P01857": "LAAAAAAAAL"}),
    )
    # The state the bug leaves behind: a design table, and no reference beside it.
    (tmp_path / "mhc_ii.tsv").write_text("peptide\tallele\n", encoding="utf-8")
    reference = tmp_path / "mhc_ii_reference.tsv"
    assert not reference.is_file()

    result = prepare_class_ii_table(tmp_path, [("binder_0", "AAAAAAAAAL")], binder_kind="protein")

    assert result["state"] == "table_present"
    assert reference.is_file(), "the reference table must be written beside an existing design table"
    assert result["reference"] == str(reference)
    body = reference.read_text(encoding="utf-8")
    assert "P02768" in body
    # The design table the operator had is untouched.
    assert (tmp_path / "mhc_ii.tsv").read_text(encoding="utf-8") == "peptide\tallele\n"


def test_a_binder_that_is_not_a_protein_gets_no_reference_table(tmp_path: Path, monkeypatch) -> None:
    """Only a protein binder has a tolerated-protein baseline to measure against."""
    folder = tmp_path / "matrices"
    folder.mkdir()
    (folder / "HLA-DRB1_0301.csv").write_text(_MATRIX, encoding="utf-8")
    monkeypatch.setattr("rfproteina.disease.tepitope.matrices_dir", lambda: folder)
    monkeypatch.setattr("rfproteina.disease.tepitope.DR_PANEL", ("DRB1*03:01",))
    monkeypatch.setattr(
        "rfproteina.disease.tepitope.reference_sequences",
        lambda: ("ready", {"P02768": "AAAAAAAAALAAAAA"}),
    )
    (tmp_path / "mhc_ii.tsv").write_text("peptide\tallele\n", encoding="utf-8")
    result = prepare_class_ii_table(tmp_path, [("binder_0", "AAAAAAAAAL")], binder_kind="peptide")
    assert result["state"] == "table_present"
    assert not (tmp_path / "mhc_ii_reference.tsv").is_file()


def test_the_upstream_script_runs_on_python_3(tmp_path: Path) -> None:
    matrices = tmp_path / "matrices"
    matrices.mkdir()
    (matrices / "HLA-DRB1_0301.csv").write_text(_MATRIX, encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "tepitope.py"
    # Length 10: the port keeps the upstream range(len - 9), so one window.
    ran = subprocess.run(
        [sys.executable, str(script), "-t", "10", "--matrices", str(matrices), "AAAAAAAAAL"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert "HLA-DRB1_0301" in ran.stdout
    # Length 9 is one core. The Python 2 range stopped one window early.
    short = subprocess.run(
        [sys.executable, str(script), "-t", "10", "--matrices", str(matrices), "AAAAAAAAA"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert short.returncode == 0, short.stderr
    assert short.stdout.count("HLA-DRB1_0301") == 1
    assert "\t1\t9\t" in short.stdout


def test_consensus_detection_rejects_the_tepitope_method() -> None:
    rows = [{"method": "tepitope", "percentile_rank": "1", "allele": "DRB1*03:01", "peptide": "AAAAAAAAAAAAAAA"}]
    assert _declares_consensus(rows) is False
