"""``--stage-tables`` has to put the table there *before* the screen reads it.

The screen reads its liability tables out of the run's own output directory.
The downloads live somewhere else entirely, under the names their servers gave
them. Nothing joined the two, so a machine holding every download screened
exactly as a bare one did -- the fold axis reported no table, which reads as a
target that was checked and found not to move.

Two things are pinned here and they fail in opposite directions:

* without the flag, nothing is written into the run directory -- staging is
  opt-in, and a run that did not ask for it must not find files appearing in
  its output;
* with the flag, the table is on disk *by the time the screen runs*, not after
  it. Staged afterwards it would be a file nobody read, and the axis would
  report untested on a directory that holds its answer.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from rfproteina.disease.cli import _stage_screen_tables

HPA_HEADER = ("Gene", "ENSG ID", "Binder", "Disease", "Class", "Control", "logFC", "p-value adjusted")


@pytest.fixture
def stocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A download directory, pointed at by DATA_PATH the way a real run finds it."""
    directory = tmp_path / "data" / "screening_tables"
    directory.mkdir(parents=True)
    with (directory / "blood_pea_disease_de.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HPA_HEADER)
        for gene, disease, log2 in (
            ("ERBB2", "Breast cancer", "0.28"),
            ("TP53", "Breast cancer", "1.5"),
        ):
            writer.writerow((gene, "ENSG0", "seq.1", disease, "Cancer", "All other", log2, "1E-3"))
    with gzip.open(directory / "UP000005640_9606.fasta.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            ">sp|P04626|ERBB2_HUMAN x OS=Homo sapiens OX=9606 GN=ERBB2 PE=1 SV=1\nMKV\n"
            ">sp|P04637|TP53_HUMAN x OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=1\nMKV\n"
        )
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    return directory


def test_staging_narrows_to_this_run_s_own_target(
    stocked: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Narrowed to the accession the screen resolved, not to the whole disease.

    The whole-disease conversion is past the fold reader's 1 MB bound, so an
    unnarrowed stage writes a file that is then refused -- present on disk and
    measuring nothing, which is the outcome this step exists to prevent.
    """
    run = tmp_path / "run"
    run.mkdir()
    _stage_screen_tables(run, "Breast cancer", "P04626")

    written = (run / "plasma_fold.tsv").read_text(encoding="utf-8").splitlines()
    assert written[0].split("\t") == ["gene", "accession", "log2_fold_change", "source"]
    assert [row.split("\t")[0] for row in written[1:]] == ["ERBB2"], "TP53 is not this run's target"
    assert "Staged plasma_fold for the screen" in capsys.readouterr().out


def test_the_table_is_readable_by_the_screen_that_follows(stocked: Path, tmp_path: Path) -> None:
    """End to end: staged, then measured through the screen's own reader."""
    from rfproteina.disease.target_liability import TargetLimits, measure_target_liability

    run = tmp_path / "run"
    run.mkdir()
    _stage_screen_tables(run, "Breast cancer", "P04626")

    record = measure_target_liability("P04626", directory=run, limits=TargetLimits(fold_max=2.0))["plasma_fold"]
    assert record["state"] == "clear"
    assert record["measurement"] == pytest.approx(0.28)
    assert "Human Protein Atlas" in str(record["fold_source"])


def test_a_disease_the_table_does_not_hold_is_said_not_guessed(
    stocked: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """This program's disease vocabulary is not HPA's, and the mismatch is real.

    An empty fold table would read as "no fold reported", indistinguishable
    from a target that genuinely does not move, so nothing is written and the
    diseases the file does hold are named.
    """
    run = tmp_path / "run"
    run.mkdir()
    _stage_screen_tables(run, "Psoriasis", "P04626")

    assert not (run / "plasma_fold.tsv").exists()
    out = capsys.readouterr().out
    assert "disease_not_in_table" in out
    assert "Breast cancer" in out, "the diseases it does hold have to be named"


def test_a_bare_machine_reports_the_absence_and_does_not_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every outcome here is a thinner screen, not a failed run."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "empty"))
    run = tmp_path / "run"
    run.mkdir()
    _stage_screen_tables(run, "Breast cancer", "P04626")

    out = capsys.readouterr().out
    assert "conversion_source_absent:blood_pea_disease_de.tsv" in out
    assert not list(run.iterdir()), "nothing is written when there is nothing to convert"


def test_without_the_proteome_the_accession_miss_is_spelled_out(
    stocked: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The silent failure, made loud.

    A symbol-keyed table is readable and matches nothing this screen asks for,
    because the screen looks its target up by accession. Readable and unmatched
    is the worst of the outcomes to leave unsaid.
    """
    (stocked / "UP000005640_9606.fasta.gz").unlink()
    run = tmp_path / "run"
    run.mkdir()
    # Narrowed by symbol, since without the proteome no accession can be
    # translated -- the point is what the staged file can then answer.
    _stage_screen_tables(run, "Breast cancer", "ERBB2")

    assert (run / "plasma_fold.tsv").is_file()
    out = capsys.readouterr().out
    assert "proteome_absent" in out
    assert "will read untested" in out
