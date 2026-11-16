"""Four readers that lost, or could not reach, a measurement they had.

Each of these came out of an audit of the axis modules, and each is the same
shape as a defect already fixed elsewhere in this package -- which is the point:
the readers were written to a shared contract and three of them had drifted off
it in a way the fourth had not.

* ``toxicology._add`` let a relation column carrying **no** directional claim
  delete a cell that a flatly contradicting one did not, against the rule
  stated eleven lines below it.
* ``toxicology._rows`` was the only table reader in the package that did not
  catch ``MemoryError``, so one oversized file ended the whole screen.
* ``oligo`` could not find a gzipped transcriptome, which is the only form the
  reference its seed arithmetic is argued for is distributed in.
* ``oligo`` opened that reference as strict ``utf-8``, so a byte-order mark --
  which every sibling reader tolerates on purpose -- refused it outright.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from rfproteina.disease.oligo import (
    TRANSCRIPTOME_FILE,
    TRANSCRIPTOME_FILE_LIMIT,
    _reference_path,
    _scan,
    _size_limit,
)
from rfproteina.disease.toxicology import measure_toxicology, panel_cut

_HERG_CUT = panel_cut("herg")


def panel(directory: Path, header: str, row: str) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "toxicology.tsv").write_text(f"{header}\n{row}\n", encoding="utf-8")
    found = measure_toxicology(directory)
    assert isinstance(found, dict)
    record = found["herg"]
    assert isinstance(record, dict)
    return record


# ------------------------------------- the cell's own bound outranks a relation


def test_an_unreadable_relation_does_not_delete_a_bound_the_cell_carries(
    tmp_path: Path,
) -> None:
    """The suppression, and the asymmetry that made it obvious.

    ``<100`` against a 10,000 nM cut is a hit by a hundredfold, and the rule
    eleven lines below the defect says so: "A bound in the cell wins over a
    relation column, because it sits on the number it qualifies." It does --
    for every relation the reader CAN read. A ``~``, which claims nothing at
    all, took the early return and deleted the reading, so a relation column
    carrying no information was more destructive than one that flatly
    contradicted the cell.
    """
    assert _HERG_CUT is not None and _HERG_CUT > 100
    contradicting = panel(tmp_path / "a", "protein\tnm\trelation", "herg\t<100\t>")
    uninformative = panel(tmp_path / "b", "protein\tnm\trelation", "herg\t<100\t~")

    assert contradicting["state"] == "hit"
    assert uninformative["state"] == "hit", "a relation claiming nothing must not delete the cell"
    assert uninformative["measurement"] == contradicting["measurement"] == pytest.approx(100.0)
    assert uninformative["qualifier"] == "<"
    assert uninformative.get("blocker") is None


def test_the_same_rescue_applies_to_a_clear(tmp_path: Path) -> None:
    """It cost readings in both directions, so both are pinned.

    A bound well clear of the cut is a measured clear, and it was being turned
    into ``reading_not_decisive`` by the same early return.
    """
    record = panel(tmp_path / "clear", "protein\tnm\trelation", "herg\t>50000\t~")
    assert record["state"] == "clear"
    assert record["measurement"] == pytest.approx(50000.0)
    assert record["qualifier"] == ">"


def test_a_bare_number_with_an_unreadable_relation_is_still_undecided(
    tmp_path: Path,
) -> None:
    """The half that was right, and had to stay right.

    Without a bound of its own, a number beside a ``~`` genuinely is not a
    reading: nothing says how far from it the truth may sit. The guard narrowed
    to ``not bound``; it was not removed.
    """
    record = panel(tmp_path / "bare", "protein\tnm\trelation", "herg\t50\t~")
    assert record["state"] == "untested"
    assert record["blocker"] == "reading_not_decisive"
    assert record["measurement"] is None


def test_a_two_column_relation_disagreement_still_settles_on_the_cell(
    tmp_path: Path,
) -> None:
    """The other route to ``relation == "other"``, through the same guard.

    A row naming two relation columns that disagree is unreadable as a
    relation, and a cell carrying its own bound is still the cell's to settle.
    """
    record = panel(
        tmp_path / "two",
        "protein\tnm\trelation\tstandard_relation",
        "herg\t<100\t<\t=",
    )
    assert record["state"] == "hit"
    assert record["measurement"] == pytest.approx(100.0)


# --------------------------------------------- one huge file is not the screen


def test_a_panel_table_too_large_to_hold_blocks_its_axis_and_no_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``MemoryError`` out of this reader used to end the run.

    ``solubility._rows`` and the fold reader in ``target_liability`` both
    already caught it; this was the one that did not, so a single oversized
    file in the output directory took every other axis and every other design
    down with it.
    """
    (tmp_path / "toxicology.tsv").write_text("protein\tnm\nherg\t50\n", encoding="utf-8")
    real = Path.read_text

    def refuse(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "toxicology.tsv":
            raise MemoryError("cannot hold it")
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", refuse)
    found = measure_toxicology(tmp_path)
    assert isinstance(found, dict), "a blocker reaches the caller rather than an exception"
    record = found["herg"]
    assert isinstance(record, dict)
    assert record["state"] == "untested"
    assert record["blocker"] is not None


def test_a_panel_table_that_parses_today_still_parses(tmp_path: Path) -> None:
    """The half of that change which could have gone wrong.

    Catching the error must not become refusing files by size: this reader has
    never had a byte bound, so one chosen now would start refusing panel tables
    that measure.
    """
    record = panel(tmp_path / "ok", "protein\tnm", "herg\t50")
    assert record["state"] == "hit"
    assert record["measurement"] == pytest.approx(50.0)


# ------------------------------------------- the reference as it is distributed


def fasta_text(records: int = 60) -> str:
    """Enough distinct nucleotide sequence to clear the reader's own floors."""
    body = ("ACGTTGCAAGGCTTACGATCGGATCCAGTTACGCATGCAAGGTTCCAAGGTTACGGATCC" * 340)[:20_400]
    return "".join(f">tx{index}\n{body}\n" for index in range(records))


@pytest.mark.parametrize("suffix", [".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz"])
def test_a_reference_beside_the_run_is_found_compressed_or_not(tmp_path: Path, suffix: str) -> None:
    """Ensembl and GENCODE ship cDNA as ``.fa.gz`` and nothing ships it unpacked.

    So the suffix list without a compressed spelling meant the one reference
    this axis is argued for was never found beside a run, and the axis stayed
    untested on a machine that had it.
    """
    path = tmp_path / f"{TRANSCRIPTOME_FILE}{suffix}"
    if suffix.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(fasta_text(1))
    else:
        path.write_text(fasta_text(1), encoding="utf-8")
    assert _reference_path(tmp_path, None) == path


def test_a_gzipped_reference_is_scanned_and_not_refused(tmp_path: Path) -> None:
    path = tmp_path / "transcriptome.fa.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(fasta_text())
    found = _scan(path, {1, 2, 3})
    assert found.blocker is None, found.blocker
    assert found.bases > 1_000_000


def test_a_byte_order_mark_does_not_refuse_a_valid_reference(tmp_path: Path) -> None:
    """Every sibling reader in this package reads ``utf-8-sig`` on purpose.

    This one read strict ``utf-8``, so a mark -- which is what a Windows editor
    writes by default -- made a valid nucleotide FASTA
    ``transcriptome_unreadable``. It would also have been counted as a non-base
    letter by the alphabet guard, which is the second way the same byte could
    have refused the file.
    """
    path = tmp_path / "transcriptome.fa"
    path.write_text("﻿" + fasta_text(), encoding="utf-8")
    found = _scan(path, {1, 2, 3})
    assert found.blocker is None, found.blocker


def test_the_same_content_scans_the_same_compressed_or_not(tmp_path: Path) -> None:
    """The digest is over letters, not bytes, so storage must not change it.

    If it did, whether a design read as an off-target hit would depend on how
    its reference happened to be stored.
    """
    text = fasta_text()
    plain = tmp_path / "plain.fa"
    plain.write_text(text, encoding="utf-8")
    packed = tmp_path / "packed.fa.gz"
    with gzip.open(packed, "wt", encoding="utf-8") as handle:
        handle.write(text)

    one = _scan(plain, {1, 2, 3})
    two = _scan(packed, {1, 2, 3})
    assert one.blocker is two.blocker is None
    assert one.bases == two.bases
    assert one.digest == two.digest, "re-storing a reference is not a different reference"


def test_a_compressed_reference_is_bounded_on_what_it_holds(tmp_path: Path) -> None:
    """The bound is argued in uncompressed FASTA bytes.

    Applied unchanged to a gzip it would be the wrong question by about
    fourfold, admitting a compressed file holding four times the content the
    limit was chosen to admit. It is a mistyped-path guard, so the share is
    deliberately generous rather than an attempt to predict a compressor.
    """
    assert _size_limit(tmp_path / "r.fa") == TRANSCRIPTOME_FILE_LIMIT
    assert _size_limit(tmp_path / "r.fa.gz") < TRANSCRIPTOME_FILE_LIMIT
    assert _size_limit(tmp_path / "r.fa.gz") == TRANSCRIPTOME_FILE_LIMIT // 4


def test_a_file_named_gz_that_is_not_gzipped_is_a_blocker_not_a_crash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transcriptome.fa.gz"
    path.write_bytes(b"ACGT not actually gzip\n")
    found = _scan(path, {1})
    assert found.blocker == "transcriptome_unreadable"
