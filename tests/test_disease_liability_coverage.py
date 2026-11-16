"""What a liability verdict is allowed to claim about a chain it part-scanned.

Every test here is a verdict that was published on evidence narrower than the
claim, and the correct outcome is a refusal naming the blocker rather than a
number. The axes themselves are the published methods, checked in
``test_disease_liability_methods.py``; this file is about coverage, provenance
and the denominators, which are this program's own rules:

- A clear is a statement about every window of the chain, so the table has to
  have scanned the chain. A hit is positive evidence and needs no coverage.
  The module already applied that rule to a density of zero and to the
  restricted Calis scan, and not to the ADA breadth or to a non-zero density.
- A verdict has to be about THIS design, so a table that names no peptide can
  support ``untested`` and never a verdict, whatever kind the design is.
- A count of distinct binding peptides of width w is a count over the chain's
  ``L - w + 1`` windows of that width, not over its L residues, and both sides
  of the reference comparison have to be divided by the same thing.
- A percentile rank is a percentage and an IC50 a concentration, so a negative
  cell is an unreadable cell, not the best binder in the table.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from rfproteina.disease.liability import (
    _fasta_records,
    _kmer_digest,
    _window_count,
    measure_sequence_liability,
)
from tests.helpers.disease import UBIQUITIN

_UBIQUITIN = UBIQUITIN
#: The first ten 9-mers of human serum albumin (P02768), which is one of the
#: three tolerated proteins the density bar may be taken from. THE RANKS GIVEN
#: FOR THEM BELOW ARE CONSTRUCTED; the sequence is the real one.
_ALBUMIN_NINE_MERS = (
    "DAHKSEVAH",
    "AHKSEVAHR",
    "HKSEVAHRF",
    "KSEVAHRFK",
    "SEVAHRFKD",
    "EVAHRFKDL",
    "VAHRFKDLG",
    "AHRFKDLGE",
    "HRFKDLGEE",
    "RFKDLGEEN",
)
#: The Paul et al. 2015 seven-allele DR panel, as the ADA axis expects it.
_PANEL = (
    "DRB1*03:01",
    "DRB1*07:01",
    "DRB1*15:01",
    "DRB3*01:01",
    "DRB3*02:02",
    "DRB4*01:01",
    "DRB5*01:01",
)


def _windows(sequence: str, width: int) -> list[str]:
    return [sequence[start : start + width] for start in range(len(sequence) - width + 1)]


def _write_class_ii(folder: Path, peptides: list[str], rank: float = 90.0) -> None:
    """A consensus class II table scoring each peptide on all seven alleles."""
    lines = ["method\tallele\tpeptide\tconsensus_percentile_rank"]
    lines += [f"consensus\t{allele}\t{peptide}\t{rank}" for peptide in peptides for allele in _PANEL]
    (folder / "mhc_ii.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_class_i(folder: Path, scanned: list[str], binders: set[str]) -> None:
    """A class I scan: every window it looked at, binding or not."""
    lines = ["allele\tpeptide\tpercentile_rank"]
    lines += [f"HLA-A*02:01\t{window}\t{0.4 if window in binders else 50.0}" for window in scanned]
    (folder / "mhc_i.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_albumin_reference(folder: Path) -> None:
    """Ten binders over the 601 nine-mer windows of a 609-residue protein."""
    lines = ["protein\tallele\tpeptide\tpercentile_rank\tlength"]
    lines += [f"P02768\tHLA-A*02:01\t{peptide}\t0.3\t609" for peptide in _ALBUMIN_NINE_MERS]
    (folder / "mhc_i_reference.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_chain_holds_the_windows_this_file_counts_on() -> None:
    """The arithmetic every assertion below rests on, stated once.

    Ubiquitin is 76 residues, so it holds 76 - 9 + 1 = 68 nine-mer windows and
    76 - 15 + 1 = 62 fifteen-mer windows, and in this chain every one of them
    is distinct -- which is what makes "the table named them all" the same
    thing as "the scan covered the chain".
    """
    assert len(_UBIQUITIN) == 76
    assert len(set(_windows(_UBIQUITIN, 9))) == 68
    assert len(set(_windows(_UBIQUITIN, 15))) == 62


def test_a_two_peptide_class_ii_table_cannot_clear_ada(tmp_path: Path) -> None:
    """ADA clear is a claim about every window, so 2 of 62 is not one.

    The rule is applied per peptide -- a median consensus rank over the seven
    panel alleles -- so "this design holds no promiscuous binder" needs every
    window of the design to have been put through it. A table holding 2 of
    ubiquitin's 62 fifteen-mer windows published that clear on 3 per cent of
    the chain, with nothing but ``peptides_scored: 2`` to hint at it.
    """
    _write_class_ii(tmp_path, _windows(_UBIQUITIN, 15)[:2])
    axes, _by_example = measure_sequence_liability([("binder_0", _UBIQUITIN)], directory=tmp_path)
    record = axes["ada"]
    assert record["state"] == "untested"
    assert record["blocker"] == "scan_incomplete"
    # The number the table did hold is still reported, as every other refusal
    # in this module reports what it read.
    assert record["measurement"] == 90.0
    assert record["peptides_scored"] == 2

    # The complete 62-window scan of the same chain is the clear.
    _write_class_ii(tmp_path, _windows(_UBIQUITIN, 15))
    axes, _by_example = measure_sequence_liability([("binder_0", _UBIQUITIN)], directory=tmp_path)
    record = axes["ada"]
    assert record["state"] == "clear"
    assert record["blocker"] is None
    assert record["peptides_scored"] == 62


def test_an_ada_hit_needs_no_coverage(tmp_path: Path) -> None:
    """The deliberate asymmetry: one promiscuous peptide is a hit on its own.

    A median of 5.0 across the seven alleles is at or below the Paul 2015
    universal threshold of 20.0, so that peptide is a promiscuous binder
    however little else was scanned. Refusing it for want of coverage would
    discard positive evidence, which is the opposite error.
    """
    _write_class_ii(tmp_path, _windows(_UBIQUITIN, 15)[:2], rank=5.0)
    axes, _by_example = measure_sequence_liability([("binder_0", _UBIQUITIN)], directory=tmp_path)
    record = axes["ada"]
    assert record["state"] == "hit"
    assert record["measurement"] == 5.0
    assert record["blocker"] is None


def test_a_partial_design_scan_cannot_clear_against_the_reference(tmp_path: Path) -> None:
    """A density is a count over the whole chain, scanned or not.

    A table covering 20 of ubiquitin's 68 windows and finding one binder in
    them divided that 1 by the whole chain and cleared at 1.3158 against
    albumin's 1.642 bar. The complete scan of the same chain holds four
    binders. A partial scan truncates the numerator and keeps the denominator,
    so it errs toward clear -- the same reason a density of zero is refused.
    """
    windows = _windows(_UBIQUITIN, 9)
    _write_albumin_reference(tmp_path)
    _write_class_i(tmp_path, windows[:20], {windows[3]})
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)], directory=tmp_path, binder_kind="protein"
    )
    record = axes["mhc_i"]
    assert record["state"] == "untested"
    assert record["blocker"] == "scan_incomplete"
    assert "reference_density" not in record


def test_density_is_quoted_per_window_not_per_residue(tmp_path: Path) -> None:
    """Both sides are counts over windows, so both are divided by windows.

    Ubiquitin holds 68 nine-mer windows and albumin 601, so per residue the
    design was understated by 9.3 per cent against the bar and the bias always
    ran toward clear, because a design is short and the tolerated proteins are
    long. Four binders of 68 windows is 5.8824 per 100 windows; ten binders of
    601 is 1.6639.
    """
    assert _window_count(76, frozenset({9})) == 68
    assert _window_count(609, frozenset({9})) == 601
    # A chain shorter than the width holds no window of it.
    assert _window_count(8, frozenset({9})) == 0
    assert _window_count(20, frozenset({9, 15})) == 12 + 6

    windows = _windows(_UBIQUITIN, 9)
    _write_albumin_reference(tmp_path)
    _write_class_i(tmp_path, windows, {windows[3], windows[20], windows[40], windows[60]})
    axes, _by_example = measure_sequence_liability(
        [("binder_0", _UBIQUITIN)], directory=tmp_path, binder_kind="protein"
    )
    record = axes["mhc_i"]
    assert record["state"] == "hit"
    assert record["measurement"] == 5.8824
    assert record["reference_density"] == 1.6639
    assert record["binding_peptides"] == 4
    # The name says which denominator the number is on, so a stored record
    # cannot be read on the other one.
    assert record["method"] == "binding_peptides_per_100_windows"


def test_a_rank_only_table_cannot_clear_a_peptide_design(tmp_path: Path) -> None:
    """A rank with no peptide is not a verdict about any molecule.

    Nothing in this module can attribute an anonymous rank to a chain, so the
    table supports ``untested`` and nothing else -- whatever kind the design
    is. The check sat after the peptide branch, so a leftover rank-only export
    cleared a peptide design at a rank of 50 while the identical file on a
    protein design was refused with the same blocker.
    """
    peptide = "MQIFVKTLT"
    (tmp_path / "mhc_i.tsv").write_text("allele\tpercentile_rank\nHLA-A*02:01\t50\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
    assert axes["mhc_i"]["state"] == "untested"
    assert axes["mhc_i"]["blocker"] == "peptide_column_absent"
    assert axes["mhc_i"]["measurement"] == 50.0

    # Naming the peptide is what makes the same number a verdict.
    (tmp_path / "mhc_i.tsv").write_text(
        f"allele\tpeptide\tpercentile_rank\nHLA-A*02:01\t{peptide}\t50\n", encoding="utf-8"
    )
    axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
    assert axes["mhc_i"]["state"] == "clear"
    assert axes["mhc_i"]["blocker"] is None


def test_a_negative_rank_is_refused_not_treated_as_the_best_binder(tmp_path: Path) -> None:
    """A rank is a percentage and an IC50 a concentration; neither is negative.

    Several exporters write -1 for a window the predictor declined, and -1
    satisfies every cutoff there is: every such window became a binder and the
    axis published -1.0 as its measurement. It is an unreadable cell like any
    other and is refused as one.
    """
    peptide = "MQIFVKTLT"
    (tmp_path / "mhc_i.tsv").write_text(
        f"allele\tpeptide\tpercentile_rank\nHLA-A*02:01\t{peptide}\t-1\n", encoding="utf-8"
    )
    axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
    assert axes["mhc_i"]["state"] == "untested"
    assert axes["mhc_i"]["blocker"] == "cell_unparsed"
    assert axes["mhc_i"]["unparsed_columns"] == ["percentile_rank"]

    # A negative affinity is refused the same way.
    (tmp_path / "mhc_i.tsv").write_text(f"allele\tpeptide\tic50_nm\nHLA-A*02:01\t{peptide}\t-5\n", encoding="utf-8")
    axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
    assert axes["mhc_i"]["state"] == "untested"
    assert axes["mhc_i"]["blocker"] == "cell_unparsed"

    # The IEDB cut itself is untouched: a rank of exactly 1.0 still binds.
    (tmp_path / "mhc_i.tsv").write_text(
        f"allele\tpeptide\tpercentile_rank\nHLA-A*02:01\t{peptide}\t1.0\n", encoding="utf-8"
    )
    axes, _by_example = measure_sequence_liability([("b", peptide)], directory=tmp_path, binder_kind="peptide")
    assert axes["mhc_i"]["state"] == "hit"
    assert axes["mhc_i"]["measurement"] == 1.0


def test_a_supplied_score_names_the_verdict_it_displaced(tmp_path: Path) -> None:
    """``immunogenicity.tsv`` turns the axis off, so the record says so.

    A supplied score has no pass cut, so the axis can never be hit or clear
    while the file is there -- the same filename means both "my scores" and
    "turn this axis off". With the class I table that drives the Calis verdict
    beside it, the record now names the computation the file displaced.
    """
    (tmp_path / "immunogenicity.tsv").write_text("peptide\tscore\nENVKAKIQD\t0.12\n", encoding="utf-8")
    _write_class_i(tmp_path, _windows(_UBIQUITIN, 9), {"ENVKAKIQD"})
    axes, _by_example = measure_sequence_liability([("binder_0", _UBIQUITIN)], directory=tmp_path)
    record = axes["immunogenicity"]
    assert record["state"] == "untested"
    assert record["blocker"] == "no_pass_cut"
    assert record["measurement"] == 0.12
    assert record["displaced_method"] == "calis_2013_class_i_9mer"
    assert record["displaced_table"] == "mhc_i"

    # With nothing beside it there is no displacement to report.
    (tmp_path / "mhc_i.tsv").unlink()
    axes, _by_example = measure_sequence_liability([("binder_0", _UBIQUITIN)], directory=tmp_path)
    assert "displaced_method" not in axes["immunogenicity"]


def test_a_supplied_ada_score_names_the_verdict_it_displaced(tmp_path: Path) -> None:
    """The same disclosure on the ADA axis, which ``ada.tsv`` turns off."""
    (tmp_path / "ada.tsv").write_text("score\n0.5\n", encoding="utf-8")
    _write_class_ii(tmp_path, _windows(_UBIQUITIN, 15))
    axes, _by_example = measure_sequence_liability([("binder_0", _UBIQUITIN)], directory=tmp_path)
    record = axes["ada"]
    assert record["state"] == "untested"
    assert record["blocker"] == "no_pass_cut"
    assert record["displaced_method"] == "mhc_ii_dr_breadth_paul_2015"
    assert record["displaced_table"] == "mhc_ii"


def test_the_proteome_digest_is_taken_once_per_loaded_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every design's record carries the digest; the proteome is hashed once.

    The digest is a sha256 of the packed k-mer array, which is 90 MB for a
    human proteome. Re-hashing it per design made the digest 8 seconds of an
    8.6 second 200-design run, against the microseconds the off-target search
    itself costs.
    """
    kmers = np.arange(1000, dtype=np.int64)
    expected = hashlib.sha256(kmers.tobytes()).hexdigest()[:16]
    taken: list[int] = []
    real = hashlib.sha256

    def counting(data: bytes) -> object:
        taken.append(len(data))
        return real(data)

    monkeypatch.setattr(hashlib, "sha256", counting)
    assert _kmer_digest(kmers) == expected
    assert _kmer_digest(kmers) == expected
    assert len(taken) == 1, "the same array is not hashed twice"

    # A different proteome still digests differently, which is the point of
    # recording it at all.
    other = np.arange(1, 1001, dtype=np.int64)
    assert _kmer_digest(other) != expected
    assert len(taken) == 2


def test_a_comment_line_and_a_hidden_header_are_not_residues() -> None:
    """Two FASTA shapes whose own text was read as a protein sequence.

    A byte-order mark sits in front of the first ``>``, so that header did not
    start a record and its accession and description became residues of the
    first record. A line opening with ``;`` is a comment -- the original FASTA
    definition allows them -- and reading one as residues does the same thing:
    either way the proteome holds 9-mers no protein in it contains, and a
    design matching one of them reads as an off-target hit.
    """
    header = ">sp|P02768|ALBU_HUMAN Serum albumin"
    assert _fasta_records(f"; a comment line\n{header}\nMKWVTFISLL\n") == ["MKWVTFISLL"]
    assert _fasta_records(f"﻿{header}\nMKWVTFISLL\n") == ["MKWVTFISLL"]
    # A comment between records is dropped without ending one.
    assert _fasta_records(f"{header}\nMKWVTFISLL\n; note\n{header}\nDAHKSEVAHR\n") == [
        "MKWVTFISLL",
        "DAHKSEVAHR",
    ]
    # A file that is nothing but comments holds no record.
    assert _fasta_records("; one\n; two\n") == []
