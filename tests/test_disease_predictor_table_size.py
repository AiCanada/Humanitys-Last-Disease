"""A predictor table too large to hold blocks one axis instead of ending the run.

``_table_rows`` reads the whole file into memory with no size bound. That is
survivable for the tables this screen is meant to read -- a scan of one design
is a handful of kilobytes -- but the download registry points at a 10.26 GB
IEDB export whose axis is ``mhc_i``, and the stem is printed right beside it.
The file an operator is most likely to drop in is therefore the one that cannot
be held, and before this it raised ``MemoryError`` out of the reader and ended
the screen: every other axis and every other design went unmeasured because one
file in the directory was too big.

The two sibling readers already got this right -- ``target_liability._rows``
catches it, and ``_proteome_kmers`` in this same module answers with its own
``proteome_too_large`` blocker -- so this is the odd one out being brought into
line rather than a new policy.

No size bound is added here on purpose. A bound would refuse files that parse
today, and refusing a table that would have measured is the failure this screen
is built to avoid; catching the error that was already fatal cannot do that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfproteina.disease.liability import _ada_record, _score_record, _table_rows


def test_a_table_too_large_to_hold_is_unreadable_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: a named refusal reaches the caller, not an exception."""
    path = tmp_path / "mhc_i.tsv"
    path.write_text("peptide\tallele\n", encoding="utf-8")

    def refuse(*_args: object, **_kwargs: object) -> str:
        raise MemoryError("cannot hold 10.26 GB")

    monkeypatch.setattr(Path, "read_text", refuse)
    assert _table_rows(path) is None


def test_the_axis_reports_it_rather_than_raising(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through a real axis, because the reader returning None is only half of it.

    An axis that turned the None into an exception of its own would leave the
    run just as dead.
    """
    (tmp_path / "ada.tsv").write_text("score\n0.5\n", encoding="utf-8")
    real = Path.read_text

    def refuse(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "ada.tsv":
            raise MemoryError("cannot hold it")
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", refuse)
    record = _score_record(tmp_path, "ada", method="supplied_table")
    assert record["state"] == "untested"
    assert record["blocker"] == "predictor_unreadable"


def test_a_table_that_parses_today_still_parses(tmp_path: Path) -> None:
    """Guards the half of this that could have gone wrong.

    Catching the error must not become refusing files by size. A table the
    reader takes today has to keep being taken, or this change would suppress
    measurements to avoid a crash.
    """
    path = tmp_path / "mhc_ii.tsv"
    path.write_text(
        "peptide\tallele\tconsensus_percentile_rank\tmethod\n" "ACDEFGHIK\tHLA-DRB1*01:01\t0.5\tconsensus\n",
        encoding="utf-8",
    )
    rows = _table_rows(path)
    assert rows is not None and len(rows) == 1
    assert rows[0]["peptide"] == "ACDEFGHIK"


def test_a_directory_with_one_huge_table_still_measures_the_other_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cost of the old behaviour, stated as a test.

    One unreadable file used to take the whole screen down with it. Here the
    unreadable ``ada`` table blocks its own axis and the class II table beside
    it is still read.
    """
    (tmp_path / "ada.tsv").write_text("score\n0.5\n", encoding="utf-8")
    (tmp_path / "mhc_ii.tsv").write_text(
        "peptide\tallele\tconsensus_percentile_rank\tmethod\n" "ACDEFGHIK\tHLA-DRB1*01:01\t0.5\tconsensus\n",
        encoding="utf-8",
    )
    real = Path.read_text

    def refuse(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "ada.tsv":
            raise MemoryError("cannot hold it")
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", refuse)
    record = _ada_record(tmp_path, [("design_0", "ACDEFGHIKLMNOP")])
    assert record["blocker"] == "predictor_unreadable"
    # And the readable table in the same directory is still readable.
    monkeypatch.undo()
    assert _table_rows(tmp_path / "mhc_ii.tsv") is not None
