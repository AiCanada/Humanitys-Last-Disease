"""Structural novelty via Foldseek: parsing, verdicts, the orchestration, and the roll-up.

Foldseek is not invoked; its search is mocked by writing the tab-separated table
the parser reads, so these run in milliseconds on CPU with no database.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rfproteina.disease import novelty
from rfproteina.disease.screening_tools import ScreeningInputError, ToolResult


def test_query_structures_finds_pdb_and_cif_only(tmp_path: Path) -> None:
    (tmp_path / "a.pdb").write_text("x", encoding="utf-8")
    (tmp_path / "b.cif").write_text("x", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    names = [path.name for path in novelty.query_structures(tmp_path)]
    assert names == ["a.pdb", "b.cif"]


def test_parse_keeps_the_highest_tmscore_per_query_and_skips_bad_rows() -> None:
    table = "\n".join(
        [
            # query, target, fident, qtmscore, ttmscore, evalue, bits
            "d1\t7aaa\t0.9\t0.42\t0.30\t1e-8\t200",
            "d1\t8bbb\t0.95\t0.77\t0.60\t1e-20\t400",
            "d1\tbad\tNA\tNA\tNA\tNA\tNA",
            "d2\t9ccc\t0.3\t0.20\t0.10\t2.0\t50",
        ]
    )
    best = novelty.parse_search_table(table)
    assert best["d1"].tmscore == 0.77
    assert best["d1"].target == "8bbb"
    assert best["d2"].tmscore == 0.20


def test_verdict_hits_at_same_fold_and_clears_below_it() -> None:
    known = novelty.StructureHit("d1", "8bbb", 0.95, 0.77, 1e-20, 400.0)
    novel = novelty.StructureHit("d2", "9ccc", 0.30, 0.20, 2.0, 50.0)
    assert novelty.verdict(known)["state"] == "hit"
    assert novelty.verdict(known)["novel"] is False
    assert novelty.verdict(novel)["state"] == "clear"
    assert novelty.verdict(None)["state"] == "clear"
    assert novelty.verdict(None)["reason"] == "no_match_within_evalue"


def test_search_raises_when_there_is_nothing_to_search(tmp_path: Path) -> None:
    with pytest.raises(ScreeningInputError, match="no_query_structures"):
        novelty.novelty_search(tmp_path, tmp_path / "db", tmp_path / "work")


def test_search_is_unavailable_when_foldseek_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "d1.pdb").write_text("x", encoding="utf-8")
    monkeypatch.setattr(novelty, "foldseek_available", lambda: False)
    with pytest.raises(novelty.NoveltyUnavailableError, match="foldseek_not_installed"):
        novelty.novelty_search(tmp_path, tmp_path / "db", tmp_path / "work")


def test_search_is_unavailable_when_the_database_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "d1.pdb").write_text("x", encoding="utf-8")
    monkeypatch.setattr(novelty, "foldseek_available", lambda: True)
    with pytest.raises(novelty.NoveltyUnavailableError, match="foldseek_database_missing"):
        novelty.novelty_search(tmp_path, tmp_path / "db_does_not_exist", tmp_path / "work")


def test_search_builds_a_record_per_design_from_mocked_foldseek(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "d1.pdb").write_text("x", encoding="utf-8")
    (tmp_path / "d2.pdb").write_text("x", encoding="utf-8")
    database = tmp_path / "db"
    database.write_text("db", encoding="utf-8")

    def fake_run_tool(command: Sequence[str], **_kwargs: object) -> ToolResult:
        out = Path(command[4])
        out.write_text("d1\t8bbb\t0.95\t0.77\t0.60\t1e-20\t400\n", encoding="utf-8")
        return ToolResult(True, 0, "", "")

    monkeypatch.setattr(novelty, "foldseek_available", lambda: True)
    monkeypatch.setattr(novelty, "run_tool", fake_run_tool)
    records = novelty.novelty_search(tmp_path, database, tmp_path / "work")
    assert records["d1"]["state"] == "hit"
    # d2 had no row in the table, so it is the honest novel-and-clear case.
    assert records["d2"]["state"] == "clear"
    assert records["d2"]["novel"] is True


def test_rollup_is_a_hit_when_any_design_is_not_novel() -> None:
    records = {
        "d1": novelty.verdict(novelty.StructureHit("d1", "t", 0.9, 0.8, 1e-9, 300.0)),
        "d2": novelty.verdict(None),
    }
    assert novelty.rollup(records)["state"] == "hit"


def test_database_requirements_lists_free_sets_with_sizes() -> None:
    reqs = novelty.database_requirements()
    assert "PDB" in reqs
    assert "CC-BY-4.0" in reqs["PDB"]["licence"]


# ------------- what the screening review of 2026-11-20 found in this axis
#
# Three defects, each reproduced against the installed Foldseek and database.


def test_a_multi_chain_design_is_looked_up_at_all() -> None:
    """The worst of the three: nothing was looked up, and that read as novel.

    Foldseek names a query entry after the file and appends the chain whenever
    the file holds more than one, so every complex the PPI pipeline writes came
    back as `<design>_A` / `<design>_E` and the file id matched neither. The
    same chain scored 0.976 alone and 0.0 inside its own complex.
    """
    hit_a = novelty.StructureHit("design_A", "4kc3-assembly1_A", 0.91, 0.42, 1e-8, 200.0)
    hit_e = novelty.StructureHit("design_E", "4kc3-assembly1_A", 0.97, 0.969, 1e-30, 900.0)

    matched = novelty.best_per_design({"design_A": hit_a, "design_E": hit_e}, ["design"])
    assert matched["design"] is not None, "a complex must not be recorded as unmatched"
    assert matched["design"].tmscore == 0.969, "the least novel chain answers, as the roll-up does"


def test_a_design_whose_own_name_ends_in_a_suffix_keeps_its_rows() -> None:
    """The prefix rule must prefer the longest design id, or `d1` steals `d1_extra`'s rows."""
    mine = novelty.StructureHit("d1_extra_A", "9xyz", 0.9, 0.8, 1e-9, 300.0)
    matched = novelty.best_per_design({"d1_extra_A": mine}, ["d1", "d1_extra"])
    assert matched["d1_extra"] is not None
    assert matched["d1"] is None


def test_a_reported_id_belonging_to_no_query_is_a_blocker_not_a_clear() -> None:
    """A naming assumption that has broken must not read as "nothing matched"."""
    stray = novelty.StructureHit("someone_else", "9xyz", 0.9, 0.8, 1e-9, 300.0)
    with pytest.raises(novelty.NoveltyUnavailableError, match="foldseek_query_id_unrecognized"):
        novelty.best_per_design({"someone_else": stray}, ["d1"])


def test_the_tm_score_is_chain_normalised_not_alignment_normalised() -> None:
    """Xu & Zhang's 0.5 is about a chain-normalised TM-score.

    Foldseek's `alntmscore` normalises by the aligned span, so a match covering
    half a design read near 1.0 on it -- values above 1.0 were observed, which a
    TM-score cannot take. `qtmscore`/`ttmscore` are the chain-normalised pair,
    and `metrics/cath_metric.py` already scores on the larger of them.
    """
    assert "qtmscore" in novelty.SEARCH_COLUMNS
    assert "ttmscore" in novelty.SEARCH_COLUMNS
    assert "alntmscore" not in novelty.SEARCH_COLUMNS

    # the larger of the pair wins, so a known chain counts either way round
    table = "d1\t8bbb\t0.95\t0.20\t0.80\t1e-20\t400"
    assert novelty.parse_search_table(table)["d1"].tmscore == 0.80


def test_the_published_evalue_matches_what_the_search_actually_runs() -> None:
    """Three independent literals today; a tight one hid candidates from TMalign.

    Foldseek applies `-e` at its structurealign stage, which runs before
    TMalign, so at the former 1e-3 an atom-exact copy of a PDB entry returned no
    rows at all and the axis called it novel.
    """
    import inspect

    published = float(novelty.NOVELTY_THRESHOLDS["evalue_max"]["value"])
    assert published == 1e6
    for fn in (novelty.build_search_command, novelty.novelty_search):
        default = inspect.signature(fn).parameters["evalue_max"].default
        assert default == published, f"{fn.__name__} runs at {default}, table publishes {published}"


def test_the_record_names_the_chain_that_answered() -> None:
    """A binder+target file is answered by whichever chain matches best, and a
    target matches its own PDB entry. Without this the hit reads as the design's."""
    hit = novelty.StructureHit("design_E", "4kc3-assembly1_A", 0.97, 0.969, 1e-30, 900.0)
    record = novelty.verdict(hit)
    assert record["matched_entry"] == "design_E"
    assert record["best_target"] == "4kc3-assembly1_A"
    assert novelty.verdict(None)["matched_entry"] is None


def test_a_binary_cif_is_refused_rather_than_silently_unscreened(tmp_path: Path) -> None:
    """Foldseek drops a .bcif without an error, and a dropped query reads novel.

    Measured: `createdb` on a directory holding one readable .cif and one .bcif
    exits 0 and writes a lookup naming only the .cif. The .bcif design then drew
    no rows, and the axis called it structurally novel because nothing had ever
    looked at it.
    """
    (tmp_path / "readable.cif").write_text("x", encoding="utf-8")
    (tmp_path / "design.bcif").write_bytes(b"\x00binary")
    with pytest.raises(ScreeningInputError, match="foldseek_cannot_read_bcif"):
        novelty.query_structures(tmp_path)
    assert ".bcif" not in novelty.STRUCTURE_SUFFIXES
