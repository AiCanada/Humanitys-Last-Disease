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
            "d1\t7aaa\t0.9\t0.42\t1e-8\t200",
            "d1\t8bbb\t0.95\t0.77\t1e-20\t400",
            "d1\tbad\tNA\tNA\tNA\tNA",
            "d2\t9ccc\t0.3\t0.20\t2.0\t50",
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
        out.write_text("d1\t8bbb\t0.95\t0.77\t1e-20\t400\n", encoding="utf-8")
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
