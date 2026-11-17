"""Biosecurity via Commec: flexible CSV parsing, the conservative mapping, orchestration.

Commec is not invoked; ``commec screen`` and ``commec flag`` are mocked. The flag
step writes the CSV the parser reads. These run in milliseconds on CPU with no
database.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rfproteina.disease import biosecurity
from rfproteina.disease.screening_tools import ScreeningInputError, ToolResult


def test_recommendation_mapping_is_conservative() -> None:
    assert biosecurity.verdict("Pass")["state"] == "clear"
    assert biosecurity.verdict("Flag")["state"] == "hit"
    assert biosecurity.verdict("Warn")["state"] == "hit"
    assert biosecurity.verdict("Error")["state"] == "untested"


def test_an_unknown_recommendation_is_untested_not_clear() -> None:
    record = biosecurity.verdict("something_new")
    assert record["state"] == "untested"
    assert "unknown_recommendation" in str(record["blocker"])


def test_flag_csv_is_parsed_with_flexible_column_names() -> None:
    text = "query_name,Overall Recommendation,notes\nd1,Pass,ok\nd2,Flag,toxin\n"
    parsed = biosecurity.parse_flag_csv(text)
    assert parsed == {"d1": "Pass", "d2": "Flag"}


def test_flag_csv_without_the_needed_columns_is_refused() -> None:
    with pytest.raises(ScreeningInputError, match="flag_csv_missing_columns"):
        biosecurity.parse_flag_csv("colA,colB\n1,2\n")
    with pytest.raises(ScreeningInputError, match="empty_flag_csv"):
        biosecurity.parse_flag_csv("   \n")


def test_screen_raises_without_records(tmp_path: Path) -> None:
    fasta = tmp_path / "x.fasta"
    fasta.write_text("", encoding="utf-8")
    with pytest.raises(ScreeningInputError, match="no_fasta_records"):
        biosecurity.biosecurity_screen(fasta, tmp_path, tmp_path / "work")


def test_screen_is_unavailable_when_commec_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">d1\nACGTACGT\n", encoding="utf-8")
    monkeypatch.setattr(biosecurity, "commec_available", lambda: False)
    with pytest.raises(biosecurity.BiosecurityUnavailableError, match="commec_not_installed"):
        biosecurity.biosecurity_screen(fasta, tmp_path, tmp_path / "work")


def test_screen_is_unavailable_when_the_database_dir_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">d1\nACGTACGT\n", encoding="utf-8")
    monkeypatch.setattr(biosecurity, "commec_available", lambda: True)
    with pytest.raises(biosecurity.BiosecurityUnavailableError, match="commec_databases_missing"):
        biosecurity.biosecurity_screen(fasta, tmp_path / "nodir", tmp_path / "work")


def test_screen_builds_a_record_per_input_from_mocked_commec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">d1\nACGTACGT\n>d2\nGGGGCCCC\n>d3\nTTTTAAAA\n", encoding="utf-8")
    databases = tmp_path / "db"
    databases.mkdir()

    def fake_run_tool(command: Sequence[str], **_kwargs: object) -> ToolResult:
        if "flag" in command:
            out = Path(command[command.index("-o") + 1])
            out.write_text("name,recommendation\nd1,Pass\nd2,Flag\n", encoding="utf-8")
        return ToolResult(True, 0, "", "")

    monkeypatch.setattr(biosecurity, "commec_available", lambda: True)
    monkeypatch.setattr(biosecurity, "run_tool", fake_run_tool)
    records = biosecurity.biosecurity_screen(fasta, databases, tmp_path / "work")
    assert records["d1"]["state"] == "clear"
    assert records["d2"]["state"] == "hit"
    # d3 got no verdict from commec; it is untested, never assumed to have passed.
    assert records["d3"]["state"] == "untested"
    assert records["d3"]["blocker"] == "no_commec_verdict"


def test_rollup_is_a_hit_when_any_record_is_flagged() -> None:
    records = {"d1": biosecurity.verdict("Pass"), "d2": biosecurity.verdict("Flag")}
    assert biosecurity.rollup(records)["state"] == "hit"


def test_requirements_are_actionable_and_name_the_local_free_tool() -> None:
    reqs = biosecurity.requirements()
    assert "pip install commec" in reqs["install"]
    assert "skip_tx" in str(reqs["skip_tx_mode"]) or "--skip-tx" in str(reqs["skip_tx_mode"])
