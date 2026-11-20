"""Shared screening plumbing: tool runner, FASTA I/O, record shape, alphabet."""

from __future__ import annotations

from pathlib import Path

import pytest

from rfproteina.disease.screening_tools import (
    CLEAR,
    alphabet_of,
    axis_record,
    read_fasta,
    resolve_tool,
    run_tool,
    tool_available,
    write_fasta,
)


def test_a_missing_tool_is_a_blocker_not_a_crash() -> None:
    result = run_tool(["definitely-not-a-real-tool-xyz123"])
    assert result.ok is False
    assert result.blocker is not None
    assert result.blocker.startswith("tool_not_installed:")


def test_a_nonzero_exit_is_reported_as_an_error_blocker() -> None:
    result = run_tool(["false"])
    assert result.ok is False
    assert result.blocker is not None
    assert result.blocker.startswith("tool_error:")


def test_a_clean_exit_passes_stdin_through() -> None:
    if not tool_available("cat"):
        pytest.skip("cat not available")
    result = run_tool(["cat"], stdin_text="hello\n")
    assert result.ok is True
    assert "hello" in result.stdout


def test_an_empty_command_is_refused() -> None:
    assert run_tool([]).blocker == "empty_command"


def test_fasta_roundtrips_and_trims_the_header_to_its_id(tmp_path: Path) -> None:
    path = tmp_path / "x.fasta"
    write_fasta([("design_1 some description", "ACDEFG"), ("design_2", "")], path)
    records = read_fasta(path.read_text(encoding="utf-8"))
    # The empty sequence is not written, and the header keeps only its first token.
    assert records == [("design_1", "ACDEFG")]


def test_fasta_with_no_header_yields_no_records() -> None:
    assert read_fasta("ACDEFG\nHIKLMN\n") == []


def test_alphabet_reads_protein_nucleic_and_empty() -> None:
    assert alphabet_of("ACDEFGHIKLMNPQRSTVWY") == "protein"
    assert alphabet_of("ACGTACGT") == "nucleic"
    assert alphabet_of("ACGUACGU") == "nucleic"
    assert alphabet_of("   ") == "empty"


def test_axis_record_has_the_three_rollup_keys_plus_extras() -> None:
    record = axis_record(CLEAR, 0.0, None, off_target=None)
    assert record["state"] == CLEAR
    assert record["measurement"] == 0.0
    assert record["blocker"] is None
    assert record["off_target"] is None


# ------------------------------------------------- the shared tool resolver


def _executable(path: Path) -> Path:
    import stat

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_resolve_prefers_the_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chosen = _executable(tmp_path / "chosen" / "widget")
    monkeypatch.setenv("WIDGET", str(chosen))
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    _executable(tmp_path / "data" / "widget" / "widget")
    assert resolve_tool("widget", "WIDGET", "widget") == str(chosen)


def test_resolve_falls_through_an_env_var_that_names_nothing_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale variable must not hide a copy that is actually there."""
    monkeypatch.setenv("WIDGET", str(tmp_path / "gone" / "widget"))
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    installed = _executable(tmp_path / "data" / "widget" / "widget")
    assert resolve_tool("widget", "WIDGET", "widget") == str(installed)


def test_resolve_finds_the_installed_copy_under_data_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIDGET", raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    installed = _executable(tmp_path / "data" / "widget" / "bin" / "widget")
    assert resolve_tool("widget", "WIDGET", "widget", "bin/widget") == str(installed)


def test_resolve_ignores_a_directory_and_a_non_executable_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIDGET", raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    (tmp_path / "data" / "widget" / "widget").mkdir(parents=True)
    assert resolve_tool("widget", "WIDGET", "widget") is None

    plain = tmp_path / "data" / "other" / "other"
    plain.parent.mkdir(parents=True, exist_ok=True)
    plain.write_text("text", encoding="utf-8")
    plain.chmod(0o644)
    assert resolve_tool("other", "OTHER", "other") is None


def test_resolve_returns_none_when_nothing_anywhere_has_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIDGET", raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    assert resolve_tool("widget-that-does-not-exist", "WIDGET", "widget") is None
