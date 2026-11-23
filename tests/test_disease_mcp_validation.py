"""The MCP validation tools expose scoped checks with no browser or model dispatch."""

import json
import os
from pathlib import Path
from uuid import UUID

import pytest

from rfproteina.disease import mcp_api, validation_checks


def _run(path: Path) -> Path:
    directory = path / "inference_samples"
    directory.mkdir(parents=True)
    (directory / "binder_0_generated.cif").write_text("data_example\n", encoding="utf-8")
    return path


def test_validation_plan_and_execution_use_the_current_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")
    (first / validation_checks.VALIDATION_FILE).write_text('{"binder_0":{"status":"sample note"}}', encoding="utf-8")
    plan = mcp_api.call_tool("plan_validation", {"directory": str(first)})
    candidate = plan["candidates"][0]["id"]
    focus = {"directory": str(first), "candidate_id": candidate}
    checks = mcp_api.call_tool("plan_validation", focus)["checks"]
    assert {item["id"] for item in checks if item["recommended"]} == {"structure_files", "experimental_record"}
    result = mcp_api.call_tool("validate_output_files", {**focus, "checks": ["structure_files", "experimental_record"]})
    assert result["results"][0]["result"] == "Run — 1 of 1 files readable as UTF-8."
    assert "bench record contains a status" in result["results"][1]["result"]
    assert "sample note" not in str(result)
    with pytest.raises(ValueError, match="candidate"):
        mcp_api.call_tool("validate_output_files", {
            "directory": str(second), "candidate_id": candidate, "checks": ["structure_files"],
        })


def test_validation_tools_refuse_outside_roots_and_parent_directories(tmp_path: Path, monkeypatch) -> None:
    checkout = tmp_path / "checkout"
    permitted = checkout / "runs"
    _run(permitted / "inside")
    outside = _run(tmp_path / "outside")
    (permitted / "linked").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(mcp_api, "_repo_root", lambda: checkout)
    monkeypatch.delenv("RFD4_MCP_ROOTS", raising=False)
    for path in (outside, permitted / "linked"):
        with pytest.raises(ValueError, match="RFD4_MCP_ROOTS"):
            mcp_api.call_tool("plan_validation", {"directory": str(path)})
    with pytest.raises(ValueError, match="concrete finished run"):
        mcp_api.call_tool("plan_validation", {"directory": str(permitted)})


def test_validation_requires_explicit_checks_within_the_read_budget(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    path = _run(tmp_path / "run")
    arguments = {"directory": str(path)}
    for keys in ([], ["unknown"], ["structure_files", "structure_files"]):
        with pytest.raises(ValueError, match="check IDs"):
            mcp_api.call_tool("validate_output_files", {**arguments, "checks": keys})
    monkeypatch.setattr(mcp_api, "_VALIDATION_BYTE_BUDGET", 1)

    def should_not_read(_path: Path) -> str:
        pytest.fail("Validation opened a structure after rejecting its workload.")

    monkeypatch.setattr(validation_checks, "_design_text", should_not_read)
    with pytest.raises(ValueError, match="read budget"):
        mcp_api.call_tool("validate_output_files", {**arguments, "checks": ["structure_files"]})


def test_mcp_run_rejects_the_browser_validation_flag() -> None:
    assert "validate_output" not in mcp_api.options_schema()["properties"]
    with pytest.raises(ValueError, match="browser panel"):
        mcp_api.run_program({"options": {"validate_output": ["runs/demo"]}})


def test_save_validation_report_creates_distinct_local_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path / "run")
    (run / validation_checks.VALIDATION_FILE).write_text('{"binder_0":{"status":"private note"}}', encoding="utf-8")
    plan = mcp_api.call_tool("plan_validation", {"directory": str(run)})
    args = {
        "directory": str(run),
        "candidate_id": plan["candidates"][0]["id"],
        "checks": ["structure_files", "experimental_record"],
    }
    checked = mcp_api.call_tool("validate_output_files", args)
    assert not (run / "validation_reports").exists()
    assert mcp_api.call_tool("list_validation_reports", {"directory": str(run)})["report_files"] == []
    first = mcp_api.call_tool("save_validation_report", args)
    first_path = run / first["report_file"]
    original = first_path.read_bytes()
    second = mcp_api.call_tool("save_validation_report", args)
    second_path = run / second["report_file"]

    assert first_path.parent == second_path.parent == run / "validation_reports"
    assert first_path != second_path
    assert first_path.read_bytes() == original
    assert set(first_path.parent.iterdir()) == {first_path, second_path}
    contents = json.loads(original)
    assert contents["schema_version"] == 1
    assert contents["candidate_id"] == args["candidate_id"]
    assert contents["results"] == checked["results"] == first["results"] == second["results"]
    assert b"private note" not in original and str(run).encode() not in original
    (first_path.parent / "validation-11111111111111111111111111111111.json").symlink_to(first_path)
    (first_path.parent / ".validation-staging.tmp").touch()
    listed = mcp_api.call_tool("list_validation_reports", {"directory": str(run)})
    assert listed["report_files"] == sorted([first["report_file"], second["report_file"]])


def test_save_validation_report_refuses_cross_run_candidates_and_linked_destinations(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")
    plan = mcp_api.call_tool("plan_validation", {"directory": str(first)})
    with pytest.raises(ValueError, match="candidate"):
        mcp_api.call_tool("save_validation_report", {
            "directory": str(second), "candidate_id": plan["candidates"][0]["id"], "checks": ["structure_files"],
        })
    assert not (second / "validation_reports").exists()

    other = tmp_path / "other"
    other.mkdir()
    (second / "validation_reports").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="inside the run"):
        mcp_api.call_tool("save_validation_report", {"directory": str(second), "checks": ["structure_files"]})
    with pytest.raises(ValueError, match="inside the run"):
        mcp_api.call_tool("list_validation_reports", {"directory": str(second)})
    assert not list(other.iterdir())


def test_save_validation_report_anchors_publication_to_the_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path / "run")
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open = os.open

    def swap_report_directory(path, *args, **kwargs):
        if path == mcp_api._VALIDATION_REPORT_DIR and (run / "validation_reports").is_dir():
            (run / "validation_reports").rename(run / "moved_reports")
            (run / "validation_reports").symlink_to(outside, target_is_directory=True)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_report_directory)
    with pytest.raises(ValueError, match="could not be saved"):
        mcp_api.call_tool("save_validation_report", {"directory": str(run), "checks": ["structure_files"]})
    assert not list(outside.iterdir())


def test_save_validation_report_detects_a_folder_moved_after_open(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path / "run")
    real_open_directory = mcp_api._open_directory

    def move_after_open(path, *, dir_fd=None):
        descriptor = real_open_directory(path, dir_fd=dir_fd)
        if path == mcp_api._VALIDATION_REPORT_DIR:
            (run / "validation_reports").rename(run / "moved_reports")
            (run / "validation_reports").mkdir()
        return descriptor

    monkeypatch.setattr(mcp_api, "_open_directory", move_after_open)
    with pytest.raises(ValueError, match="directory changed"):
        mcp_api.call_tool("save_validation_report", {"directory": str(run), "checks": ["structure_files"]})
    assert not list((run / "moved_reports").iterdir())
    assert not list((run / "validation_reports").iterdir())


def test_save_validation_report_detects_a_replaced_staging_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path / "run")
    real_link = os.link
    real_open_directory = mcp_api._open_directory

    def replace_before_link(source, destination, *, src_dir_fd, dst_dir_fd, follow_symlinks):
        os.unlink(source, dir_fd=src_dir_fd)
        replacement = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=src_dir_fd,
        )
        try:
            os.write(replacement, b"replacement")
        finally:
            os.close(replacement)
        return real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    def arm_replacement_after_open(path, *, dir_fd=None):
        descriptor = real_open_directory(path, dir_fd=dir_fd)
        if path == mcp_api._VALIDATION_REPORT_DIR:
            monkeypatch.setattr(os, "link", replace_before_link)
        return descriptor

    monkeypatch.setattr(mcp_api, "_open_directory", arm_replacement_after_open)
    with pytest.raises(ValueError, match="changed before publication"):
        mcp_api.call_tool("save_validation_report", {"directory": str(run), "checks": ["structure_files"]})
    assert not list((run / "validation_reports").iterdir())


def test_save_validation_report_cannot_replace_an_existing_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path / "run")
    fixed_id = UUID(int=0)
    monkeypatch.setattr(mcp_api, "uuid4", lambda: fixed_id)
    args = {"directory": str(run), "checks": ["structure_files"]}
    saved = mcp_api.call_tool("save_validation_report", args)
    path = run / saved["report_file"]
    original = path.read_bytes()
    with pytest.raises(ValueError, match="could not be saved"):
        mcp_api.call_tool("save_validation_report", args)
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


def test_validation_can_resolve_recommended_or_all_installed_checks_atomically(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path / "run")
    plan = mcp_api.call_tool("plan_validation", {"directory": str(run)})

    recommended = {item["id"] for item in plan["checks"] if item["installed"] and item["recommended"]}
    selected = mcp_api.call_tool(
        "validate_output_files", {"directory": str(run), "selection": "recommended"}
    )
    assert {item["id"] for item in selected["results"]} == recommended

    installed = {item["id"] for item in plan["checks"] if item["installed"]}
    all_selected = mcp_api.call_tool(
        "validate_output_files", {"directory": str(run), "selection": "all_installed"}
    )
    assert {item["id"] for item in all_selected["results"]} == installed


@pytest.mark.parametrize("selection", [None, [], {}, 7])
def test_selection_must_be_a_named_mode(tmp_path, monkeypatch, selection) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path / "run")
    with pytest.raises(ValueError, match="selection must be"):
        mcp_api.call_tool("validate_output_files", {"directory": str(run), "selection": selection})


def test_validation_selection_and_explicit_checks_are_mutually_exclusive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path / "run")
    with pytest.raises(ValueError, match="not both"):
        mcp_api.call_tool(
            "validate_output_files",
            {
                "directory": str(run),
                "checks": ["structure_files"],
                "selection": "recommended",
            },
        )


def test_list_validation_reports_rejects_an_oversized_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    monkeypatch.setattr(mcp_api, "_VALIDATION_REPORT_ENTRY_LIMIT", 1)
    run = _run(tmp_path / "run")
    folder = run / "validation_reports"
    folder.mkdir()
    (folder / "first.tmp").touch()
    (folder / "second.tmp").touch()
    with pytest.raises(ValueError, match="Too many entries"):
        mcp_api.call_tool("list_validation_reports", {"directory": str(run)})


@pytest.mark.skipif(os.scandir not in os.supports_fd, reason="Listing uses directory descriptors on this platform.")
def test_list_validation_reports_rejects_a_directory_swapped_before_open(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path / "run")
    folder = run / "validation_reports"
    folder.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open = os.open

    def swap_on_open(path, *args, **kwargs):
        if path == folder:
            folder.rename(run / "moved")
            folder.symlink_to(outside, target_is_directory=True)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_on_open)
    with pytest.raises(ValueError, match="could not be listed"):
        mcp_api.call_tool("list_validation_reports", {"directory": str(run)})


def test_receipt_link_does_not_qualify_an_mcp_validation_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = tmp_path / "run"
    run.mkdir()
    target = tmp_path / "attachment.json"
    target.touch()
    (run / validation_checks.SYNTHESIS_REVIEW_RECEIPT).symlink_to(target)
    with pytest.raises(ValueError, match="concrete finished run"):
        mcp_api.call_tool("plan_validation", {"directory": str(run)})
