"""Bounded MCP access to artifacts produced inside one approved finished run."""

import base64
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from tests.helpers.disease import load_mcp_server as _server


def _run(root: Path) -> Path:
    run = root / "run"
    (run / "inference_samples" / "binder_0").mkdir(parents=True)
    return run


def _validated(tool: str, value: dict[str, object]) -> dict[str, object]:
    schema = {str(item["name"]): item["outputSchema"] for item in _server()._TOOLS}[tool]
    jsonschema.validate(value, schema)
    return value


def test_output_files_are_listed_and_read_in_bounded_chunks(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease.mcp_api import call_tool

    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path)
    artifact = run / "inference_samples" / "binder_0" / "binder_0_generated.cif"
    artifact.write_text("abcdefghij", encoding="utf-8")

    listed = _validated("list_output_files", call_tool("list_output_files", {"directory": str(run)}))
    relative = artifact.relative_to(run).as_posix()
    assert listed["files"] == [{"path": relative, "bytes": 10, "media_type": "chemical/x-structure"}]
    first = _validated(
        "read_output_file",
        call_tool("read_output_file", {"directory": str(run), "path": relative, "max_bytes": 4}),
    )
    assert first["content"] == "abcd"
    assert first["encoding"] == "utf-8"
    assert first["next_offset"] == 4
    second = call_tool(
        "read_output_file",
        {"directory": str(run), "path": relative, "offset": first["next_offset"], "max_bytes": 20},
    )
    assert second["content"] == "efghij"
    assert second["next_offset"] is None


def test_paged_output_verifies_the_whole_file_across_revisions(tmp_path, monkeypatch):
    from rfproteina.disease.mcp_api import call_tool

    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path / "samples")
    report = run / "result.tsv"
    original = "unit\tµM\n".encode()
    report.write_bytes(original)
    args = {"directory": str(run), "path": report.name}
    first = call_tool("read_output_file", {**args, "max_bytes": 6})
    expected = hashlib.sha256(original).hexdigest()
    assert first["file_sha256"] == expected
    assert base64.b64decode(first["content"]) == original[:6] and first["encoding"] == "base64"
    tail = call_tool("read_output_file", {**args, "offset": first["next_offset"], "expected_sha256": expected})
    assert base64.b64decode(tail["content"]) == original[6:] and tail["next_offset"] is None
    assert tail["file_sha256"] == expected
    report.write_bytes(original.replace(b"M", b"L"))
    with pytest.raises(ValueError, match="changed since the previous read"):
        call_tool("read_output_file", {**args, "offset": first["next_offset"], "expected_sha256": expected})
    report.write_bytes(b"")
    empty = call_tool("read_output_file", args)
    assert empty["bytes_returned"] == 0 and empty["next_offset"] is None
    assert empty["file_sha256"] == empty["chunk_sha256"] == hashlib.sha256(b"").hexdigest()


def test_listing_continuation_detects_an_added_output(tmp_path, monkeypatch):
    from rfproteina.disease import mcp_api

    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path / "samples")
    for name in ("b.csv", "c.csv"):
        (run / name).write_text("score\n")
    args = {"directory": str(run), "limit": 1}
    first = mcp_api.list_output_files(args)
    following = {**args, "offset": first["next_offset"], "expected_sha256": first["inventory_sha256"]}
    assert mcp_api.list_output_files(following)["files"][0]["path"] == "c.csv"
    (run / "a.csv").write_text("score\n")
    with pytest.raises(ValueError, match="inventory has changed"):
        mcp_api.list_output_files(following)
    assert mcp_api.list_output_files(args)["total_files"] == 3
    monkeypatch.setattr(mcp_api, "_OUTPUT_ENTRY_LIMIT", 2)
    with pytest.raises(ValueError, match="too many output files"):
        mcp_api.list_output_files(args)


def test_output_reader_returns_binary_as_base64_and_paginates_inventory(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease.mcp_api import call_tool

    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path)
    binary = run / "inference_samples" / "binder_0" / "binder_0.bcif"
    binary.write_bytes(b"\xff\x00\x01")
    (run / "generation.json").write_text("{}", encoding="utf-8")
    first = call_tool("list_output_files", {"directory": str(run), "limit": 1})
    assert len(first["files"]) == 1 and first["next_offset"] == 1 and first["total_files"] == 2
    second = call_tool("list_output_files", {"directory": str(run), "offset": first["next_offset"], "limit": 1})
    assert len(second["files"]) == 1 and second["next_offset"] is None
    read = call_tool(
        "read_output_file",
        {"directory": str(run), "path": binary.relative_to(run).as_posix()},
    )
    assert read["encoding"] == "base64"
    assert base64.b64decode(read["content"]) == b"\xff\x00\x01"


def test_output_access_rejects_links_traversal_hidden_secrets_and_secret_content(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease.mcp_api import call_tool

    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (run / "linked.txt").symlink_to(outside)
    (run / ".env").write_text("PASSWORD=nope", encoding="utf-8")
    (run / "api_keys.json").write_text("{}", encoding="utf-8")
    (run / "id_ed25519").write_text("not a real key", encoding="utf-8")
    (run / "putty.ppk").write_text("not a real key", encoding="utf-8")
    credential = run / "result.txt"
    authorization = "Bearer " + "x" * 16
    credential.write_text(f"Authorization: {authorization}", encoding="utf-8")

    paths = {item["path"] for item in call_tool("list_output_files", {"directory": str(run)})["files"]}
    assert "linked.txt" not in paths
    assert ".env" not in paths
    assert "api_keys.json" not in paths
    assert "id_ed25519" not in paths
    assert "putty.ppk" not in paths
    with pytest.raises(ValueError, match="inside"):
        call_tool("read_output_file", {"directory": str(run), "path": "../outside.txt"})
    with pytest.raises(ValueError, match="regular file"):
        call_tool("read_output_file", {"directory": str(run), "path": "linked.txt"})
    with pytest.raises(ValueError, match="credential"):
        call_tool("read_output_file", {"directory": str(run), "path": "result.txt"})
    with pytest.raises(ValueError, match="credential"):
        call_tool(
            "read_output_file",
            {"directory": str(run), "path": "result.txt", "offset": 25, "max_bytes": 1},
        )


@pytest.mark.parametrize(
    "content",
    [
        "prefix password=correct-horse-battery-staple suffix",
        "-----BEGIN " + "OPENSSH PRIVATE KEY-----\nnot-a-real-key",
    ],
)
def test_output_reader_classifies_the_whole_file_before_returning_a_slice(
    tmp_path: Path, monkeypatch, content: str
) -> None:
    from rfproteina.disease.mcp_api import call_tool

    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path)
    artifact = run / "result.txt"
    artifact.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="credential"):
        call_tool(
            "read_output_file",
            {"directory": str(run), "path": "result.txt", "offset": len(content) - 1, "max_bytes": 1},
        )


def test_output_reader_refuses_files_too_large_to_classify_safely(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease import mcp_api

    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path)
    artifact = run / "large.bin"
    with artifact.open("wb") as stream:
        stream.truncate(mcp_api._OUTPUT_CLASSIFY_MAX + 1)
    with pytest.raises(ValueError, match="too large"):
        mcp_api.call_tool("read_output_file", {"directory": str(run), "path": "large.bin", "max_bytes": 1})


def test_output_access_requires_an_approved_concrete_run(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease.mcp_api import call_tool

    run = _run(tmp_path / "outside")
    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path / "approved"))
    with pytest.raises(ValueError, match="outside"):
        call_tool("list_output_files", {"directory": str(run)})


def test_verdict_reader_does_not_follow_a_screen_symlink_outside_the_run(tmp_path: Path, monkeypatch) -> None:
    from rfproteina.disease.mcp_api import call_tool

    monkeypatch.setenv("RFD4_MCP_ROOTS", str(tmp_path))
    run = _run(tmp_path)
    outside = tmp_path / "outside-screen.json"
    outside.write_text(json.dumps({"call": "screened_clear", "secret_detail": "outside"}), encoding="utf-8")
    (run / "screen.json").symlink_to(outside)
    result = call_tool("read_verdict", {"directory": str(run), "include_record": True})
    assert len(result["runs"]) == 1
    assert "regular file" in result["runs"][0]["error"]
    assert "verdict" not in result["runs"][0]
