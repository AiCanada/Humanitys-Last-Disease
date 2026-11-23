"""Humanitys-Last-Disease is an MCP server. These tests do not start the browser or a shell."""

import io
import json
import os
import time
from pathlib import Path

import jsonschema
import pytest
import tomllib

from tests.helpers.disease import UBIQUITIN, exchange_gated, exchange_with, ok_handler
from tests.helpers.disease import load_mcp_server as _server

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "humanitys_last_disease_mcp.py"
#: Spelled out here rather than imported, so the protocol tests do not pay the
#: package import that ``mcp_api.TOOL_NAMES`` would cost them.
_TOOL_NAMES = (
    "list_options",
    "run",
    "screen_directory",
    "read_verdict",
    "describe_architecture",
    "retrieve_databases",
    "retrieve_structures",
    "read_sequence_measures",
    "read_results",
    "list_output_files",
    "read_output_file",
    "backend_status",
    "declare_corpora",
    "plan_validation",
    "validate_output_files",
    "save_validation_report",
    "list_validation_reports",
    "plan_screening",
    "run_selected_screens",
)


def _exchange(messages, handler):
    return exchange_with(_server(), messages, handler)


def test_the_server_names_itself_and_lists_its_tools():
    """An agent discovers the tools from the protocol, not from a help printout."""
    _raw, replies = _exchange(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 0, "method": "ping"},
        ],
        ok_handler,
    )
    assert [reply["id"] for reply in replies] == [1, 2, 0]
    started = replies[0]["result"]
    assert started["serverInfo"]["name"] == "Humanitys-Last-Disease"
    assert started["protocolVersion"] == "2025-06-18"
    assert "Do not shell out" in started["instructions"]
    names = [tool["name"] for tool in replies[1]["result"]["tools"]]
    assert names[:4] == ["list_options", "run", "screen_directory", "read_verdict"]
    assert names == list(_TOOL_NAMES)
    assert replies[2]["result"] == {}


def test_program_text_stays_out_of_the_protocol_stream(capsys):
    """A print from the program must not be framed as an MCP message.

    The server now holds the client's stdout and points ``sys.stdout`` at
    stderr for the life of the loop, so the text cannot reach stdout at all
    rather than merely being written somewhere else first. That matters because
    importing the program's package prints a GPU line to stdout, and the
    import happens on the first tool call, inside this loop.
    """

    def noisy(_name, _arguments):
        print("Screen leaked into the protocol")
        return {"ok": True}

    raw, replies = _exchange(
        [{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "read_verdict", "arguments": {}}}],
        noisy,
    )
    captured = capsys.readouterr()
    assert b"leaked" not in raw
    assert replies[0]["result"]["isError"] is False
    assert "leaked" not in captured.out, "a print must not be able to reach the protocol stream"
    assert "leaked" in captured.err, "it still has to be readable as a diagnostic"


def test_an_import_banner_on_stdout_cannot_corrupt_the_stream(capsys):
    """The real failure this guards: ``import rfproteina`` prints to stdout.

    ``src/rfproteina/__init__.py`` imports torch, and that import writes a GPU
    line to stdout. Both ``list_options`` and ``_call_main`` import the package
    outside any redirect, so on a fresh process the first tool call used to put
    a line that is not an MCP message into the stream the client is parsing.
    The suite could not see it because importing the test module imports the
    package first, leaving the import inside the call a no-op.
    """

    def importer(_name, _arguments):
        print("GPU information: NVIDIA RTX A6000, 8, 6, 45, 84, 300, 2100")
        return {"ok": True}

    raw, replies = _exchange(
        [{"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {"name": "read_verdict", "arguments": {}}}],
        importer,
    )
    for line in raw.splitlines():
        if line.strip():
            json.loads(line)
    assert b"GPU information" not in raw
    assert "GPU information" in capsys.readouterr().err
    assert replies[0]["result"]["isError"] is False


def test_an_unknown_tool_is_a_protocol_error():
    """A name no tool has is the request's fault, so it is a JSON-RPC error and not a tool result.

    It used to come back as a result with ``isError``, which is how a tool that
    ran and failed is reported. The specification lists an unknown tool under
    protocol errors and gives -32602 as its example, and the conformance checks
    count the result form as a failure.
    """
    _raw, replies = _exchange(
        [{"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "human", "arguments": {}}}],
        ok_handler,
    )
    assert "result" not in replies[0]
    assert replies[0]["error"] == {"code": -32602, "message": "Unknown tool: human"}


def test_a_content_length_frame_is_accepted():
    server = _server()
    body = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"}).encode("utf-8")
    frame = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    outgoing = io.BytesIO()
    assert server.serve(stdin=io.BytesIO(frame), stdout=outgoing, handler=ok_handler) == 0
    reply = json.loads(outgoing.getvalue().decode("ascii").strip())
    assert reply["id"] == 7
    assert reply["result"] == {}


def test_a_notification_is_never_answered():
    """JSON-RPC forbids a reply to a message with no id, whatever it asks for.

    Only ``notifications/`` prefixed methods were being skipped, so a bare
    ``ping`` or an unknown method sent as a notification drew a response with
    ``"id": null`` -- a reply to something the client is not waiting for.
    """
    _raw, replies = _exchange(
        [
            {"jsonrpc": "2.0", "method": "ping"},
            {"jsonrpc": "2.0", "method": "nope/there"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 5, "method": "ping"},
        ],
        ok_handler,
    )
    assert [reply["id"] for reply in replies] == [5], f"only the request may be answered, got {replies}"


def test_unreadable_json_is_a_parse_error_and_not_an_invalid_request():
    """A frame that is not JSON and a frame that is JSON but not a request differ."""
    server = _server()
    outgoing = io.BytesIO()
    raw = b"{ this is not json }\n" + json.dumps({"jsonrpc": "2.0", "id": 6}).encode() + b"\n"
    server.serve(stdin=io.BytesIO(raw), stdout=outgoing, handler=ok_handler)
    replies = [json.loads(line) for line in outgoing.getvalue().splitlines() if line.strip()]
    assert replies[0]["error"]["code"] == -32700
    assert replies[1]["error"]["code"] == -32600


def test_a_version_this_server_does_not_speak_is_refused_with_the_list_it_does():
    """The modern era rejects per request, and names what would work."""
    _raw, replies = _exchange(
        [
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/list",
                "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "1900-01-01"}},
            }
        ],
        ok_handler,
    )
    error = replies[0]["error"]
    assert error["code"] == -32022
    assert error["data"]["requested"] == "1900-01-01"
    assert "2026-07-28" in error["data"]["supported"]


def test_the_handshake_offers_the_newest_era_it_speaks_not_the_oldest():
    """An unsupported handshake version got answered with 2024-11-05.

    The rule is to answer with the latest the server supports. Answering with
    the oldest negotiated every current client down to a revision that has no
    structured tool results, which is the whole point of this server.
    """
    _raw, replies = _exchange(
        [
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "initialize",
                "params": {"protocolVersion": "1.0.0", "capabilities": {}},
            }
        ],
        ok_handler,
    )
    server = _server()
    assert replies[0]["result"]["protocolVersion"] == server.LEGACY_VERSIONS[0]
    assert replies[0]["result"]["protocolVersion"] != server.LEGACY_VERSIONS[-1]
    assert "2024-11-05" in server.LEGACY_VERSIONS, "the old revision is still spoken, just not offered first"


def test_a_modern_request_needs_no_handshake_and_is_marked_complete():
    """Revision 2026-07-28 removed initialize. A tool call arrives cold."""
    _raw, replies = _exchange(
        [
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "read_verdict",
                    "arguments": {"directory": "."},
                    "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"},
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/list",
                "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}},
            },
        ],
        ok_handler,
    )
    # Correlated by id, not by position: a tool call is answered by the worker
    # thread while the read loop goes straight on to the next message, so the
    # list is answered first. JSON-RPC never promised an order.
    by_id = {reply["id"]: reply["result"] for reply in replies}
    called, listed = by_id[10], by_id[12]
    assert called["resultType"] == "complete"
    assert called["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "Humanitys-Last-Disease"
    assert listed["ttlMs"] > 0 and listed["cacheScope"] in {"public", "private"}
    assert [tool["name"] for tool in listed["tools"]] == list(_TOOL_NAMES)


def test_a_handshake_result_carries_none_of_the_modern_fields():
    """Those field names postdate the handshake revisions."""
    _raw, replies = _exchange([{"jsonrpc": "2.0", "id": 13, "method": "tools/list"}], ok_handler)
    listed = replies[0]["result"]
    assert "resultType" not in listed
    assert "ttlMs" not in listed
    assert "_meta" not in listed


def test_discover_is_answered_so_a_dual_era_client_does_not_fall_back():
    """A client probes with server/discover and reads the error to pick an era.

    Answering ``-32601`` to the probe is exactly what tells a dual-era client
    this is a handshake-only server, and a modern-only client then fails
    outright.
    """
    _raw, replies = _exchange(
        [
            {
                "jsonrpc": "2.0",
                "id": 14,
                "method": "server/discover",
                "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}},
            }
        ],
        ok_handler,
    )
    found = replies[0]["result"]
    assert found["resultType"] == "complete"
    assert found["supportedVersions"][0] == "2026-07-28"
    assert "2025-06-18" in found["supportedVersions"]
    assert found["capabilities"]["tools"] == {"listChanged": False}
    assert found["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "Humanitys-Last-Disease"
    assert "Do not shell out" in found["instructions"]


def test_every_tool_declares_the_shape_of_its_structured_result():
    """The verdict is read from the schema, not guessed from the JSON text."""
    from rfproteina.disease import mcp_api

    server = _server()
    by_name = {str(tool["name"]): tool for tool in server._TOOLS}
    assert set(by_name) == set(_TOOL_NAMES)
    for name, tool in by_name.items():
        assert tool["outputSchema"]["type"] == "object", f"{name} declares no output shape"
    jsonschema.validate(mcp_api.list_options(), by_name["list_options"]["outputSchema"])
    verdict = by_name["read_verdict"]["outputSchema"]["properties"]["runs"]["items"]["properties"]["verdict"]
    fields = set(verdict["properties"])
    assert set(mcp_api._VERDICT_FIELDS) <= fields, "the schema has drifted from what _verdict returns"
    axis = verdict["properties"]["sequence_liability"]["additionalProperties"]["properties"]
    assert set(mcp_api._AXIS_FIELDS) == set(axis)
    assert by_name["screen_directory"]["outputSchema"]["required"] == [
        "program",
        "directory",
        "complete",
        "runs",
        "exit_code",
        "messages",
    ]
    assert "exit_code" not in by_name["read_verdict"]["outputSchema"]["properties"]


def test_the_options_schema_covers_every_option_and_offers_no_key():
    """An agent selects any option from the schema and the client can validate it."""
    import re

    from rfproteina.disease.cli import build_parser
    from rfproteina.disease.mcp_api import options_schema

    schema = options_schema()
    properties = schema["properties"]
    selectable = {
        action.dest
        for action in build_parser()._actions
        if action.option_strings
        and action.dest not in {"help", "ui", "validate_output"}
        and not re.search(r"(?i)key|token|secret|password", action.dest)
    }
    assert set(properties) == selectable
    assert schema["additionalProperties"] is False
    assert properties["run"]["type"] == "boolean"
    assert properties["research"]["type"] == "array"
    assert properties["research"]["items"] == {"type": "string"}
    assert "protein" in properties["binder_kind"]["enum"]
    assert all(not re.search(r"(?i)key|token|secret", name) for name in properties)


def test_redaction_keeps_prose_and_still_removes_a_peptide():
    """Nine letters in any case is not a sequence test, it is most long words.

    The twenty residue codes omit only B, J, O, U, X and Z, so the old rule
    rewrote "difference", "clearance" and "percentage" as omitted sequences.
    Across the stored screens it matched twenty-one runs, all of them prose.
    """
    from rfproteina.disease.mcp_api import _redact

    prose = "The difference between the reference clearance and the measured percentage."
    assert _redact(prose) == prose
    assert _redact("reference_proteins_pooled names the pooled proteins.").startswith("reference_proteins")
    assert _redact("SIINFEKLV") == "[sequence omitted]", "a nine-residue epitope is still a sequence"
    assert _redact(UBIQUITIN) == "[sequence omitted]"
    assert _redact("acdefghiklmnpqrstvwyacdefghik") == "[sequence omitted]", "lower case at twenty still counts"


def test_the_read_loop_keeps_answering_while_a_tool_call_runs():
    """The regression that made a long call unusable.

    Generation runs for hours and the whole server used to sit inside it: no
    ping could be answered and, because a cancellation on stdio is a message
    rather than a closed stream, nothing could ask it to stop. The ping must come
    back first here, which it can only do if the loop never blocked.
    """

    def slow(_name, _arguments):
        time.sleep(0.5)
        return {"ok": True}

    _raw, replies = _exchange(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "read_verdict", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        ],
        slow,
    )
    assert [reply["id"] for reply in replies] == [2, 1], "the ping waited for the tool call"
    assert replies[1]["result"]["isError"] is False


def test_progress_is_reported_as_the_program_prints_it(monkeypatch):
    """A client that asks for progress gets each step while the call is open."""
    server = _server()
    monkeypatch.setattr(server, "_PROGRESS_EVERY_S", 0.0)

    def talkative(_name, _arguments):
        report = server._CURRENT.report
        for step in ("Reading the archive", "Scoring binder_0", "Writing the screen"):
            report(step)
        return {"ok": True}

    _raw, sent = exchange_with(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "screen_directory",
                    "arguments": {"directory": "."},
                    "_meta": {"progressToken": "tok-1"},
                },
            }
        ],
        talkative,
    )
    notices = [item for item in sent if item.get("method") == "notifications/progress"]
    answers = [item for item in sent if "id" in item]
    assert [notice["params"]["message"] for notice in notices] == [
        "Reading the archive",
        "Scoring binder_0",
        "Writing the screen",
    ]
    assert all(notice["params"]["progressToken"] == "tok-1" for notice in notices)
    values = [notice["params"]["progress"] for notice in notices]
    assert values == sorted(values), "progress must rise every time"
    assert len(set(values)) == len(values), "and never repeat a value"
    assert len(answers) == 1, "one response, after the notifications"
    assert sent.index(answers[0]) == len(sent) - 1, "progress must stop once the call is answered"


def test_no_progress_is_sent_when_the_client_did_not_ask_for_it():
    """Progress may only reference a token from an active request."""
    server = _server()

    def talkative(_name, _arguments):
        server._CURRENT.report("Reading the archive")
        return {"ok": True}

    _raw, replies = exchange_with(
        server,
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "read_verdict", "arguments": {}}}],
        talkative,
    )
    assert all("method" not in reply for reply in replies)
    assert replies[0]["id"] == 1


def test_a_cancelled_call_stops_and_is_never_answered():
    """The spec: stop processing, free what was held, send no response."""
    server = _server()
    reached = []

    def interruptible(_name, _arguments):
        report = server._CURRENT.report
        for step in range(50):
            # The pause is what makes this deterministic: the read loop has the
            # cancellation long before the program would have finished.
            time.sleep(0.02)
            report(f"step {step}")
            reached.append(step)
        return {"ok": True}

    _raw, replies = exchange_with(
        server,
        [
            {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "read_verdict", "arguments": {}}},
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": 7, "reason": "no longer needed"},
            },
            {"jsonrpc": "2.0", "id": 8, "method": "ping"},
        ],
        interruptible,
    )
    assert [reply["id"] for reply in replies] == [8], f"the cancelled call was answered: {replies}"
    assert len(reached) < 50, "the program was left to run to the end"


def test_an_unknown_cancellation_is_ignored_rather_than_answered():
    """A cancellation can arrive after the work finished. That is not an error."""
    _raw, replies = _exchange(
        [
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 999}},
            {"jsonrpc": "2.0", "method": "notifications/cancelled"},
            {"jsonrpc": "2.0", "id": 3, "method": "ping"},
        ],
        ok_handler,
    )
    assert [reply["id"] for reply in replies] == [3]


def test_a_progress_token_is_a_string_or_an_integer_and_nothing_else():
    """Echoing a float back made this server the one sending a bad token."""
    server = _server()
    assert server._progress_token({"_meta": {"progressToken": 42}}) == 42
    assert server._progress_token({"_meta": {"progressToken": "a"}}) == "a"
    assert server._progress_token({"_meta": {"progressToken": 2.5}}) is None
    assert server._progress_token({"_meta": {"progressToken": True}}) is None
    assert server._progress_token({"_meta": {}}) is None
    assert server._progress_token(None) is None


def test_a_child_process_cannot_reach_the_protocol_stream(tmp_path):
    """The hazard redirect_stdout does not cover: an inherited file descriptor.

    For a real generation the program starts rfd4-generate without redirecting
    it, and that child inherits descriptor 1. Under the old code descriptor 1
    was the client's pipe, so the child's Hydra log and resolved config -- with
    absolute checkpoint and output paths, never scrubbed -- went into the
    protocol stream, and a write without a trailing newline landed inside a
    frame rather than between two.

    Run in a subprocess because it moves descriptors 0 and 1, which is not
    something to do to the test runner.
    """
    import subprocess
    import sys

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import importlib.util, json, os, subprocess, sys\n"
        f"spec = importlib.util.spec_from_file_location('srv', {str(_SCRIPT)!r})\n"
        "srv = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(srv)\n"
        "channel = srv._take_stdout()\n"
        "srv._take_stdin()\n"
        "subprocess.run([sys.executable, '-c', \"import sys; print('CHILD-ON-FD-1'); \"\n"
        "                'sys.stdout.write(\"NO-NEWLINE\")'], check=False)\n"
        "print('PARENT-VIA-SYS-STDOUT')\n"
        "srv._Writer(channel).send({'jsonrpc': '2.0', 'id': 1, 'result': {'ok': True}})\n",
        encoding="utf-8",
    )
    done = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert "CHILD-ON-FD-1" not in done.stdout, "a child reached the protocol stream"
    assert "NO-NEWLINE" not in done.stdout, "a child could still split a frame"
    assert "PARENT-VIA-SYS-STDOUT" not in done.stdout
    assert "CHILD-ON-FD-1" in done.stderr, "the child's output must still be readable"
    # What is left on the protocol stream is exactly one frame, and it parses.
    frames = [line for line in done.stdout.splitlines() if line.strip()]
    assert len(frames) == 1, f"expected one clean frame, got {frames}"
    assert json.loads(frames[0]) == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}


def test_a_batch_frame_answers_every_request_in_it():
    """A top-level array is a JSON-RPC batch, not one invalid request.

    Rejecting the array answered none of the ids inside it, so each request in
    the batch waited for its own timeout -- while this server advertises a
    revision whose base protocol requires batches to be accepted.
    """
    server = _server()
    outgoing = io.BytesIO()
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "read_verdict", "arguments": {}},
        },
    ]
    raw = json.dumps(batch).encode() + b"\n"
    assert server.serve(stdin=io.BytesIO(raw), stdout=outgoing, handler=ok_handler) == 0
    frames = [json.loads(line) for line in outgoing.getvalue().splitlines() if line.strip()]
    assert len(frames) == 1
    replies = frames[0]
    assert isinstance(replies, list)
    assert {reply["id"] for reply in replies} == {1, 2, 3}, "the notification inside must stay unanswered"


def test_a_call_cancelled_while_queued_is_never_started():
    """The worker takes one call at a time, so a second can be withdrawn first."""
    server = _server()
    started = []

    def slow(_name, _arguments):
        started.append(_arguments.get("directory"))
        time.sleep(0.4)
        return {"ok": True}

    _raw, replies = exchange_with(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "read_verdict", "arguments": {"directory": "first"}},
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "read_verdict", "arguments": {"directory": "second"}},
            },
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 2}},
        ],
        slow,
    )
    assert started == ["first"], f"the withdrawn call was started anyway: {started}"
    assert [reply["id"] for reply in replies] == [1]


def test_a_tool_that_prints_nothing_is_still_not_answered_once_cancelled():
    """read_verdict and list_options get no progress callback, so nothing in
    them can notice a cancellation. It has to be noticed before the response."""
    server = _server()

    def quiet(_name, _arguments):
        time.sleep(0.3)
        return {"ok": True}

    _raw, replies = exchange_with(
        server,
        [
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "list_options", "arguments": {}}},
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 4}},
            {"jsonrpc": "2.0", "id": 5, "method": "ping"},
        ],
        quiet,
    )
    assert [reply["id"] for reply in replies] == [5], f"the cancelled call was answered: {replies}"


def test_a_quiet_call_keeps_reporting_so_the_client_can_hold_its_timeout(monkeypatch):
    """Generation prints that it is starting, then nothing for hours.

    A client may reset its timeout when progress arrives, so repeating the last
    line is the difference between a long call and one that looks dead.
    """
    server = _server()
    monkeypatch.setattr(server, "_HEARTBEAT_S", 0.05)
    monkeypatch.setattr(server, "_PROGRESS_EVERY_S", 0.0)

    def quiet_after_one_line(_name, _arguments):
        server._CURRENT.report("Starting 2 local generation request(s).")
        time.sleep(0.4)
        return {"ok": True}

    _raw, sent = exchange_with(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "run", "arguments": {}, "_meta": {"progressToken": 7}},
            }
        ],
        quiet_after_one_line,
    )
    notices = [item for item in sent if item.get("method") == "notifications/progress"]
    assert len(notices) > 1, f"the call went silent: {notices}"
    assert all(notice["params"]["message"].startswith("Starting 2 local") for notice in notices)
    values = [notice["params"]["progress"] for notice in notices]
    assert values == sorted(values) and len(set(values)) == len(values)
    assert sent[-1].get("id") == 1, "the heartbeat must stop before the response"


def test_a_frame_split_across_pipe_writes_is_read_whole(tmp_path):
    """A BytesIO hands over everything at once. A pipe does not.

    ``read(n)`` on a raw stream returns what has arrived, not what was asked
    for, so a Content-Length frame delivered in more than one piece came back
    truncated and the message was dropped. Every test here used a BytesIO, which
    cannot show that; this one uses a real pipe and delivers the body in three
    pieces.
    """
    import os
    import threading

    server = _server()
    body = json.dumps({"jsonrpc": "2.0", "id": 5, "method": "ping", "params": {"pad": "x" * 4000}}).encode()
    frame = b"Content-Length: %d\r\n\r\n" % len(body) + body
    read_fd, write_fd = os.pipe()

    def feed():
        with os.fdopen(write_fd, "wb", buffering=0) as sink:
            for start in range(0, len(frame), 1500):
                sink.write(frame[start : start + 1500])
                time.sleep(0.05)

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    # Raw on purpose: the reader must not depend on being handed a buffered one.
    with os.fdopen(read_fd, "rb", buffering=0) as incoming:
        frames = list(server._messages(incoming))
    writer.join(timeout=5)
    assert len(frames) == 1, f"the frame was split into {len(frames)} pieces"
    assert frames[0] is not server._UNREADABLE, "the frame was lost"
    assert frames[0]["id"] == 5
    assert len(frames[0]["params"]["pad"]) == 4000


def test_a_frame_that_ends_early_is_a_parse_error_not_a_hang():
    """A Content-Length larger than what arrives must end the stream."""
    server = _server()
    body = b'{"jsonrpc":"2.0","id":6,"method":"ping"}'
    truncated = b"Content-Length: 9999\r\n\r\n" + body
    frames = list(server._messages(io.BytesIO(truncated)))
    assert frames == [server._UNREADABLE]


def test_the_windows_launch_is_the_linux_server_not_a_shell():
    server = _server()
    command = server.wsl_command(Path("."))
    assert command[1:3] == ["--cd", "."]
    assert ".venv/bin/python" in command
    assert "scripts/humanitys_last_disease_mcp.py" in command
    assert "human" not in command
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "shell=True" not in text
    assert "subprocess.run" not in text


def test_the_mcp_files_point_at_this_server_and_carry_no_secret():
    mcp = json.loads((_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    grok = (_ROOT / ".grok" / "config.toml").read_text(encoding="utf-8")
    entry = mcp["mcpServers"]["Humanitys-Last-Disease"]
    assert entry["command"] == "python"
    assert entry["args"] == ["-u", "scripts/humanitys_last_disease_mcp.py"]
    assert "[mcp_servers.Humanitys-Last-Disease]" in grok
    assert 'args = ["-u", "scripts/humanitys_last_disease_mcp.py"]' in grok
    combined = json.dumps(mcp) + grok
    assert "sk-" not in combined
    assert "/home/" not in combined
    assert "Users" not in combined
    configured = tomllib.loads(grok)["mcp_servers"]["Humanitys-Last-Disease"]["tool_timeouts"]
    assert set(configured) == set(_TOOL_NAMES)


def test_options_are_flags_and_not_a_shell_line():
    from rfproteina.disease.mcp_api import argv_from_options

    argv = argv_from_options(
        {
            "disease": "Hay fever",
            "accept_only": True,
            "pick": "98",
            "run": False,
            "confirm_inference": False,
            "research": ["nvidia", "openrouter"],
            "screen": "disease_runs/example",
        }
    )
    assert argv[:2] == ["--disease", "Hay fever"]
    assert "--ui" not in argv
    assert "--run" not in argv
    assert "--confirm-inference" not in argv
    assert "--accept-only" in argv
    assert argv[argv.index("--research") + 1] == "nvidia"
    assert argv.count("--research") == 2
    screen = argv[argv.index("--screen") + 1]
    assert Path(screen).as_posix().endswith("disease_runs/example")
    assert argv.count(screen) == 1
    with pytest.raises(ValueError, match="API keys") as rejected:
        argv_from_options({"api_key": "nope"})
    assert "nope" not in str(rejected.value)
    with pytest.raises(ValueError, match="newline"):
        argv_from_options({"disease": "one\ntwo"})
    with pytest.raises(ValueError, match="Unknown option"):
        argv_from_options({"cure": True})


def test_run_refuses_a_prompt_and_the_browser_without_calling_the_program(monkeypatch):
    from rfproteina.disease import mcp_api

    def refused(_argv):
        raise AssertionError("the program was called")

    monkeypatch.setattr(mcp_api, "_call_main", refused)
    with pytest.raises(ValueError, match="disease"):
        mcp_api.run_program({"options": {}})
    with pytest.raises(ValueError, match="browser"):
        mcp_api.run_program({"options": {"ui": True, "disease": "Asthma", "accept_only": True}})
    with pytest.raises(ValueError, match="accept_only"):
        mcp_api.run_program({"options": {"disease": "Asthma"}})
    with pytest.raises(ValueError, match="API keys"):
        mcp_api.run_program({"options": {"disease": "Asthma", "accept_only": True, "grok_key": "nope"}})


def test_list_options_covers_the_parser_and_hides_keys():
    from rfproteina.disease.cli import build_parser
    from rfproteina.disease.mcp_api import list_options

    catalog = list_options()
    names = {option["name"] for option in catalog["options"]}
    parser_names = {
        action.dest
        for action in build_parser()._actions
        if action.option_strings and action.dest not in {"help", "ui", "validate_output"}
    }
    assert names == parser_names
    by_name = {option["name"]: option for option in catalog["options"]}
    assert "protein" in by_name["binder_kind"]["choices"]
    assert by_name["run"]["kind"] == "boolean"
    assert by_name["research"]["kind"] == "array"
    assert all("key" not in option["name"] for option in catalog["options"])
    # The factual legend, which is what the notes are for. The disclaimer that
    # used to follow it was asked out of this flow.
    notes = " ".join(catalog["notes"])
    assert "screened_clear is every axis in that bar clear" in notes
    assert "not a cure" not in notes


def _finished_run(root: Path) -> Path:
    run = root / "gene_1ABC_1"
    sample = run / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text(
        "data_design\n#\nloop_\n_entity_poly.entity_id\n"
        "_entity_poly.pdbx_strand_id\n_entity_poly.pdbx_seq_one_letter_code_can\n"
        f"1 A {UBIQUITIN}\n#\n",
        encoding="utf-8",
    )
    metrics = run / "inference_metrics"
    metrics.mkdir(parents=True)
    (metrics / "per_example_metrics.csv").write_text(
        "example_id,chain_of_interest\nbinder_0,A\n",
        encoding="utf-8",
    )
    (run / "gene_1ABC_1.job.json").write_text(
        json.dumps({"binder_kind": "protein", "target_id": "P05113", "generation_started": True}),
        encoding="utf-8",
    )
    return run


def test_screen_directory_returns_the_verdict_without_a_shell(tmp_path, monkeypatch, capsys):
    """The verdict is the screen record. The program's printed lines are not the result."""
    import rfproteina.disease.cli as cure
    from rfproteina.disease.mcp_api import read_verdict, screen_directory

    def refused(*_args, **_kwargs):
        raise AssertionError("screen must not start a process")

    monkeypatch.setattr(cure.subprocess, "run", refused)
    monkeypatch.setattr(cure.subprocess, "Popen", refused)
    monkeypatch.delenv("HUMAN_PROTEOME", raising=False)
    # DATA_PATH has to point somewhere empty, or this asserts something about
    # the developer's machine rather than about the code. conftest loads
    # .env.local with override=True, so without this the screen resolves the
    # installed MHCflurry and its allele list and runs a real class I scan,
    # which shells out and trips the refusal above. The assertion is not
    # relaxed by this: the screen still must not start a process, it is just
    # no longer handed a tool to start one with.
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "absent"))
    monkeypatch.setenv("HUMANITYS_LAST_DISEASE_LOG", str(tmp_path / "mcp.log"))
    run = _finished_run(tmp_path)
    result = screen_directory({"directory": str(run), "log_file": str(tmp_path / "mcp.log")})
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Screen " not in captured.err
    assert result["exit_code"] == 0
    assert result["complete"] is True
    assert len(result["runs"]) == 1
    verdict = result["runs"][0]["verdict"]
    assert verdict["call"] == "untested"
    assert isinstance(verdict["states"], dict)
    assert UBIQUITIN not in json.dumps(result)
    stored = (run / "screen.json").read_bytes()
    again = read_verdict({"directory": str(run)})
    assert again["runs"][0]["verdict"]["call"] == "untested"
    assert (run / "screen.json").read_bytes() == stored
    assert "messages" not in again


def test_read_verdict_redacts_a_sequence_and_a_personal_path(tmp_path):
    """A phrase match no longer takes the whole line with it.

    Replacing the entire string on a phrase hit discarded the measurement beside
    it, and the screen does not write cure claims in the first place -- not
    writing one is a promise of the module. Sequences, secrets and personal
    paths are still removed, which is what the returned record can really carry.
    """
    from rfproteina.disease.mcp_api import read_verdict

    run = tmp_path / "gene_1ABC_1"
    run.mkdir()
    peptide = "ACDEFGHIKLMNPQRSTVWYACDEFGHIK"
    private_home = "/".join(("", "home", tmp_path.name))
    (run / "screen.json").write_text(
        json.dumps(
            {
                "call": "untested",
                "bar": "sequence",
                "states": {"immunogenicity": "untested"},
                "untested": ["immunogenicity"],
                "hits": [],
                "note": peptide,
                "measured": "Screened clear on 6 axes. Two sentences, both kept.",
                "where": f"{private_home}/secret",
            }
        ),
        encoding="utf-8",
    )
    before = (run / "screen.json").read_bytes()
    result = read_verdict({"directory": str(run), "include_record": True})
    encoded = json.dumps(result)
    record = result["runs"][0]["verdict"]["record"]
    assert peptide not in encoded
    assert private_home not in encoded
    assert result["runs"][0]["verdict"]["call"] == "untested"
    assert record["note"] == "[sequence omitted]"
    # The whole line used to come back as "[omitted]", measurement and all.
    assert record["measured"] == "Screened clear on 6 axes. Two sentences, both kept."
    assert (run / "screen.json").read_bytes() == before


def test_stopping_work_ends_a_running_generation_child():
    """Generation is silent for hours, so the printed-line check never fires.

    Before this there was no handle on the child at all: a cancellation set a
    flag that nothing in the generation phase ever read, so the run continued to
    the end and the worker stayed occupied. ``subprocess_runner`` now holds the
    child while it runs.
    """
    import subprocess
    import sys
    import threading as thread_module

    from rfproteina.disease import dispatch

    codes = []

    def run_a_long_child():
        codes.append(dispatch.subprocess_runner([sys.executable, "-c", "import time; time.sleep(120)"]))

    worker = thread_module.Thread(target=run_a_long_child, daemon=True)
    worker.start()
    for _ in range(200):
        if dispatch._running:
            break
        time.sleep(0.02)
    assert dispatch._running, "the child was never registered"
    assert dispatch.stop_running() == 1
    worker.join(timeout=30)
    assert not worker.is_alive(), "the child outlived the stop"
    assert codes and codes[0] != 0, f"a terminated child must not look successful: {codes}"
    assert not dispatch._running, "the finished child was not released"
    # Nothing left to stop, and saying so must not raise.
    assert dispatch.stop_running() == 0
    assert isinstance(subprocess.Popen, type)


def test_cancelling_the_running_call_stops_its_child():
    """The cancellation has to land while the call is inside the program.

    Held back until the handler says it has started. Sent any earlier the call
    is still queued, the worker skips it, and there is correctly nothing to
    stop -- a real case, but not the one a long generation needs.
    """
    import threading as thread_module

    server = _server()
    stopped = []
    started = thread_module.Event()
    server._stop_work = lambda: stopped.append(True)

    def quiet(_name, _arguments):
        started.set()
        time.sleep(0.3)
        return {"ok": True}

    replies = exchange_gated(
        server,
        [
            ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "run", "arguments": {}}}, None),
            ({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}}, started),
            ({"jsonrpc": "2.0", "id": 2, "method": "ping"}, None),
        ],
        quiet,
    )
    assert stopped == [True], "the running call's child was not stopped"
    assert [reply["id"] for reply in replies] == [2], "the cancelled call must not be answered"


def test_only_the_running_call_has_its_child_stopped():
    """A withdrawn queued call must not end the work of the one in flight."""
    import threading as thread_module

    server = _server()
    stopped = []
    started = thread_module.Event()
    server._stop_work = lambda: stopped.append(True)

    def quiet(_name, _arguments):
        started.set()
        time.sleep(0.3)
        return {"ok": True}

    replies = exchange_gated(
        server,
        [
            ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "run", "arguments": {}}}, None),
            ({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "run", "arguments": {}}}, None),
            ({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 2}}, started),
        ],
        quiet,
    )
    assert stopped == [], "cancelling a queued call stopped the running call's child"
    assert [reply["id"] for reply in replies] == [1], "only the call that ran may be answered"


def test_the_server_carries_no_cure_disclaimer():
    """Asked out of this flow. The factual legend stays."""
    server = _server()
    text = json.dumps(server._TOOLS) + server._INSTRUCTIONS
    assert "not a cure" not in text
    assert "screened_clear means every axis in the named bar is clear" in text
    assert "Do not shell out" in server._INSTRUCTIONS


def test_program_output_capture_is_bounded_and_marks_truncation():
    from rfproteina.disease import mcp_api

    stream = mcp_api._LineStream(None)
    text = "x" * (mcp_api._CAPTURE_MAX_CHARS + 100)
    assert stream.write(text) == len(text)
    stream.flush_partial()
    assert stream.lines == [mcp_api._CAPTURE_MARKER]
    assert stream.raw == []
    assert stream._partial == ""


def test_an_absurd_content_length_is_a_parse_error_and_does_not_wait():
    """A negative or huge length is not read. The read would block until those bytes arrived."""
    server = _server()
    for header in (b"Content-Length: -1\r\n\r\n", b"Content-Length: 999999999\r\n\r\n"):
        outgoing = io.BytesIO()
        assert server.serve(stdin=io.BytesIO(header), stdout=outgoing, handler=ok_handler) == 0
        replies = [json.loads(line) for line in outgoing.getvalue().splitlines() if line.strip()]
        assert len(replies) == 1
        assert replies[0]["error"]["code"] == -32700


def test_a_relay_without_pipes_stops_the_process(monkeypatch):
    server = _server()
    stopped = {}

    class _Process:
        stdin = None
        stdout = None

        def kill(self):
            stopped["kill"] = True

        def wait(self, timeout=None):
            stopped["wait"] = timeout
            return 1

    monkeypatch.setattr(server.subprocess, "Popen", lambda *_args, **_kwargs: _Process())
    assert server._relay([str(_SCRIPT)]) == 1
    assert stopped == {"kill": True, "wait": 5}


def test_an_unknown_disease_does_not_ask_forever(tmp_path, monkeypatch):
    import rfproteina.disease.cli as cure
    from rfproteina.disease import mcp_api

    monkeypatch.setattr(cure, "search_diseases", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(cure, "match_diseases", lambda *_args, **_kwargs: [])
    with pytest.raises(ValueError, match="not found"):
        mcp_api.run_program(
            {
                "options": {
                    "disease": "NotADisease",
                    "accept_only": True,
                    "log_file": str(tmp_path / "mcp.log"),
                }
            }
        )


def test_a_non_finite_measurement_does_not_hide_the_verdict():
    from rfproteina.disease.mcp_api import _verdict

    view = _verdict(
        {
            "call": "untested",
            "sequence_liability": {
                "immunogenicity": {"state": "untested", "measurement": float("inf"), "blocker": "nan"},
            },
        },
        include_record=False,
    )
    encoded = json.dumps(view, allow_nan=False)
    assert "Infinity" not in encoded
    assert view["call"] == "untested"
    assert view["sequence_liability"]["immunogenicity"]["measurement"] is None


def test_a_missing_directory_is_an_error_and_a_failed_rescreen_is_marked(tmp_path, monkeypatch):
    from rfproteina.disease import mcp_api

    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="not on this computer"):
        mcp_api.screen_directory({"directory": str(missing)})

    run = tmp_path / "gene_1ABC_1"
    run.mkdir()
    screen = run / "screen.json"
    screen.write_text(
        json.dumps({"call": "untested", "bar": "sequence", "states": {}, "untested": [], "hits": []}),
        encoding="utf-8",
    )
    untouched = screen.read_bytes()

    def failed(_argv, on_line=None):
        return 1, ["Screen was not written for gene_1ABC_1: boom"]

    monkeypatch.setattr(mcp_api, "_call_main", failed)
    result = mcp_api.screen_directory({"directory": str(run)})
    assert result["exit_code"] == 1
    assert result["runs"][0]["written"] is False
    assert result["runs"][0]["verdict"]["call"] == "untested"
    assert screen.read_bytes() == untouched

    def replaced(_argv, on_line=None):
        screen.write_text(
            json.dumps({"call": "untested", "bar": "sequence", "states": {"ada": "clear"}, "untested": [], "hits": []}),
            encoding="utf-8",
        )
        return 0, []

    past = time.time() - 10
    os.utime(screen, (past, past))
    monkeypatch.setattr(mcp_api, "_call_main", replaced)
    again = mcp_api.screen_directory({"directory": str(run)})
    assert again["runs"][0]["written"] is True
    assert again["runs"][0]["verdict"]["states"]["ada"] == "clear"
