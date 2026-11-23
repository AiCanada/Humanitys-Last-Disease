"""The MCP server keeps answering when something around a call goes wrong.

Four ways the loop could stop being a server without saying so. An error of its
own while answering killed the one worker, so every later call waited forever.
A closed stream told a running call to stop with a flag it only reads when the
program prints, and generation prints nothing, so its child outlived the server.
A line with no newline was read whole, however long. And a client that went
away mid-reply ended the process with a traceback.

And one answer that was the wrong kind. A call naming no tool, or not shaped
as a call, is the request's own fault and is a JSON-RPC error. It came back as
a tool result with ``isError``, which is how a tool that ran and failed is
reported.

And the entry point on Windows, which passes the client's stream to the server
in WSL, read that stream a full block at a time. A client's first message is
short, so it was never passed on and no session got past ``initialize``.

These tests load the server file alone. None of them imports the program.
"""

import io
import json
import os
import threading

from tests.helpers.disease import exchange_gated, exchange_with, ok_handler
from tests.helpers.disease import load_mcp_server as _server


def test_a_blank_roots_setting_uses_the_checkout_output_directories(monkeypatch):
    server = _server()
    monkeypatch.setenv("RFD4_MCP_ROOTS", "   ")
    monkeypatch.setenv(server._SERVE_ENV, "1")
    monkeypatch.setattr(server, "serve", lambda: 0)
    assert server.main() == 0
    checkout = server._repo().resolve()
    assert os.environ["RFD4_MCP_ROOTS"].split(os.pathsep) == [str(checkout / n) for n in ("runs", "disease_runs")]


def test_launch_uses_the_virtualenv_prefix_when_python_is_a_symlink(tmp_path, monkeypatch):
    server = _server()
    env = tmp_path / ".venv"
    (env / "bin").mkdir(parents=True)
    base_python = tmp_path / "python"
    base_python.touch()
    python = env / "bin" / "python"
    python.symlink_to(base_python)
    monkeypatch.setattr(server.sys, "executable", str(base_python))
    monkeypatch.setattr(server.sys, "prefix", str(tmp_path))
    assert server.venv_python(tmp_path) == python
    monkeypatch.setattr(server.sys, "prefix", str(env))
    assert server.venv_python(tmp_path) is None


def test_windows_launcher_passes_native_and_posix_roots_as_literal_arguments(monkeypatch):
    server = _server()
    monkeypatch.setenv("RFD4_MCP_ROOTS", r"D:\screening runs;E:\assays")
    paths = {r"D:\screening runs": "/data/d/screening runs", r"E:\assays": "/data/e/assays"}
    converted = []

    def translate(args, **kwargs):
        assert args[1:4] == ["--exec", "/usr/bin/wslpath", "-u"] and kwargs["timeout"] > 0
        converted.append(args[-1])
        return paths[args[-1]] + "\n"

    monkeypatch.setattr(server.subprocess, "check_output", translate)
    assert "RFD4_MCP_ROOTS=/data/d/screening runs:/data/e/assays" in server.wsl_command(server._repo())
    monkeypatch.setenv("RFD4_MCP_ROOTS", "/srv/results;/mnt/d/assays")
    assert "RFD4_MCP_ROOTS=/srv/results:/mnt/d/assays" in server.wsl_command(server._repo())
    assert converted == list(paths)


def _call(number: int, name: str = "read_verdict") -> dict:
    return {"jsonrpc": "2.0", "id": number, "method": "tools/call", "params": {"name": name, "arguments": {}}}


def test_the_worker_outlives_an_error_of_the_servers_own():
    """One worker answers every call, so it has to survive a failure that is not the tool's."""
    server = _server()
    decorate = server._decorate
    failed = []

    private_home = "/".join(("", "home", "fixture-user"))

    def once(result, *, modern, cacheable=False):
        if not failed:
            failed.append(True)
            raise RuntimeError(f"a fault while answering, with {private_home}/secret in it")
        return decorate(result, modern=modern, cacheable=cacheable)

    server._decorate = once
    raw, replies = exchange_with(server, [_call(1), _call(2)], ok_handler)
    by_id = {reply["id"]: reply for reply in replies}
    assert by_id[1]["error"] == {"code": -32603, "message": "Internal error."}
    assert by_id[2]["result"]["isError"] is False, "the second call was never answered"
    assert private_home.encode() not in raw


def test_an_outstanding_call_keeps_its_request_id():
    server = _server()
    outgoing = io.BytesIO()
    started = threading.Event()
    release = threading.Event()
    handled = []

    def held(name, arguments):
        handled.append(name)
        started.set()
        assert release.wait(10), "the test did not release the accepted call"
        return ok_handler(name, arguments)

    calls = server._Calls(server._Writer(outgoing), held)
    try:
        assert server._reply_to(_call(1), calls) is None
        assert started.wait(10), "the accepted call did not start"
        assert server._reply_to(_call(1), calls) == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Request ID is already in use."},
        }
        ping = {"jsonrpc": "2.0", "id": 1.0, "method": "ping"}
        assert server._reply_to(ping, calls)["error"]["code"] == -32600
    finally:
        release.set()
        calls.shutdown()

    assert handled == ["read_verdict"]
    replies = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert len(replies) == 1
    assert replies[0]["id"] == 1
    assert replies[0]["result"]["isError"] is False


def test_replies_in_a_batch_reserve_their_ids_until_the_frame_is_sent():
    server = _server()
    ping = {"jsonrpc": "2.0", "id": 3, "method": "ping"}
    wire = io.BytesIO((json.dumps([ping, ping]) + "\n" + json.dumps(ping) + "\n").encode())
    replies = io.BytesIO()
    assert server.serve(stdin=wire, stdout=replies, handler=ok_handler) == 0
    batch, following = (json.loads(line) for line in replies.getvalue().splitlines())
    assert batch[0]["id"] == following["id"] == 3 and batch[0]["result"] == following["result"] == {}
    assert batch[1]["id"] is None and batch[1]["error"]["code"] == -32600


def test_the_call_backlog_is_bounded_while_one_call_runs():
    server = _server()
    server._MAX_OUTSTANDING_CALLS = 1
    outgoing = io.BytesIO()
    started = threading.Event()
    release = threading.Event()
    handled = []

    def held(name, arguments):
        handled.append(name)
        started.set()
        assert release.wait(10), "the test did not release the accepted call"
        return ok_handler(name, arguments)

    calls = server._Calls(server._Writer(outgoing), held)
    try:
        assert server._reply_to(_call(1), calls) is None
        assert started.wait(10), "the accepted call did not start"
        assert server._reply_to(_call(2), calls) == {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32000, "message": "Server is busy."},
        }
    finally:
        release.set()
        calls.shutdown()

    assert handled == ["read_verdict"]
    replies = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert [reply["id"] for reply in replies] == [1]


def test_closing_the_stream_ends_what_a_running_call_started():
    """The flag is read when the program prints. Generation does not print, so its child is ended."""
    server = _server()
    server._SHUTDOWN_GRACE_S = 0.2
    server._SHUTDOWN_CANCEL_S = 10.0
    child_ended = threading.Event()
    stopped = []

    def stop() -> None:
        stopped.append(True)
        child_ended.set()

    server._stop_work = stop

    def generation(_name, _arguments):
        # Silent for as long as its child runs, the way a real generation is.
        assert child_ended.wait(30), "nothing ended the child"
        return {"program": "Humanitys-Last-Disease"}

    _raw, replies = exchange_with(server, [_call(1, "run")], generation)
    assert stopped == [True]
    assert replies == [], "a call the client let go of is not answered"


def test_a_stream_that_closes_with_nothing_running_stops_nothing():
    server = _server()
    stopped = []
    server._stop_work = lambda: stopped.append(True)
    _raw, replies = exchange_with(server, [_call(1)], ok_handler)
    assert stopped == []
    assert replies[0]["result"]["isError"] is False


def test_nothing_is_loaded_to_stop_a_program_that_was_never_started():
    """Asking the program to stop must not be what imports it."""
    import sys

    server = _server()
    loaded = []
    server._api = lambda: loaded.append(True)
    saved = sys.modules.pop("rfproteina.disease.dispatch", None)
    try:
        server._stop_work()
        assert loaded == []
    finally:
        if saved is not None:
            sys.modules["rfproteina.disease.dispatch"] = saved


def test_a_line_with_no_end_is_one_bad_frame_and_the_next_is_answered():
    server = _server()
    server._MAX_FRAME = 4096
    ping = json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping"}).encode("utf-8") + b"\n"
    raw = b"x" * 20_000 + b"\n" + ping
    outgoing = io.BytesIO()
    assert server.serve(stdin=io.BytesIO(raw), stdout=outgoing, handler=ok_handler) == 0
    replies = [json.loads(line) for line in outgoing.getvalue().splitlines() if line.strip()]
    assert replies[0]["error"]["code"] == -32700
    assert replies[0]["id"] is None
    assert replies[1] == {"jsonrpc": "2.0", "id": 9, "result": {}}
    assert len(replies) == 2


def test_nonfinite_numbers_and_oversized_integers_are_bad_frames():
    server = _server()
    bad = [
        b'{"jsonrpc":"2.0","id":NaN,"method":"ping"}\n',
        b'{"jsonrpc":"2.0","id":Infinity,"method":"ping"}\n',
        b'{"jsonrpc":"2.0","id":1e400,"method":"ping"}\n',
        b'{"jsonrpc":"2.0","id":' + b"9" * 5000 + b',"method":"ping"}\n',
    ]
    ping = b'{"jsonrpc":"2.0","id":9,"method":"ping"}\n'
    outgoing = io.BytesIO()
    assert server.serve(stdin=io.BytesIO(b"".join([*bad, ping])), stdout=outgoing, handler=ok_handler) == 0
    replies = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert [reply["error"]["code"] for reply in replies[:-1]] == [-32700] * len(bad)
    assert replies[-1] == {"jsonrpc": "2.0", "id": 9, "result": {}}


def test_a_line_is_never_read_past_the_frame_limit():
    """The reader asks the stream for a bounded line, so the bound holds before any parsing."""
    server = _server()
    server._MAX_FRAME = 1024
    asked = []

    class Counting(io.BytesIO):
        def readline(self, size=-1):
            asked.append(size)
            return super().readline(size)

    stream = Counting(b"y" * 5000 + b"\n")
    assert list(server._messages(stream)) == [server._UNREADABLE]
    assert asked and all(size == 1025 for size in asked)


def test_a_frame_of_exactly_the_limit_is_still_read():
    server = _server()
    body = json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}).encode("utf-8")
    server._MAX_FRAME = len(body)
    outgoing = io.BytesIO()
    assert server.serve(stdin=io.BytesIO(body + b"\n"), stdout=outgoing, handler=ok_handler) == 0
    assert json.loads(outgoing.getvalue())["id"] == 3


def test_content_length_headers_use_bounded_reads_and_preserve_the_next_frame():
    server = _server()
    server._MAX_FRAME_HEADERS = 128
    body = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
    next_frame = b'{"jsonrpc":"2.0","id":2,"method":"ping"}\n'
    first_header = b"Content-Length: %d\r\n" % len(body)
    header = b"Content-Type: application/json\r\n"
    limits = []

    class Bounded(io.BytesIO):
        def readline(self, size=-1):
            assert size > 0
            limits.append(size)
            return super().readline(size)

    stream = Bounded(first_header + header + b"\r\n" + body + next_frame)
    assert [message["id"] for message in server._messages(stream)] == [1, 2]
    remaining = server._MAX_FRAME_HEADERS - len(first_header)
    assert limits[1:3] == [remaining + 1, remaining - len(header) + 1]


def test_content_length_requires_a_separator_within_the_header_budget():
    server = _server()
    body = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
    header = (b"Content-Length: %d\r\n" % len(body)) + b"Content-Type: application/json\r\n"
    server._MAX_FRAME_HEADERS = len(header)

    assert list(server._messages(io.BytesIO(header + b"\r\n" + body))) == [server._UNREADABLE]


def test_a_client_that_went_away_is_not_a_crash():
    """Writing to a closed pipe ends the loop quietly, with the worker shut down."""
    server = _server()

    class Gone(io.BytesIO):
        def write(self, _data):
            raise BrokenPipeError("the client closed its end")

    ping = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode("utf-8") + b"\n"
    assert server.serve(stdin=io.BytesIO(ping + ping), stdout=Gone(), handler=ok_handler) == 0


def test_progress_to_a_client_that_stopped_reading_does_not_fail_the_tool():
    """Progress is sent from inside the program's print. A dead pipe there is not the program's error."""
    server = _server()

    class Deaf:
        def send(self, _message):
            raise BrokenPipeError("nobody is reading")

    progress = server._Progress(Deaf(), "token", threading.Event())
    progress.line("Starting generation")
    progress.heartbeat()


def test_the_windows_relay_passes_on_a_short_message_without_waiting_for_more():
    """A client sends one message and waits for its answer. The relay must not wait for a full block.

    Read with ``read``, a buffered pipe is held until 64 KiB arrive or it closes,
    so a server started on Windows never received ``initialize``. The source here
    is buffered the way ``sys.stdin.buffer`` is, and the pipe is left open.
    """
    server = _server()
    read_end, write_end = os.pipe()
    source = os.fdopen(read_end, "rb")
    arrived = threading.Event()

    class Sink(io.BytesIO):
        def write(self, data):
            arrived.set()
            return super().write(data)

    sink = Sink()
    copier = threading.Thread(target=server._copy, args=(source, sink), kwargs={"close_destination": False})
    copier.daemon = True
    copier.start()
    try:
        os.write(write_end, b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
        assert arrived.wait(5), "the message was held back until the stream closed"
        assert sink.getvalue() == b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n'
    finally:
        os.close(write_end)
        copier.join(5)
        source.close()
    assert not copier.is_alive(), "the copy did not end when the stream closed"


def test_every_tool_says_whether_it_only_reads():
    """Clients can distinguish a local additive write from network-capable tools."""
    server = _server()
    hints = {tool["name"]: tool["annotations"] for tool in server._TOOLS}
    assert set(hints) == {tool["name"] for tool in server._TOOLS}
    writers = {"run", "screen_directory", "save_validation_report"}
    open_world = {"run", "screen_directory"}
    for name, annotation in hints.items():
        assert isinstance(annotation["title"], str) and annotation["title"]
        assert annotation["readOnlyHint"] is (name not in writers), name
        assert annotation["openWorldHint"] is (name in open_world), name
    assert hints["run"]["destructiveHint"] is True
    assert hints["run"]["idempotentHint"] is False
    assert hints["screen_directory"]["idempotentHint"] is False
    assert hints["save_validation_report"]["destructiveHint"] is False
    assert hints["save_validation_report"]["idempotentHint"] is False
    listed = server._reply_to({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, None)
    assert all("annotations" in tool for tool in listed["result"]["tools"])


def test_a_call_that_no_tool_can_be_given_is_a_protocol_error():
    """The request is at fault, so the answer is -32602 and no tool is reached."""
    reached = []

    def tool(name, _arguments):
        reached.append(name)
        return {"program": "Humanitys-Last-Disease"}

    frames = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "no_such_tool", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": ["read_verdict"]},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call"},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": 7, "arguments": {}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "read_verdict", "arguments": [1]}},
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "read_verdict", "arguments": None}},
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "read_verdict"}},
    ]
    _raw, replies = exchange_with(_server(), frames, tool)
    by_id = {reply["id"]: reply for reply in replies}
    assert by_id[1]["error"] == {"code": -32602, "message": "Unknown tool: no_such_tool"}
    for number in (2, 3, 4, 5):
        assert by_id[number]["error"]["code"] == -32602, number
        assert "result" not in by_id[number]
    # No arguments at all, and null for them, are both a call with none.
    assert by_id[6]["result"]["isError"] is False
    assert by_id[7]["result"]["isError"] is False
    assert reached == ["read_verdict", "read_verdict"]


def test_request_ids_have_mcp_identifier_types():
    server = _server()
    for request_id in (None, False, True, [], {}, 0.25, float("nan"), float("inf")):
        reply = server._reply_to({"jsonrpc": "2.0", "id": request_id, "method": "ping"}, None)
        assert reply == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid request."},
        }
    for request_id in (0, -2, 1.0, "1"):
        assert "result" in server._reply_to({"jsonrpc": "2.0", "id": request_id, "method": "ping"}, None)
    assert server._key(1) == server._key(1.0) != server._key("1")


def test_requesting_a_background_task_does_not_run_a_foreground_tool():
    server = _server()
    message = _call(1)
    message["params"]["task"] = {}
    reply = server._reply_to(message, None)
    assert reply["id"] == 1 and reply["error"]["code"] == -32601


def test_what_a_tool_makes_of_its_arguments_stays_a_result_a_model_can_read():
    """A refusal by the tool is the tool's answer. A model reads it and corrects the call."""

    def refusing(_name, _arguments):
        raise ValueError("Name the directory to read.")

    _raw, replies = exchange_with(_server(), [_call(1)], refusing)
    assert "error" not in replies[0]
    assert replies[0]["result"]["isError"] is True
    assert replies[0]["result"]["content"][0]["text"] == "Name the directory to read."


def test_a_tool_refusal_does_not_return_paths_credentials_or_control_lines():
    secret = "sk-" + "a" * 20

    private_home = "/".join(("", "home", "fixture-user"))
    bearer = "bearer-value"
    authorization = f"Bearer {bearer}"
    password = "example-password"

    def refusing(_name, _arguments):
        raise ValueError(
            f"Could not read {private_home}/private/run with {secret}, "
            f"Authorization: {authorization}, and password={password}.\nTry another run."
        )

    raw, replies = exchange_with(_server(), [_call(1)], refusing)
    text = replies[0]["result"]["content"][0]["text"]
    assert replies[0]["result"]["isError"] is True
    assert private_home not in text
    assert secret not in text
    assert bearer not in text
    assert password not in text
    assert "\n" not in text
    assert "<path>" in text
    assert text.count("[redacted]") == 3
    assert all(value not in raw for value in (private_home.encode(), secret.encode(), bearer.encode(), password.encode()))


def test_an_oversized_batch_is_refused_before_any_item_runs():
    server = _server()
    server._MAX_BATCH_ITEMS = 2
    reached = []

    def handler(name, arguments):
        reached.append((name, arguments))
        return ok_handler(name, arguments)

    outgoing = io.BytesIO()
    batch = [_call(number) for number in range(1, 4)]
    raw = json.dumps(batch).encode() + b"\n"
    assert server.serve(stdin=io.BytesIO(raw), stdout=outgoing, handler=handler) == 0
    assert reached == []
    assert [json.loads(line) for line in outgoing.getvalue().splitlines()] == [
        {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Batch request has too many items."},
        }
    ]


def test_an_unknown_name_is_echoed_as_one_short_run_of_text():
    _raw, replies = exchange_with(_server(), [_call(1, "x\ny" + "z" * 300)], ok_handler)
    message = replies[0]["error"]["message"]
    assert message.startswith("Unknown tool: x y")
    assert "\n" not in message
    assert len(message) == len("Unknown tool: ") + 80


def test_the_same_refusal_is_given_to_a_request_that_carries_its_version():
    """The stateless revision changes how a request names its version, not what an unknown tool is."""
    frame = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "nope",
            "arguments": {},
            "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"},
        },
    }
    _raw, replies = exchange_with(_server(), [frame], ok_handler)
    assert replies[0]["error"] == {"code": -32602, "message": "Unknown tool: nope"}


def test_an_unknown_tool_is_refused_without_waiting_for_the_call_in_flight():
    """The refusal comes from the read loop, so it does not queue behind hours of generation."""
    server = _server()
    server._SHUTDOWN_GRACE_S = 0.2
    started = threading.Event()
    release = threading.Event()
    server._stop_work = release.set

    def slow(_name, _arguments):
        started.set()
        assert release.wait(30), "the running call was never told to stop"
        return {"program": "Humanitys-Last-Disease"}

    replies = exchange_gated(
        server,
        [(_call(1, "run"), None), (_call(2, "no_such_tool"), started)],
        slow,
    )
    assert replies == [{"jsonrpc": "2.0", "id": 2, "error": {"code": -32602, "message": "Unknown tool: no_such_tool"}}]
