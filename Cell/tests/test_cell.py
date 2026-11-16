"""Tests for the Cell relay. Standard library only; run with the Windows Python:

    python -m unittest discover -s Cell/tests -v

The relay is started as a real process on a free port and driven over HTTP, against
``fake_panel.FakePanel``. ``RealPanelTest`` drives it against the real panel pages
inside WSL, and is skipped when WSL or the checkout's virtual environment is absent.
"""

from __future__ import annotations

import gzip
import http.client
import json
import os
import queue
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELL = HERE.parent
sys.path.insert(0, str(CELL))
sys.path.insert(0, str(HERE))

import cell_pages  # noqa: E402
import cell_relay  # noqa: E402
from fake_panel import FakePanel  # noqa: E402

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class RunningRelay:
    """One relay process on a free port."""

    def __init__(self, *extra: str) -> None:
        self.port = free_port()
        self.log = Path(tempfile.mkdtemp(prefix="cell-test-")) / "relay.log"
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(CELL / "cell_relay.py"), "--listen-port", str(self.port),
             "--log-file", str(self.log), *extra],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=NO_WINDOW,
        )
        deadline = time.monotonic() + 15
        while True:
            try:
                status, _headers, _body = self.get("/cell/ping")
                if status == 200:
                    return
            except OSError:
                pass
            if self.proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("relay did not start: " + (self.proc.stderr.read().decode() if self.proc.stderr else ""))
            time.sleep(0.1)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None,
                timeout: float = 20.0) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            data = response.read()
            return response.status, {k.lower(): v for k, v in response.getheaders()}, data
        finally:
            connection.close()

    def get(self, path: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
        return self.request("GET", path, headers=headers)

    def post(self, path: str, body: bytes = b"", headers: dict[str, str] | None = None,
             kind: str = "application/x-www-form-urlencoded") -> tuple[int, dict[str, str], bytes]:
        sent = {"Origin": self.origin, "Content-Type": kind}
        sent.update(headers or {})
        return self.request("POST", path, body=body, headers=sent)

    def status(self, query: str = "") -> dict:
        return json.loads(self.get("/cell/status" + query)[2])

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=10)
        if self.proc.stderr:
            self.proc.stderr.close()


def wait_for(check, seconds: float = 15.0, every: float = 0.1):
    deadline = time.monotonic() + seconds
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(every)


class UnitTest(unittest.TestCase):
    def test_host_of(self) -> None:
        self.assertEqual(cell_relay.host_of("100.64.0.5:8765"), "100.64.0.5")
        self.assertEqual(cell_relay.host_of("Name.Tail.ts.net."), "name.tail.ts.net")
        self.assertEqual(cell_relay.host_of("[FD7A::1]:8765"), "fd7a::1")
        self.assertEqual(cell_relay.host_of("fd7a:115c::1"), "fd7a:115c::1")
        self.assertEqual(cell_relay.host_of("localhost:8766"), "localhost")

    def test_local_path_never_leaves_the_relay(self) -> None:
        self.assertEqual(cell_relay.local_path("/step/3"), "/step/3")
        self.assertEqual(cell_relay.local_path("http://127.0.0.1:41000/step/3?x=1"), "/step/3?x=1")
        self.assertEqual(cell_relay.local_path("//evil.example/x"), "/x")
        self.assertEqual(cell_relay.local_path("https://evil.example"), "/")
        self.assertEqual(cell_relay.local_path("/\\evil.example/x"), "/x")
        self.assertEqual(cell_relay.local_path("\\\\evil.example"), "/")
        self.assertEqual(cell_relay.local_path("http://[bad"), "/")

    def test_inject_adds_tags_once_and_keeps_the_page(self) -> None:
        page = b"<html><head><title>x</title></head><body><form></form><script>1</script></body></html>"
        out = cell_pages.inject(page)
        self.assertIn(b'rel="apple-touch-icon"', out)
        self.assertIn(b'rel="manifest"', out)
        self.assertLess(out.index(b"apple-touch-icon"), out.index(b"</head>"))
        self.assertLess(out.index(b"<script>1</script>"), out.index(b"/cell/panel.js"))
        self.assertEqual(out.count(b"/cell/panel.js"), 1)
        self.assertEqual(cell_pages.inject(b"plain text"), b"plain text")

    def test_icon_is_a_valid_png(self) -> None:
        for size in (48, 180, 512):
            data = cell_pages.png_icon(size)
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            width, height, depth, colour = struct.unpack(">IIBB", data[16:26])
            self.assertEqual((width, height, depth, colour), (size, size, 8, 2))
            start = data.index(b"IDAT") + 4
            length = struct.unpack(">I", data[start - 8:start - 4])[0]
            raw = zlib.decompress(data[start:start + length])
            self.assertEqual(len(raw), size * (size * 3 + 1))

    def test_manifest_starts_at_the_address_that_always_lands_right(self) -> None:
        manifest = json.loads(cell_pages.manifest())
        self.assertEqual(manifest["start_url"], "/cell/go")
        self.assertEqual(manifest["display"], "standalone")

    def _proxy(self, *chunks: bytes) -> tuple[str | None, bytes]:
        left, right = socket.socketpair()
        try:
            for chunk in chunks:
                left.sendall(chunk)
                time.sleep(0.03)
            right.settimeout(2)
            source = cell_relay.read_proxy_line(right)
            rest = b""
            if source is not None:
                right.settimeout(0.3)
                try:
                    rest = right.recv(100)
                except OSError:
                    rest = b""
            return source, rest
        finally:
            left.close()
            right.close()

    def test_proxy_line(self) -> None:
        self.assertEqual(self._proxy(b"PROXY TCP4 100.64.0.7 100.64.0.1 5555 8765\r\nGET /"), ("100.64.0.7", b"GET /"))
        self.assertEqual(self._proxy(b"PRO", b"XY TCP6 fd7a::7 fd7a::1 5 6\r\n", b"GET"), ("fd7a::7", b"GET"))
        self.assertEqual(self._proxy(b"GET / HTTP/1.1\r\n")[0], "")
        self.assertEqual(self._proxy(b"POST / HTTP/1.1\r\n")[0], "")
        self.assertIsNone(self._proxy(b"PROXY UNKNOWN\r\nGET /")[0])
        self.assertIsNone(self._proxy(b"PROXY TCP4 --json=false 100.64.0.1 5 6\r\nGET /")[0])
        self.assertIsNone(self._proxy(b"PROXY TCP4 not-an-address 100.64.0.1 5 6\r\n")[0])
        self.assertIsNone(self._proxy(b"PROXY " + b"x" * 200 + b"\r\n")[0])

    def test_gate_admits_only_the_owner_and_fails_closed(self) -> None:
        gate = cell_relay.Gate("tailscale")
        answers = {
            ("status", "--json", "--peers=false"): {"Self": {"UserID": 7}},
            ("whois", "--json", "100.64.0.7"): {"Node": {"User": 7, "ComputedName": "phone"}},
            ("whois", "--json", "100.64.0.9"): {"Node": {"User": 9, "ComputedName": "guest"}},
        }
        gate._json = lambda *arguments: answers.get(arguments, {})  # type: ignore[method-assign]
        self.assertTrue(gate.allows("100.64.0.7"))
        self.assertFalse(gate.allows("100.64.0.9"))
        self.assertFalse(gate.allows("100.64.0.99"))
        self.assertEqual(gate.name("100.64.0.7"), "phone (100.64.0.7)")
        # Tailscale not answering is "ask again", not "someone else's device", and is not remembered.
        silent = cell_relay.Gate("tailscale")
        silent._json = lambda *arguments: {}  # type: ignore[method-assign]
        self.assertEqual(silent.check("100.64.0.7"), cell_relay.Gate.UNKNOWN)
        self.assertFalse(silent.allows("100.64.0.7"))
        silent._json = lambda *arguments: answers.get(arguments, {})  # type: ignore[method-assign]
        self.assertEqual(silent.check("100.64.0.7"), cell_relay.Gate.ALLOW)
        # Without the owner rule every device Tailscale can name gets in, and still gets its name.
        open_gate = cell_relay.Gate("tailscale", owner_only=False)
        open_gate._json = lambda *arguments: answers.get(arguments, {})  # type: ignore[method-assign]
        self.assertTrue(open_gate.allows("100.64.0.9"))
        self.assertEqual(open_gate.name("100.64.0.9"), "guest (100.64.0.9)")
        self.assertFalse(open_gate.allows("100.64.0.99"))

    def test_output_lines_follow_carriage_returns_like_a_terminal(self) -> None:
        import cell_program

        program = cell_program.Program(distro="", repo="", command="x")
        for chunk in ("first line\r", "\nsecond", " half\n10%\r20%\r", "30%\nwaiting\r"):
            program._feed(chunk)
        seen = program.output(0)
        self.assertEqual(seen["lines"], ["first line", "second half", "30%"])
        self.assertEqual(seen["partial"], "waiting")

    def test_only_the_programs_own_first_lines_set_the_panel_and_the_log(self) -> None:
        import cell_program

        program = cell_program.Program(distro="", repo="", command="x")
        program._proc = type("Alive", (), {"poll": lambda self: None})()  # type: ignore[assignment]
        for line in ("CELL-PID 4242", "Run log: /home/u/.config/Humanitys-Last-Disease/Humanitys-Last-Disease.log",
                     "note: Option panel: http://127.0.0.1:9/", "Option panel: http://127.0.0.1:41000/",
                     "Option panel: http://127.0.0.1:9/", "Run log: /etc/passwd"):
            program._feed(line + "\n")
        self.assertEqual(program.panel_port(), 41000)
        self.assertEqual(program._run_log, "/home/u/.config/Humanitys-Last-Disease/Humanitys-Last-Disease.log")
        self.assertEqual(program._linux_pid, 4242)
        self.assertNotIn("CELL-PID 4242", program.output(0)["lines"])

    def test_a_relay_without_a_panel_is_not_taken_for_a_panel(self) -> None:
        self.assertTrue(cell_relay.panel_shaped({"active": False, "done": 0, "total": 0, "step": 0}))
        self.assertFalse(cell_relay.panel_shaped({"active": False, "done": 0, "total": 0, "cell": "2.1"}))
        self.assertFalse(cell_relay.panel_shaped(["active", "done", "total"]))


class ProxyTest(unittest.TestCase):
    """The relay in front of a panel that answers at once."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = FakePanel()
        cls.relay = RunningRelay("--panel-port", str(cls.panel.port), "--no-discovery", "--allow-host", "phone.example",
                                 "--max-upload-mb", "1")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.relay.stop()

    def test_unknown_host_is_refused(self) -> None:
        status, _headers, _body = self.relay.get("/", {"Host": "evil.example"})
        self.assertEqual(status, 403)
        status, _headers, _body = self.relay.get("/cell/ping", {"Host": "phone.example:8765"})
        self.assertEqual(status, 200)

    def test_root_goes_to_the_current_step_and_pages_carry_the_phone_tags(self) -> None:
        status, headers, _body = self.relay.get("/")
        self.assertEqual(status, 303)
        self.assertRegex(headers["location"], r"^/step/\d+$")
        status, headers, body = self.relay.get(headers["location"])
        self.assertEqual(status, 200)
        self.assertIn(b"Fake panel step", body)
        self.assertIn(b"/cell/panel.js", body)
        self.assertIn(b"/cell/app.webmanifest", body)
        self.assertEqual(headers["x-frame-options"], "DENY")
        self.assertEqual(headers["cache-control"], "no-store")

    def test_post_needs_a_cell_page_as_sender(self) -> None:
        plain = {"Content-Type": "application/x-www-form-urlencoded"}
        before = len(self.panel.posts)
        self.assertEqual(self.relay.request("POST", "/", b"a=1", plain)[0], 403)
        for origin in ("http://evil.example", "null", "http://127.0.0.1.evil.example"):
            self.assertEqual(self.relay.request("POST", "/", b"a=1", {**plain, "Origin": origin})[0], 403, origin)
        self.assertEqual(self.relay.request("POST", "/", b"a=1", {**plain, "Referer": "http://evil.example/x"})[0], 403)
        self.assertEqual(len(self.panel.posts), before)
        status, headers, _body = self.relay.request("POST", "/", b"a=1", {**plain, "Referer": self.relay.origin + "/step/0"})
        self.assertEqual(status, 303)
        named = {"Origin": "http://phone.example:8765", "Host": "phone.example:8765"}
        status, headers, _body = self.relay.post("/", b"a=1", named)
        self.assertEqual(status, 303)
        self.assertRegex(headers["location"], r"^/step/\d+$")
        self.assertEqual(len(self.panel.posts), before + 2)
        # The same host on another port is another program's page, and so is any other host.
        for origin in (f"http://127.0.0.1:{self.relay.port + 1}", "http://127.0.0.1", "http://localhost:3000",
                       "http://phone.example:8765", f"ftp://127.0.0.1:{self.relay.port}"):
            self.assertEqual(self.relay.request("POST", "/", b"a=1", {**plain, "Origin": origin})[0], 403, origin)
        self.assertEqual(len(self.panel.posts), before + 2)

    def test_a_stale_step_address_never_rewinds_the_run(self) -> None:
        self.relay.post("/", b"a=1")
        self.relay.post("/", b"a=1")
        current = self.panel.step
        self.assertGreaterEqual(current, 2)
        rewinds = self.panel.rewinds
        for path in ("/step/0", f"/step/{current - 1}", f"/step/{current + 5}", "/anything", "/step/0/"):
            status, headers, _body = self.relay.get(path)
            self.assertEqual(status, 303, path)
            self.assertEqual(headers["location"], f"/step/{current}", path)
        self.assertEqual(self.panel.rewinds, rewinds)
        self.assertEqual(self.panel.step, current)

    def test_progress_and_where(self) -> None:
        data = json.loads(self.relay.get("/progress")[2])
        self.assertEqual(data["step"], self.panel.step)
        self.assertFalse(data["closed"])
        self.assertEqual(json.loads(self.relay.get("/cell/where")[2])["go"], f"/step/{self.panel.step}")
        self.assertEqual(self.relay.get("/cell/go")[1]["location"], f"/step/{self.panel.step}")

    def test_viewer_script_is_compressed_and_revalidated(self) -> None:
        status, headers, body = self.relay.get("/viewer/3dmol.js", {"Accept-Encoding": "gzip"})
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("content-encoding"), "gzip")
        self.assertIn(b"fake viewer", gzip.decompress(body))
        self.assertIn("max-age", headers["cache-control"])
        status, _headers, body = self.relay.get("/viewer/3dmol.js", {"If-None-Match": headers["etag"]})
        self.assertEqual((status, body), (304, b""))
        status, headers, body = self.relay.get("/structure/0123456789abcdef")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"data_fake"))
        self.assertNotIn("content-encoding", headers)

    def test_static_files_and_icons(self) -> None:
        for path, kind in (("/cell/panel.js", "application/javascript"), ("/cell/panel.css", "text/css"),
                           ("/cell/app.webmanifest", "application/manifest+json"), ("/cell/icon-180.png", "image/png"),
                           ("/apple-touch-icon.png", "image/png"), ("/favicon.ico", "image/png")):
            status, headers, body = self.relay.get(path)
            self.assertEqual(status, 200, path)
            self.assertTrue(headers["content-type"].startswith(kind), path)
            self.assertTrue(body, path)

    def test_two_requests_on_one_connection(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", self.relay.port, timeout=10)
        try:
            for _ in range(3):
                connection.request("GET", "/progress")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn("step", json.loads(response.read()))
        finally:
            connection.close()

    def test_head_and_method_and_size_limits(self) -> None:
        status, headers, body = self.relay.request("HEAD", "/cell/")
        self.assertEqual((status, body), (200, b""))
        self.assertGreater(int(headers["content-length"]), 1000)
        self.assertEqual(self.relay.request("DELETE", "/")[0], 405)
        self.assertEqual(self.relay.post("/", b"x" * (1024 * 1024 + 10))[0], 413)

    def test_hub_page_and_status(self) -> None:
        status, _headers, body = self.relay.get("/cell/")
        self.assertEqual(status, 200)
        self.assertIn(b"Start program", body)
        self.assertIn(b'data-mode="hub"', body)
        report = self.relay.status()
        self.assertTrue(report["panel"]["open"])
        self.assertEqual(report["program"]["state"], "idle")
        self.assertEqual(report["device"], "this computer")
        self.assertEqual(self.relay.get("/cell/working?t=nope")[0], 303)
        self.assertEqual(self.relay.get("/cell/nothing")[0], 404)

    def test_requests_from_another_site_are_refused_unless_they_open_a_page(self) -> None:
        cross = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "image"}
        for path in ("/", "/step/0", "/cell/log", "/progress", "/cell/status"):
            self.assertEqual(self.relay.get(path, cross)[0], 403, path)
        near = {"Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Dest": "empty"}
        self.assertEqual(self.relay.get("/progress", near)[0], 403)
        opening = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"}
        self.assertEqual(self.relay.get("/", opening)[0], 303)
        for site in ("same-origin", "none"):
            self.assertEqual(self.relay.get("/progress", {"Sec-Fetch-Site": site})[0], 200, site)

    def test_a_refused_request_is_answered_before_its_body_is_sent(self) -> None:
        with socket.create_connection(("127.0.0.1", self.relay.port), timeout=5) as raw:
            raw.sendall(
                b"POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: http://evil.example\r\n"
                b"Content-Type: application/octet-stream\r\nContent-Length: 900000\r\n\r\n"
            )
            raw.settimeout(5)
            self.assertIn(b" 403 ", raw.recv(200).split(b"\r\n", 1)[0])

    def test_malformed_requests_get_an_answer_and_no_traceback(self) -> None:
        for head in (b"GET /viewer/\xff HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n",
                     b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nUser-Agent: a\rb\r\n\r\n",
                     b"POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: \xb2\r\n\r\n",
                     b"GET / HTTP/1.1\r\n folded: x\r\n\r\n"):
            with socket.create_connection(("127.0.0.1", self.relay.port), timeout=5) as raw:
                raw.sendall(head)
                raw.settimeout(5)
                self.assertIn(b" 400 ", raw.recv(200).split(b"\r\n", 1)[0], head)
        self.assertEqual(self.relay.get("/cell/status?since=%C2%B2")[0], 200)
        self.assertNotIn("Traceback", self.relay.log.read_text(encoding="utf-8"))

    def test_an_asset_address_the_panel_does_not_know_is_not_found(self) -> None:
        self.assertEqual(self.relay.get("/viewer/other.js")[0], 404)
        self.assertEqual(self.relay.get("/closed")[1]["location"], f"/step/{self.panel.step}")

    def test_not_modified_carries_no_length(self) -> None:
        etag = self.relay.get("/cell/panel.js")[1]["etag"]
        status, headers, body = self.relay.get("/cell/panel.js", {"If-None-Match": etag})
        self.assertEqual((status, body), (304, b""))
        self.assertNotIn("content-length", headers)
        self.assertEqual(self.relay.get("/progress")[0], 200)

    def test_a_page_request_tells_the_panel_it_is_a_mirror(self) -> None:
        self.relay.get(f"/step/{self.panel.step}")
        self.assertEqual(self.panel.last_get_headers.get("x-cell-no-rewind"), "1")

    def test_shutdown_is_for_this_computer_only(self) -> None:
        with socket.create_connection(("127.0.0.1", self.relay.port), timeout=5) as raw:
            raw.sendall(
                b"PROXY TCP4 100.64.0.7 100.64.0.1 5555 8765\r\n"
                b"POST /cell/shutdown HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: http://127.0.0.1\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
            self.assertIn(b" 403 ", raw.recv(200).split(b"\r\n", 1)[0])
        self.assertIsNone(self.relay.proc.poll())


class SlowPanelTest(unittest.TestCase):
    """A panel that takes longer than the relay lets a phone wait."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = FakePanel(delay=2.0)
        cls.relay = RunningRelay("--panel-port", str(cls.panel.port), "--no-discovery", "--slow-after", "0.5")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.relay.stop()

    def test_slow_submission_goes_to_the_working_page_and_then_on(self) -> None:
        began = time.monotonic()
        status, headers, _body = self.relay.post("/", b"a=1")
        self.assertEqual(status, 303)
        self.assertLess(time.monotonic() - began, 1.5)
        working = headers["location"]
        self.assertRegex(working, r"^/cell/working\?t=[0-9a-f]{24}$")
        token = working.split("=", 1)[1]
        status, _headers, body = self.relay.get(working)
        self.assertEqual(status, 200)
        self.assertIn(b'data-mode="working"', body)
        # Anything that would show a panel page now shows the working page instead.
        for path in ("/", "/step/0", "/cell/"):
            self.assertEqual(self.relay.get(path)[1]["location"], working, path)
        self.assertEqual(json.loads(self.relay.get("/cell/where")[2])["go"], working)
        # A second submission is not sent to the program.
        self.assertEqual(self.relay.post("/", b"b=2")[1]["location"], working)
        report = self.relay.status("?t=" + token)
        self.assertEqual(report["follow"]["state"], "pending")
        self.assertTrue(report["panel"]["progress"]["active"])
        done = wait_for(lambda: (r := self.relay.status("?t=" + token))["follow"]["state"] == "done" and r, 10)
        self.assertEqual(done["follow"]["go"], "/step/1")
        self.assertEqual(len(self.panel.posts), 1)
        self.assertEqual(self.relay.get("/step/1")[0], 200)
        self.assertEqual(self.relay.status("?t=" + "0" * 24)["follow"], {"state": "unknown", "go": "/step/1"})


class LatePanelTest(unittest.TestCase):
    """A panel whose /progress answer is late for a moment."""

    def test_one_late_answer_is_not_a_closed_panel_and_the_form_is_still_sent(self) -> None:
        panel = FakePanel()
        relay = RunningRelay("--panel-port", str(panel.port), "--no-discovery")
        try:
            self.assertEqual(json.loads(relay.get("/progress")[2])["closed"], False)
            panel.progress_delay = 4.0
            status, _headers, body = relay.get("/progress")
            self.assertEqual(status, 503)
            self.assertNotIn(b"closed", body)
            self.assertIsNone(json.loads(relay.get("/cell/where")[2])["go"])
            self.assertEqual(relay.get("/step/0")[0], 503)
            before = len(panel.posts)
            status, headers, _body = relay.post("/", b"a=1")
            self.assertEqual(status, 303)
            self.assertNotEqual(headers["location"], "/cell/")
            wait_for(lambda: len(panel.posts) == before + 1, 10)
            panel.progress_delay = 0.0
            wait_for(lambda: relay.get("/progress")[0] == 200, 15)
            self.assertFalse(json.loads(relay.get("/progress")[2])["closed"])
        finally:
            panel.progress_delay = 0.0
            relay.stop()


class ClosingPanelTest(unittest.TestCase):
    def test_close_and_a_vanished_panel_land_on_the_cell_page(self) -> None:
        panel = FakePanel()
        relay = RunningRelay("--panel-port", str(panel.port), "--no-discovery")
        try:
            status, headers, _body = relay.post("/", b"close=1")
            self.assertEqual((status, headers["location"]), (303, "/cell/"))
            wait_for(lambda: json.loads(relay.get("/progress")[2])["closed"], 10)
            self.assertEqual(relay.get("/")[1]["location"], "/cell/")
            self.assertEqual(relay.get("/closed")[1]["location"], "/cell/")
            self.assertEqual(relay.post("/", b"a=1")[1]["location"], "/cell/")
            self.assertFalse(relay.status()["panel"]["open"])
        finally:
            relay.stop()


class OwnerGateTest(unittest.TestCase):
    """The relay behind a stand-in for Tailscale Serve's PROXY line and ``tailscale whois``."""

    @classmethod
    def setUpClass(cls) -> None:
        folder = Path(tempfile.mkdtemp(prefix="cell-gate-"))
        script = folder / "fake_tailscale.py"
        script.write_text(
            "import json, sys\n"
            "a = sys.argv[1:]\n"
            "if a[:1] == ['status']: print(json.dumps({'Self': {'UserID': 7}}))\n"
            "elif a[:1] == ['whois']:\n"
            "    user = 7 if a[-1] in ('100.64.0.7', 'fd7a::7') else 9\n"
            "    print(json.dumps({'Node': {'User': user, 'ComputedName': 'dev-' + str(user)}}))\n",
            encoding="utf-8",
        )
        shim = folder / "tailscale.cmd"
        shim.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="ascii")
        cls.shim = str(shim)
        cls.panel = FakePanel()
        cls.relay = RunningRelay("--panel-port", str(cls.panel.port), "--no-discovery", "--owner-only", cls.shim)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.relay.stop()

    def _through_serve(self, line: bytes, path: str = "/cell/status") -> bytes:
        with socket.create_connection(("127.0.0.1", self.relay.port), timeout=20) as raw:
            raw.sendall(line + f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".encode())
            data = b""
            while True:
                chunk = raw.recv(65536)
                if not chunk:
                    return data
                data += chunk

    def test_owner_device_gets_in_and_is_named(self) -> None:
        answer = self._through_serve(b"PROXY TCP4 100.64.0.7 100.64.0.1 5555 8765\r\n")
        self.assertTrue(answer.startswith(b"HTTP/1.1 200 "))
        self.assertIn(b"dev-7 (100.64.0.7)", answer)
        self.assertTrue(self._through_serve(b"PROXY TCP6 fd7a::7 fd7a::1 5555 8765\r\n").startswith(b"HTTP/1.1 200 "))

    def test_other_devices_and_bad_lines_are_refused(self) -> None:
        for line in (b"PROXY TCP4 100.64.0.9 100.64.0.1 5555 8765\r\n", b"PROXY UNKNOWN\r\n", b"PROXY garbage\r\n"):
            answer = self._through_serve(line)
            self.assertTrue(answer.startswith(b"HTTP/1.1 403 "), line)
            self.assertIn(b"Not allowed", answer)

    def test_this_computer_needs_no_line(self) -> None:
        status, _headers, body = self.relay.get("/cell/ping")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["gate"], "owner")

    def test_every_tailnet_device_mode_still_names_devices_and_keeps_shutdown_local(self) -> None:
        relay = RunningRelay("--panel-port", str(self.panel.port), "--no-discovery", "--tailscale", self.shim,
                             "--any-tailnet-device")
        try:
            self.assertEqual(json.loads(relay.get("/cell/ping")[2])["gate"], "tailnet")
            with socket.create_connection(("127.0.0.1", relay.port), timeout=20) as raw:
                raw.sendall(b"PROXY TCP4 100.64.0.9 100.64.0.1 5555 8765\r\n"
                            b"GET /cell/status HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
                answer = b""
                while chunk := raw.recv(65536):
                    answer += chunk
            self.assertTrue(answer.startswith(b"HTTP/1.1 200 "))
            self.assertIn(b"dev-9 (100.64.0.9)", answer)
            with socket.create_connection(("127.0.0.1", relay.port), timeout=20) as raw:
                raw.sendall(b"PROXY TCP4 100.64.0.9 100.64.0.1 5555 8765\r\n"
                            b"POST /cell/shutdown HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: http://127.0.0.1\r\n"
                            b"Content-Length: 0\r\nConnection: close\r\n\r\n")
                self.assertIn(b" 403 ", raw.recv(200).split(b"\r\n", 1)[0])
            self.assertIsNone(relay.proc.poll())
        finally:
            relay.stop()


class ProgramTest(unittest.TestCase):
    """Start, watch, talk to and stop a program from the Cell page's endpoints."""

    def test_full_cycle(self) -> None:
        argv = json.dumps([sys.executable, "-u", str(HERE / "fake_panel.py")])
        relay = RunningRelay("--program-argv", argv, "--no-discovery")
        try:
            self.assertEqual(relay.get("/")[1]["location"], "/cell/")
            self.assertEqual(relay.status()["program"]["state"], "idle")
            self.assertEqual(relay.request("POST", "/cell/start", b"", {"Content-Length": "0"})[0], 403)
            answer = json.loads(relay.post("/cell/start")[2])
            self.assertTrue(answer["ok"], answer)
            report = wait_for(lambda: (r := relay.status())["panel"]["open"] and r, 20)
            self.assertEqual(report["program"]["state"], "running")
            self.assertTrue(report["program"]["managed"])
            self.assertTrue(any("Option panel" in line for line in report["output"]["lines"]))
            self.assertFalse(any("CELL-PID" in line for line in report["output"]["lines"]))
            self.assertFalse(json.loads(relay.post("/cell/start")[2])["ok"])
            self.assertEqual(relay.get("/")[1]["location"], "/step/0")
            self.assertEqual(relay.post("/", b"a=1")[1]["location"], "/step/1")
            self.assertTrue(json.loads(relay.post("/cell/stdin", b"line=hello+phone")[2])["ok"])
            seen = report["output"]["next"]
            wait_for(lambda: any("heard: hello phone" in line for line in relay.status(f"?since={seen}")["output"]["lines"]), 10)
            self.assertEqual(json.loads(relay.get("/cell/log")[2])["lines"], [])
            self.assertTrue(json.loads(relay.post("/cell/stop")[2])["ok"])
            ended = wait_for(lambda: (r := relay.status())["program"]["state"] == "exited" and r, 20)
            self.assertFalse(ended["panel"]["open"])
            self.assertEqual(relay.get("/")[1]["location"], "/cell/")
            self.assertTrue(json.loads(relay.post("/cell/start")[2])["ok"])
            wait_for(lambda: relay.status()["panel"]["open"], 20)
            # The relay's own shutdown stops the program first, then the relay.
            self.assertEqual(relay.post("/cell/shutdown")[0], 200)
            relay.proc.wait(timeout=40)
        finally:
            relay.stop()


def _wsl_python() -> list[str] | None:
    """The command that runs the checkout's Python inside WSL, or None when it is absent."""
    if os.environ.get("CELL_SKIP_WSL") or os.name != "nt":
        return None
    repo = cell_relay.default_repo()
    python = repo + "/.venv/bin/python"
    try:
        found = subprocess.run(["wsl.exe", "-e", "test", "-x", python], capture_output=True, timeout=60,
                               creationflags=NO_WINDOW, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return ["wsl.exe", "--cd", repo, "-e", python] if found.returncode == 0 else None


class RealPanelTest(unittest.TestCase):
    """The relay in front of the real panel pages, served by a real BrowserSession in WSL."""

    def test_real_pages_through_the_relay(self) -> None:
        python = _wsl_python()
        if python is None:
            self.skipTest("WSL or the checkout's .venv is not available")
        # The harness keeps every side effect (key store, log, uploads) in this folder.
        drive, rest = os.path.splitdrive(tempfile.mkdtemp(prefix="cell-real-"))
        scratch = "/mnt/" + drive[0].lower() + rest.replace("\\", "/")
        harness = subprocess.Popen(
            [*python, cell_relay.default_repo() + "/Cell/tests/panel_harness.py",
             "--delay", "3", "--steps", "3", "--scratch", scratch, "--max-seconds", "180"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=NO_WINDOW,
        )
        relay = None
        try:
            # Read on a thread: a WSL that is slow to start must fail this test, not hang it.
            lines: queue.Queue[str] = queue.Queue()
            assert harness.stdout is not None
            stream = harness.stdout
            threading.Thread(
                target=lambda: [lines.put(raw.decode("utf-8", "replace")) for raw in iter(stream.readline, b"")],
                daemon=True,
            ).start()
            port = None
            deadline = time.monotonic() + 120
            while port is None and time.monotonic() < deadline:
                try:
                    line = lines.get(timeout=1.0)
                except queue.Empty:
                    if harness.poll() is not None:
                        break
                    continue
                if " PORT " in line:
                    port = int(line.split("port=", 1)[1].split()[0])
            self.assertIsNotNone(port, "the panel harness did not report a port")
            relay = RunningRelay("--panel-port", str(port), "--no-discovery", "--slow-after", "1")
            wait_for(lambda: relay.status()["panel"]["open"], 30)
            self.assertEqual(relay.get("/")[1]["location"], "/step/0")
            status, _headers, body = relay.get("/step/0")
            self.assertEqual(status, 200)
            self.assertIn(b'name="disease"', body)
            self.assertIn(b'<form data-step="0"', body)
            self.assertIn(b"/cell/panel.js", body)
            self.assertIn(b"apple-touch-icon", body)
            boundary = "cellboundary123"
            parts = "".join(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
                for name, value in (("disease", "Asthma"), ("binder_kind", "protein"), ("where", "local"))
            ) + f"--{boundary}--\r\n"
            status, headers, _body = relay.post("/", parts.encode(), kind=f"multipart/form-data; boundary={boundary}")
            self.assertEqual(status, 303)
            self.assertTrue(headers["location"].startswith("/cell/working?t="), headers["location"])
            token = headers["location"].split("=", 1)[1]
            saw_progress = False
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                report = relay.status("?t=" + token)
                progress = report["panel"]["progress"]
                saw_progress = saw_progress or bool(progress and progress["active"])
                if report["follow"]["state"] == "done":
                    break
                time.sleep(0.3)
            self.assertEqual(report["follow"], {"state": "done", "go": "/step/1"})
            self.assertTrue(saw_progress, "the real panel's progress never showed as active")
            status, _headers, body = relay.get("/step/1")
            self.assertEqual(status, 200)
            self.assertIn(b'<form data-step="1"', body)
            self.assertIn(b'name="__nav" value="back"', body)
            # A stale address is answered with the current one, and the real panel is not rewound.
            self.assertEqual(relay.get("/step/0")[1]["location"], "/step/1")
            self.assertEqual(json.loads(relay.get("/progress")[2])["step"], 1)
            # Asked directly, with the header the relay sends, the real panel answers
            # an earlier step with the current page instead of rewinding the run.
            direct = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            try:
                direct.request("GET", "/step/0", headers={"X-Cell-No-Rewind": "1"})
                answer = direct.getresponse()
                answer.read()
                self.assertEqual((answer.status, answer.getheader("Location")), (303, "/step/1"))
            finally:
                direct.close()
            time.sleep(1.0)
            self.assertEqual(json.loads(relay.get("/progress")[2])["step"], 1)
        finally:
            if relay is not None:
                relay.stop()
            harness.kill()
            try:
                harness.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
