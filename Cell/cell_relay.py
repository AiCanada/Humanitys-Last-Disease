"""Cell relay: the phone's front door to the ``human`` option panel.

The panel listens on ``127.0.0.1`` at a port that changes on every start, and it was
written for one browser on the same computer. This relay holds a fixed port, is
published to the tailnet by Tailscale Serve, and does what a phone needs:

* It is an HTTP proxy, not a pipe. It checks who is calling (only this user's own
  tailnet devices), which name was called (so a rebinding web page is refused), and
  where a request came from (so another site cannot post a form, or drive the relay
  with requests it cannot read the answers to).
* A page the program takes long to answer is not left hanging. iOS gives up on a
  request after 60 quiet seconds, and a generation run takes far longer. The relay
  keeps the submission going and sends the phone to a page that shows the progress
  and the output, then on to the next panel page when it is ready.
* A stale address never rewinds the run. The panel treats ``GET /step/3`` as "go back
  to step 3". A phone reloads old tabs on its own, so the relay answers any step
  address other than the current one with the current one. The page's own Back
  button still goes back.
* A panel that is slow for a moment is not a panel that closed. Only a panel that
  refuses connections, says it closed, or stays silent for a while is treated as gone.
* ``/cell/`` is the control page: start the program, stop it, read its output.

Standard library only. Runs with the Windows Python; the program runs in WSL.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import ipaddress
import json
import logging
import logging.handlers
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from urllib.parse import parse_qs, urlsplit

import cell_pages
from cell_program import Program

VERSION = "2.1"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
#: Processes that can own the panel's listening port on Windows.
#: ``wslrelay.exe`` is the Windows side of a port opened inside WSL.
PANEL_OWNERS = frozenset({"wslrelay.exe", "python.exe", "pythonw.exe", "python3.exe", "human.exe"})
#: Keys the panel's ``/progress`` answer always carries.
PROGRESS_KEYS = frozenset({"active", "done", "total"})
PROGRESS_FIELDS = ("active", "done", "total", "label", "action")
LOOPBACK = frozenset({"127.0.0.1", "::1"})
REDIRECTS = frozenset({301, 302, 303, 307, 308})
MAX_LINE = 16384
MAX_HEADER_BYTES = 65536
HEADER_TIMEOUT = 30.0
IDLE_TIMEOUT = 75.0
#: Connections served at once. More than this is a fault somewhere, not a phone.
MAX_CONNECTIONS = 128
#: Seconds between searches for a panel that Cell did not start. Each search lists
#: the computer's listening ports, so it runs in the background and not per request.
SEARCH_EVERY = 5.0
#: Seconds a known panel may stay silent before it is treated as gone.
GONE_AFTER = 10.0
#: Checks, five seconds apart, that a held submission's panel may fail in a row.
HELD_SILENCE = 6
TOKEN = re.compile(r"^[0-9a-f]{24}$")
STEP_PATH = re.compile(r"^/step/([0-9]{1,6})/?$")
DIGITS = re.compile(r"^[0-9]{1,18}$")
#: A request target is printable ASCII. Anything else cannot be sent on to the panel.
TARGET = re.compile(r"^[\x21-\x7e]+$")
#: Control characters a header value may not carry. A tab is allowed.
HEADER_CONTROL = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")
COMPRESSIBLE = ("text/", "application/javascript", "application/json", "application/manifest+json", "chemical/")
SECURITY_HEADERS = (
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "same-origin"),
)

_SMALL_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">{refresh}
<title>Cell</title></head><body style="font-family:system-ui,sans-serif;margin:2rem 1rem;line-height:1.5">
<h1>{title}</h1>
<p>{text}</p>
<p><a href="/cell/go">Try again</a></p>
</body></html>
"""
DENIED_PAGE = _SMALL_PAGE.format(
    refresh="",
    title="Not allowed",
    text="This device is not signed in to Tailscale as the owner of the computer.",
).encode("utf-8")
CHECKING_PAGE = _SMALL_PAGE.format(
    refresh='\n<meta http-equiv="refresh" content="5">',
    title="One moment",
    text="The computer could not check this device with Tailscale yet. This page tries again by itself.",
).encode("utf-8")
BUSY_PAGE = _SMALL_PAGE.format(
    refresh='\n<meta http-equiv="refresh" content="2">',
    title="One moment",
    text="The panel did not answer. This page tries again by itself.",
).encode("utf-8")

log = logging.getLogger("cell")


# --------------------------------------------------------------------- HTTP types
class HttpError(Exception):
    """A request the relay answers itself with an error status."""

    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or HTTPStatus(status).phrase)
        self.status = status


class PanelGone(Exception):
    """No panel is listening: none is known, or the known port refused the connection."""


class PanelSlow(Exception):
    """The panel accepted the connection and then did not answer in time, or broke off."""


@dataclass
class Request:
    method: str
    target: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    keep_alive: bool
    #: Bytes of body the head announced. The body is read after the request is admitted.
    size: int = 0
    body: bytes = b""
    #: The tailnet device behind the connection, or "" for a caller on this computer.
    source: str = ""
    #: Shown on the Cell page.
    device: str = ""
    #: True for a caller on this computer that did not come through Tailscale Serve.
    local: bool = False

    def first(self, name: str) -> str:
        values = self.query.get(name) or [""]
        return values[0]


@dataclass
class Response:
    status: int
    body: bytes = b""
    content_type: str = "text/plain; charset=utf-8"
    headers: list[tuple[str, str]] = field(default_factory=list)
    etag: str = ""
    cache: str = "no-store"
    keep_alive: bool = True
    #: Called after the response is on the wire.
    after: object = None


@dataclass
class Answer:
    """What the panel sent back."""

    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "application/octet-stream")

    @property
    def is_html(self) -> bool:
        return self.content_type.casefold().startswith("text/html")


@dataclass
class Look:
    """One reading of the panel: ``open``, ``closed``, ``slow`` or ``gone``."""

    kind: str
    data: dict[str, object] | None = None

    @property
    def step(self) -> int | None:
        value = self.data.get("step") if self.data else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None


def redirect(location: str) -> Response:
    return Response(303, b"", headers=[("Location", location)])


def json_response(data: object, status: int = 200) -> Response:
    return Response(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")


def text_response(status: int, message: str) -> Response:
    return Response(status, (message + "\n").encode("utf-8"))


def html_response(body: bytes, status: int = 200) -> Response:
    return Response(status, body, "text/html; charset=utf-8")


def local_path(location: str) -> str:
    """The path of a redirect target. A host in it is dropped, so it stays on this relay.

    Browsers read a backslash in an address as a slash, so ``/\\\\host`` would
    leave the relay as surely as ``//host``. Both come out as one leading slash.
    """
    try:
        parts = urlsplit(location.replace("\\", "/"))
    except ValueError:
        return "/"
    path = "/" + parts.path.lstrip("/")
    return path + ("?" + parts.query if parts.query else "")


def host_of(value: str) -> str:
    """The host name in a Host header or an origin, lower-cased, without port or brackets."""
    value = value.strip().casefold()
    if value.startswith("["):
        return value[1:].split("]", 1)[0]
    return value.rsplit(":", 1)[0].rstrip(".") if value.count(":") == 1 else value.rstrip(".")


def read_head(reader, max_body: int) -> Request | None:
    """Read one request line and its headers. ``None`` means the caller closed the connection.

    The body is not read here. A request that will be refused is refused before
    its body is taken in, so a page elsewhere cannot make the relay hold an upload
    it is about to throw away.
    """
    line = reader.readline(MAX_LINE + 1)
    while line in (b"\r\n", b"\n"):
        line = reader.readline(MAX_LINE + 1)
    if not line:
        return None
    if len(line) > MAX_LINE or not line.endswith(b"\n"):
        raise HttpError(414)
    parts = line.decode("latin-1").split()
    if len(parts) != 3 or not parts[2].startswith("HTTP/1.") or not TARGET.match(parts[1]):
        raise HttpError(400)
    method, target, version = parts
    headers: dict[str, str] = {}
    total = 0
    while True:
        raw = reader.readline(MAX_LINE + 1)
        if not raw:
            raise HttpError(400)
        total += len(raw)
        if len(raw) > MAX_LINE or total > MAX_HEADER_BYTES:
            raise HttpError(431)
        if raw in (b"\r\n", b"\n"):
            break
        name, colon, value = raw.decode("latin-1").partition(":")
        value = value.strip()
        if not colon or not name.strip() or name != name.strip() or HEADER_CONTROL.search(value):
            raise HttpError(400)
        key = name.casefold()
        headers[key] = f"{headers[key]}, {value}" if key in headers else value
    if "transfer-encoding" in headers:
        raise HttpError(501, "Chunked requests are not accepted.")
    length = headers.get("content-length", "0")
    if not DIGITS.match(length):
        raise HttpError(400)
    size = int(length)
    if size > max_body:
        raise HttpError(413, "The upload is larger than Cell accepts.")
    try:
        split = urlsplit(target)
    except ValueError:
        raise HttpError(400) from None
    wants_close = headers.get("connection", "").casefold()
    keep_alive = "close" not in wants_close and (version == "HTTP/1.1" or "keep-alive" in wants_close)
    return Request(
        method=method.upper(),
        target=target,
        path=split.path or "/",
        query=parse_qs(split.query, keep_blank_values=True),
        headers=headers,
        keep_alive=keep_alive,
        size=size,
    )


def read_body(reader, client: socket.socket, request: Request) -> None:
    """Read the body the head announced, for a request that has been admitted."""
    if not request.size:
        return
    if request.headers.get("expect", "").casefold() == "100-continue":
        client.sendall(b"HTTP/1.1 100 Continue\r\n\r\n")
    body = reader.read(request.size)
    if len(body) != request.size:
        raise HttpError(400)
    request.body = body


# ----------------------------------------------------------------- tailnet gate
def _run(command: list[str], encoding: str = "utf-8", timeout: float = 15.0) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            creationflags=NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.decode(encoding, errors="replace")


class Gate:
    """Decides which tailnet devices get in, and knows their names.

    Tailscale Serve is always started with PROXY protocol v1, so each connection it
    forwards begins with one line naming the device that opened it. That line is
    trusted only from loopback, where Tailscale Serve connects from.

    With ``owner_only`` a device is admitted only when it belongs to this computer's
    own Tailscale user. A device shared into the tailnet by someone else, or a
    tagged device, is refused. Without it every device Tailscale can name is
    admitted. When Tailscale itself does not answer, the result is ``UNKNOWN``:
    the caller is asked to wait, and nothing is remembered.
    """

    ALLOW, DENY, UNKNOWN = "allow", "deny", "unknown"
    ALLOWED_FOR = 600.0
    DENIED_FOR = 20.0
    OWNER_FOR = 600.0

    def __init__(self, tailscale: str, *, owner_only: bool = True) -> None:
        self._tailscale = tailscale
        self.owner_only = owner_only
        self._lock = threading.Lock()
        self._owner: tuple[int, float] | None = None
        self._known: dict[str, tuple[bool, float, str]] = {}

    def _json(self, *arguments: str) -> dict[str, object]:
        try:
            data = json.loads(_run([self._tailscale, *arguments]) or "{}")
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _owner_id(self) -> int | None:
        """This computer's Tailscale user, or ``None`` when Tailscale does not answer."""
        with self._lock:
            if self._owner is not None and time.monotonic() - self._owner[1] < self.OWNER_FOR:
                return self._owner[0]
        own = self._json("status", "--json", "--peers=false").get("Self")
        if not isinstance(own, dict) or not isinstance(own.get("UserID"), int):
            return None
        with self._lock:
            self._owner = (own["UserID"], time.monotonic())
        return own["UserID"]

    def check(self, address: str) -> str:
        """``ALLOW``, ``DENY`` or ``UNKNOWN`` for one device address."""
        with self._lock:
            known = self._known.get(address)
            if known is not None:
                age = time.monotonic() - known[1]
                if age < (self.ALLOWED_FOR if known[0] else self.DENIED_FOR):
                    return self.ALLOW if known[0] else self.DENY
        # Tailscale is asked outside the lock: one slow answer must not make every
        # other connection wait behind it.
        owner = self._owner_id()
        if owner is None:
            log.info("Tailscale did not answer; %s is asked to wait", address)
            return self.UNKNOWN
        node = self._json("whois", "--json", address).get("Node")
        node = node if isinstance(node, dict) else {}
        allowed = bool(node) and (not self.owner_only or node.get("User") == owner)
        name = str(node.get("ComputedName") or node.get("Name") or "unknown device")
        with self._lock:
            if len(self._known) > 256:
                self._known.clear()
            self._known[address] = (allowed, time.monotonic(), name)
        log.info("%s: %s (%s)", "allowed" if allowed else "REFUSED", name, address)
        return self.ALLOW if allowed else self.DENY

    def allows(self, address: str) -> bool:
        return self.check(address) == self.ALLOW

    def name(self, address: str) -> str:
        with self._lock:
            known = self._known.get(address)
        return f"{known[2]} ({address})" if known else address


def read_proxy_line(client: socket.socket) -> str | None:
    """Read a PROXY v1 line when the connection starts with one, and return its source address.

    ``""`` means the connection has no such line: it was opened on this computer.
    ``None`` means a line that could not be read; the connection is refused.
    The address is returned only when it is one: it is later handed to the
    ``tailscale`` command, and text that is not an address has no business there.
    """
    marker = b"PROXY "
    deadline = time.monotonic() + 10.0
    try:
        while True:
            peeked = client.recv(len(marker), socket.MSG_PEEK)
            if not peeked or time.monotonic() > deadline:
                return None
            if peeked != marker[: len(peeked)]:
                return ""
            if len(peeked) == len(marker):
                break
            time.sleep(0.01)
        line = b""
        while not line.endswith(b"\r\n"):
            if len(line) > 107:
                return None
            byte = client.recv(1)
            if not byte:
                return None
            line += byte
    except OSError:
        return None
    parts = line.decode("ascii", errors="replace").split()
    if len(parts) < 6 or parts[1] not in {"TCP4", "TCP6"}:
        return None
    try:
        return str(ipaddress.ip_address(parts[2]))
    except ValueError:
        return None


# ------------------------------------------------------------------ panel locator
def _windows_candidates() -> list[int]:
    """Loopback ports that a possible panel process is listening on."""
    names: dict[str, str] = {}
    for line in _run(["tasklist", "/fo", "csv", "/nh"], "mbcs").splitlines():
        parts = [part.strip('"') for part in line.split('","')]
        if len(parts) >= 2:
            names[parts[1]] = parts[0].casefold()
    ports: list[int] = []
    for line in _run(["netstat", "-ano", "-p", "TCP"], "mbcs").splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[4].isascii() or not parts[4].isdigit():
            continue
        host, _, port = parts[1].rpartition(":")
        if host not in {"127.0.0.1", "0.0.0.0"} or not DIGITS.match(port) or parts[2] != "0.0.0.0:0":
            continue
        if names.get(parts[4], "") in PANEL_OWNERS:
            ports.append(int(port))
    return ports


def _wsl_candidates() -> list[int]:
    """Listening ports inside WSL distributions that are already running.

    Used when the Windows port table does not show the forwarded port. A stopped
    distribution is left stopped.
    """
    running = _run(["wsl.exe", "-l", "--running", "-q"], "utf-16-le")
    ports: list[int] = []
    for distro in (name.strip("\x00﻿ \r") for name in running.splitlines()):
        if not distro or distro.startswith("docker-desktop"):
            continue
        for line in _run(["wsl.exe", "-d", distro, "-e", "ss", "-ltnH"]).splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            host, _, port = parts[3].rpartition(":")
            if DIGITS.match(port) and host in {"127.0.0.1", "0.0.0.0", "*", "[::]"}:
                ports.append(int(port))
    return ports


def panel_shaped(data: object) -> bool:
    """True for the panel's own ``/progress`` answer.

    A Cell relay with no panel answers the same address with the same keys, and
    marks its answer ``cell``. That one is not a panel, and adopting it would
    point this relay at another relay.
    """
    return isinstance(data, dict) and PROGRESS_KEYS <= data.keys() and "cell" not in data


def is_panel(port: int) -> bool:
    """True when ``127.0.0.1:port`` answers ``/progress`` as an open panel does."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1.5)
    try:
        connection.request("GET", "/progress")
        response = connection.getresponse()
        if response.status != 200:
            return False
        data = json.loads(response.read(65536))
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        connection.close()
    # ``stand_in`` marks the test suite's own panel, which a relay in service must
    # never adopt when the tests happen to run on the same computer.
    if not panel_shaped(data) or data.get("closed") is not False or data.get("stand_in"):
        return False
    step = data.get("step")
    return isinstance(step, int) and not isinstance(step, bool)


class Panel:
    """Knows where the panel is, and talks to it."""

    def __init__(self, program: Program, *, fixed_port: int | None, discover: bool, own_port: int) -> None:
        self._program = program
        self._fixed = fixed_port
        self._discover = discover
        self._own = own_port
        self._lock = threading.Lock()
        self._found: int | None = None
        self._searching = False
        self._searched = 0.0
        #: When each port was first seen silent. Cleared by its next answer.
        self._silent: dict[int, float] = {}

    def port(self) -> int | None:
        """The panel's port: the program Cell started, else a fixed one, else a found one.

        A panel started on the computer is looked for in the background. This call
        only reports what the last search found, so no request waits on a search.
        """
        managed = self._program.panel_port()
        if managed is not None:
            return managed
        if self._fixed is not None:
            return self._fixed
        with self._lock:
            if self._found is not None or not self._discover:
                return self._found
            if self._searching or time.monotonic() - self._searched < SEARCH_EVERY:
                return None
            self._searching = True
        threading.Thread(target=self._search_job, name="cell-panel-search", daemon=True).start()
        return None

    def _search_job(self) -> None:
        found = None
        try:
            found = self._search()
        except Exception:
            log.exception("the search for a panel failed")
        with self._lock:
            self._searching = False
            self._searched = time.monotonic()
            if found is not None and self._found is None:
                self._found = found
        if found is not None:
            log.info("panel found on 127.0.0.1:%s (started on the computer)", found)

    def _search(self) -> int | None:
        seen: set[int] = {self._own}
        for source in (_windows_candidates, _wsl_candidates):
            for port in source():
                if port in seen:
                    continue
                seen.add(port)
                if is_panel(port):
                    return port
        return None

    def _forget(self, port: int) -> None:
        with self._lock:
            if self._found == port:
                self._found = None
            self._silent.pop(port, None)

    def fetch(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float | None = 30.0,
        port: int | None = None,
    ) -> Answer:
        """One request to the panel. ``timeout=None`` waits as long as the program takes."""
        if port is None:
            port = self.port()
        if port is None:
            raise PanelGone
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
        try:
            try:
                connection.connect()
            except OSError as error:
                self._forget(port)
                raise PanelGone from error
            sock = connection.sock
            sock.settimeout(timeout)
            if timeout is None:
                # A held submission can be quiet for hours. Keep-alive probes are
                # what tell a panel that is working from a connection that died.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if hasattr(socket, "SIO_KEEPALIVE_VALS"):
                    sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 30_000, 5_000))
            try:
                connection.request(method, path, body=body, headers=headers or {})
                response = connection.getresponse()
                data = response.read()
            except (OSError, ValueError, http.client.HTTPException) as error:
                raise PanelSlow from error
            found = {name.casefold(): value for name, value in response.getheaders()}
            return Answer(response.status, found, data)
        finally:
            connection.close()

    def look(self) -> Look:
        """Ask the panel where it is. Asked fresh every time: the panel is on this computer.

        ``slow`` is a known panel that did not answer this once. It becomes ``gone``
        only after staying silent for ``GONE_AFTER`` seconds, so one late answer
        does not throw a phone off the page it is filling in.
        """
        port = self.port()
        if port is None:
            return Look("gone")
        try:
            data = json.loads(self.fetch("GET", "/progress", timeout=3.0, port=port).body)
        except PanelGone:
            return Look("gone")
        except (PanelSlow, ValueError):
            data = None
        if not panel_shaped(data):
            now = time.monotonic()
            with self._lock:
                since = self._silent.setdefault(port, now)
                found = self._found == port
            if now - since < GONE_AFTER:
                return Look("slow")
            if found:
                # Something else is listening there now. Look for the panel again.
                self._forget(port)
            return Look("gone")
        with self._lock:
            self._silent.pop(port, None)
        return Look("closed" if data.get("closed") else "open", data)


# ------------------------------------------------------------- slow submissions
class Pending:
    """One page sent to the program that has not been answered yet."""

    def __init__(self, port: int | None) -> None:
        self.token = secrets.token_hex(12)
        self.port = port
        self.created = time.monotonic()
        self.finished = 0.0
        self.done = threading.Event()
        self.go = "/"
        self.answer: Answer | None = None
        self._lock = threading.Lock()

    def finish(self, go: str, answer: Answer | None = None) -> bool:
        """Settle where the phone goes next. The first call wins; a later one is ignored."""
        with self._lock:
            if self.done.is_set():
                return False
            self.go = go
            self.answer = answer
            self.finished = time.monotonic()
            self.done.set()
            return True


class Pendings:
    KEEP_DONE = 1800.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._all: dict[str, Pending] = {}

    def new(self, port: int | None) -> Pending:
        pending = Pending(port)
        with self._lock:
            now = time.monotonic()
            for token in [t for t, p in self._all.items() if p.done.is_set() and now - p.finished > self.KEEP_DONE]:
                del self._all[token]
            self._all[pending.token] = pending
        return pending

    def get(self, token: str) -> Pending | None:
        with self._lock:
            return self._all.get(token)

    def active(self) -> Pending | None:
        """The newest submission still being worked on."""
        with self._lock:
            waiting = [p for p in self._all.values() if not p.done.is_set()]
        return max(waiting, key=lambda p: p.created) if waiting else None


# ------------------------------------------------------------------------- relay
class Relay:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.port: int = arguments.listen_port
        self.max_body: int = arguments.max_upload_mb * 1024 * 1024
        self.slow_after: float = arguments.slow_after
        local_argv = json.loads(arguments.program_argv) if arguments.program_argv else None
        self.program = Program(
            distro=arguments.distro,
            repo=arguments.repo,
            command=arguments.program,
            local_argv=local_argv,
            logger=log,
        )
        self.panel = Panel(
            self.program,
            fixed_port=arguments.panel_port,
            discover=not arguments.no_discovery,
            own_port=arguments.listen_port,
        )
        self.pendings = Pendings()
        tailscale = arguments.owner_only or arguments.tailscale
        self.gate = Gate(tailscale, owner_only=not arguments.any_tailnet_device) if tailscale else None
        self.binds = [item for item in arguments.bind if item not in LOOPBACK]
        self.allow_peers = frozenset(arguments.allow_peer)
        self.hosts = frozenset(
            host_of(name) for name in ["127.0.0.1", "localhost", "::1", *self.binds, *arguments.allow_host] if name
        )
        #: Held while a page is checked and fetched, and while a submission is taken.
        #: A submission through this relay can then never move the panel on between
        #: the check of the step and the request that names it.
        self._nav = threading.Lock()
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        self._viewer: dict[tuple[int | None, str], Response] = {}
        self._gzipped: dict[str, bytes] = {}
        self._static = self._build_static()

    @property
    def gate_name(self) -> str:
        if self.gate is None:
            return "none"
        return "owner" if self.gate.owner_only else "tailnet"

    # -------------------------------------------------------------- static
    @staticmethod
    def _build_static() -> dict[str, Response]:
        def fixed(body: bytes, content_type: str, cache: str) -> Response:
            return Response(200, body, content_type, etag='"' + hashlib.sha1(body).hexdigest()[:16] + '"', cache=cache)

        script = "application/javascript; charset=utf-8"
        day = "public, max-age=86400"
        touch = fixed(cell_pages.png_icon(180), "image/png", day)
        return {
            "/cell/panel.js": fixed(cell_pages.PANEL_JS.encode("utf-8"), script, "no-cache"),
            "/cell/panel.css": fixed(cell_pages.PANEL_CSS.encode("utf-8"), "text/css; charset=utf-8", "no-cache"),
            "/cell/app.webmanifest": fixed(cell_pages.manifest(), "application/manifest+json", "no-cache"),
            "/cell/icon-180.png": touch,
            "/cell/icon-192.png": fixed(cell_pages.png_icon(192), "image/png", day),
            "/cell/icon-512.png": fixed(cell_pages.png_icon(512), "image/png", day),
            "/apple-touch-icon.png": touch,
            "/apple-touch-icon-precomposed.png": touch,
            "/favicon.ico": fixed(cell_pages.png_icon(48), "image/png", day),
        }

    # ---------------------------------------------------------- connections
    def serve(self, address: str) -> None:
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        server = socket.socket(family, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            # On Windows this stops another program from binding the same port over this one.
            server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((address, self.port))
        server.listen(128)
        log.info("Cell relay %s listening on %s:%s (admits: %s)", VERSION, address, self.port, self.gate_name)
        while True:
            try:
                client, peer = server.accept()
            except OSError:
                time.sleep(0.05)
                continue
            if not self._slots.acquire(blocking=False):
                client.close()
                continue
            try:
                threading.Thread(target=self._connection, args=(client, peer[0]), daemon=True).start()
            except RuntimeError:
                # No thread could be started. Drop this caller and keep serving.
                self._slots.release()
                client.close()

    def _connection(self, client: socket.socket, peer: str) -> None:
        try:
            client.settimeout(HEADER_TIMEOUT)
            source = ""
            if peer in LOOPBACK:
                # Tailscale Serve connects from loopback and names the device first.
                found = read_proxy_line(client)
                if found is None:
                    self._refuse(client, html_response(DENIED_PAGE, 403))
                    return
                source = found
                if source and self.gate is not None:
                    verdict = self.gate.check(source)
                    if verdict == Gate.UNKNOWN:
                        self._refuse(client, html_response(CHECKING_PAGE, 503))
                        return
                    if verdict != Gate.ALLOW:
                        self._refuse(client, html_response(DENIED_PAGE, 403))
                        return
            elif peer not in self.allow_peers:
                self._refuse(client, html_response(DENIED_PAGE, 403))
                return
            if source:
                device = self.gate.name(source) if self.gate is not None else source
            else:
                device = "this computer" if peer in LOOPBACK else peer
            reader = client.makefile("rb", 65536)
            first = True
            while True:
                client.settimeout(HEADER_TIMEOUT if first else IDLE_TIMEOUT)
                first = False
                try:
                    request = read_head(reader, self.max_body)
                    if request is None:
                        return
                    request.source = source
                    request.device = device
                    request.local = peer in LOOPBACK and not source
                    refusal = self._precheck(request)
                    if refusal is not None:
                        self._refuse(client, refusal, request)
                        return
                    read_body(reader, client, request)
                except HttpError as error:
                    self._refuse(client, text_response(error.status, str(error)))
                    return
                try:
                    response = self._dispatch(request)
                except Exception:
                    log.exception("request failed: %s %s", request.method, request.path)
                    response = text_response(500, "Cell hit an error. See cell.log on the computer.")
                keep = request.keep_alive and response.keep_alive
                self._write(client, request, response, keep_alive=keep)
                if callable(response.after):
                    response.after()
                if not keep:
                    return
        except (OSError, ValueError):
            return
        finally:
            self._slots.release()
            try:
                client.close()
            except OSError:
                pass

    def _refuse(self, client: socket.socket, response: Response, request: Request | None = None) -> None:
        self._write(client, request, response, keep_alive=False)
        self._linger(client)

    @staticmethod
    def _linger(client: socket.socket) -> None:
        """Let the caller finish sending before the connection closes.

        Closing while the caller is still sending, an upload that is too large for
        example, resets the connection, and the caller never sees the answer.
        """
        try:
            client.shutdown(socket.SHUT_WR)
            client.settimeout(1.0)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and client.recv(65536):
                pass
        except OSError:
            pass

    def _write(self, client: socket.socket, request: Request | None, response: Response, *, keep_alive: bool) -> None:
        body = response.body
        status = response.status
        headers = list(response.headers)
        if response.etag:
            headers.append(("ETag", response.etag))
            if request is not None and request.headers.get("if-none-match", "") == response.etag:
                status, body = 304, b""
        accepts = request.headers.get("accept-encoding", "").casefold() if request is not None else ""
        kind = response.content_type.casefold()
        if status != 304 and len(body) >= 1024 and "gzip" in accepts and kind.startswith(COMPRESSIBLE):
            body = self._gzip(body, response.etag)
            headers.append(("Content-Encoding", "gzip"))
        if kind.startswith(COMPRESSIBLE):
            headers.append(("Vary", "Accept-Encoding"))
        if status != 304:
            # A 304 has no body, and a length on it would describe the wrong thing.
            headers.append(("Content-Type", response.content_type))
            headers.append(("Content-Length", str(len(body))))
        headers.append(("Cache-Control", response.cache))
        headers.append(("Connection", "keep-alive" if keep_alive else "close"))
        headers.extend(SECURITY_HEADERS)
        head = f"HTTP/1.1 {status} {HTTPStatus(status).phrase}\r\n"
        head += "".join(f"{name}: {value}\r\n" for name, value in headers) + "\r\n"
        if request is not None and request.method == "HEAD":
            body = b""
        try:
            client.sendall(head.encode("latin-1", errors="replace") + body)
        except OSError:
            pass

    def _gzip(self, body: bytes, etag: str) -> bytes:
        """Compress, remembering the result for bodies that never change."""
        if not etag:
            return gzip.compress(body, 5)
        cached = self._gzipped.get(etag)
        if cached is None:
            cached = gzip.compress(body, 6)
            if len(self._gzipped) > 32:
                self._gzipped.clear()
            self._gzipped[etag] = cached
        return cached

    # ------------------------------------------------------------- routing
    def _precheck(self, request: Request) -> Response | None:
        """Refuse a request that is not for this relay, before its body is read."""
        if host_of(request.headers.get("host", "")) not in self.hosts:
            # A page on some other site that resolves its own name to this address
            # arrives with that name here. It is not one of ours.
            return text_response(403, "Cell does not answer to this address.")
        if request.method not in {"GET", "HEAD", "POST"}:
            return text_response(405, "Method not allowed.")
        site = request.headers.get("sec-fetch-site", "").casefold()
        if site and site not in {"same-origin", "none"}:
            # Sent by another site's page. It cannot read the answer, but the request
            # itself would still do its work here, so only opening a page is allowed.
            opening = (
                request.method == "GET"
                and request.headers.get("sec-fetch-mode", "").casefold() == "navigate"
                and request.headers.get("sec-fetch-dest", "").casefold() == "document"
            )
            if not opening:
                return text_response(403, "This request came from another site.")
        if request.method == "POST" and not self._same_origin(request):
            return text_response(403, "This form did not come from a Cell page.")
        return None

    @staticmethod
    def _same_origin(request: Request) -> bool:
        """True when a POST was sent by a page of this relay, at the address it was sent to.

        Every current browser names the sending page in Origin on a POST. Scheme,
        host and port must all be this relay's: another program's page on the same
        computer has the same host and a different port, and is not a Cell page.
        A POST that names no page at all is refused as well.
        """
        origin = request.headers.get("origin", "")
        if origin:
            if origin.casefold() == "null":
                return False
            sender = origin
        else:
            sender = request.headers.get("referer", "")
        if not sender:
            return False
        try:
            parts = urlsplit(sender)
        except ValueError:
            return False
        called = request.headers.get("host", "").strip().casefold()
        return parts.scheme in {"http", "https"} and bool(called) and parts.netloc.casefold() == called

    def _dispatch(self, request: Request) -> Response:
        if request.path in self._static:
            return self._static[request.path]
        if request.path == "/cell" or request.path.startswith("/cell/"):
            return self._hub(request)
        return self._panel(request)

    @staticmethod
    def _forwarded(request: Request) -> dict[str, str]:
        kept = {}
        for name, sent in (
            ("content-type", "Content-Type"),
            ("user-agent", "User-Agent"),
            ("accept", "Accept"),
            ("accept-language", "Accept-Language"),
        ):
            value = request.headers.get(name, "")
            if value and value.isascii() and value.isprintable():
                kept[sent] = value
        return kept

    def where(self) -> str | None:
        """The address a phone should be showing right now, or ``None`` when that is not known."""
        waiting = self.pendings.active()
        if waiting is not None:
            return "/cell/working?t=" + waiting.token
        look = self.panel.look()
        if look.kind == "slow":
            return None
        if look.kind != "open" or look.step is None:
            return "/cell/"
        return f"/step/{look.step}"

    # --------------------------------------------------------------- panel
    def _panel(self, request: Request) -> Response:
        path = request.path
        if request.method == "POST":
            return self._action(request)
        if path == "/progress":
            return self._progress()
        if path.startswith("/viewer/") or path.startswith("/structure/"):
            return self._asset(request)
        return self._page(request)

    def _progress(self) -> Response:
        look = self.panel.look()
        if look.kind == "slow":
            # The page's own script ignores an answer it cannot read and asks again.
            return text_response(503, "The panel did not answer.")
        if look.kind == "gone":
            # A panel page that is still open learns the panel is gone and leaves
            # for /closed, which lands on the Cell page. Marked, so that another
            # relay looking for a panel does not take this answer for one.
            return json_response(
                {
                    "active": False,
                    "done": 0,
                    "total": 0,
                    "label": "",
                    "action": "",
                    "step": None,
                    "closed": True,
                    "cell": VERSION,
                }
            )
        return json_response(look.data)

    def _asset(self, request: Request) -> Response:
        path = request.path
        key = (self.panel.port(), path)
        if path.startswith("/viewer/") and key in self._viewer:
            return self._viewer[key]
        try:
            answer = self.panel.fetch("GET", path, headers=self._forwarded(request))
        except (PanelGone, PanelSlow):
            return text_response(503, "The panel is not open.")
        if answer.status in REDIRECTS:
            # The panel answers an address it does not know with its current page.
            # A script or a structure is never a page.
            return text_response(404, "Not found.")
        if answer.status != 200:
            return Response(answer.status, answer.body, answer.content_type)
        if path.startswith("/viewer/"):
            # The viewer script is the same file on every page. The panel sends it
            # uncached each time, which is half a megabyte per page on a phone.
            response = Response(
                200,
                answer.body,
                answer.content_type,
                etag='"' + hashlib.sha1(answer.body).hexdigest()[:16] + '"',
                cache="public, max-age=86400",
            )
            if len(self._viewer) > 4:
                self._viewer.clear()
            self._viewer[key] = response
            return response
        return Response(200, answer.body, answer.content_type)

    def _page(self, request: Request) -> Response:
        with self._nav:
            waiting = self.pendings.active()
            if waiting is not None:
                return redirect("/cell/working?t=" + waiting.token)
            look = self.panel.look()
            if look.kind == "slow":
                return html_response(BUSY_PAGE, 503)
            if look.kind != "open" or look.step is None:
                return redirect("/cell/")
            current = look.step
            asked = STEP_PATH.match(request.path)
            if asked is None or int(asked.group(1)) != current:
                # Never forwarded: to the panel an earlier step means "go back there".
                # /closed is not forwarded either: an open panel would answer it with
                # its closing page.
                return redirect(f"/step/{current}")
            headers = self._forwarded(request)
            # Tells the panel not to treat this as its own browser going back. The
            # panel can move on between the check above and this request when the
            # desktop window sends a page; a panel that knows the header then
            # answers with the current page instead of rewinding.
            headers["X-Cell-No-Rewind"] = "1"
            try:
                answer = self.panel.fetch("GET", f"/step/{current}", headers=headers, timeout=8.0)
            except PanelGone:
                return redirect("/cell/")
            except PanelSlow:
                return html_response(BUSY_PAGE, 503)
        return self._from_panel(answer)

    def _from_panel(self, answer: Answer) -> Response:
        if answer.status in REDIRECTS:
            return redirect(local_path(answer.headers.get("location", "/")))
        if answer.is_html:
            return html_response(cell_pages.inject(answer.body), answer.status)
        return Response(answer.status, answer.body, answer.content_type)

    def _action(self, request: Request) -> Response:
        """A submitted panel page. The program may take hours to answer it."""
        with self._nav:
            waiting = self.pendings.active()
            if waiting is not None:
                # The panel would make this one wait for the first and give it the
                # same page, so it is not sent at all.
                return redirect("/cell/working?t=" + waiting.token)
            look = self.panel.look()
            if look.kind in {"gone", "closed"}:
                # No panel, or one that is shutting down: there is nothing to answer.
                return redirect("/cell/")
            # A panel that is merely slow to say where it is still gets the form.
            # Dropping what was typed because of one late answer is the worse error.
            pending = self.pendings.new(self.panel.port())
        threading.Thread(
            target=self._hold,
            args=(pending, request.target, self._forwarded(request), request.body),
            name="cell-submit",
            daemon=True,
        ).start()
        if pending.done.wait(self.slow_after):
            return redirect(pending.go)
        return redirect("/cell/working?t=" + pending.token)

    def _hold(self, pending: Pending, target: str, headers: dict[str, str], body: bytes) -> None:
        """Send one submission and watch over it until it is answered or its panel is gone.

        The answer can take hours, and nothing else on the phone works while it is
        awaited. So the wait must end when the panel it was sent to is no longer
        there, even if that connection never says so.
        """
        threading.Thread(
            target=self._send, args=(pending, target, headers, body), name="cell-submit-send", daemon=True
        ).start()
        silent = 0
        while not pending.done.wait(5.0):
            if self.panel.port() != pending.port:
                log.info("a held submission was dropped: its panel is gone")
                pending.finish("/cell/")
                return
            silent = silent + 1 if self.panel.look().kind in {"slow", "gone"} else 0
            if silent >= HELD_SILENCE:
                log.info("a held submission was dropped: its panel stopped answering")
                pending.finish("/cell/")
                return

    def _send(self, pending: Pending, target: str, headers: dict[str, str], body: bytes) -> None:
        try:
            try:
                answer = self.panel.fetch("POST", target, headers=headers, body=body, timeout=None, port=pending.port)
            except (PanelGone, PanelSlow):
                pending.finish("/cell/")
                return
            if answer.status in REDIRECTS:
                pending.finish(local_path(answer.headers.get("location", "/")))
                return
            if answer.is_html and self.panel.look().kind in {"gone", "closed"}:
                # The answer to Close is a page that only shuts a desktop window.
                pending.finish("/cell/")
                return
            pending.finish("/cell/result?t=" + pending.token, answer)
        except Exception:
            log.exception("submission failed")
            pending.finish("/cell/")

    # ----------------------------------------------------------------- hub
    def _status(self, request: Request) -> Response:
        since = request.first("since")
        # The program first, the panel second: a program reported as ended is then
        # never paired with a panel that was still open a moment earlier.
        program = self.program.state()
        look = self.panel.look()
        is_open = look.kind in {"open", "slow"}
        status: dict[str, object] = {
            "version": VERSION,
            "device": request.device,
            "gate": self.gate_name,
            "program": program,
            "panel": {
                "open": is_open,
                "busy": look.kind == "slow",
                "step": look.step,
                "progress": (
                    {name: look.data.get(name) for name in PROGRESS_FIELDS} if look.kind == "open" and look.data else None
                ),
            },
            "output": self.program.output(int(since) if DIGITS.match(since) else 0),
        }
        token = request.first("t")
        if token:
            pending = self.pendings.get(token) if TOKEN.match(token) else None
            if pending is None:
                status["follow"] = {"state": "unknown", "go": self.where() or "/cell/"}
            elif pending.done.is_set():
                status["follow"] = {"state": "done", "go": pending.go}
            else:
                status["follow"] = {"state": "pending"}
        return json_response(status)

    def _hub(self, request: Request) -> Response:
        path = request.path.rstrip("/") or "/cell"
        if request.method == "POST":
            return self._hub_post(request, path)
        if path == "/cell":
            waiting = self.pendings.active()
            if waiting is not None:
                return redirect("/cell/working?t=" + waiting.token)
            return html_response(cell_pages.hub_page())
        if path == "/cell/working":
            token = request.first("t")
            if not TOKEN.match(token):
                return redirect("/cell/")
            return html_response(cell_pages.hub_page("working", token))
        if path == "/cell/status":
            return self._status(request)
        if path == "/cell/where":
            return json_response({"go": self.where()})
        if path == "/cell/go":
            target = self.where()
            return redirect(target) if target else html_response(BUSY_PAGE, 503)
        if path == "/cell/log":
            lines, note = self.program.tail_log()
            return json_response({"lines": lines, "note": note})
        if path == "/cell/result":
            pending = self.pendings.get(request.first("t")) if TOKEN.match(request.first("t")) else None
            if pending is None or pending.answer is None:
                return redirect("/")
            return self._from_panel(pending.answer)
        if path == "/cell/ping":
            return json_response({"cell": VERSION, "program": self.program.state()["state"], "gate": self.gate_name})
        return text_response(404, "Not found.")

    def _hub_post(self, request: Request, path: str) -> Response:
        if path == "/cell/start":
            if not self.program.alive() and self.panel.look().kind in {"open", "slow"}:
                return json_response({"ok": False, "message": "The program is already running on the computer."})
            ok, message = self.program.start()
            return json_response({"ok": ok, "message": message})
        if path == "/cell/stop":
            external = None if self.program.alive() else self.panel.port()
            ok, message = self.program.stop(external_port=external)
            return json_response({"ok": ok, "message": message})
        if path == "/cell/stdin":
            fields = parse_qs(request.body.decode("utf-8", errors="replace"), keep_blank_values=True)
            ok, message = self.program.send_line((fields.get("line") or [""])[0])
            return json_response({"ok": ok, "message": message})
        if path == "/cell/shutdown":
            if not request.local:
                return text_response(403, "Only the computer itself may stop the relay.")
            return Response(200, b'{"ok": true}', "application/json", keep_alive=False, after=self._shutdown)
        return text_response(404, "Not found.")

    def _shutdown(self) -> None:
        """Stop the program cleanly, then end the relay. Called by Stop-Cell.ps1."""
        log.info("shutdown requested from this computer")
        if self.program.alive():
            self.program.stop()
            deadline = time.monotonic() + 30.0
            while self.program.alive() and time.monotonic() < deadline:
                time.sleep(0.2)
        logging.shutdown()
        os._exit(0)


# -------------------------------------------------------------------------- main
def default_repo() -> str:
    """The checkout that holds this folder, as WSL names it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    drive, rest = os.path.splitdrive(root)
    if len(drive) == 2 and drive[1] == ":":
        return "/mnt/" + drive[0].lower() + rest.replace("\\", "/")
    return root.replace("\\", "/")


def configure_logging(path: str | None) -> None:
    log.setLevel(logging.INFO)
    layout = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    if path:
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=1_000_000, backupCount=2, encoding="utf-8"
        )
        handler.setFormatter(layout)
        log.addHandler(handler)
    if sys.stdout is not None:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(layout)
        log.addHandler(console)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--listen-port", type=int, default=8766, help="fixed port this relay listens on")
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        metavar="ADDRESS",
        help="extra address to listen on: this computer's address inside a WireGuard tunnel. Needs --allow-peer.",
    )
    parser.add_argument(
        "--allow-peer",
        action="append",
        default=[],
        metavar="ADDRESS",
        help="a caller admitted on an extra --bind address: the phone's address inside that tunnel. Repeat for more.",
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="NAME",
        help="a name or address phones use for this computer, such as its Tailscale address. Repeat for more.",
    )
    parser.add_argument(
        "--owner-only",
        metavar="TAILSCALE_EXE",
        default=None,
        help="admit only this user's own tailnet devices, named by Tailscale Serve's PROXY protocol line",
    )
    parser.add_argument(
        "--tailscale",
        metavar="TAILSCALE_EXE",
        default=None,
        help="the tailscale command, used to name and check tailnet devices. Implied by --owner-only.",
    )
    parser.add_argument(
        "--any-tailnet-device",
        action="store_true",
        help="admit every device Tailscale can name, not only this user's own",
    )
    parser.add_argument("--panel-port", type=int, default=None, help="use this panel port and skip the search")
    parser.add_argument("--no-discovery", action="store_true", help="do not look for a panel started on the computer")
    parser.add_argument("--distro", default="", help="WSL distribution the program runs in (default: the default one)")
    parser.add_argument("--repo", default=default_repo(), help="the checkout, as WSL names it")
    parser.add_argument("--program", default=".venv/bin/human", help="command that starts the program, run in --repo")
    parser.add_argument("--program-argv", default="", help=argparse.SUPPRESS)
    parser.add_argument("--max-upload-mb", type=int, default=200, help="largest form or upload accepted")
    parser.add_argument(
        "--slow-after",
        type=float,
        default=2.5,
        help="seconds a sent page may take before the phone is shown the working page",
    )
    parser.add_argument("--log-file", default=None, help="write the relay log here")
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    for address in arguments.bind:
        try:
            unspecified = ipaddress.ip_address(address).is_unspecified
        except ValueError:
            parser.error(f"--bind takes an address, not {address!r}.")
        if unspecified:
            parser.error("--bind takes one tunnel address, not every address of the computer.")
    if [item for item in arguments.bind if item not in LOOPBACK] and not arguments.allow_peer:
        parser.error("--bind needs --allow-peer: say which caller is admitted on that address.")
    if arguments.any_tailnet_device and not (arguments.tailscale or arguments.owner_only):
        parser.error("--any-tailnet-device needs --tailscale.")

    configure_logging(arguments.log_file)
    try:
        relay = Relay(arguments)
        for address in relay.binds:
            threading.Thread(target=relay.serve, args=(address,), daemon=True).start()
        relay.serve("127.0.0.1")
    except KeyboardInterrupt:
        return 0
    except OSError as error:
        log.error("Cell relay could not start: %s", error)
        return 1
    except Exception:
        log.exception("Cell relay stopped on an unexpected error")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
