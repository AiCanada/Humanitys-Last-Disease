"""Grok Bot on this computer, with no API key.

Submit question opens the Grok Bot desktop app and pastes the question box into
the focused Chief of Staff chat. The reply is read back from that window.
``grokbot://`` links cannot carry a prompt, so the text rides on stdin to
``scripts/paste-grok-bot.ps1`` and the reply comes from
``scripts/read-desktop-reply.ps1``. The program that is opened is
``GROK_BOT_LINK`` or Grok Bot's own link, never a path from the request.
Nothing here is written to the run log.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

GROK_BOT = "grokbot"
GROK_BOT_MODEL = "chief-of-staff"
GROK_BOT_LABEL = "Grok Bot"
APP_LINK = "grokbot://app/v1/open"
PASTE_MAX = 16 * 1024
OPEN_INTERVAL_S = 3.0
SETTLE_S = 2.5
#: The paste script's own fixed waits, which this timeout must not cut short:
#: 12000 waiting for the window, 700 for a restore from the taskbar, 450 after
#: the topmost pin, a 5000 deadline looking for the foreground, then 120, four
#: clicks of 90, 350 after the clicks, 800 while the chat tree is published,
#: 80 and 800 after the paste.
PASTE_BUDGET_S = (12_000 + 700 + 450 + 5_000 + 120 + 4 * 90 + 350 + 800 + 80 + 800) / 1000
#: A backstop for a script that hangs, not a limit on a script doing its work.
#: Set below the budget above it fired during a slow but succeeding run, and a
#: terminated script skips the cleanup that unpins the window, so the window
#: was left on top of everything while the caller reported a failed paste.
PASTE_TIMEOUT_S = PASTE_BUDGET_S + 15.0
FIND_MS = 60_000
REPLY_TIMEOUT_MS = 10 * 60_000
QUIET_MS = 15_000
READ_SLACK_S = 30.0
STDOUT_MAX = 1024 * 1024

AskDesktop = Callable[[str], str]
Clock = Callable[[], float]
Exists = Callable[[str], bool]

_TAG = re.compile(r"^GEV-[0-9a-f]{6,16}$")
_SHORTCUT = re.compile(r"^[A-Za-z]:\\.*\.(?:lnk|exe)$", re.IGNORECASE)
_MOUNT = re.compile(r"/mnt/([A-Za-z])/(.*)")
_DRIVE = re.compile(r"([A-Za-z]):\\(.*)")
_UNSAFE = re.compile(r"[\u0000-\u001f\u007f\"]")

_open_lock = threading.Lock()
_turn_lock = threading.Lock()
_last_open = -OPEN_INTERVAL_S

_TASK_PREFIX = "Chief of Staff, act on this request from PROJECT H.U.M.A.N.\n\n"
_TASK_SUFFIX = (
    "\n\nAnswer in this chat. Do not invent a database result, a PDB id, an "
    "amino-acid sequence, a synthesis step, or a probability of cure. If you "
    "do not know, say so.\n\n"
)


def repo_root() -> Path:
    """The checkout that holds the paste and reply scripts."""
    return Path(__file__).resolve().parents[3]


def paste_script() -> Path:
    """The script that pastes into the focused Grok Bot composer."""
    return repo_root() / "scripts" / "paste-grok-bot.ps1"


def reply_script() -> Path:
    """The script that reads the reply after the pasted marker."""
    return repo_root() / "scripts" / "read-desktop-reply.ps1"


def reset_open_claim() -> None:
    """Forget the last open. Tests use this so one case does not block the next."""
    global _last_open
    with _open_lock:
        _last_open = -OPEN_INTERVAL_S


def claim_open(now: float) -> bool:
    """True when this open is at least ``OPEN_INTERVAL_S`` after the previous one."""
    global _last_open
    with _open_lock:
        if now - _last_open < OPEN_INTERVAL_S:
            return False
        _last_open = now
        return True


def new_tag(avoid: str = "") -> str:
    """A marker the reply reader accepts, and which the question does not already contain."""
    for _ in range(8):
        tag = "GEV-" + secrets.token_hex(4)
        if _TAG.fullmatch(tag) and tag not in avoid:
            return tag
    raise ValueError("Grok Bot could not be opened.")


def grok_bot_task(question: str, tag: str) -> str:
    """The paste: the request, the same limits as a model answer, then the marker.

    The marker is the last line. The reader treats the text after the message
    that carries it as the reply. A question that does not fit keeps its end,
    so the marker and the request to act both stay.
    """
    if not _TAG.fullmatch(tag):
        raise ValueError("Grok Bot could not be opened.")
    suffix = f"{_TASK_SUFFIX}[{tag}]"
    room = PASTE_MAX - len(_TASK_PREFIX) - len(suffix)
    if room < 1:
        raise ValueError("The request is too long to paste into Grok Bot.")
    body = question.strip()
    if len(body) > room:
        body = body[-room:]
    text = f"{_TASK_PREFIX}{body}{suffix}"
    if len(text) > PASTE_MAX or not text.endswith(f"[{tag}]"):
        raise ValueError("The request is too long to paste into Grok Bot.")
    return text


def launch_target(env: Mapping[str, str] | None = None, *, exists: Exists | None = None) -> tuple[str, str]:
    """What to open: a local ``.lnk`` or ``.exe``, or Grok Bot's own link.

    A configured path that is not a shortcut on this computer is refused.
    It is not replaced, and it is not taken from the request.
    """
    source = os.environ if env is None else env
    configured = str(source.get("GROK_BOT_LINK", "")).strip()
    if not configured:
        return (APP_LINK, "app-link")
    if len(configured) > 1024 or _UNSAFE.search(configured) or _SHORTCUT.fullmatch(configured) is None:
        return ("", "refused")
    check = path_exists if exists is None else exists
    if not check(configured):
        return ("", "refused")
    return (configured, "shortcut")


def path_exists(value: str) -> bool:
    """True when ``value`` is a file here, including a Windows path seen from WSL."""
    if Path(value).is_file():
        return True
    match = _DRIVE.fullmatch(value)
    if match is None or sys.platform == "win32":
        return False
    mounted = Path(f"/mnt/{match.group(1).lower()}") / match.group(2).replace("\\", "/")
    return mounted.is_file()


def windows_path(path: Path) -> str:
    """A path Windows PowerShell can open. A WSL ``/mnt/<drive>`` path is converted."""
    text = path.resolve().as_posix()
    match = _MOUNT.fullmatch(text)
    if match is None:
        return str(path.resolve())
    return f"{match.group(1).upper()}:\\" + match.group(2).replace("/", "\\")


def powershell_command(script: Path, exe: str) -> list[str]:
    """How the script is started. The task is not an argument."""
    return [
        exe,
        "-NoProfile",
        "-STA",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-WindowStyle",
        "Hidden",
        "-File",
        windows_path(script),
    ]


def ask_grok_bot(
    question: str,
    *,
    open_app: Callable[[str], None] | None = None,
    paste: Callable[[str], bool] | None = None,
    read_reply: Callable[[str], str] | None = None,
    sleep: Callable[[float], None] | None = None,
    env: Mapping[str, str] | None = None,
    clock: Clock | None = None,
    exists: Exists | None = None,
) -> str:
    """Paste the question to Chief of Staff and return the text read from the window.

    Injected callables are the tests' seams. The defaults open this computer's
    Grok Bot, paste through the script, and read the reply. One real paste runs
    at a time. The question is not written to the log.
    """
    if open_app is None and paste is None and read_reply is None:
        with _turn_lock:
            return _ask(question, open_app, paste, read_reply, sleep, env, clock, exists)
    return _ask(question, open_app, paste, read_reply, sleep, env, clock, exists)


def _ask(
    question: str,
    open_app: Callable[[str], None] | None,
    paste: Callable[[str], bool] | None,
    read_reply: Callable[[str], str] | None,
    sleep: Callable[[float], None] | None,
    env: Mapping[str, str] | None,
    clock: Clock | None,
    exists: Exists | None,
) -> str:
    if not question.strip():
        raise ValueError("Type a question for the checked models.")
    target, kind = launch_target(env, exists=exists)
    if kind == "refused" or not target:
        raise ValueError("GROK_BOT_LINK is not a Grok Bot shortcut on this computer.")
    real = open_app is None
    if real and not claim_open((clock or time.time)()):
        raise ValueError("Grok Bot is opening.")
    tag = new_tag(question)
    text = grok_bot_task(question, tag)
    pause = time.sleep if sleep is None else sleep
    opener = _open_target if open_app is None else open_app
    sender = _paste_task if paste is None else paste
    reader = _read_reply if read_reply is None else read_reply
    try:
        opener(target)
    except OSError:
        raise ValueError("Grok Bot could not be opened.") from None
    pause(SETTLE_S)
    try:
        pasted = sender(text)
    except OSError:
        raise ValueError("The request could not be pasted into Grok Bot.") from None
    if pasted is not True:
        raise ValueError("The request could not be pasted into Grok Bot.")
    try:
        reply = reader(tag)
    except OSError:
        raise ValueError("Grok Bot's reply did not come back.") from None
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("Grok Bot's reply did not come back.")
    return reply.strip()


def windows_powershell(env: Mapping[str, str] | None = None) -> str:
    """PowerShell 5.1 by its full path, on Windows or from WSL. Empty when it is absent."""
    source = os.environ if env is None else env
    if sys.platform == "win32":
        root = str(source.get("SystemRoot") or source.get("SYSTEMROOT") or "")
        if not re.fullmatch(r"[A-Za-z]:\\Windows\\?", root):
            return ""
        candidate = Path(root.rstrip("\\")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    else:
        candidate = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    return str(candidate) if candidate.is_file() else ""


def windows_explorer(env: Mapping[str, str] | None = None) -> str:
    """Explorer by its full path. Empty when this computer has no Windows desktop."""
    source = os.environ if env is None else env
    if sys.platform == "win32":
        root = str(source.get("SystemRoot") or source.get("SYSTEMROOT") or "")
        if not re.fullmatch(r"[A-Za-z]:\\Windows\\?", root):
            return ""
        candidate = Path(root.rstrip("\\")) / "explorer.exe"
    else:
        candidate = Path("/mnt/c/Windows/explorer.exe")
    return str(candidate) if candidate.is_file() else ""


def _open_target(target: str) -> None:
    opener = windows_explorer()
    if not opener:
        raise ValueError("This computer cannot open Grok Bot.")
    try:
        subprocess.Popen(
            [opener, target],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except OSError:
        raise ValueError("Grok Bot could not be opened.") from None


def _paste_task(text: str) -> bool:
    if not text or len(text) > PASTE_MAX:
        return False
    result = _run_script(paste_script(), text, PASTE_TIMEOUT_S, "The request could not be pasted into Grok Bot.")
    return result.returncode == 0


def _read_reply(tag: str) -> str:
    if not _TAG.fullmatch(tag):
        raise ValueError("Grok Bot's reply did not come back.")
    job = json.dumps(
        {
            "app": "grok-bot",
            "tag": tag,
            "timeoutMs": REPLY_TIMEOUT_MS,
            "findMs": FIND_MS,
            "quietMs": QUIET_MS,
            "copy": False,
        },
        separators=(",", ":"),
    )
    result = _run_script(
        reply_script(),
        job,
        (REPLY_TIMEOUT_MS / 1000) + READ_SLACK_S,
        "Grok Bot's reply did not come back.",
    )
    return _reply_text(result.stdout)


def _run_script(script: Path, stdin: str, timeout: float, failure: str) -> subprocess.CompletedProcess[bytes]:
    exe = windows_powershell()
    if not exe or not script.is_file():
        raise ValueError("This computer cannot open Grok Bot.")
    try:
        return subprocess.run(
            powershell_command(script, exe),
            input=stdin.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError(failure) from None


def _reply_text(raw: bytes) -> str:
    if len(raw) > STDOUT_MAX:
        raise ValueError("Grok Bot's reply did not come back.")
    said: object = None
    for line in reversed(raw.decode("utf-8", errors="replace").splitlines()):
        if not line.strip():
            continue
        try:
            said = json.loads(line)
        except json.JSONDecodeError:
            continue
        break
    if isinstance(said, dict) and said.get("event") == "reply" and isinstance(said.get("b64"), str):
        try:
            decoded = base64.b64decode(said["b64"], validate=True).decode("utf-8").strip()
        except (ValueError, UnicodeDecodeError):
            decoded = ""
        if decoded:
            return decoded
    raise ValueError("Grok Bot's reply did not come back.")
