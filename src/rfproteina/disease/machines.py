"""Remote machines this computer can already reach with ssh.

Generation runs on a remote machine through ``ssh <host>``, so the names worth
offering are the ones ssh itself already resolves: the ``Host`` entries in the
ssh config and the hosts recorded in ``known_hosts``. Nothing is probed over the
network. A scan would be slow, would say nothing about whether ssh can actually
log in, and would reach machines the user never asked about.

Only those two files are read. No key material is opened, and what is read stays
on this computer: the panel that shows the list is served on localhost only.

An ``Include`` directive in the ssh config is not followed, so a host defined
only in an included file is not listed. The free-text box next to the list takes
any name, so such a host can still be used.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

#: Most names offered at once, so a long known_hosts cannot fill the page.
MACHINE_LIMIT = 200
#: Names that mean this machine, which the dropdown adds on its own.
LOCAL_NAMES = frozenset({"local", "localhost", "127.0.0.1", "::1", "0.0.0.0"})
#: A pattern stands for a set of machines, so it names no single one to offer.
_PATTERN_MARKS = ("*", "?", "!")


def ssh_directory() -> Path:
    """Where the ssh files are read from. ``RFD4_SSH_DIR`` selects another one."""
    override = os.environ.get("RFD4_SSH_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".ssh"


def discover_remote_machines(query: str = "", *, limit: int = MACHINE_LIMIT) -> list[str]:
    """Host names ssh already knows, narrowed by ``query`` and sorted.

    A missing or unreadable file is skipped, because having no ssh config is not
    an error. ``query`` matches anywhere in the name, ignoring case. Duplicates
    are removed case-insensitively and the first spelling seen is kept.
    """
    directory = ssh_directory()
    found: dict[str, str] = {}
    for name in _config_hosts(directory / "config") + _known_hosts(directory / "known_hosts"):
        key = name.casefold()
        if key in LOCAL_NAMES or key in found:
            continue
        found[key] = name
    wanted = query.strip().casefold()
    names = [name for key, name in found.items() if not wanted or wanted in key]
    return sorted(names, key=str.casefold)[:limit]


def _config_hosts(path: Path) -> list[str]:
    """``Host`` names from an ssh config, without the wildcard patterns."""
    hosts: list[str] = []
    for line in _lines(path):
        parts = line.replace("=", " ").split()
        if len(parts) < 2 or parts[0].casefold() != "host":
            continue
        for pattern in parts[1:]:
            if not any(mark in pattern for mark in _PATTERN_MARKS):
                hosts.append(pattern)
    return hosts


def _known_hosts(path: Path) -> list[str]:
    """Host names from ``known_hosts``. A hashed entry names nothing readable."""
    hosts: list[str] = []
    for line in _lines(path):
        parts = line.split()
        if not parts:
            continue
        # A marker such as @cert-authority or @revoked precedes the name field.
        field = parts[1] if parts[0].startswith("@") and len(parts) > 1 else parts[0]
        if field.startswith("|"):
            continue
        for entry in field.split(","):
            name = _plain_host(entry)
            if name and not any(mark in name for mark in _PATTERN_MARKS):
                hosts.append(name)
    return hosts


def _plain_host(entry: str) -> str:
    """``[host]:port`` and a bare ``host`` both come back as ``host``."""
    text = entry.strip()
    if text.startswith("[") and "]" in text:
        return text[1 : text.index("]")].strip()
    return text


#: How long the network folder may stay open before the page stops waiting.
BROWSE_TIMEOUT_S = 600.0
_HOST_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,251}[A-Za-z0-9])?")
#: Windows File Explorer rooted at Network. The person picks a computer. The
#: dialog returns that computer; this program does not scan the network itself.
_EXPLORER_BROWSE = (
    "$shell = New-Object -ComObject Shell.Application\n"
    "$folder = $shell.BrowseForFolder(0, 'Select a remote machine', 0, 18)\n"
    "if ($null -eq $folder) { exit 2 }\n"
    "$path = [string]$folder.Self.Path\n"
    "$name = [string]$folder.Self.Name\n"
    "if ($path) { [Console]::Out.WriteLine($path) } else { [Console]::Out.WriteLine($name) }\n"
)


def machine_name_from_explorer(text: str) -> str:
    """The computer name in a Network selection. A local folder names nothing.

    ``\\\\host`` and ``\\\\host\\share`` both come back as ``host``. A drive
    path, this machine, or a name ssh could read as an option comes back empty.
    """
    raw = text.strip().strip('"').strip()
    if not raw or any(ord(char) < 32 for char in raw):
        return ""
    if raw.startswith("\\\\"):
        host = raw[2:].split("\\", 1)[0].strip()
    elif "\\" in raw or "/" in raw or ":" in raw:
        return ""
    else:
        host = raw
    if not host or ".." in host or host.casefold() in LOCAL_NAMES:
        return ""
    if _HOST_NAME.fullmatch(host) is None:
        return ""
    return host


def pick_remote_machine() -> str:
    """Open Windows File Explorer on Network and return the chosen computer.

    An empty string is a cancelled dialog. A local folder is refused, because
    it is not a machine ssh can be asked to run on. This computer's own names
    are refused the same way: the dropdown is what adds this machine.
    """
    from rfproteina.disease.grok_bot import windows_powershell

    exe = windows_powershell()
    if not exe:
        raise ValueError("Windows File Explorer is not on this computer.")
    try:
        result = subprocess.run(
            [exe, "-NoProfile", "-Sta", "-Command", _EXPLORER_BROWSE],
            capture_output=True,
            timeout=BROWSE_TIMEOUT_S,
            check=False,
            cwd=_dialog_cwd(),
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("Windows File Explorer did not return a machine.") from None
    if result.returncode == 2:
        return ""
    text = result.stdout.decode("utf-8", errors="replace").strip()
    first = text.splitlines()[0].strip() if text else ""
    host = machine_name_from_explorer(first)
    if host:
        return host
    if not first:
        return ""
    raise ValueError("Choose a computer in Network. A local folder is not a remote machine.")


def _dialog_cwd() -> str | None:
    """A Windows directory the dialog can start from. The repo mount is not one."""
    for candidate in (Path("/mnt/c/Windows"), Path(r"C:\Windows")):
        if candidate.is_dir():
            return str(candidate)
    return None


def _lines(path: Path) -> list[str]:
    """Content lines of one file, without comments. An unreadable file is empty."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            kept.append(line)
    return kept
