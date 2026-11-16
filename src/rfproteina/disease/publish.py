"""Keep API keys and personal paths out of files that can be uploaded.

Text that is saved for a later search, training set, or repository upload
is passed through :func:`scrub_for_upload` first. A path inside this checkout
is stored relative to the repository. The key store itself is refused when
its directory is inside a repository.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from contextlib import suppress
from functools import cache
from pathlib import Path

_SECRET = re.compile(
    r"(?i)(?:"
    r"sk-[A-Za-z0-9_\-]{16,}|"
    r"xai-[A-Za-z0-9_\-]{16,}|"
    r"nvapi-[A-Za-z0-9_\-]{16,}|"
    r"hf_[A-Za-z0-9]{16,}|"
    r"ghp_[A-Za-z0-9]{16,}|"
    r"github_pat_[A-Za-z0-9_]{16,}|"
    r"AKIA[0-9A-Z]{16}"
    r")"
)
_BEARER = re.compile(r"(?i)(authorization:\s*bearer\s+)\S+")
# The separator is matched one or twice because an encoded JSON document spells
# a single backslash as two. Scrubbing the values before they are encoded is the
# real defence; this keeps a document that was scrubbed afterwards from reading
# as clean when it is not.
#
# Each of these is case-insensitive because nothing upstream normalises the case
# of a path. ``--out-dir`` is a ``type=Path`` with no ``resolve()``, so whatever
# the user typed is what reaches the archive record: ``c:\users\name\runs``
# stayed exactly that, matched neither the exact prefix comparison below nor a
# pattern written for ``Users``, and the account name was written into
# ``generation.json`` -- the file meant for a later upload. Over-matching is the
# safe direction. On a case-sensitive filesystem ``/HOME/name`` is a different
# directory from ``/home/name``, so replacing it too is wrong in principle, but
# it is wrong towards removing a name that was not personal, never towards
# keeping one that is.
_WIN_USER = re.compile(r"(?i)[A-Za-z]:[/\\]{1,2}Users[/\\]{1,2}[^/\\\s\"']+")
_WSL_USER = re.compile(r"(?i)/mnt/[A-Za-z]/Users/[^/\s\"']+")
_UNIX_HOME = re.compile(r"(?i)/home/[^/\s\"']+")
_REDACTED = "[redacted]"
_HOME = "<home>"


class RepositoryKeyError(Exception):
    """An API key directory was inside a git repository."""


def write_text_atomically(path: Path, text: str) -> None:
    """Replace a file's whole contents, or leave the old contents alone.

    A plain write truncates first, so an interrupted one leaves a half-written
    file where a complete one was. For a screen that is worse than no file: a
    reader cannot tell a truncated verdict from a missing one, and a verdict is
    the thing someone decides what to synthesise on.

    What is guaranteed is the *swap*: a reader sees either the whole previous
    contents or the whole new contents, never a mixture, and a failure anywhere
    leaves the previous contents. ``os.replace`` provides that on POSIX and on
    Windows, unlike ``Path.rename``, which refuses an existing target on
    Windows. The contents are not flushed to the disk itself, so this is not a
    guarantee against power loss -- only against a failed, interrupted or
    concurrent write, which are the ones that happen.

    The temporary file is created in the destination's own directory, because a
    replace across filesystems fails, and it carries the process id so two
    processes writing one directory cannot collide on it.
    """
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        # A successful replace consumed the temp. Anything else leaves it: a
        # failed write, a failed replace, or an interrupt between the two. The
        # cleanup is in `finally` rather than `except OSError` so that a
        # KeyboardInterrupt does not strand it, and its own failure is ignored
        # so it cannot mask whatever went wrong first.
        if temp.exists():
            with suppress(OSError):
                temp.unlink()


def repository_contains(path: Path) -> bool:
    """Whether ``path`` is inside a checkout of this project or any git repo."""
    try:
        current = path.resolve()
    except OSError:
        current = path
    for parent in (current, *current.parents):
        if (parent / ".git").exists() or (parent / ".project-root").is_file():
            return True
    return False


def refuse_repository_key_store(directory: Path) -> None:
    """Raise when keys would be written inside a repository."""
    if repository_contains(directory):
        raise RepositoryKeyError(
            "API keys cannot be saved inside a repository. " "Set RFD4_KEY_DIR to a directory outside this project."
        )


def repository_relative(path: Path) -> str:
    """Return a checkout-relative path when ``path`` lives in this repository.

    A path outside the checkout is returned unchanged so a temporary test
    directory still opens. Machine prefixes are not written into tracked files.
    """
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    root = _repository_root()
    if root is not None:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            return str(resolved)
    return str(path)


def scrub_for_upload(text: str) -> str:
    """Remove API keys and personal paths from text that may be uploaded."""
    cleaned = _SECRET.sub(_REDACTED, text)
    cleaned = _BEARER.sub(r"\1" + _REDACTED, cleaned)
    prefixes = _prefix_pattern(_personal_prefixes())
    if prefixes is not None:
        cleaned = prefixes.sub(_HOME, cleaned)
    cleaned = _WIN_USER.sub(_HOME, cleaned)
    cleaned = _WSL_USER.sub(_HOME, cleaned)
    cleaned = _UNIX_HOME.sub(_HOME, cleaned)
    return cleaned


def scrub_path_prefix(path: str) -> str:
    """Remove a personal prefix from the start of one path, and nothing after it.

    :func:`scrub_for_upload` is for free text, where a home directory can sit
    anywhere in a sentence, so it replaces one wherever it finds it. A path is
    not free text. Only its start can name this machine's user, and replacing a
    ``/home/<name>`` segment further in deletes a component: two run
    directories on a shared mount, ``/data/home/a/run`` and ``/data/home/b/run``,
    both came back as ``/data<home>/run``, and neither string was a path that
    exists. A reader uses the path to tell two runs apart and to name one in
    its next call, so the rest is returned exactly as it was given.
    """
    prefixes = _prefix_pattern(_personal_prefixes())
    if prefixes is not None:
        found = prefixes.match(path)
        # A prefix ends at a separator. Without this the home of ``ann`` would
        # be cut out of ``/home/anna``.
        if found is not None and path[found.end() : found.end() + 1] in {"", "/", "\\"}:
            return _HOME + path[found.end() :]
    for pattern in (_WIN_USER, _WSL_USER, _UNIX_HOME):
        found = pattern.match(path)
        if found is not None:
            return _HOME + path[found.end() :]
    return path


@cache
def _prefix_pattern(prefixes: tuple[str, ...]) -> re.Pattern[str] | None:
    """One case-insensitive alternation over ``prefixes``, longest first.

    An exact ``in``/``replace`` pass could not match a prefix the user spelled
    in another case, which is the whole of the leak described above
    :data:`_WIN_USER`. ``re`` takes the first branch that matches and
    :func:`_personal_prefixes` already orders longest first, so a longer prefix
    still wins over a prefix of itself.

    Cached on the prefix tuple, which is cheap to rebuild but not to compile.
    """
    ordered = [prefix for prefix in prefixes if prefix]
    if not ordered:
        return None
    return re.compile("|".join(re.escape(prefix) for prefix in ordered), re.IGNORECASE)


def scrubbed_json(payload: object, *, indent: int = 2) -> str:
    """JSON text, with keys and personal paths removed, ending in a newline.

    Every string in the payload is scrubbed *before* it is encoded. Scrubbing
    only the finished document misses a Windows path, because ``json.dumps``
    writes one backslash as two and ``C:\\\\Users\\\\name`` does not match a
    pattern written for ``C:\\Users\\name``. A path written with forward
    slashes was scrubbed either way, which is what hid this on a WSL run.

    The encoded document is scrubbed as well, so a value that reaches the
    document without passing through this walk is still covered.
    """
    return scrub_for_upload(json.dumps(_scrubbed_values(payload), indent=indent) + "\n")


def _scrubbed_values(value: object) -> object:
    """``value`` with :func:`scrub_for_upload` applied to every string inside it."""
    if isinstance(value, str):
        return scrub_for_upload(value)
    if isinstance(value, Mapping):
        return {_scrubbed_values(key): _scrubbed_values(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_scrubbed_values(item) for item in value]
    return value


def _personal_prefixes() -> tuple[str, ...]:
    """Absolute prefixes that identify this machine, longest first."""
    found: list[str] = []
    home = Path.home()
    found.extend((str(home), home.as_posix()))
    profile = os.environ.get("USERPROFILE", "").strip()
    if profile:
        found.append(profile)
        found.append(profile.replace("\\", "/"))
    root = _repository_root()
    if root is not None:
        found.extend((str(root), root.as_posix()))
    # A prefix with backslashes is also spelled with doubled backslashes once a
    # JSON document has encoded it, so both spellings are replaced.
    found.extend(prefix.replace("\\", "\\\\") for prefix in tuple(found) if "\\" in prefix)
    unique = [prefix for prefix in found if prefix and prefix not in {".", "/"}]
    unique.sort(key=len, reverse=True)
    return tuple(dict.fromkeys(unique))


@cache
def _repository_root() -> Path | None:
    """The checkout this module lives in, found once per process.

    Cached because it walks the filesystem and every scrubbed string paid for
    that walk. :func:`scrub_for_upload` calls :func:`_personal_prefixes` per
    call and :func:`_scrubbed_values` calls it per string, so a 238-row
    ``returned_structures.json`` -- 5620 strings -- did 5620 walks and took 58
    seconds where encoding the document and scrubbing it once took 28
    milliseconds. The panel pays that before drawing the structures page and
    again on every pass through it. Measured on this checkout: 14.5 ms a walk
    under WSL on ``/mnt``, against 0.003 ms for the rest of the prefix list.

    Caching is safe to the point of being invisible: the answer depends only on
    ``__file__``, which cannot change while the process runs.
    :func:`_personal_prefixes` deliberately stays uncached so that ``HOME`` or
    ``USERPROFILE`` changing is still picked up.
    """
    start = Path(__file__).resolve()
    for parent in start.parents:
        if (parent / ".project-root").is_file() or (parent / ".git").exists():
            return parent
    return None
