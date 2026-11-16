"""Getting an AlphaFold model back, by whichever route is open.

Retrieval is built as several routes rather than one, because the step a person
has already taken differs: they may have the file on disk, or the text on their
clipboard, or only an accession.

* :func:`from_text` -- paste the mmCIF or PDB straight in.
* :func:`from_path` -- a downloaded ``.cif``, ``.pdb``, compressed form, or a
  ``.zip`` holding a model.
* :func:`from_accession` -- a model the AlphaFold database already holds.
* :func:`newest_download` -- the archive that just landed, found without being
  told where it is.

Every result records how it arrived. An overlay captioned as a model from one
service, drawn from a file somebody pasted, would be the page asserting
something nobody checked, so the provenance travels with the coordinates.
"""

from __future__ import annotations

import gzip
import os
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rfproteina.disease.results_compare import chain_labels, looks_like_pdb

#: AlphaFold database, which serves precomputed predictions and nothing else.
AFDB_PREDICTION = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
#: A structure past this is not a single model of one protein. The whole point
#: of these routes is a model of one chain or a small complex.
TEXT_LIMIT = 64 * 1024 * 1024
#: Members of a server archive that are not models, and are expected rather
#: than surprising.
_NOT_MODELS = ("terms_of_use", "job_request", "readme", "license")


@dataclass(frozen=True)
class Retrieved:
    """One retrieved model, or why there is none."""

    text: str = ""
    #: Where it came from, in words, for the caption under the overlay.
    source: str = ""
    #: ``afdb``, ``paste``, ``file`` or ``archive`` -- what kind of route.
    route: str = ""
    chains: tuple[str, ...] = ()
    blocker: str = ""
    #: True only where the content was served by the service it claims to be
    #: from. A pasted or picked file is never verified, and saying so is the
    #: difference between a caption and an assertion.
    verified: bool = False

    @property
    def found(self) -> bool:
        return bool(self.text) and not self.blocker

    def caption(self) -> str:
        if not self.found:
            return _BLOCKER_TEXT.get(self.blocker, self.blocker or "Nothing retrieved.")
        trust = "" if self.verified else " Not verified: this is the file as supplied."
        return f"{self.source}.{trust}"


_BLOCKER_TEXT = {
    "empty": "Nothing was supplied.",
    "not_a_structure": (
        "That is not an mmCIF or PDB structure. A model from the AlphaFold database is the "
        "file behind its cifUrl or pdbUrl; a model from the AlphaFold Server is inside the "
        "archive it hands you."
    ),
    "no_chains": "The structure parsed but named no chain, so there is nothing to overlay.",
    "too_large": "That file is larger than a single model of one protein; it was not read.",
    "not_found": "No file at that path.",
    "unreadable": "That file could not be read.",
    "no_model_in_archive": (
        "The archive held no model. An AlphaFold Server archive carries its models beside "
        "its terms_of_use and job_request; an archive of only those is a job that produced "
        "nothing."
    ),
    "no_accession": "No accession to look up.",
    "afdb_no_entry": (
        "The AlphaFold database holds no model for that accession. A designed binder has "
        "none by construction: the database is keyed by accession and its only "
        "sequence-keyed lookup is an exact checksum match."
    ),
    "afdb_unreachable": "The AlphaFold database did not answer.",
    "afdb_no_model_url": "The database answered but named no model file.",
}


def _structure_record(text: str, *, source: str, route: str, verified: bool) -> Retrieved:
    """Common tail: refuse what is not a structure, and name its chains."""
    if not text.strip():
        return Retrieved(blocker="empty", route=route)
    if len(text) > TEXT_LIMIT:
        return Retrieved(blocker="too_large", route=route)
    if not looks_like_pdb(text) and "_atom_site." not in text:
        return Retrieved(blocker="not_a_structure", route=route)
    chains = chain_labels(text)
    if not chains:
        return Retrieved(blocker="no_chains", route=route)
    return Retrieved(text=text, source=source, route=route, chains=chains, verified=verified)


def from_text(text: str, *, label: str = "Pasted structure") -> Retrieved:
    """A model pasted in as mmCIF or PDB text.

    Never verified, and the caption says so. This route exists precisely
    because it asks nothing of the machine -- no path to type, no archive to
    unpack, no request to make -- so the one thing it cannot do is vouch for
    what arrived.
    """
    return _structure_record(text, source=label, route="paste", verified=False)


def from_path(path: Path, *, opener: Callable[[Path], bytes] | None = None) -> Retrieved:
    """A downloaded model: a structure file, or a server archive holding one."""
    read = opener or (lambda target: target.read_bytes())
    if not path.is_file():
        return Retrieved(blocker="not_found", route="file")
    try:
        size = path.stat().st_size
    except OSError:
        return Retrieved(blocker="unreadable", route="file")
    if size > TEXT_LIMIT:
        return Retrieved(blocker="too_large", route="file")
    if path.suffix.lower() == ".zip":
        return _from_archive(path)
    try:
        raw = read(path)
        if path.suffix.lower() == ".gz":
            raw = gzip.decompress(raw)
    except (OSError, EOFError, gzip.BadGzipFile, MemoryError):
        return Retrieved(blocker="unreadable", route="file")
    return _structure_record(
        raw.decode("utf-8", errors="replace"),
        source=f"Opened {path.name}",
        route="file",
        verified=False,
    )


def _from_archive(path: Path) -> Retrieved:
    """The first model in an AlphaFold Server archive.

    Members are ranked so a ``model_0`` is preferred over a later one and the
    archive's own paperwork is never mistaken for a structure. A member naming
    ``..`` or an absolute path is skipped: an archive is untrusted input, and
    this does not write anything out of it, but it does not read a name that
    was built to escape either.
    """
    try:
        with zipfile.ZipFile(path) as bundle:
            names = [
                name
                for name in bundle.namelist()
                if not name.endswith("/")
                and ".." not in Path(name).parts
                and not Path(name).is_absolute()
                and Path(name).suffix.lower() in {".cif", ".pdb"}
                and not any(skip in Path(name).name.lower() for skip in _NOT_MODELS)
            ]
            if not names:
                return Retrieved(blocker="no_model_in_archive", route="archive")
            names.sort(key=lambda name: (_model_rank(name), name))
            chosen = names[0]
            raw = bundle.read(chosen)
    except (zipfile.BadZipFile, OSError, KeyError, MemoryError):
        return Retrieved(blocker="unreadable", route="archive")
    return _structure_record(
        raw.decode("utf-8", errors="replace"),
        source=f"Read {Path(chosen).name} from {path.name}",
        route="archive",
        verified=False,
    )


def _model_rank(name: str) -> tuple[int, str]:
    """Prefer an explicitly numbered first model, then anything else."""
    lowered = Path(name).name.lower()
    for index in range(5):
        if f"model_{index}" in lowered:
            return index, lowered
    return 9, lowered


def from_accession(
    accession: str,
    *,
    read_json: Callable[[str], object],
    read_bytes: Callable[[str], bytes],
) -> Retrieved:
    """A model the AlphaFold database already holds for this accession.

    Both readers are injected so this needs no network in a test and so the
    caller decides what a request looks like -- the project has one reader for
    public JSON already, with its own rate limiting and its own refusal to
    leak anything about the machine it runs on.

    This is the only route that can call itself verified: the content came from
    the service the caption names, keyed by the accession asked for.
    """
    wanted = accession.strip().upper()
    if not wanted:
        return Retrieved(blocker="no_accession", route="afdb")
    try:
        found = read_json(AFDB_PREDICTION.format(accession=wanted))
    except Exception:
        return Retrieved(blocker="afdb_unreachable", route="afdb")
    record = found[0] if isinstance(found, list) and found else found
    if not isinstance(record, dict):
        return Retrieved(blocker="afdb_no_entry", route="afdb")
    url = record.get("cifUrl") or record.get("pdbUrl")
    if not isinstance(url, str) or not url:
        return Retrieved(blocker="afdb_no_model_url", route="afdb")
    try:
        raw = read_bytes(url)
    except Exception:
        return Retrieved(blocker="afdb_unreachable", route="afdb")
    entry = record.get("entryId")
    version = record.get("latestVersion")
    named = str(entry) if isinstance(entry, str) else wanted
    plddt = record.get("globalMetricValue")
    extra = f", mean pLDDT {plddt:.1f}" if isinstance(plddt, int | float) else ""
    return _structure_record(
        raw.decode("utf-8", errors="replace"),
        source=f"Retrieved {named} from the AlphaFold database (version {version}{extra})",
        route="afdb",
        verified=True,
    )

# ------------------------------------------------- finding it without being told


#: Where a browser drops a download. Derived rather than written down, so no
#: home directory is embedded in this file.
def download_directories() -> tuple[Path, ...]:
    """The folders a downloaded archive is likely to be sitting in.

    Both sides of the fence when running under WSL: the browser that visited
    the server is a Windows browser and saves to the Windows profile, while
    this code reads a Linux path. ``USERPROFILE`` is translated rather than
    guessed at, and a directory that is not there is simply left out.
    """
    found: list[Path] = []
    home = Path.home()
    found.extend((home / "Downloads", home))
    profile = os.environ.get("USERPROFILE", "").strip()
    if profile:
        drive, _, rest = profile.partition(":")
        if len(drive) == 1 and rest:
            mount = Path("/mnt") / drive.lower() / rest.replace("\\", "/").strip("/")
            found.extend((mount / "Downloads", mount))
        found.append(Path(profile) / "Downloads")
    # Under WSL ``USERPROFILE`` is usually not set at all, and the browser that
    # visited the server is a Windows browser saving to a Windows profile. So
    # the mounted profiles are searched by shape rather than by name: no home
    # directory is written into this file, and the shared or template profiles
    # Windows ships are skipped because nobody's download lands there.
    for mount in sorted(Path("/mnt").glob("*/Users/*/Downloads")):
        if mount.parent.name.casefold() in {"default", "default user", "public", "all users"}:
            continue
        found.append(mount)
    # And the one that matters most, asked for rather than assumed. Downloads
    # is a redirectable shell folder and is not always under the profile at
    # all: this machine's points at another drive entirely, so every guess
    # above misses it and an archive that has just landed is reported absent.
    redirected = windows_downloads()
    if redirected is not None:
        found.insert(0, redirected)
    return tuple(
        unique
        for unique in dict.fromkeys(found)
        if _is_directory(unique)
    )


def _is_directory(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def newest_download(
    *,
    directories: Sequence[Path] | None = None,
    within_seconds: float | None = None,
    now: float | None = None,
    service: str = "",
) -> Path | None:
    """The most recently written AlphaFold-looking download, or None.

    So the loop closes without being told where the file is: the server has no
    submission API and its terms close automated access to its pages, which
    leaves exactly two steps a person must take -- upload, and click download.
    Everything after that second click can happen by itself, and this is the
    part that makes it.

    ``within_seconds`` bounds how old a file may be. Unbounded, a stale archive
    from last week would be picked up and overlaid as though it were the fold
    just asked for, which is the wrong answer delivered confidently. The caller
    sets the window; nothing is chosen outside it.
    """
    where = download_directories() if directories is None else tuple(directories)
    stamp = time.time() if now is None else now
    best: tuple[float, Path] | None = None
    for directory in where:
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            allowed = _SERVICE_SUFFIXES.get(service, (".zip", ".cif", ".pdb"))
            if entry.suffix.lower() not in allowed:
                continue
            if not _looks_like_alphafold(entry.name, service):
                continue
            try:
                if not entry.is_file():
                    continue
                written = entry.stat().st_mtime
                if entry.stat().st_size == 0:
                    continue
            except OSError:
                continue
            if within_seconds is not None and stamp - written > within_seconds:
                continue
            if best is None or written > best[0]:
                best = (written, entry)
    return None if best is None else best[1]


#: Names the server's own downloads carry. Matched loosely, because the exact
#: shape is the service's to change, but matched at all -- a folder of
#: unrelated zips is not a place to pick the newest thing and hope.
_ALPHAFOLD_NAMES = ("fold_", "alphafold", "af3", "af-", "model_")
#: Per service, because the two leave differently shaped files and a download
#: handed to the wrong overlay is captioned as something it is not. The server
#: gives ``fold_<job>.zip``; the database gives ``AF-<accession>-F1-model_v6``
#: as a ``.cif`` or ``.pdb``. Matching only the general marks above would let
#: the newest server archive answer a database lookup.
_SERVICE_NAMES = {
    "af3": ("fold_",),
    "af2": ("af-",),
}
_SERVICE_SUFFIXES = {
    "af3": (".zip",),
    "af2": (".cif", ".pdb"),
}


def _looks_like_alphafold(name: str, service: str = "") -> bool:
    lowered = name.lower()
    marks = _SERVICE_NAMES.get(service, _ALPHAFOLD_NAMES)
    return any(mark in lowered for mark in marks)


#: Downloads, as Windows itself records it.
_DOWNLOADS_GUID = "{374DE290-123F-4565-9164-39C4925E467B}"
_SHELL_FOLDERS = (
    "HKCU" + chr(92) + "Software" + chr(92) + "Microsoft" + chr(92) + "Windows"
    + chr(92) + "CurrentVersion" + chr(92) + "Explorer" + chr(92) + "User Shell Folders"
)


def windows_downloads() -> Path | None:
    """Where Windows says Downloads is, as a path readable from here.

    Asked of the registry rather than assumed, because the assumption fails
    silently. Downloads is a redirectable shell folder: on a machine that has
    moved it, there is no ``~/Downloads`` to find, so a guessed location
    reports the archive absent when it is sitting there under another drive.
    Measured on this one, it is on a different drive from the profile.

    Returns None off Windows, or when the answer cannot be read or is not a
    directory. None rather than a guess: the caller would search a guessed
    folder and report an absence it never established.
    """
    if not (sys.platform.startswith("win") or _under_wsl()):
        return None
    try:
        completed = subprocess.run(
            ["reg.exe", "query", _SHELL_FOLDERS, "/v", _DOWNLOADS_GUID],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        if _DOWNLOADS_GUID not in line:
            continue
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        candidate = _as_local_path(os.path.expandvars(parts[2].strip()))
        if candidate is not None and _is_directory(candidate):
            return candidate
    return None


def _under_wsl() -> bool:
    version = Path("/proc/version")
    try:
        return version.is_file() and "microsoft" in version.read_text(
            encoding="utf-8", errors="ignore"
        ).casefold()
    except OSError:
        return False


def _as_local_path(windows_path: str) -> Path | None:
    """A Windows path as this process can open it, on Windows or under WSL."""
    text = windows_path.strip().strip('"')
    if not text:
        return None
    if sys.platform.startswith("win"):
        return Path(text)
    drive, _, rest = text.partition(":")
    if len(drive) != 1 or not drive.isalpha():
        return None
    return Path("/mnt") / drive.lower() / rest.replace(chr(92), "/").strip("/")
