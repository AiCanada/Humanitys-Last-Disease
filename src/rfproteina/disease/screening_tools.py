"""Shared plumbing for the external-tool screens: novelty, off-target, biosecurity.

These three screens wrap command-line bioinformatics tools (Foldseek, MMseqs2,
Commec) the way ``mhc_tables.py`` wraps MHCflurry: a tool is used only when it is
already on PATH, this module installs nothing and downloads nothing, and an
absent tool or database produces an ``untested`` record whose ``blocker`` names
exactly what is missing rather than a guess or a crash.

The large reference databases these tools need are documented, with sizes and
licences, in ``docs/screening-implementation-handoff.md``. They are fetched by an
operator, never by this module.

Every record this family of screens emits is in the shape
``combinations.rollup_axis`` reads: a ``state`` of ``hit``/``clear``/``untested``,
a ``measurement`` number or ``None``, and a ``blocker`` string or ``None``. The
roll-up rule that turns a batch of per-design records into one axis is the shared
one in ``combinations.py``; it is not copied here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

#: The three axis states, matching ``combinations.HIT``/``CLEAR``/``UNTESTED``.
HIT = "hit"
CLEAR = "clear"
UNTESTED = "untested"

#: Overrides :func:`screening_code_root` outright, for a machine that keeps the
#: third-party code somewhere else entirely.
SCREENING_CODE_ENV = "SCREENING_CODE_ROOT"

#: The repository folder that holds third-party screening *code*. The name has
#: a space in it and is therefore not importable; that is deliberate, because
#: nothing here is imported as a Python package -- these are upstream sources
#: and compiled binaries, resolved by path.
SCREENING_CODE_DIRNAME = "screening subpackage"

#: Marker files that identify the checkout root, in priority order.
_ROOT_MARKERS = (".project-root", "pyproject.toml")


def checkout_root() -> Path | None:
    """The repository root, found by walking up for ``.project-root``.

    Deliberately NOT ``Path(__file__).resolve().parents[N]``. A fixed parent
    count is silently wrong the moment a module's depth changes -- it resolves
    to some other directory rather than raising -- and six screens in this
    package previously carried that bug. An upward search for a marker file is
    correct at any depth.

    Returns ``None`` when no marker is found, so a caller falls back rather
    than inventing a path.
    """
    here = Path(__file__).resolve()
    for folder in (here, *here.parents):
        for marker in _ROOT_MARKERS:
            if (folder / marker).exists():
                return folder
    return None


def screening_code_root() -> Path | None:
    """Where third-party screening *code* lives, or ``None`` if nowhere known.

    ``SCREENING_CODE_ROOT`` wins, then ``<checkout>/screening subpackage``.
    The directory is returned only when it exists, so an un-migrated machine
    gets ``None`` and :func:`tool_code_dir` falls back to ``DATA_PATH``.
    """
    named = os.environ.get(SCREENING_CODE_ENV, "").strip()
    if named:
        candidate = Path(named)
        return candidate if candidate.is_dir() else None
    root = checkout_root()
    if root is None:
        return None
    candidate = root / SCREENING_CODE_DIRNAME
    return candidate if candidate.is_dir() else None


def data_root() -> Path | None:
    """``DATA_PATH`` as a path, or ``None`` when it is unset or empty.

    Weights, example inputs, reference tables and databases stay here. Only
    code moved into the checkout, so a tool is resolved from two roots and
    this is the data one.
    """
    root = os.environ.get("DATA_PATH", "").strip()
    return Path(root) if root else None


def data_dir(*parts: str) -> Path:
    """``DATA_PATH`` (else ``<checkout>/data``), with ``parts`` joined on.

    Always returns a path, unlike :func:`data_root`, because every caller needs
    somewhere to look even when nothing is configured; whether anything is
    actually there is a separate question each one answers for itself.

    The fallback comes from :func:`checkout_root` -- an upward search for
    ``.project-root`` -- and not from a fixed ``parents[N]`` count. For a module
    sitting directly in ``src/rfproteina/disease/`` the two agree exactly, so
    this changes no present behaviour. What it removes is a latent silent
    failure: a fixed parent count resolves to some other directory, with no
    error raised, the moment a module's depth changes.
    """
    root = data_root()
    if root is None:
        checkout = checkout_root()
        root = checkout / "data" if checkout is not None else Path("data")
    return root.joinpath(*parts)


def _roots() -> tuple[Path, ...]:
    """The roots to search for a tool, in precedence order.

    ``DATA_PATH`` comes first and the relocated code second, for two reasons.
    ``DATA_PATH`` is set explicitly by an operator, and explicit configuration
    beating an implicit checkout-relative guess is the precedence every other
    resolver in this package already uses. It also keeps tests honest: a test
    that points ``DATA_PATH`` at a temporary directory must see that directory,
    not whatever happens to be installed in the developer's own checkout.
    """
    return tuple(root for root in (data_root(), screening_code_root()) if root is not None)


def tool_code_dir(tool: str) -> Path | None:
    """The directory holding ``tool``'s third-party code, or ``None``.

    Answers "where is it", never "where should it be": a directory is returned
    only when it exists. Keeping both roots in play means a machine whose code
    has not been relocated still resolves its tools, so the relocation can
    happen in either order with no window in which nothing works.

    Use :func:`tool_code_path` instead when a specific file is wanted. A
    relocation leaves the old directory behind, empty, and an empty directory
    satisfies this function while containing nothing.
    """
    for base in _roots():
        candidate = base / tool
        if candidate.is_dir():
            return candidate
    return None


def tool_code_path(tool: str, *relatives: str) -> Path | None:
    """The first of ``relatives`` that exists under any root, else ``None``.

    Each root is searched for every candidate before moving on, so a root that
    merely has the tool's directory does not shadow a root that has the actual
    file. That distinction matters after a relocation, which leaves an empty
    directory at the old location.
    """
    for base in _roots():
        for relative in relatives:
            candidate = base / tool / relative
            if candidate.exists():
                return candidate
    return None


#: The twenty standard residues. How a non-standard residue is handled is each
#: screen's call; the set lives here so the screens agree on what "standard" is.
STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
#: The nucleotide alphabet (DNA and RNA, plus the ``N`` ambiguity code).
NUCLEOTIDES = frozenset("ACGTUN")


class ScreeningInputError(ValueError):
    """Input a screen cannot read: no records, an empty or wrong-alphabet sequence."""


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one external-tool invocation.

    ``ok`` is true only on a clean exit. ``blocker`` is a short machine-readable
    reason on any failure and ``None`` on success, so a caller turns a failure
    straight into an ``untested`` axis record instead of raising.
    """

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    blocker: str | None = None


def tool_available(name: str) -> bool:
    """Whether ``name`` is on PATH, checked with :func:`shutil.which`, never assumed.

    PATH only. A screen whose tool may also be installed under one of the roots
    in :func:`_roots` wants :func:`resolve_tool`, which looks in both.
    """
    return shutil.which(name) is not None


def _is_executable(path: Path) -> bool:
    """A file this process may execute. Both halves matter: a directory named
    like the binary satisfies the first test and not the second."""
    return path.is_file() and os.access(path, os.X_OK)


def resolve_tool(name: str, env_var: str, *relatives: str) -> str | None:
    """The executable for ``name``, or ``None`` when this machine has none.

    One precedence for every external-tool screen, the one
    ``offtarget.mmseqs_command`` established: an explicit ``env_var`` wins, then
    a copy under one of the roots :func:`tool_code_path` searches --
    ``DATA_PATH/<name>`` first, then the checkout's ``screening subpackage`` --
    and ``PATH`` last.

    Here rather than in each screen because PATH-only was not a choice any
    screen made, it was an omission: an installer that puts a binary under
    ``DATA_PATH`` leaves a screen that only asks ``shutil.which`` reporting the
    tool missing while it sits on disk, which is exactly what the novelty screen
    did. ``relatives`` are the paths to try inside the tool's directory, since a
    release tarball may unpack to ``<name>`` or to ``bin/<name>``.

    An ``env_var`` naming something unusable falls through to the search rather
    than failing: it may name a bare command on PATH, which is why
    :func:`shutil.which` is tried on it before giving up on it.
    """
    named = os.environ.get(env_var, "").strip()
    if named:
        candidate = Path(named)
        if _is_executable(candidate):
            return str(candidate)
        found = shutil.which(named)
        if found:
            return found
    if relatives:
        inside = tool_code_path(name, *relatives)
        if inside is not None and _is_executable(inside):
            return str(inside)
    return shutil.which(name)


def run_tool(
    command: Sequence[str],
    *,
    timeout_s: float = 1800.0,
    cwd: Path | None = None,
    stdin_text: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ToolResult:
    """Run a fixed-shape argument list and capture its output.

    ``command`` is always a list and is never passed through a shell, so there is
    no shell for an argument to inject into. A missing executable, a non-zero
    exit, a timeout and an OS error each return a :class:`ToolResult` with
    ``ok=False`` and a blocker rather than raising, because to the caller they
    all mean the same thing: the measurement did not happen.
    """
    if not command:
        return ToolResult(False, -1, "", "", blocker="empty_command")
    exe = command[0]
    if shutil.which(exe) is None:
        return ToolResult(False, 127, "", "", blocker=f"tool_not_installed:{exe}")
    try:
        finished = subprocess.run(
            list(command),
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, -1, "", "", blocker=f"tool_timeout:{exe}")
    except OSError as error:
        return ToolResult(False, -1, "", str(error), blocker=f"tool_failed:{exe}")
    if finished.returncode != 0:
        return ToolResult(
            False,
            finished.returncode,
            finished.stdout,
            finished.stderr,
            blocker=f"tool_error:{exe}:{finished.returncode}",
        )
    return ToolResult(True, 0, finished.stdout, finished.stderr)


def axis_record(
    state: str,
    measurement: int | float | None,
    blocker: str | None,
    **extra: object,
) -> dict[str, object]:
    """One per-design axis record in the shape ``combinations.rollup_axis`` reads."""
    record: dict[str, object] = {"state": state, "measurement": measurement, "blocker": blocker}
    record.update(extra)
    return record


def read_fasta(text: str) -> list[tuple[str, str]]:
    """``(id, sequence)`` pairs from FASTA text, with the ``>`` and any description trimmed.

    The id is the first whitespace-delimited token of the header, which is how a
    search tool reports a query, so a hit maps back onto the record. Blank lines
    are skipped and wrapped sequence lines are joined. Text with no ``>`` header
    returns an empty list rather than guessing one record out of loose lines.
    """
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].split()[0] if line[1:].split() else ""
            chunks = []
        elif header is not None:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def write_fasta(records: Iterable[tuple[str, str]], path: Path) -> int:
    """Write ``(id, sequence)`` records to ``path`` as 60-column FASTA; return the count written.

    An empty sequence is skipped rather than written as a header with no
    residues, which a search tool rejects.
    """
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for name, sequence in records:
            clean = "".join((sequence or "").split()).upper()
            if not clean:
                continue
            handle.write(f">{name}\n")
            for start in range(0, len(clean), 60):
                handle.write(clean[start : start + 60] + "\n")
            count += 1
    return count


def alphabet_of(sequence: str) -> str:
    """``"protein"``, ``"nucleic"`` or ``"empty"`` for a sequence.

    A sequence over the nucleotide alphabet alone reads as nucleic; anything that
    uses a residue outside it reads as protein. This cannot tell a short protein
    of only A/C/G/T from DNA, which is why each screen states which alphabet it
    requires rather than trusting this to disambiguate a ten-mer.
    """
    clean = "".join((sequence or "").split()).upper()
    if not clean:
        return "empty"
    if set(clean) <= NUCLEOTIDES:
        return "nucleic"
    return "protein"
