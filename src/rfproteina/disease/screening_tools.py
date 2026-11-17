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

import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

#: The three axis states, matching ``combinations.HIT``/``CLEAR``/``UNTESTED``.
HIT = "hit"
CLEAR = "clear"
UNTESTED = "untested"

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
    """Whether ``name`` is on PATH, checked with :func:`shutil.which`, never assumed."""
    return shutil.which(name) is not None


def run_tool(
    command: Sequence[str],
    *,
    timeout_s: float = 1800.0,
    cwd: Path | None = None,
    stdin_text: str | None = None,
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
