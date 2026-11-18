"""Biosecurity screening of synthesis constructs, via Commec (the Common Mechanism).

Before a designed construct is ordered as DNA, it should be screened for sequences
of concern -- toxins and regulated-pathogen genes. This stage wraps Commec, the
International Biosecurity and Biosafety Initiative for Science (IBBIS) Common
Mechanism: a free, open-source (MIT), locally run screener that transfers no data
off the machine.

``commec screen`` runs a nucleotide FASTA through a biorisk HMM search, a
regulated-pathogen taxonomy search (BLASTX/DIAMOND against NCBI nr and BLASTN
against core_nt), and a low-concern clear-back step, writing one
``<record>.screen.json`` per sequence. ``commec flag`` collates those into a CSV
whose recommendation per record is ``Pass``, ``Flag`` or ``Warn``. This module
maps those onto axis states and rolls them up.

The mapping is conservative on purpose: ``Warn`` is treated as a hit, not a clear,
because for a biosecurity gate a missed concern is the costly error. The mapping
is a module constant (:data:`RECOMMENDATION_STATE`) so a reviewer can see and
change it.

Nothing here downloads a database. The biorisk and low-concern databases are
small (IBBIS states the full Commec download is under 10 GB) and fetched with
``commec setup``; the regulated-taxonomy step additionally needs the NCBI ``nr``
and ``core_nt`` databases, which are large and operator-provided -- run with
``--skip-tx`` to screen on the biorisk and low-concern databases alone. If Commec
is not installed, or the database directory is absent, the stage returns
``untested`` with a blocker naming what is missing. See :func:`requirements`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from rfproteina.disease.combinations import rollup_axis
from rfproteina.disease.screening_tools import (
    CLEAR,
    HIT,
    UNTESTED,
    ScreeningInputError,
    alphabet_of,
    axis_record,
    data_dir,
    read_fasta,
    run_tool,
    tool_code_path,
)

COMMEC = "commec"
#: An executable path. It wins over the copy installed under ``DATA_PATH``.
COMMEC_ENV = "COMMEC"
#: The database directory. It wins over ``DATA_PATH/commec/databases``.
COMMEC_DATABASES_ENV = "COMMEC_DATABASES"

#: Commec recommendation -> axis state. ``warn`` maps to a hit deliberately: a
#: lower-confidence concern still needs human review, and for a safety gate the
#: conservative direction is to flag it. ``error`` is not a verdict, so it is
#: untested. Keys are compared case-folded.
RECOMMENDATION_STATE: dict[str, str] = {
    "pass": CLEAR,
    # The installed ``commec flag`` writes this when ``--skip-tx`` skipped taxonomy.
    "pass (skipped taxonomy)": CLEAR,
    "flag": HIT,
    "warn": HIT,
    # ScreenStatus.WARN is the word "Warning", not "Warn".
    "warning": HIT,
    "error": UNTESTED,
}

#: Candidate header names for the record-id column in a ``commec flag`` CSV. The
#: exact spelling is parsed flexibly and should be verified against a real run.
_ID_COLUMNS = ("name", "query", "query_name", "record", "sequence", "sequence_name", "id", "target")
#: Candidate header names for the recommendation column.
# ``flag`` is the overall column in ``screen_pipeline_status.csv``. It is matched
# exactly, before the substring fallback, so ``virus_flag`` is not selected.
_RECOMMENDATION_COLUMNS = (
    "flag",
    "recommendation",
    "recommend",
    "status",
    "outcome",
    "overall",
    "screen_status",
)


class BiosecurityUnavailableError(RuntimeError):
    """Commec or its database is absent. Carries the blocker to record as untested."""

    def __init__(self, blocker: str) -> None:
        super().__init__(blocker)
        self.blocker = blocker


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _install_root() -> Path:
    """``DATA_PATH/commec``: the virtual environment and the 39 GB of databases.

    Neither moved into the checkout. Only Commec's bundled search binaries did,
    and :func:`_tools_root` resolves those separately.
    """
    return data_dir("commec")


def _tools_root() -> Path | None:
    """Commec's bundled HMMER/BLAST/Infernal binaries, wherever they are.

    These are third-party compiled code, so they moved into the checkout's
    ``screening subpackage/commec/tools``; the databases they search did not.
    ``None`` when no copy is on disk, which leaves ``PATH`` untouched.

    Resolved by artifact rather than by directory: ``DATA_PATH/commec`` still
    exists here -- it holds the virtual environment and the databases -- so
    asking for that directory first would shadow the relocated ``tools`` and
    report them missing.
    """
    tools = tool_code_path("commec", "tools")
    return tools if tools is not None and tools.is_dir() else None


def commec_command() -> str | None:
    """The Commec executable, or None when this machine has none.

    ``COMMEC`` wins, then ``DATA_PATH/commec/venv`` (the copy
    ``scripts/install_commec.py`` installs), then ``PATH``.
    """
    named = os.environ.get(COMMEC_ENV, "").strip()
    if named:
        candidate = Path(named)
        if _executable(candidate):
            return str(candidate)
        found = shutil.which(named)
        if found:
            return found
    venv = _install_root() / "venv"
    for relative in ("bin/commec", "Scripts/commec.exe", "Scripts/commec"):
        candidate = venv / relative
        if _executable(candidate):
            return str(candidate)
    return shutil.which(COMMEC)


def commec_databases() -> Path | None:
    """The installed database directory, or None when it is not on disk."""
    named = os.environ.get(COMMEC_DATABASES_ENV, "").strip()
    if named:
        path = Path(named)
        return path if path.is_dir() else None
    path = _install_root() / "databases"
    return path if path.is_dir() else None


def commec_available() -> bool:
    """Whether a Commec executable can be resolved. Checked, never assumed."""
    return commec_command() is not None


def _screen_environment() -> dict[str, str]:
    """Process environment plus the HMMER, BLAST, and Infernal binaries.

    ``scripts/install_commec.py`` fetches databases. The search programs are
    unpacked under ``screening subpackage/commec/tools`` (or, on a machine
    whose code has not been relocated, ``DATA_PATH/commec/tools``) because
    Commec calls ``hmmscan``, ``blastx``, ``blastn``, and ``cmscan`` from
    ``PATH``. BLAST also needs its shared libraries on ``LD_LIBRARY_PATH``.

    When no copy of the tools is on disk, ``PATH`` is left exactly as it was
    rather than prepended with directories that do not exist.
    """
    env = os.environ.copy()
    tools = _tools_root()
    if tools is None:
        return env
    bindirs = [tools / "usr/lib/infernal", tools / "usr/bin"]
    libdirs = [tools / "usr/lib/x86_64-linux-gnu", tools / "usr/lib/ncbi-blast+"]
    bins = [str(path) for path in bindirs if path.is_dir()]
    libs = [str(path) for path in libdirs if path.is_dir()]
    if bins:
        env["PATH"] = os.pathsep.join([*bins, env.get("PATH", "")])
    if libs:
        previous = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*libs, previous] if previous else libs)
    return env


def build_screen_command(
    fasta: Path,
    databases_dir: Path,
    output_json: Path,
    *,
    commec: str = COMMEC,
    skip_tx: bool = True,
    threads: int = 1,
) -> list[str]:
    """The ``commec screen`` argument list. ``--skip-tx`` omits the large NCBI taxonomy step."""
    command = [
        commec,
        "screen",
        "-d",
        str(databases_dir),
        "-o",
        str(output_json),
        "-t",
        str(threads),
    ]
    if skip_tx:
        command.append("--skip-tx")
    command.append(str(fasta))
    return command


def build_flag_command(screen_dir: Path, output_dir: Path, *, commec: str = COMMEC) -> list[str]:
    """The ``commec flag`` argument list.

    The directory is positional. ``-o`` is the directory where Commec writes
    ``screen_pipeline_status.csv``, whose overall column is ``flag``.
    """
    return [commec, "flag", str(screen_dir), "-o", str(output_dir)]


def _header_index(header: list[str], candidates: tuple[str, ...]) -> int | None:
    lowered = [name.strip().casefold() for name in header]
    for candidate in candidates:
        if candidate in lowered:
            return lowered.index(candidate)
    # A looser match for a recommendation column whose header embeds the word.
    for position, name in enumerate(lowered):
        if any(candidate in name for candidate in candidates):
            return position
    return None


def parse_flag_csv(text: str) -> dict[str, str]:
    """Map each record id to its recommendation string from a ``commec flag`` CSV.

    The id and recommendation columns are found by name, case-insensitively,
    because the exact header spelling is Commec's to change. A file whose header
    carries neither raises :class:`ScreeningInputError`, rather than returning an
    empty map that would read as "everything screened clear".
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ScreeningInputError("empty_flag_csv")
    header = next(iter(lines)).split(",")
    id_index = _header_index(header, _ID_COLUMNS)
    rec_index = _header_index(header, _RECOMMENDATION_COLUMNS)
    if id_index is None or rec_index is None:
        raise ScreeningInputError("flag_csv_missing_columns")
    recommendations: dict[str, str] = {}
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) <= max(id_index, rec_index):
            continue
        record_id = fields[id_index].strip()
        if record_id:
            recommendations[record_id] = fields[rec_index].strip()
    return recommendations


def verdict(recommendation: str) -> dict[str, object]:
    """One record's biosecurity axis from its Commec recommendation.

    An unrecognised recommendation is untested, not clear, so a changed Commec
    vocabulary surfaces as "not screened" rather than as a silent pass.
    """
    key = recommendation.strip().casefold()
    state = RECOMMENDATION_STATE.get(key)
    if state is None:
        return axis_record(UNTESTED, None, f"unknown_recommendation:{key or 'blank'}", recommendation=recommendation)
    blocker = None if state != UNTESTED else f"commec_{key}"
    return axis_record(state, None, blocker, recommendation=recommendation)


def biosecurity_screen(
    fasta_path: Path,
    databases_dir: Path,
    work_dir: Path,
    *,
    commec: str = COMMEC,
    skip_tx: bool = True,
    threads: int = 1,
    timeout_s: float = 7200.0,
) -> dict[str, dict[str, object]]:
    """A biosecurity record per record in ``fasta_path``.

    Raises :class:`ScreeningInputError` when the FASTA holds no record. Raises
    :class:`BiosecurityUnavailableError` when Commec is not installed or the
    database directory is absent, so the caller records every input ``untested``
    with that one blocker. A record Commec does not report on is ``untested`` with
    a ``no_commec_verdict`` blocker, never assumed to have passed.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.is_file():
        raise ScreeningInputError("fasta_missing")
    records = read_fasta(fasta_path.read_text(encoding="utf-8"))
    if not records:
        raise ScreeningInputError("no_fasta_records")
    if commec == COMMEC:
        resolved = commec_command()
        if resolved is None:
            raise BiosecurityUnavailableError("commec_not_installed")
    else:
        resolved = commec
    if not Path(databases_dir).is_dir():
        raise BiosecurityUnavailableError("commec_databases_missing")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    screen_out = work_dir / "screen"
    screen_out.mkdir(parents=True, exist_ok=True)
    flag_dir = work_dir / "flags"
    screen_env = _screen_environment()
    screen_result = run_tool(
        build_screen_command(
            fasta_path, Path(databases_dir), screen_out, commec=resolved, skip_tx=skip_tx, threads=threads
        ),
        timeout_s=timeout_s,
        env=screen_env,
    )
    if not screen_result.ok:
        raise BiosecurityUnavailableError(screen_result.blocker or "commec_screen_failed")
    flag_result = run_tool(
        build_flag_command(screen_out, flag_dir, commec=resolved),
        timeout_s=timeout_s,
        env=screen_env,
    )
    if not flag_result.ok:
        raise BiosecurityUnavailableError(flag_result.blocker or "commec_flag_failed")
    csv_out = flag_dir / "screen_pipeline_status.csv"
    text = csv_out.read_text(encoding="utf-8") if csv_out.is_file() else flag_result.stdout
    recommendations = parse_flag_csv(text)
    out: dict[str, dict[str, object]] = {}
    for record_id, sequence in records:
        recommendation = recommendations.get(record_id)
        if recommendation is None:
            out[record_id] = axis_record(
                UNTESTED, None, "no_commec_verdict", input_alphabet=alphabet_of(sequence)
            )
            continue
        record = verdict(recommendation)
        record["input_alphabet"] = alphabet_of(sequence)
        out[record_id] = record
    return out


def rollup(records: dict[str, dict[str, object]]) -> dict[str, object]:
    """One biosecurity axis for the whole batch, by the shared roll-up rule."""
    empty = axis_record(UNTESTED, None, "designs_absent")
    return rollup_axis(list(records), records, empty)


def unavailable_axis(blocker: str) -> dict[str, object]:
    """The axis a caller records when the stage could not run at all."""
    return axis_record(UNTESTED, None, blocker)


def requirements() -> dict[str, object]:
    """Exactly what the Commec stage needs, so an unavailable path is actionable."""
    return {
        "install": "pip install commec  (MIT; https://github.com/ibbis-bio/common-mechanism)",
        "databases_biorisk_benign": (
            "commec setup -d <dir>  (biorisk + low-concern + control-list databases; "
            "IBBIS states the full Commec download is under 10 GB; MIT)"
        ),
        "databases_taxonomy": (
            "NCBI nr (BLASTX/DIAMOND) and core_nt (BLASTN), operator-provided and large; "
            "needed only without --skip-tx"
        ),
        "external_tools": "hmmer, diamond or blast, cmscan (Commec calls these; install per Commec docs)",
        "skip_tx_mode": "run with skip_tx=True to screen on biorisk + low-concern databases alone",
        "input": "nucleotide FASTA (DNA/RNA); back-translate protein designs before screening",
        "alternative": (
            "SecureDNA (securedna.org): free hosted screening via a client with issued keys; "
            "not a local-only free database, and it sends a hashed form off-machine, so Commec is "
            "the local-and-free option implemented here"
        ),
    }
